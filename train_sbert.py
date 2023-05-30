import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer

from utils.custom_datasets import VoiceDescriptionDataset
from utils.helper_functions import sentence_features_to_device, get_train_test_dataframes

def train_sbert(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_directory = os.path.join('runs', "sbert_" + datetime.now().strftime("%b%d_%H_%M_%S"))
    os.makedirs(output_directory)
    writer = SummaryWriter(output_directory)
    print(f'Use CUDA: {hparams["use_cuda"]}')

    def collate_fn(batch):
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels

    train_dataframe, test_dataframe = get_train_test_dataframes(hparams["text_similarity_df_dir"])
    print(len(train_dataframe), len(test_dataframe))
    train_dataloader = DataLoader(VoiceDescriptionDataset(train_dataframe, hparams), batch_size=128, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(VoiceDescriptionDataset(test_dataframe, hparams), batch_size=128, collate_fn=collate_fn, shuffle=True)

    model = SentenceTransformer(hparams["sbert_model"], device=device)
    model = model.to(device=device)
    if hparams["sbert_checkpoint"] != "":
        print("sbert loaded!")
        model.load_state_dict(torch.load(hparams["sbert_checkpoint"], map_location=device)["state_dict"])

    criterion = nn.L1Loss()
    criterion = criterion.to(device=device)
    learning_rate = hparams["text_learning_rate"]
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    tensorboard_train_step = 0
    max_grad_norm = 1.

    for epoch in range(hparams["epochs"]):
        iteration = epoch + 1
        print(f'Epoch: {iteration}')

        model.train()
        running_loss = 0
        for i, (sentence_features, labels) in enumerate(train_dataloader):
            sentence_features = list(map(lambda batch: sentence_features_to_device(batch, device), sentence_features))
            labels = labels.to(device)
            
            optimizer.zero_grad()

            embeddings = [model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            loss = criterion(torch.cosine_similarity(embeddings[0], embeddings[1]), labels.view(-1))
            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_grad_norm)
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), tensorboard_train_step)
            tensorboard_train_step += 1
            if i % 50 == 49:
                print("TRAIN - Epoch: {}; Iter: {}; Loss: {:.6f}".format(iteration, i, running_loss/49))
                running_loss = 0
                
        print("EVALUATION")
        model.eval()
        with torch.no_grad():
            running_val_loss = 0
            for i, (sentence_features, labels) in enumerate(test_dataloader):
                sentence_features = list(map(lambda batch: sentence_features_to_device(batch, device), sentence_features))
                labels = labels.to(device)

                embeddings = [model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
                loss = criterion(torch.cosine_similarity(embeddings[0], embeddings[1]), labels.view(-1))
                running_val_loss += loss.item()

            avg_val_loss = running_val_loss/(i+1)
            print(f'Validation loss {iteration}: {avg_val_loss}')
            writer.add_scalar("Loss/test", avg_val_loss, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                checkpoint_path = os.path.join(output_directory, f'checkpoint_{iteration}')

                print(f'Saving model and optimizer state at iteration {iteration} to {checkpoint_path}')
                torch.save({'iteration': iteration,
                            'state_dict': model.state_dict(),
                            'criterion': criterion.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'learning_rate': learning_rate}, checkpoint_path)

    checkpoint_path = os.path.join(output_directory, f'overfit_checkpoint_{iteration}')    
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)