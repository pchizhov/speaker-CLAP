import os
from datetime import datetime
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sentence_transformers import SentenceTransformer

from utils.custom_datasets import AnnotationsDataset
from utils.helper_functions import sentence_features_to_device, get_train_test_dataframes
from utils.speaker_encoder import SpeakerEncoder

def train_encoders(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_directory = os.path.join('runs', "encoders_" + datetime.now().strftime("%b%d_%H_%M_%S"))
    os.makedirs(output_directory)
    writer = SummaryWriter(output_directory)
    print(f'Use CUDA: {hparams["use_cuda"]}')

    text_model = SentenceTransformer(hparams["sbert_model"], device=device)
    text_model = text_model.to(device=device)
    if hparams["sbert_checkpoint"] != "":
        print("sbert loaded!")
        text_model.load_state_dict(torch.load(hparams["sbert_checkpoint"], map_location=device)["state_dict"])
    text_learning_rate = hparams["text_learning_rate"]
    text_optimizer = optim.AdamW(text_model.parameters(), lr=text_learning_rate)

    # processor = Wav2Vec2Processor.from_pretrained(hparams["wav2vec2_model"])
    processor = Wav2Vec2FeatureExtractor.from_pretrained(hparams["wav2vec2_model"])
    wav2vec2 = Wav2Vec2Model.from_pretrained(hparams["wav2vec2_model"]).to(device)
    audio_model = SpeakerEncoder(wav2vec2, processor, hparams, device).to(device)
    if hparams["wav2vec2_checkpoint"] != "":
        print("wav2vec2 loaded!")
        audio_model.load_state_dict(torch.load(hparams["wav2vec2_checkpoint"], map_location=device)["state_dict"])
    audio_learning_rate = hparams["audio_learning_rate"]
    audio_optimizer = optim.AdamW(audio_model.parameters(), lr=audio_learning_rate)

    def collate_fn(batch):
        audio_paths = []
        sentence_features = []
        labels = []

        for example in batch:
            audio_path = example[0]
            audio_paths.append(audio_path)

            text = example[1]
            sentence_features.append(text)

            label = example[2]
            labels.append(label)

        labels = torch.tensor(labels, dtype=torch.float32).to(device=device)

        return audio_paths, sentence_features, labels

    train_dataframe, test_dataframe = get_train_test_dataframes(hparams["annotations_df_dir"])
    # train_dataframe = train_dataframe[:int(len(train_dataframe)*0.5)]
    print(len(train_dataframe), len(test_dataframe))
    train_dataloader = DataLoader(AnnotationsDataset(train_dataframe, text_model, device, hparams), 
                                batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(AnnotationsDataset(test_dataframe, text_model, device, hparams), 
                                batch_size=8, shuffle=True, collate_fn=collate_fn)

    criterion = nn.L1Loss()
    criterion = criterion.to(device=device)

    best_val_loss = float('inf')
    tensorboard_train_step = 0
    tensorboard_val_step = 0
    max_grad_norm = 1.

    for epoch in range(hparams["epochs"]):
        iteration = epoch + 1
        print(f'Epoch: {iteration}')

        criterion.train()
        audio_model.train()
        text_model.train()

        if hparams["freeze_model"] == "wav2vec2":
            audio_model.eval()
        if hparams["freeze_model"] == "sbert":
            text_model.eval()
        
        running_loss = 0
        for i, (audio, sentence_features, similarity) in enumerate(train_dataloader):
            sentence_features = list(map(lambda batch: sentence_features_to_device(batch, device), sentence_features))

            audio_optimizer.zero_grad()
            text_optimizer.zero_grad()

            vec_1 = audio_model(audio)
            vec_2 = torch.cat([text_model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features])
            loss = criterion(torch.cosine_similarity(vec_1, vec_2), similarity)
            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_grad_norm)
            if hparams["freeze_model"] != "wav2vec2":
                audio_optimizer.step()
            if hparams["freeze_model"] != "sbert":
                text_optimizer.step()

            del audio, sentence_features, similarity, vec_1, vec_2
            torch.cuda.empty_cache()
            gc.collect()


            writer.add_scalar("Loss/train", loss.item(), tensorboard_train_step)
            tensorboard_train_step += 1

            if i % 200 == 199:
                print("TRAIN - Epoch: {}; Iter: {}; Loss: {:.6f}".format(iteration, i, running_loss/200))
                running_loss = 0

            if i % 2000 == 1999:
                print("EVALUATION")
                criterion.eval()
                audio_model.eval()
                text_model.eval()
                with torch.no_grad():
                    running_val_loss = 0
                    for j, (audio, sentence_features, similarity) in enumerate(test_dataloader):
                        sentence_features = list(map(lambda batch: sentence_features_to_device(batch, device), sentence_features))

                        vec_1 = audio_model(audio)
                        vec_2 = torch.cat([text_model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features])
                        loss = criterion(torch.cosine_similarity(vec_1, vec_2), similarity)
                        running_val_loss += loss.item()

                    avg_val_loss = running_val_loss/(j+1)
                    print(f'Validation loss {iteration}: {avg_val_loss}')
                    writer.add_scalar("Loss/test", avg_val_loss, tensorboard_val_step)
                    tensorboard_val_step += 1

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss

                        audio_checkpoint_path = os.path.join(output_directory, f'wav2vec2_checkpoint_{tensorboard_val_step}')
                        text_checkpoint_path = os.path.join(output_directory, f'sbert_checkpoint_{tensorboard_val_step}')

                        print(f'Saving models and optimizer state at iteration {tensorboard_val_step} to {output_directory}')
                        torch.save({'iteration': tensorboard_val_step,
                                    'state_dict': audio_model.state_dict(),
                                    'criterion': criterion.state_dict(),
                                    'optimizer': audio_optimizer.state_dict(),
                                    'learning_rate': audio_learning_rate}, audio_checkpoint_path)
                        torch.save({'iteration': tensorboard_val_step,
                                    'state_dict': text_model.state_dict(),
                                    'criterion': criterion.state_dict(),
                                    'optimizer': text_optimizer.state_dict(),
                                    'learning_rate': text_learning_rate}, text_checkpoint_path)