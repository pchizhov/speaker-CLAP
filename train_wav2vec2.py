import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from torch.utils.tensorboard import SummaryWriter

from utils.custom_datasets import AudioDataset
from utils.helper_functions import get_train_test_dataframes
from utils.speaker_encoder import SpeakerEncoder

def train_wav2vec2(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_directory = os.path.join('runs', "wav2vec2_" + datetime.now().strftime("%b%d_%H_%M_%S"))
    os.makedirs(output_directory)
    writer = SummaryWriter(output_directory)
    print(f'Use CUDA: {hparams["use_cuda"]}')

    train_dataframe, test_dataframe = get_train_test_dataframes(hparams["audio_similarity_df_dir"])
    # train_dataframe = train_dataframe[:int(len(train_dataframe)*0.4)]
    print(len(train_dataframe), len(test_dataframe))
    train_dataloader = DataLoader(AudioDataset(train_dataframe, device, hparams), batch_size=8, shuffle=True)
    test_dataloader = DataLoader(AudioDataset(test_dataframe, device, hparams), batch_size=8, shuffle=True)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(hparams["wav2vec2_model"])
    wav2vec2 = Wav2Vec2Model.from_pretrained(hparams["wav2vec2_model"]).to(device)
    model = SpeakerEncoder(wav2vec2, processor, hparams, device).to(device)
    if hparams["wav2vec2_checkpoint"] != "":
        print("wav2vec2 loaded!")
        model.load_state_dict(torch.load(hparams["wav2vec2_checkpoint"], map_location=device)["state_dict"])
    criterion = nn.L1Loss()
    learning_rate = hparams["audio_learning_rate"]
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    tensorboard_train_step = 0
    tensorboard_val_step = 0

    for epoch in range(hparams["epochs"]):
        iteration = epoch + 1
        print(f'Epoch: {iteration}')

        model.train()
        running_loss = 0
        for i, (audio1, audio2, similarity) in enumerate(train_dataloader):
            optimizer.zero_grad()

            vec_1 = model(audio1)
            vec_2 = model(audio2)
            loss = criterion(torch.cosine_similarity(vec_1, vec_2), similarity)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), tensorboard_train_step)
            tensorboard_train_step += 1
            if i % 200 == 199:
                print("TRAIN - Epoch: {}; Iter: {}; Loss: {:.6f}".format(iteration, i, running_loss/200))
                running_loss = 0
            
            if i % 2000 == 1999:
                print("EVALUATION")
                model.eval()
                with torch.no_grad():
                    running_val_loss = 0
                    for j, (audio1, audio2, similarity) in enumerate(test_dataloader):
                        vec_1 = model(audio1)
                        vec_2 = model(audio2)
                        loss = criterion(torch.cosine_similarity(vec_1, vec_2), similarity)
                        running_val_loss += loss.item()

                    avg_val_loss = running_val_loss/(j+1)
                    print(f'Validation loss {iteration}: {avg_val_loss}')
                    writer.add_scalar("Loss/test", avg_val_loss, tensorboard_val_step)
                    tensorboard_val_step += 1

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss

                        checkpoint_path = os.path.join(output_directory, f'checkpoint_{tensorboard_val_step}')

                        print(f'Saving model and optimizer state at iteration {tensorboard_val_step} to {checkpoint_path}')
                        torch.save({'iteration': tensorboard_val_step,
                                    'state_dict': model.state_dict(),
                                    'criterion': criterion.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'learning_rate': learning_rate}, checkpoint_path)

    checkpoint_path = os.path.join(output_directory, f'overfit_checkpoint_{tensorboard_val_step}')
    print(f'Saving model and optimizer state at iteration {tensorboard_val_step} to {checkpoint_path}')
    torch.save({'iteration': tensorboard_val_step,
                'state_dict': model.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)
