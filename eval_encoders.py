import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sentence_transformers import SentenceTransformer

from utils.custom_datasets import AnnotationsDataset
from utils.helper_functions import sentence_features_to_device
from utils.speaker_encoder import SpeakerEncoder

def eval_encoders(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use CUDA: {hparams["use_cuda"]}')

    text_model = SentenceTransformer(hparams["sbert_model"], device=device)
    text_model = text_model.to(device=device)
    text_model.load_state_dict(torch.load(hparams["sbert_checkpoint"], map_location=device)["state_dict"])

    processor = Wav2Vec2Processor.from_pretrained(hparams["wav2vec2_model"])
    wav2vec2 = Wav2Vec2Model.from_pretrained(hparams["wav2vec2_model"]).to(device)
    audio_model = SpeakerEncoder(wav2vec2, processor, hparams, device).to(device)
    audio_model.load_state_dict(torch.load(hparams["wav2vec2_checkpoint"], map_location=device)["state_dict"])

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


    test_dataframe = pd.read_csv(hparams["annotations_CS_df_dir"], index_col=0).reset_index(drop=True)
    test_dataframe.dropna(inplace=True)
    test_dataframe = test_dataframe.reset_index(drop=True)

    print("test_dataframe length:", len(test_dataframe))

    test_dataloader = DataLoader(AnnotationsDataset(test_dataframe, text_model, device, hparams), 
                                batch_size=1, shuffle=True, collate_fn=collate_fn)

    criterion = nn.L1Loss()
    criterion = criterion.to(device=device)

    loss_list = []

    criterion.eval()
    audio_model.eval()
    text_model.eval()
    with torch.no_grad():
        running_val_loss = 0
        for i, (audio, sentence_features, similarity) in enumerate(test_dataloader):
            sentence_features = list(map(lambda batch: sentence_features_to_device(batch, device), sentence_features))

            vec_1 = audio_model(audio)
            vec_2 = torch.cat([text_model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features])
            cos = torch.cosine_similarity(vec_1, vec_2)
            loss = criterion(cos, similarity)
            running_val_loss += loss.item()
            loss_list.append(loss.item())
            print(similarity, cos, loss.item())
            # if loss.item() > 0.9:
            #     print(vec_1[:, :15], vec_2[:, :15])
            #     print()
            # if similarity == 1. and cos < 0.1:
            #     print(audio[0].split("/")[-1], "\n")

        avg_val_loss = running_val_loss/(i+1)
        print(f"Average Loss: {avg_val_loss}")

    print("loss_list length:", len(loss_list))
    
    test_dataframe['loss'] = loss_list
    test_dataframe.to_csv("annotations_control_samples_positive_loss.csv")