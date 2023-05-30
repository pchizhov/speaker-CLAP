import os

import torch
from torch.utils.data import Dataset
from sentence_transformers import InputExample

class AudioDataset(Dataset):
    def __init__(self, dataframe, device, hparams):
        self.dataframe = dataframe
        self.device = device
        self.hparams = hparams

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio1 = self.dataframe["audio1"].iloc[idx]
        audio_path1 = os.path.join(self.hparams["audio_path"], audio1)
        speaker1 = self.dataframe["speaker1"].iloc[idx]

        audio2 = self.dataframe["audio2"].iloc[idx]
        audio_path2 = os.path.join(self.hparams["audio_path"], audio2)
        speaker2 = self.dataframe["speaker2"].iloc[idx]
        similarity = torch.tensor(
            1., dtype=torch.float32) if speaker1 == speaker2 else torch.tensor(0., dtype=torch.float32)

        return self._fix_path(audio_path1), self._fix_path(audio_path2), similarity.to(device=self.device)

    def _fix_path(self, path):
        if os.path.isfile(path[:-4] + ".mp3"):
            return path[:-4] + ".mp3"
        else:
            vctk_speaker_id = path.split("_")[1].split("/")[-1]
            src_folder = self.hparams["VCTK_dir"]
            return os.path.join(src_folder, vctk_speaker_id, path[:-4].split("/")[-1] + "_mic1.flac")
        

class VoiceDescriptionDataset(Dataset):
    def __init__(self, dataframe, hparams):
        self.dataframe = dataframe
        self.annotation_1 = dataframe["transcription1"]
        self.annotation_2 = dataframe["transcription2"]
        self.similarity = dataframe["similarity"]
        self.hparams = hparams

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        annotation_1 = self.annotation_1.iloc[idx]
        annotation_2 = self.annotation_2.iloc[idx]

        speaker1 = self.dataframe["speaker1"].iloc[idx]
        speaker2 = self.dataframe["speaker2"].iloc[idx]
        similarity = torch.tensor(
            1.) if speaker1 == speaker2 else torch.tensor(0.)

        return InputExample(texts=[annotation_1, annotation_2], label=similarity)
    

class AnnotationsDataset(Dataset):
    def __init__(self, dataframe, text_model, device, hparams):
        self.device = device
        self.text_model = text_model
        self.dataframe = dataframe
        self.audio_path = dataframe["audio"]
        self.transcription = dataframe["transcription"]
        self.similarity = dataframe["similarity"]
        self.hparams = hparams

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_path = self.audio_path.iloc[idx]
        audio_path = os.path.join(self.hparams["audio_path"], audio_path)
        return audio_path, self.text_model.tokenize([self.transcription.iloc[idx]]), torch.tensor(self.similarity.iloc[idx], device=self.device)