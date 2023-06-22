import argparse
import torch

from train_both_encoders import train_encoders
from train_sbert import train_sbert
from train_wav2vec2 import train_wav2vec2
from eval_encoders import eval_encoders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', help="both / wav2vec2 / sbert", default="both", type=str)
    parser.add_argument('--freeze_model', nargs='?', help="wav2vec2 / sbert", default="", type=str)
    parser.add_argument('--epochs', nargs='?', default=100, type=int)

    parser.add_argument('--annotations_df_dir', nargs='?', default="/gpfs/space/home/lastovko/train_encoders/datasets/annotations_similarity_df.csv", type=str)
    parser.add_argument('--annotations_CS_df_dir', nargs='?', default="/gpfs/space/home/lastovko/train_encoders/datasets/annotations_control_samples_positive.csv", type=str)
    parser.add_argument('--audio_similarity_df_dir', nargs='?', default="/gpfs/space/home/lastovko/train_encoders/datasets/audio_similarity_df.csv", type=str)
    parser.add_argument('--text_similarity_df_dir', nargs='?', default="/gpfs/space/home/lastovko/train_encoders/datasets/text_similarity_df.csv", type=str)
    parser.add_argument('--audio_path', nargs='?', default="/gpfs/space/home/lastovko/AUDIO_DATA/clips", type=str)
    parser.add_argument('--VCTK_dir', nargs='?', default="/gpfs/space/home/lastovko/VCTK_data/wav48_silence_trimmed/", type=str)

    parser.add_argument('--wav2vec2_model', nargs='?', default="facebook/wav2vec2-large-960h-lv60-self", type=str)
    parser.add_argument('--wav2vec2_checkpoint', nargs='?', default="", type=str)
    parser.add_argument('--sbert_model', nargs='?', default="all-mpnet-base-v2", type=str)
    parser.add_argument('--sbert_checkpoint', nargs='?', default="", type=str)

    parser.add_argument('--wav2vec2_output_dim', nargs='?', default=1024, type=int)
    parser.add_argument('--speaker_encoder_dim', nargs='?', default=768, type=int)

    parser.add_argument('--text_learning_rate', nargs='?', default=1e-7, type=float)
    parser.add_argument('--audio_learning_rate', nargs='?', default=4e-6, type=float)

    args = parser.parse_args()
    hparams = vars(args)
    hparams["use_cuda"] = torch.cuda.is_available()

    if hparams["freeze_model"] == "sbert":
        hparams["text_learning_rate"] = 0.
    if hparams["freeze_model"] == "wav2vec2":
        hparams["audio_learning_rate"] = 0.

    return hparams

if __name__ == '__main__':
    hparams = parse_args()

    if hparams["model"] == "both":
        print(hparams["sbert_checkpoint"])
        print(hparams["wav2vec2_checkpoint"])
        train_encoders(hparams)
    elif hparams["model"] == "wav2vec2":
        train_wav2vec2(hparams)
    elif hparams["model"] == "sbert":
        train_sbert(hparams)
    elif hparams["model"] == "eval":
        eval_encoders(hparams)
