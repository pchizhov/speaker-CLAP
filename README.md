# speaker-CLAP

CLAP - Contrastive Language–Audio Pre-training


## Preparation

Before starting the training a user needs to download audio dataset: VCTK and Common Voice.<br>
Audio files referenced in the annotation dataset must be extracted into separate folder for joint training. The path to these extracted files will later be passed as the `--audio_path` argument into the script.<br>
In order to perform standalone training of the audio encoder, a path to the folder with VCTK dataset has to be provided as `--VCTK_dir` argument.

## Running

The most convenient way to run the training is by shell script. An example of the script is the following:

```
python -u main.py \
--model both \
--epochs 200 \
--wav2vec2_checkpoint runs/wav2vec2_Apr03_09_10_17/checkpoint_87 \
--sbert_checkpoint runs/sbert_Apr01_21_09_34/checkpoint_87 \
--audio_learning_rate 1e-5 \
--text_learning_rate 1e-6
```

`--model` argument defines the training method: `both` for joint training; `wav2vec2` for audio training; `sbert` for text encoder training.
`--wav2vec2_checkpoint` and `--sbert_checkpoint` are used to defined paths to fine-tuned models. If the path is not provided, a pre-trained model from HuggingFace will be downloaded.

Another useful arguments:
- `--VCTK_dir` - path to VCTK directory. Used only for audio encoder standalone training.
- `--wav2vec2_model` and `--sbert_model` - paths to HuggingFace models. Example: `"facebook/wav2vec2-large-960h-lv60-self"`
- `--freeze_model` - used to apply freezing of the gradients of either of the models during training. Optional argument. Example: `"wav2vec2"` or `"sbert"`
- `--audio_path` - path to voice recordings from both VCTK and Common Voice dataset. Used for joint training of both encoders as well as during standalone training of the audio encoder.
