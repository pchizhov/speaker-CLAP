import torch
import torch.nn as nn
import torchaudio

class SpeakerEncoder(nn.Module):
    def __init__(self, wav2vec2, processor, hparams, device):
        super(SpeakerEncoder, self).__init__()
        self.wav2vec2 = wav2vec2
        self.processor = processor
        self.device = device
        self.target_sampling_rate = processor.sampling_rate
        
        # linear layer in case Wav2Vec2 has different output dimension from the text encoder. Can be removed if not needed 
        self.linear = nn.Linear(hparams["wav2vec2_output_dim"], hparams["speaker_encoder_dim"])

    def forward(self, paths):
        # transform audio file to numpy array
        wav_tensors = [self._speech_file_to_array_fn(path) for path in paths]

        # preprocess batch of numpy arrays. return pytorch tensors and apply padding based on the "longest" array
        processed_inputs = self.processor(wav_tensors, return_tensors="pt", padding=True, sampling_rate=self.target_sampling_rate).to(self.device)

        # encode preprocessed inputs. returns matrix 'batch_size * padding_length * wav2vec2_out_dim'
        encoded_inputs = self.wav2vec2(**processed_inputs).last_hidden_state

        # pool max values. return matrix 'batch_size * wav2vec2_out_dim'
        pooled_inputs = encoded_inputs.max(dim=1).values

        # can be removed if linear layer is not needed
        output = self.linear(pooled_inputs)

        return output

    def _speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().detach().numpy()
        return speech