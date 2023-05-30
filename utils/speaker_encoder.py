import torch
import torch.nn as nn
import torchaudio

class SpeakerEncoder(nn.Module):
    def __init__(self, wav2vec2, processor, hparams, device):
        super(SpeakerEncoder, self).__init__()
        self.wav2vec2 = wav2vec2
        self.processor = processor
        self.device = device
        
        # self.target_sampling_rate = processor.feature_extractor.sampling_rate
        self.target_sampling_rate = processor.sampling_rate
        
        self.linear = nn.Linear(hparams["wav2vec2_output_dim"], hparams["speaker_encoder_dim"])

    def forward(self, paths):
        # batch_size = len(paths)
        wav_tensors = [self._speech_file_to_array_fn(path) for path in paths]
        
        # processed_inputs = [self.processor(wav_tensor, return_tensors="pt", padding="longest", sampling_rate=self.target_sampling_rate) for wav_tensor in wav_tensors]
        processed_inputs = self.processor(wav_tensors, return_tensors="pt", padding=True, sampling_rate=self.target_sampling_rate).to(self.device)
        
        # input_values = [processed_input.input_values.to(self.device) for processed_input in processed_inputs]
        # attention_masks = [processed_input.attention_mask.to(self.device) for processed_input in processed_inputs]
        
        # model_output = torch.cat([self.wav2vec2(input_values[i], attention_mask=attention_masks[i]).last_hidden_state.max(dim=1).values for i in range(batch_size)])
        output = torch.cat([self.wav2vec2(**processed_inputs).last_hidden_state.max(dim=1).values])

        output = self.linear(output)

        return output

    def _speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        # speech = resampler(speech_array).squeeze().to(self.device)
        speech = resampler(speech_array).squeeze().detach().numpy()
        return speech