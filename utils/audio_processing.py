import torchaudio.transforms as T
import torch

def preprocess_audio(waveform, sample_rate):
    if sample_rate != 16000:
        waveform = T.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_spec = T.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=128)(waveform)
    mel_spec = torch.log1p(mel_spec)

    mel_spec = torch.nn.functional.interpolate(mel_spec.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)
    return mel_spec.squeeze(0)
