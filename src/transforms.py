import torch
from torchaudio import transforms

class MelSpectrogramTransform:  

    def __init__(
        self,
        sample_rate=44_100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    ):

        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __call__(self, waveform):
        return self.mel_spectrogram(waveform)


class LogMelSpectrogramTransform:  

    def __init__(
        self,
        sample_rate=44_100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    ):

        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __call__(self, waveform):
        mel = self.mel_spectrogram(waveform)

        log_mel = torch.log(
            mel.clamp(min=1e-9)
        )

        return log_mel


class SpectrogramToImage:
    def __call__(self, spectrogram):
        return spectrogram.repeat(3, 1, 1)