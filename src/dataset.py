from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.nn import functional
from torch.utils.data import Dataset
from torchaudio import transforms


class ESC50Dataset(Dataset):
    def __init__(
        self,
        audio_dir,
        metadata_path,
        folds,
        transform=None,
        target_sample_rate=44_100,
        num_samples=220_500,
    ):

        self.audio_dir = Path(audio_dir)
        self.metadata = pd.read_csv(metadata_path)

        self.metadata = self.metadata[
            self.metadata["fold"].isin(folds)
        ].reset_index(drop=True)

        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]
        audio_path = self.audio_dir / row["filename"]

        waveform, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
        )

        waveform = torch.from_numpy(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        waveform = self._convert_to_mono(waveform)
        waveform = self._resample(waveform, sample_rate)
        waveform = self._fix_length(waveform)

        if self.transform is not None:
            waveform = self.transform(waveform)

        label = int(row["target"])

        return waveform, label

    def _convert_to_mono(self, waveform):

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def _resample(self, waveform, sample_rate):

        if sample_rate != self.target_sample_rate:
            resampler = transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            waveform = resampler(waveform)

        return waveform

    def _fix_length(self, waveform):

        current_length = waveform.shape[-1]

        if current_length < self.num_samples:
            padding_amount = self.num_samples - current_length
            waveform = functional.pad(waveform, (0, padding_amount))

        elif current_length > self.num_samples:
            waveform = waveform[..., : self.num_samples]

        return waveform