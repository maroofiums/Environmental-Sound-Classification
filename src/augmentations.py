import torch


class ComposeAudio:
    """
    Apply multiple audio transformations sequentially.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, waveform):
        for transform in self.transforms:
            waveform = transform(waveform)

        return waveform


class RandomGain:
    """
    Randomly increase or decrease waveform amplitude.

    gain_range:
        Minimum and maximum gain multiplier.
    """

    def __init__(self, gain_range=(0.8, 1.2), probability=0.5):
        self.gain_range = gain_range
        self.probability = probability

    def __call__(self, waveform):
        if torch.rand(1).item() > self.probability:
            return waveform

        gain = torch.empty(1).uniform_(
            self.gain_range[0],
            self.gain_range[1],
        ).item()

        waveform = waveform * gain

        # Keep values within the valid waveform range
        waveform = waveform.clamp(-1.0, 1.0)

        return waveform


class AddGaussianNoise:
    """
    Add Gaussian noise to the waveform.
    """

    def __init__(self, noise_factor=0.005, probability=0.5):
        self.noise_factor = noise_factor
        self.probability = probability

    def __call__(self, waveform):
        if torch.rand(1).item() > self.probability:
            return waveform

        noise = torch.randn_like(waveform)
        waveform = waveform + self.noise_factor * noise

        waveform = waveform.clamp(-1.0, 1.0)

        return waveform

class RandomTimeShift:
    """
    Shift the waveform left or right by a random number of samples.

    The empty region is filled with zeros.
    """

    def __init__(
        self,
        max_shift_seconds=0.5,
        sample_rate=44_100,
        probability=0.5,
    ):
        self.max_shift_samples = int(
            max_shift_seconds * sample_rate
        )
        self.probability = probability

    def __call__(self, waveform):
        if torch.rand(1).item() > self.probability:
            return waveform

        shift = torch.randint(
            -self.max_shift_samples,
            self.max_shift_samples + 1,
            (1,),
        ).item()

        if shift == 0:
            return waveform

        shifted_waveform = torch.zeros_like(waveform)

        if shift > 0:
            # Move audio to the right
            shifted_waveform[..., shift:] = (
                waveform[..., :-shift]
            )
        else:
            # Move audio to the left
            shifted_waveform[..., :shift] = (
                waveform[..., -shift:]
            )

        return shifted_waveform