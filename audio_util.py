import essentia
from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, MelBands
import numpy as np
import functools

import rich.progress


class AudioFeatureLoader:
    def __init__(
        self,
        sample_rate=44100.0,
        window_sizes=[1024, 2048, 4096],
        mel_bands=80,
        mel_lowest_frequency=27.5,
        mel_highest_frequency=16000.0,
        frame_hop=441,
        log_scale=True,
        use_tqdm=False,
    ):

        self.sample_rate = sample_rate
        self.window_sizes = window_sizes
        self.mel_bands = mel_bands
        self.mel_lowest_frequency = mel_lowest_frequency
        self.mel_highest_frequency = mel_highest_frequency
        self.frame_hop = frame_hop
        self.log_scale = log_scale
        self.use_tqdm = use_tqdm

        self.pipelines = self._create_pipelines()

    # A pipeline is a sequence of functions that should be applied to each frame
    # This function returns the pipelines for the different window sizes
    def _create_pipelines(self):
        pipelines = []

        for window_size in self.window_sizes:
            # Apply fade-in and fade-out of a frame
            window = Windowing(size=window_size, type="blackmanharris62")

            # Compute the frequencies the frame
            spectrum = Spectrum(size=window_size)

            # Convert the window_sizesto mel scale for better human perception
            # we use window_size/2 + 1 since we effectively can only deal with
            # half of the frequency spectrum due to aliasing in the sampling
            mel = MelBands(
                inputSize=(window_size // 2) + 1,
                numberBands=self.mel_bands,
                lowFrequencyBound=self.mel_lowest_frequency,
                highFrequencyBound=self.mel_highest_frequency,
                sampleRate=self.sample_rate,
            )

            pipelines.append([window, spectrum, mel])
        return pipelines

    def load(self, audio_path):
        loader = MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
        samples = loader()

        feature_channels = []

        for window_size, pipeline in zip(self.window_sizes, self.pipelines):
            frames = FrameGenerator(samples, window_size, self.frame_hop)

            frames = rich.progress.track(frames, description='[bold][red]OOOOOOO[/red][/bold]') if self.use_tqdm else frames

            feats = [_run_pipeline(pipeline, frame) for frame in frames]
            feature_channels.append(feats)

        # Transpose to move channels to axis 2 instead of axis 0
        feat_channels = np.transpose(np.stack(feature_channels), (1, 2, 0))

        # Apply numerically-stable log-scaling
        # Value 1e-16 comes from DDC
        if self.log_scale:
            feat_channels = np.log(feat_channels + 1e-16)

        return feat_channels.astype(np.float32)


def _run_pipeline(pipeline, value):
    return functools.reduce(lambda x, f: f(x), pipeline, value)

def resample_features(frames, beat_start_end_times, frames_per_beat=32, radius=0, frames_per_second=100):
    result = []

    starts = np.array([start for start, _ in beat_start_end_times])
    ends = np.array([end for _, end in beat_start_end_times])

    # TODO: use min of frames as default_value instead of log(eps)
    result = _resample_features_at_beat(frames, starts, ends, radius, frames_per_beat, frames_per_second, default_value=np.log(1e-16))

    return np.transpose(result, (1, 0, 2, 3))



def _resample_features_at_beat(frames, start, end, radius, frames_per_beat, frames_per_second, default_value):
    frame_count = frames.shape[0]
    assert frame_count > 0

    start, end = (start*frames_per_second).astype(int), (end*frames_per_second).astype(int)

    indices = np.linspace(start - radius, end + radius, num = frames_per_beat, endpoint = False).astype(int)

    clipped_indices = np.clip(indices, 0, frame_count - 1)

    bad_indices = (indices < 0) | (indices >= frame_count)

    result = frames[clipped_indices]

    result[bad_indices] = default_value

    return result
