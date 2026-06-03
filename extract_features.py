
import essentia
from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, MelBands
import numpy as np
import functools

# A pipeline is a sequence of functions that should be applied to each frame
# This function returns the pipelines for the different window sizes
def create_pipelines(sample_rate=44100.0,
                     window_sizes=[1024, 2048, 4096],
                     mel_bands=80,
                     mel_lowest_frequency=27.5,
                     mel_highest_frequency=16000.0):

    pipelines = []

    for window_size in window_sizes:
        # Apply fade-in and fade-out of a frame 
        window = Windowing(size=window_size, type='blackmanharris62')

        # Compute the frequencies the frame
        spectrum = Spectrum(size=window_size)

        # Convert the window_sizesto mel scale for better human perception
        # we use window_size/2 + 1 since we effectively can only deal with
        # half of the frequency spectrum due to aliasing in the sampling
        mel = MelBands(inputSize=(window_size // 2) + 1,
                       numberBands=mel_bands,
                       lowFrequencyBound=mel_lowest_frequency,
                       highFrequencyBound=mel_highest_frequency,
                       sampleRate=sample_rate)

        pipelines.append([window, spectrum, mel])
    return pipelines

def run_pipeline(pipeline, value):
    return functools.reduce(lambda x, f: f(x), pipeline, value)
    
def extract_mel_features(audio_path, pipelines, sample_rate=44100.0, frame_hop=512, window_sizes=[1024, 2048, 4096], log_scale=True):
    loader = MonoLoader(filename=audio_path, sampleRate=sample_rate)
    samples = loader()

    feature_channels = []

    for window_size, pipeline in zip(window_sizes, analyzers):
        frames = FrameGenerator(samples, window_size, frame_hop)
        feats = [run_pipeline(pipeline, frame) for frame in frames]
        feat_channels.append(feats)

    # Transpose to move channels to axis 2 instead of axis 0
    feat_channels = np.transpose(np.stack(feat_channels), (1, 2, 0))

    # Apply numerically-stable log-scaling
    # Value 1e-16 comes from DDC
    if log_scale:
        feat_channels = np.log(feat_channels + 1e-16)

    return feat_channels

