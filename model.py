import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import glob
    import pickle
    import numpy as np

    return F, glob, nn, np, pickle, torch


@app.cell
def _(torch):
    print('HasCuda:',torch.cuda.is_available() )
    return


@app.cell(hide_code=True)
def _(glob, pickle):
    all_stepfiles = []
    stepfile_paths = glob.glob('data/parsed/*.ssc.bin')
    for _path in stepfile_paths:
        with open(_path, 'rb') as _f:
            all_stepfiles.append(pickle.load(_f))
    return all_stepfiles, stepfile_paths


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    def get_feature_path_for(refined_stepfile_path):
        assert str(refined_stepfile_path).endswith(".ssc.bin")
        return Path("data/features") / (Path(Path(refined_stepfile_path).stem).stem + ".feat.bin")


    return (get_feature_path_for,)


@app.cell
def _(pickle):
    class LoadFeaturesCached():
        def __init__(self):
            self.cache = {}

        def __call__(self, path):
            if path in self.cache:
                return self.cache[path]

            with open(path, 'rb') as f:
                features = pickle.load(f)
                self.cache[path] = features
                return features


    return (LoadFeaturesCached,)


@app.cell
def _(get_feature_path_for, np, torch):
    FRAMES_PER_SECOND = 100 

    from math import floor
    from collections import namedtuple 
    import itertools

    ChartStats = namedtuple('ChartStats', ['len_frames', 'first_frame_index', 'last_frame_index'])
    StepfileStats = namedtuple('StepfileStats', ['len_frames', 'chart_stats'])


    # TODO: use default value as minimum value from features for each file, since
    # files have varying silences.
    DEFAULT_VALUE = np.log(1e-16)

    class PIUC_CNN_Onset_Dataset(torch.utils.data.Dataset):
    
        def __len__(self):
            return self.len_frames


        
        def __init__(self, stepfiles, paths, loader):
            all_features = [loader(get_feature_path_for(path)) for path in paths]
        
            def get_chart_stats(chart, features):
                first_frame_index = floor(chart.steps[0].time_in_seconds * FRAMES_PER_SECOND)
                last_frame_index = floor(chart.steps[-1].time_in_seconds * FRAMES_PER_SECOND)

                len_frames = last_frame_index - first_frame_index + 1

                assert len(chart.steps) >= 2
                assert first_frame_index >= 0
                assert last_frame_index < features.shape[0]
                assert first_frame_index < last_frame_index

                # inclusive
                return ChartStats(len_frames, first_frame_index, last_frame_index)
            
            def get_stepfile_stats(stepfile, features):
                assert len(stepfile.charts) > 0

                chart_stats = [get_chart_stats(chart, features) for chart in stepfile.charts]
                len_frames = sum(stat.len_frames for stat in chart_stats)
            
                return StepfileStats(len_frames, chart_stats)


            stepfile_stats = [get_stepfile_stats(stepfile, features) 
                              for features, stepfile in zip(all_features, stepfiles)]
            len_frames = sum(stat.len_frames for stat in stepfile_stats) 
        
            self.stepfile_stats = stepfile_stats
            self.len_frames = len_frames
            self.loader = loader
            self.all_features = all_features 
            self.stepfiles = stepfiles

            self.stepfile_len_sums = list(itertools.accumulate(stats.len_frames for stats in stepfile_stats))
            self.stepfile_lens = list((stats.len_frames for stats in stepfile_stats))


        

        def _get_target_chart(self, file, file_stats, target_index, total):
            target_chart = None
            for chart, chart_stats in zip(file.charts, file_stats.chart_stats):
                if target_index < total + chart_stats.len_frames:
                    target_chart = (chart, chart_stats)
                    break
                total += chart_stats.len_frames

            assert target_chart is not None
            return target_chart, total


        def _get_target_stepfile(self, target_index):
            if target_index < self.stepfile_len_sums[0]:
                return (self.stepfiles[0], self.stepfile_stats[0], self.all_features[0]), 0

            l = 0
            r = len(self.stepfile_len_sums)-1

            assert not (target_index < self.stepfile_len_sums[l])
            assert target_index < self.stepfile_len_sums[r]

            while l+1 != r:
                m = (l + r) // 2

                if target_index < self.stepfile_len_sums[m]:
                    r = m
                else:
                    l = m
                
            assert not (target_index < self.stepfile_len_sums[l])
            assert target_index < self.stepfile_len_sums[r]

            return (self.stepfiles[r], self.stepfile_stats[r], self.all_features[r]), self.stepfile_len_sums[l]

        def _get_frame_context(self, file_features, feature_index):
            context_radius = 7
            # +1 because it is inclusive
            indices = np.arange(feature_index-context_radius, feature_index+context_radius+1).astype(int)
            indices_bad = (indices < 0) | (indices > file_features.shape[0])
            indices_clipped = np.clip(indices, 0, file_features.shape[0])
            result = file_features[indices_clipped]
            result[indices_bad] = DEFAULT_VALUE
            return result

        def __getitem__(self, target_index):
            assert isinstance(target_index, int)
            assert target_index < len(self)

            (file, file_stats, file_features), total_before_file = self._get_target_stepfile(target_index)
            # we now know what stepfile the target index is, now we need to pick the right chart

            (chart, chart_stats), total_before_chart = self._get_target_chart(file, file_stats, target_index, total_before_file)
            # we now know what chart the target index is, now we need to pick the right frame

            feature_index = target_index - total_before_chart + chart_stats.first_frame_index

            frames = self._get_frame_context(file_features, feature_index)

        

            return frames

    return (PIUC_CNN_Onset_Dataset,)


@app.cell
def _(dataset):
    s0 = dataset[0]
    s0.shape
    # np.float32(-4397.8184)
    return (s0,)


@app.cell
def _(dataset, s0):
    import tqdm

    for i in tqdm.tqdm(range(0, len(dataset)), total=len(dataset)):
        break
        try:
            _d = dataset[i]
        except IndexError as e:
            print(e)
            print(i)
            break
    
        if _d.shape != s0.shape:
            raise Exception('Expected database[{}].shape to equal {}, got {}', i, s0.shape, _d.shape)
        
        
    


    return


@app.cell
def _(dataset):
    len(dataset)
    # np.float64(14124822.0)
    return


@app.cell
def _(LoadFeaturesCached):
    loader = LoadFeaturesCached()
    return (loader,)


@app.cell
def _(PIUC_CNN_Onset_Dataset, all_stepfiles, loader, stepfile_paths):
    dataset = PIUC_CNN_Onset_Dataset(all_stepfiles, stepfile_paths, loader)
    return (dataset,)


@app.cell
def _(nn, torch):
    class PumpItUpConvolutionCNNOnset(nn.Module):
        def __init__(self):
            super().__init__()

            # Input: 
            # - X: (Batch x 3 x 15 x 80) tensor
            # - Difficuly: (Batch x 25) tensor representing the one-hot difficulty of the chart
            # Where:
            # - 15 is the context window length, 7 frames before target frame, target frame, 7 frames after target frame
            # - 3 is the number of audio channels 
            # - 80 are the actiavtion values for each frequency, measured in logarithm, with frequency in mel scale
            # Output: (Batch) tensor containing the [0,1] value for whether a frame should be placed at this position

            self.convolution = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=10,
                    kernel_size=(7,3)
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3)),
                nn.Conv2d(
                    in_channels=10,
                    out_channels=20,
                    kernel_size=(3,3)
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3)),
            )

            # 1120 checked from manual obvservation of previous layer
            cnn_output_len_flattened = 1120
            max_difficulty = 25

            self.mlp = nn.Sequential(
                nn.Linear(in_features=cnn_output_len_flattened + max_difficulty, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=1)
            )

        def forward(self, x, difficulty):
            # Batch x 3 x 15 x 80
            convolved = self.convolution(x)

            # Batch x 20 x T x F
            flattened = torch.flatten(convolved, start_dim=1)

            # Batch x (20 * T * F)
            with_difficulty = torch.concat([flattened, difficulty], dim=1)

            # Batch x (20 * T * F + 25)
            linear_result = self.mlp(with_difficulty)

            # Batch x 1
            return nn.functional.sigmoid(linear_result)

    return (PumpItUpConvolutionCNNOnset,)


@app.cell
def _(PumpItUpConvolutionCNNOnset):
    cnn_model = PumpItUpConvolutionCNNOnset()
    return (cnn_model,)


@app.cell
def _(cnn_difficulty_data, cnn_model, cnn_sample_data):
    cnn_model(cnn_sample_data, cnn_difficulty_data)
    return


@app.cell
def _(np, torch):
    cnn_sample_data = torch.tensor(np.random.randn(3, 15, 80)[None, :]).type(torch.float32)
    cnn_sample_data.shape
    return (cnn_sample_data,)


@app.cell
def _(F, torch):
    cnn_difficulty_data = F.one_hot(torch.tensor([15]), num_classes=25).type(torch.float32)
    cnn_difficulty_data.shape
    return (cnn_difficulty_data,)


@app.cell
def _(nn, torch):
    class PumpItUpConvolutionOnset(nn.Module):
        def __init__(self):
            super().__init__()

            # Input: 
            # - X: (Batch x UnrollLength x 3 x 15 x 80) 
            # - Difficuly: (Batch x 25)
            # Output: (Batch x UnrollLength) 
            # See PumpItUpConvolutionCNNOnset for more info on these numbers

            self.convolution = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=10,
                    kernel_size=(7,3)
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3)),
                nn.Conv2d(
                    in_channels=10,
                    out_channels=20,
                    kernel_size=(3,3)
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3)),
            )

            cnn_output_len_flattened = 1120
            max_difficulty = 25
            rnn_size = 200
            unroll_length = 100

            self.rnn_projection = nn.Linear(
                in_features=cnn_output_len_flattened+max_difficulty,
                out_features=rnn_size
            )

            self.lstm = nn.LSTM(
                input_size=rnn_size,
                hidden_size=rnn_size,
                num_layers=2,
                batch_first=True,
                dropout=0.5
            )

            self.mlp = nn.Sequential(
                nn.Linear(in_features=rnn_size, out_features=256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=128, out_features=1)
            )

        def forward(self, x, difficulty):
            assert len(x.shape) == 5

            batch = x.shape[0]
            unroll_length = x.shape[1]
            assert x.shape[2] == 3
            assert x.shape[3] == 15
            assert x.shape[4] == 80

            assert len(difficulty.shape) == 2
            assert difficulty.shape[0] == batch
            assert difficulty.shape[1] == 25

            # convolution functions cannot handle input with arbitrary shape prefixes,
            # so we need to flatten the batch and the unroll, and then unflatten it

            # Batch x UnrollLength x 3 x 15 x 80
            _x = torch.flatten(x, start_dim=0, end_dim=1)
            # (Batch * UnrollLength) x 3 x 15 x 80 
            _convolved = self.convolution(_x)
            # (Batch * UnrollLength) x 20 x T x F
            convolved = torch.unflatten(_convolved, 0, (batch, unroll_length))

            # Batch x UnrollLength x 20 x T x F
            flattened = torch.flatten(convolved, start_dim=2)

            # Difficulty: Batch x 25
            repeated_difficulty = torch.reshape(difficulty, (batch, 1, 25)).repeat((1, unroll_length, 1))

            # Batch x UnrollLength x (20 * T * F)
            with_difficulty = torch.concat([flattened, repeated_difficulty], dim=2)

            # Batch x UnrollLength x (20 * T * F + 25)
            projected = self.rnn_projection(with_difficulty)

            # Batch x UnrollLength x RNNSize
            after_lstm, _ = self.lstm(projected)

            # Batch x UnrollLength x RNNSize
            _after_lstm = torch.flatten(after_lstm, start_dim=0, end_dim=1)
            # (Batch * UnrollLength) x RNNSize
            _linear_result = self.mlp(_after_lstm)
            # (Batch * UnrollLength) x 1
            linear_result = torch.reshape(_linear_result, (batch, unroll_length))

            # Batch x UnrollLength
            return nn.functional.sigmoid(linear_result)


    return (PumpItUpConvolutionOnset,)


@app.cell
def _(full_difficulty_data, full_model, full_model_data):
    full_model(full_model_data, full_difficulty_data)
    return


@app.cell
def _(PumpItUpConvolutionOnset):
    full_model = PumpItUpConvolutionOnset()
    return (full_model,)


@app.cell
def _(np, torch):
    full_model_data = torch.tensor(np.random.randn(1, 100, 3, 15, 80)).type(torch.float32)
    full_model_data.shape
    return (full_model_data,)


@app.cell
def _(F, torch):
    full_difficulty_data = F.one_hot(torch.tensor([15]), num_classes=25).type(torch.float32)
    full_difficulty_data.shape
    return (full_difficulty_data,)


@app.cell
def _(full_difficulty_data, torch):
    torch.reshape(full_difficulty_data, (1, 1, 25)).repeat((1, 100, 1)).shape
    return


if __name__ == "__main__":
    app.run()
