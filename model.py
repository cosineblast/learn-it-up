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
    for _file in glob.glob('data/parsed/*.ssc.bin'):
        with open(_file, 'rb') as _f:
            all_stepfiles.append(pickle.load(_f))
    return


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
