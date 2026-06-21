
import torch.nn as nn
import numpy as np
import torch

class PumpItUpConvolutionCNNOnset(nn.Module):
    def __init__(self, dropout=0.5, channel_is_last=False):
        super().__init__()

        # Input: 
        # - X: (Batch x 3 x 15 x 80) tensor
        # - Difficuly: (Batch x 25) tensor representing the one-hot difficulty of the chart
        # Where:
        # - 15 is the context window length, 7 frames before target frame, target frame, 7 frames after target frame
        # - 3 is the number of audio channels 
        # - 80 are the actiavtion values for each frequency, measured in logarithm, with frequency in mel scale
        # Output: (Batch) tensor containing the log-odds of the [0,1] value for whether a frame should be placed at this position
        #
        # Additionally, if X is of the shape (Batch x Unroll x 3 x 15 x 80), it is flattened to (Batch * Unroll x 3 x 15 x 80) for convenience.
        # 
        # Parameters:
        # - dropout: Dropout to use after fully connected linear layers.
        # - channel_is_last: True if the channel dimension of the input tensor X is the last one of the tensor, instead of the first.
        # In this case, the input tensor is of shape (Batch x 15 x 80 x 3)

        self.channel_is_last = channel_is_last

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
            nn.Dropout(dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, x, difficulty):
        # (Batch x 3 x 15 x 80), (Batch x 25)
        # or (Batch x Unroll x 3 x 15 x 80), (Batch x 25)
        if len(x.shape) == 5:
            unroll = x.shape[1]
            x = torch.flatten(x, start_dim=0, end_dim=1)
            difficulty = difficulty.repeat_interleave(unroll, dim=0)

        if self.channel_is_last:
            # (Batch, 15, 80, 3) -> (Batch, 3, 15, 80)
            x = x.transpose(1, 3).transpose(2, 3)
            
        # Batch x 3 x 15 x 80
        convolved = self.convolution(x)

        # Batch x 20 x T x F
        flattened = torch.flatten(convolved, start_dim=1)

        # Batch x (20 * T * F)
        with_difficulty = torch.concat([flattened, difficulty], dim=1)

        # Batch x (20 * T * F + 25)
        linear_result = self.mlp(with_difficulty)

        # Batch x 1
        output = torch.flatten(linear_result)

        # Batch
        return output

# unused for now, cnn alone gives us ok performance, we will try this in the future.
# currently unused because of the different input tensor shape, for which we have not
# yet implemented a dataset class for
class PumpItUpConvolutionLSTMOnset(nn.Module):
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
        return linear_result


from dataclasses import dataclass

class PumpItUpConvolutionSelectionLSTM(nn.Module):

    @dataclass
    class Settings:
        rnn_size: int = 128
        rnn_layers: int = 2
        delta_features: int = 3
        
    def __init__(self, settings = Settings()):
        super().__init__()

        # Input: 
        # - X: (Batch x UnrollLength x 5 x 4) tensor with the bag-of-arrows representation of the steps
        # - DeltaTime: (Batch x UnrollLength x 3) value representing the amout of time in seconds since the last and next step, and whether it is the first step or not
        # Where:
        # - 5 is the number of steps of the game (down left, up left, center, up right, down right)
        # - 4 is the one-hot for the arrow (disabled, step, start hold, end hold)
        # Output: (Batch x UnrollLength x 4^5) tensor containing the pre-softmax scores for the next step to be placed

        self.rnn_projection = nn.Linear(
            in_features=5*4 + settings.delta_features,
            out_features=settings.rnn_size,
        )

        self.lstm = nn.LSTM(
            input_size=settings.rnn_size,
            hidden_size=settings.rnn_size,
            num_layers=settings.rnn_layers,
            batch_first=True,
            dropout=0.5
        )

        self.out_projection = nn.Sequential(
            nn.Linear(
                in_features=settings.rnn_size,
                out_features=4**5
            )
        )

    def forward(self, x, delta, initial_state=None, return_state=False):
        batch = x.shape[0]
        unroll = x.shape[1]
        assert x.shape[2] == 5
        assert x.shape[3] == 4

        assert delta.shape[0] == batch
        assert delta.shape[1] == unroll
        assert delta.shape[2] == 3

        # Batch x Unroll x 5 x 4
        flattened_x = torch.flatten(x, start_dim=2)

        # X: Batch x Unroll x 20
        # Diff: Batch x Unroll x 2
        concated = torch.concat([flattened_x, delta], dim=2)

        # Batch x Unroll x 22
        projected = self.rnn_projection(concated)

        # Batch x Unroll x RNNSize
        after_lstm, state = self.lstm(projected, initial_state)

        # Batch x Unroll x RNNSize
        _after_lstm = torch.flatten(after_lstm, start_dim=0, end_dim=1)
        # (Batch * Unroll) x RNNSize
        _final_layer = self.out_projection(_after_lstm)
        # (Batch * Unroll) x 4^5
        final_layer = torch.unflatten(_final_layer, 0, (batch, unroll)) 

        if return_state:
            # Batch x Unroll x 4^5
            return final_layer, state
        else:
            return final_layer
        
