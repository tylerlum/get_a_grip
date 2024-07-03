import torch
import torch.nn as nn
from functools import partial
from typing import List, Tuple

from enum import Enum, auto


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build model
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(
            zip([input_dim] + list(hidden_dims[:-1]), hidden_dims)
        ):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        assert x.shape[-1] == self.input_dim, f"{x.shape} != {self.input_dim}"
        for layer in self.layers:
            x = layer(x)
        return x


class FiLMLayer(nn.Module):
    """
    A PyTorch implementation of a FiLM layer.
    """

    def __init__(
        self,
        num_output_dims: int,
        in_channels: int,
        conditioning_dim: int,
        hidden_dims: Tuple[int, ...] = (32,),
    ):
        super().__init__()
        self.num_output_dims = num_output_dims
        self.in_channels = in_channels
        self.conditioning_dim = conditioning_dim
        self.hidden_dims = hidden_dims

        # Build model
        self.gamma = MLP(
            conditioning_dim, hidden_dims, in_channels
        )  # Map conditioning dimension to scaling for each channel.
        self.beta = MLP(conditioning_dim, hidden_dims, in_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        # Compute FiLM parameters
        assert conditioning.shape[-1] == self.conditioning_dim
        batch_dims = conditioning.shape[:-1]

        assert x.shape[: -(self.num_output_dims + 1)] == batch_dims
        assert x.shape[-(self.num_output_dims + 1)] == self.in_channels

        gamma = self.gamma(conditioning)
        beta = self.beta(conditioning)

        # Do unsqueezing to make sure dimensions match; run e.g., twice for 2D FiLM.
        for _ in range(self.num_output_dims):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        assert (
            gamma.shape
            == beta.shape
            == batch_dims + (self.in_channels,) + (1,) * self.num_output_dims
        )

        # Apply FiLM
        return gamma * x + beta


class CNN2DFiLM(nn.Module):
    """
    A vanilla 2D CNN with FiLM conditioning layers
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        conv_channels: List[int],
        conditioning_dim: int,
        num_in_channels: int,
        pooling=nn.MaxPool2d(kernel_size=2),
        film_hidden_layers: Tuple[int, ...] = (32,),
        dropout_every: int = 1,
        pooling_every: int = 2,
        condition_every: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.conditioning_dim = conditioning_dim  # Note conditioning can only be 1D.
        self.num_in_channels = num_in_channels
        self.pooling = pooling
        self.film_hidden_layers = film_hidden_layers
        self.dropout_every = dropout_every
        self.pooling_every = pooling_every
        self.condition_every = condition_every

        # Build model
        self.conv_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(
            zip([self.num_in_channels] + conv_channels[:-1], conv_channels)
        ):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.conv_layers.append(nn.ReLU())
            if i % self.dropout_every == 0:
                self.conv_layers.append(nn.Dropout2d(p=0.1, inplace=False))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            if i % self.condition_every == 0:
                self.conv_layers.append(
                    FiLMLayer(
                        2,
                        out_channels,
                        conditioning_dim,
                        hidden_dims=film_hidden_layers,
                    )
                )
            if i % self.pooling_every == 0:
                self.conv_layers.append(self.pooling)

        # Compute output shape
        with torch.no_grad():
            self.output_shape = self.get_output_shape()

    def get_output_shape(self):
        # Compute output shape
        x = torch.zeros((1, self.num_in_channels, *self.input_shape))
        x = self.forward(x, torch.zeros((1, self.conditioning_dim)))

        return x.shape[1:]

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        """
        Forward pass for FiLM 2D CNN.

        Args:
            x: input tensor of shape (batch_size, num_in_channels, *input_shape)
            conditioning: conditioning tensor of shape (batch_size, conditioning_dim)
        """
        assert x.shape[-2:] == self.input_shape
        assert x.shape[-3] == self.num_in_channels
        assert conditioning.shape[-1] == self.conditioning_dim

        # Reshape batch dims
        if len(x.shape) >= 4:
            batch_dims = x.shape[:-3]
            assert batch_dims == conditioning.shape[:-1]

            x = x.reshape(-1, *x.shape[-3:])
            conditioning = conditioning.reshape(-1, *conditioning.shape[-1:])
        else:
            batch_dims = None

        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, conditioning)
            else:
                x = layer(x)

        # Reshape batch dims
        if batch_dims is not None:
            x = x.reshape(*batch_dims, *x.shape[-3:])  # Batch dims, n_c, n_w, n_h

        return x


class CNN1DFiLM(nn.Module):
    def __init__(
        self,
        seq_len: int,
        conv_channels: List[int],
        conditioning_dim: int,
        num_in_channels: int,
        kernel_size: int = 3,
        pooling=nn.MaxPool1d(kernel_size=3),
        film_hidden_layers: Tuple[int, ...] = (32,),
        pooling_every: int = 2,
        dropout_every: int = 1,
        condition_every: int = 1,
        use_residual_conections: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.conv_channels = conv_channels
        self.conditioning_dim = conditioning_dim
        self.kernel_size = kernel_size
        self.num_in_channels = num_in_channels
        self.pooling = pooling
        self.dropout_every = dropout_every
        self.pooling_every = pooling_every
        self.condition_every = condition_every

        # Build model
        self.conv_layers = nn.ModuleList()
        for ii, (in_channels, out_channels) in enumerate(
            zip([self.num_in_channels] + conv_channels[:-1], conv_channels)
        ):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=self.kernel_size, padding=1
                )
            )
            self.conv_layers.append(nn.ReLU())
            if ii % self.dropout_every == 0:
                self.conv_layers.append(torch.nn.Dropout(p=0.1, inplace=False))
            self.conv_layers.append(nn.BatchNorm1d(out_channels))
            if ii % self.condition_every == 0:
                self.conv_layers.append(
                    FiLMLayer(
                        1,
                        out_channels,
                        conditioning_dim,
                        film_hidden_layers,
                    )
                )

            if ii % self.pooling_every == 0:
                self.conv_layers.append(self.pooling)

        # Compute output shape
        with torch.no_grad():
            self.output_shape = self.get_output_shape()

    def get_output_shape(self):
        # Compute output shape
        x = torch.zeros((1, self.num_in_channels, self.seq_len))
        x = self.forward(x, torch.zeros((1, self.conditioning_dim)))
        return x.shape[1:]

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        assert x.shape[-1] == self.seq_len
        assert x.shape[-2] == self.num_in_channels
        assert conditioning.shape[-1] == self.conditioning_dim

        # Reshape batch dims
        if len(x.shape) >= 3:
            batch_dims = x.shape[:-2]
            assert batch_dims == conditioning.shape[:-1]
            x = x.reshape(-1, *x.shape[-2:])
            conditioning = conditioning.reshape(-1, *conditioning.shape[-1:])
        else:
            batch_dims = None

        for layer in self.conv_layers:
            if isinstance(layer, FiLMLayer):
                x = layer(x, conditioning)
            else:
                x = layer(x)

        # Reshape batch dims
        if batch_dims is not None:
            x = x.reshape(*batch_dims, *x.shape[-2:])

        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, _ = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        return out[:, -1, :]


if __name__ == "__main__":
    # Full stack: [B, n_f, n_z, n_x, n_y] input
    B, n_f, n_z, n_x, n_y = 5, 10, 15, 16, 17
    conditioning_dim = 7

    x = torch.randn(B, n_f, n_z, n_x, n_y)
    conditioning = torch.randn(B, n_f, conditioning_dim)

    input_shape = (n_x, n_y)
    conv2d_channels = [32, 64, 128]
    in_channels_2d = 1

    cnn2d_film = CNN2DFiLM(
        input_shape, conv2d_channels, conditioning_dim, in_channels_2d
    )
    print(cnn2d_film)

    cnn2d_output_shape = cnn2d_film.get_output_shape()
    assert cnn2d_output_shape[0] == (conv2d_channels[-1])

    print(f"cnn2d_output_shape: {cnn2d_output_shape}")

    print(f"x.shape: {x.shape}")
    print(f"conditioning.shape: {conditioning.shape}")

    x_2d = x.unsqueeze(-3)  # Add channel dimension
    conditioning_2d = conditioning.unsqueeze(2).expand(
        -1, -1, n_z, -1
    )  # Add z dimension to cond.

    out_2d = cnn2d_film(x_2d, conditioning_2d)
    print(f"out.shape: {out_2d.shape}")
    assert out_2d.shape == (B, n_f, n_z, *cnn2d_output_shape)

    # Permute and reshape out_2d -> in_1d.
    out_2d = out_2d.flatten(-3, -1)  # Flatten CNN channels, xy dim.
    out_2d = out_2d.permute(0, 1, 3, 2)  # Permute to (B, n_f, n_c, n_z)

    conv1d_channels = [13, 6]
    cnn1d_film = CNN1DFiLM(n_z, conv1d_channels, conditioning_dim, out_2d.shape[-2])
    print(cnn1d_film)
    output_shape1d = cnn1d_film.get_output_shape()
    print(f"output_shape1d: {output_shape1d}")

    out_1d = cnn1d_film(out_2d, conditioning)
    # assert out_1d.shape == (B, n_f, conv1d_channels[-1], n_z)
    print(f"out_1d.shape: {out_1d.shape}")
