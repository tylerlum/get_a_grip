from enum import Enum, auto
from typing import Tuple

import torch
import torch.nn as nn


### ENUMS ###
class ConvOutputTo1D(Enum):
    FLATTEN = auto()  # (N, C, H, W) -> (N, C*H*W)
    AVG_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    AVG_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)
    MAX_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    MAX_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)


class PoolType(Enum):
    MAX = auto()
    AVG = auto()


### Small Modules ###
class Mean(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=self.dim)


class Max(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=self.dim).values


### HELPER FUNCTIONS ###
def mlp(
    num_inputs: int,
    num_outputs: int,
    hidden_layers: Tuple[int, ...],
    activation=nn.ReLU,
    output_activation=nn.Identity,
) -> nn.Sequential:
    layers = []
    layer_sizes = [num_inputs] + list(hidden_layers) + [num_outputs]
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def conv_encoder(
    input_shape: Tuple[int, ...],
    conv_channels: Tuple[int, ...],
    pool_type: PoolType = PoolType.MAX,
    dropout_prob: float = 0.0,
    conv_output_to_1d: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
    activation=nn.ReLU,
) -> nn.Module:
    # Input: Either (n_channels, n_dims) or (n_channels, height, width) or (n_channels, depth, height, width)

    # Validate input
    assert 2 <= len(input_shape) <= 4
    n_input_channels = input_shape[0]
    n_spatial_dims = len(input_shape[1:])

    # Layers for different input sizes
    n_spatial_dims_to_conv_layer_map = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    n_spatial_dims_to_maxpool_layer_map = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d,
    }
    n_spatial_dims_to_avgpool_layer_map = {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d,
    }
    n_spatial_dims_to_dropout_layer_map = {
        # 1: nn.Dropout1d,  # Not in some versions of torch
        2: nn.Dropout2d,
        3: nn.Dropout3d,
    }
    n_spatial_dims_to_adaptivemaxpool_layer_map = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }
    n_spatial_dims_to_adaptiveavgpool_layer_map = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }

    # Setup layer types
    conv_layer = n_spatial_dims_to_conv_layer_map[n_spatial_dims]
    if pool_type == PoolType.MAX:
        pool_layer = n_spatial_dims_to_maxpool_layer_map[n_spatial_dims]
    elif pool_type == PoolType.AVG:
        pool_layer = n_spatial_dims_to_avgpool_layer_map[n_spatial_dims]
    else:
        raise ValueError(f"Invalid pool_type = {pool_type}")
    dropout_layer = n_spatial_dims_to_dropout_layer_map[n_spatial_dims]

    # Conv layers
    layers = []
    n_channels = [n_input_channels] + list(conv_channels)
    for i in range(len(n_channels) - 1):
        layers += [
            conv_layer(
                in_channels=n_channels[i],
                out_channels=n_channels[i + 1],
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            activation(),
            pool_layer(kernel_size=2, stride=2),
        ]
        if dropout_prob != 0.0:
            layers += [dropout_layer(p=dropout_prob)]

    # Convert from (n_channels, X) => (Y,)
    if conv_output_to_1d == ConvOutputTo1D.FLATTEN:
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.AVG_POOL_SPATIAL:
        adaptiveavgpool_layer = n_spatial_dims_to_adaptiveavgpool_layer_map[
            n_spatial_dims
        ]
        layers.append(
            adaptiveavgpool_layer(output_size=tuple([1 for _ in range(n_spatial_dims)]))
        )
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.MAX_POOL_SPATIAL:
        adaptivemaxpool_layer = n_spatial_dims_to_adaptivemaxpool_layer_map[
            n_spatial_dims
        ]
        layers.append(
            adaptivemaxpool_layer(output_size=tuple([1 for _ in range(n_spatial_dims)]))
        )
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.AVG_POOL_CHANNEL:
        channel_dim = 1
        layers.append(Mean(dim=channel_dim))
        layers.append(nn.Flatten(start_dim=1))
    elif conv_output_to_1d == ConvOutputTo1D.MAX_POOL_CHANNEL:
        channel_dim = 1
        layers.append(Max(dim=channel_dim))
        layers.append(nn.Flatten(start_dim=1))
    else:
        raise ValueError(f"Invalid conv_output_to_1d = {conv_output_to_1d}")

    return nn.Sequential(*layers)


def main() -> None:
    # Define constants
    batch_size = 8
    input_dim_mlp = 16
    output_dim_mlp = 32
    hidden_layers_mlp = (64, 128)

    input_channels_conv = 3
    depth = 10
    height = 32
    width = 32
    conv_channels = (8, 16, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create MLP model
    mlp_model = mlp(input_dim_mlp, output_dim_mlp, hidden_layers_mlp).to(device)
    mlp_input = torch.randn(batch_size, input_dim_mlp, device=device)
    mlp_output = mlp_model(mlp_input)

    # Check and print MLP shapes
    print(f"MLP Input Shape: {mlp_input.shape}")
    print(f"MLP Output Shape: {mlp_output.shape}")
    assert mlp_input.shape == (batch_size, input_dim_mlp), "MLP input shape mismatch"
    assert mlp_output.shape == (batch_size, output_dim_mlp), "MLP output shape mismatch"

    # Create Conv Encoder model
    input_shape_conv = (input_channels_conv, width, height, depth)
    conv_model = conv_encoder(
        input_shape_conv, conv_channels, conv_output_to_1d=ConvOutputTo1D.FLATTEN
    ).to(device)
    conv_input = torch.randn(batch_size, *input_shape_conv, device=device)
    conv_output = conv_model(conv_input)

    # Check and print Conv Encoder shapes
    print(f"Conv Encoder Input Shape: {conv_input.shape}")
    print(f"Conv Encoder Output Shape: {conv_output.shape}")
    conv_output_dim = conv_output.shape[1]
    assert conv_input.shape == (
        batch_size,
        *input_shape_conv,
    ), "Conv Encoder input shape mismatch"
    assert conv_output.shape == (
        batch_size,
        conv_output_dim,
    ), "Conv Encoder output shape mismatch"


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    from get_a_grip.utils.seed import set_seed

    set_seed(43)

    LAUNCH_WITH_IPDB = False
    if LAUNCH_WITH_IPDB:
        with launch_ipdb_on_exception():
            main()
    else:
        main()
