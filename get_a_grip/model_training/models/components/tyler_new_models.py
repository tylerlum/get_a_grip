import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from nerf_grasping.models.FiLM_resnet import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    _resnet,
    BasicBlock,
    Bottleneck,
)
from nerf_grasping.models.FiLM_resnet_1d import ResNet1D
from torchvision.transforms import Lambda, Compose
from enum import Enum, auto
from functools import partial, cached_property, lru_cache
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from torchinfo import summary


def dataclass_to_kwargs(dataclass_instance: Any) -> Dict[str, Any]:
    return (
        {key: value for key, value in dataclass_instance.__dict__["_content"].items()}
        if dataclass_instance is not None
        else {}
    )


### ENUMS ###
class ConvOutputTo1D(Enum):
    FLATTEN = auto()  # (N, C, H, W) -> (N, C*H*W)
    AVG_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    AVG_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)
    MAX_POOL_SPATIAL = auto()  # (N, C, H, W) -> (N, C, 1, 1) -> (N, C)
    MAX_POOL_CHANNEL = auto()  # (N, C, H, W) -> (N, 1, H, W) -> (N, H*W)
    SPATIAL_SOFTMAX = auto()  # (N, C, H, W) -> (N, C, H, W) -> (N, 2*C)


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
        return torch.max(x, dim=self.dim)


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature: float = 1.0, output_variance: bool = False) -> None:
        super().__init__()
        self.temperature = temperature
        self.output_variance = output_variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Either (batch_size, n_channels, width) or (batch_size, n_channels, height, width)
        assert len(x.shape) in [3, 4]
        batch_size, n_channels = x.shape[:2]
        spatial_indices = [i for i in range(2, len(x.shape))]
        spatial_dims = x.shape[2:]

        # Softmax over spatial dimensions
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = nn.Softmax(dim=-1)(x / self.temperature)
        x = x.reshape(batch_size, n_channels, *spatial_dims)

        # Create spatial grid
        mesh_grids = torch.meshgrid(
            *[torch.linspace(-1, 1, dim, device=x.device) for dim in spatial_dims]
        )

        # Sanity check
        for mesh_grid in mesh_grids:
            assert mesh_grid.shape == spatial_dims

        # Get coords
        outputs = []
        for mesh_grid in mesh_grids:
            mesh_grid = mesh_grid.reshape(1, 1, *mesh_grid.shape)
            coord = torch.sum(x * mesh_grid, dim=spatial_indices)
            outputs.append(coord)

        # Get variance
        if self.output_variance:
            for mesh_grid in mesh_grids:
                mesh_grid = mesh_grid.reshape(1, 1, *mesh_grid.shape)
                coord = torch.sum(x * (mesh_grid**2), dim=spatial_indices)
                outputs.append(coord)

        # Stack
        outputs = torch.stack(outputs, dim=-1)
        expected_output_shape = (
            (batch_size, n_channels, len(spatial_dims))
            if not self.output_variance
            else (batch_size, n_channels, len(spatial_dims) * 2)
        )
        assert outputs.shape == expected_output_shape

        return outputs


CHANNEL_DIM = 1
CONV_2D_OUTPUT_TO_1D_MAP = {
    ConvOutputTo1D.FLATTEN: nn.Identity,
    ConvOutputTo1D.AVG_POOL_SPATIAL: partial(nn.AdaptiveAvgPool2d, output_size=(1, 1)),
    ConvOutputTo1D.AVG_POOL_CHANNEL: partial(Mean, dim=CHANNEL_DIM),
    ConvOutputTo1D.MAX_POOL_SPATIAL: partial(nn.AdaptiveMaxPool2d, output_size=(1, 1)),
    ConvOutputTo1D.MAX_POOL_CHANNEL: partial(Max, dim=CHANNEL_DIM),
    ConvOutputTo1D.SPATIAL_SOFTMAX: partial(
        SpatialSoftmax, temperature=1.0, output_variance=False
    ),
}
CONV_1D_OUTPUT_TO_1D_MAP = {
    ConvOutputTo1D.FLATTEN: nn.Identity,
    ConvOutputTo1D.AVG_POOL_SPATIAL: partial(nn.AdaptiveAvgPool1d, output_size=1),
    ConvOutputTo1D.AVG_POOL_CHANNEL: partial(Mean, dim=CHANNEL_DIM),
    ConvOutputTo1D.MAX_POOL_SPATIAL: partial(nn.AdaptiveMaxPool1d, output_size=1),
    ConvOutputTo1D.MAX_POOL_CHANNEL: partial(Max, dim=CHANNEL_DIM),
    ConvOutputTo1D.SPATIAL_SOFTMAX: partial(
        SpatialSoftmax, temperature=1.0, output_variance=False
    ),
}


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


class FiLMGenerator(nn.Module):
    num_beta_gamma = 2  # one scale and one bias

    def __init__(
        self,
        film_input_dim: int,
        num_params_to_film: int,
        hidden_layers: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self.film_input_dim = film_input_dim
        self.num_params_to_film = num_params_to_film
        self.film_output_dim = self.num_beta_gamma * num_params_to_film

        self.mlp = mlp(
            num_inputs=self.film_input_dim,
            num_outputs=self.film_output_dim,
            hidden_layers=hidden_layers,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(x.shape) == 2
        assert x.shape[1] == self.film_input_dim
        batch_size = x.shape[0]

        # Use delta-gamma so baseline is gamma=1
        film_output = self.mlp(x)
        beta, delta_gamma = torch.chunk(film_output, chunks=self.num_beta_gamma, dim=1)
        gamma = delta_gamma + 1.0
        assert beta.shape == gamma.shape == (batch_size, self.num_params_to_film)

        return beta, gamma


### 2D ENCODERS ###


@dataclass
class ConvEncoder2DConfig:
    use_pretrained: bool
    pooling_method: ConvOutputTo1D
    film_hidden_layers: Tuple[int, ...]


def resnet_small(*, weights=None, progress: bool = True, **kwargs: Any):
    return _resnet(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        weights=weights,
        progress=progress,
        **kwargs,
    )


def resnet_smaller(*, weights=None, progress: bool = True, **kwargs: Any):
    planes_per_layer = [16, 32, 64, 128]
    return _resnet(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        weights=weights,
        progress=progress,
        planes_per_layer=planes_per_layer,
        **kwargs,
    )


def resnet_smallest(*, weights=None, progress: bool = True, **kwargs: Any):
    planes_per_layer = [4, 8, 16, 32]
    return _resnet(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        weights=weights,
        progress=progress,
        planes_per_layer=planes_per_layer,
        **kwargs,
    )


class ResnetType2d(Enum):
    RESNET18 = auto()
    RESNET34 = auto()
    RESNET_SMALL = auto()
    RESNET_SMALLER = auto()
    RESNET_SMALLEST = auto()


class ConvEncoder2D(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conditioning_dim: Optional[int] = None,
        use_pretrained: bool = True,
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        film_hidden_layers: Tuple[int, ...] = (64, 64),
        resnet_type: ResnetType2d = ResnetType2d.RESNET18,
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, height, width)
        self.input_shape = input_shape
        self.conditioning_dim = conditioning_dim
        self.use_pretrained = use_pretrained
        self.pooling_method = pooling_method

        assert len(input_shape) == 3
        n_channels, height, width = input_shape
        assert n_channels == 1

        # Create conv architecture
        if resnet_type == ResnetType2d.RESNET18:
            weights = ResNet18_Weights.DEFAULT if self.use_pretrained else None
            weights_transforms = (
                [weights.transforms(antialias=True)] if weights is not None else []
            )
            self.img_preprocess = Compose(
                [Lambda(lambda x: x.repeat(1, 3, 1, 1))] + weights_transforms
            )
            self.conv_2d = resnet18(weights=weights)
        elif resnet_type == ResnetType2d.RESNET34:
            weights = ResNet34_Weights.DEFAULT if self.use_pretrained else None
            weights_transforms = (
                [weights.transforms(antialias=True)] if weights is not None else []
            )
            self.img_preprocess = Compose(
                [Lambda(lambda x: x.repeat(1, 3, 1, 1))] + weights_transforms
            )
            self.conv_2d = resnet34(weights=weights)
        elif resnet_type == ResnetType2d.RESNET_SMALL:
            assert not self.use_pretrained
            self.img_preprocess = Compose(
                [Lambda(lambda x: x.repeat(1, 3, 1, 1))]
            )
            self.conv_2d = resnet_small(weights=None)
        elif resnet_type == ResnetType2d.RESNET_SMALLER:
            self.img_preprocess = Compose(
                [Lambda(lambda x: x.repeat(1, 3, 1, 1))]
            )
            assert not self.use_pretrained
            self.conv_2d = resnet_smaller(weights=None)
        elif resnet_type == ResnetType2d.RESNET_SMALLEST:
            self.img_preprocess = Compose(
                [Lambda(lambda x: x.repeat(1, 3, 1, 1))]
            )
            assert not self.use_pretrained
            self.conv_2d = resnet_smallest(weights=None)
        else:
            raise ValueError(f"Invalid resnet_type = {resnet_type}")

        self.conv_2d.avgpool = CONV_2D_OUTPUT_TO_1D_MAP[self.pooling_method]()
        self.conv_2d.fc = nn.Identity()

        # Create FiLM generator
        if self.conditioning_dim is not None and self.num_film_params is not None:
            self.film_generator = FiLMGenerator(
                film_input_dim=self.conditioning_dim,
                num_params_to_film=self.num_film_params,
                hidden_layers=film_hidden_layers,
            )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch_size, n_channels, height, width)
        assert len(x.shape) == 4
        batch_size, _, _, _ = x.shape

        # Ensure valid use of conditioning
        assert (conditioning is None and self.conditioning_dim is None) or (
            conditioning is not None
            and self.conditioning_dim is not None
            and conditioning.shape == (batch_size, self.conditioning_dim)
        )

        # FiLM
        if conditioning is not None:
            beta, gamma = self.film_generator(conditioning)
            assert (
                beta.shape == gamma.shape == (batch_size, self.conv_2d.num_film_params)
            )
        else:
            beta, gamma = None, None

        # Conv
        x = self.img_preprocess(x)
        x = self.conv_2d(x, beta=beta, gamma=gamma)
        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    @property
    def num_film_params(self) -> Optional[int]:
        return (
            self.conv_2d.num_film_params
            if hasattr(self.conv_2d, "num_film_params")
            else None
        )

    @lru_cache
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = (
            torch.randn(example_batch_size, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


### 1D ENCODERS ###


@dataclass
class ConvEncoder1DConfig:
    pooling_method: ConvOutputTo1D
    film_hidden_layers: Tuple[int, ...]
    base_filters: int
    kernel_size: int
    stride: int
    groups: int
    n_block: int
    downsample_gap: int
    increasefilter_gap: int
    use_batchnorm: bool
    use_dropout: bool


class ConvEncoder1D(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        conditioning_dim: Optional[int] = None,
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        film_hidden_layers: Tuple[int, ...] = (64, 64),
        base_filters: int = 32,
        kernel_size: int = 4,
        stride: int = 2,
        groups: int = 1,
        n_block: int = 8,
        downsample_gap: int = 4,
        increasefilter_gap: int = 2,
        use_batchnorm: bool = True,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, seq_len)
        self.input_shape = input_shape
        self.conditioning_dim = conditioning_dim
        self.pooling_method = pooling_method

        assert len(input_shape) == 2
        n_channels, seq_len = input_shape

        # Create conv architecture
        self.conv_1d = ResNet1D(
            in_channels=n_channels,
            seq_len=seq_len,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            n_block=n_block,
            n_classes=2,  # Not used
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_batchnorm=use_batchnorm,
            use_dropout=use_dropout,
            verbose=False,
        )
        # Set equivalent pooling setting
        self.conv_1d.avgpool = CONV_1D_OUTPUT_TO_1D_MAP[self.pooling_method]()
        self.conv_1d.fc = nn.Identity()

        # Create FiLM generator
        if self.conditioning_dim is not None and self.num_film_params is not None:
            self.film_generator = FiLMGenerator(
                film_input_dim=self.conditioning_dim,
                num_params_to_film=self.num_film_params,
                hidden_layers=film_hidden_layers,
            )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch_size, n_channels, seq_len)
        assert len(x.shape) == 3
        batch_size, _, _ = x.shape

        # Ensure valid use of conditioning
        assert (conditioning is None and self.conditioning_dim is None) or (
            conditioning is not None
            and self.conditioning_dim is not None
            and conditioning.shape == (batch_size, self.conditioning_dim)
        )

        # FiLM
        if conditioning is not None:
            beta, gamma = self.film_generator(conditioning)
            assert (
                beta.shape == gamma.shape == (batch_size, self.conv_1d.num_film_params)
            )
        else:
            beta, gamma = None, None

        # Conv
        x = self.conv_1d(x, beta=beta, gamma=gamma)
        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    @property
    def num_film_params(self) -> Optional[int]:
        return self.conv_1d.num_film_params

    @lru_cache
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = (
            torch.randn(example_batch_size, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


@dataclass
class TransformerEncoder1DConfig:
    pooling_method: ConvOutputTo1D
    n_heads: int
    n_emb: int
    p_drop_emb: float
    p_drop_attn: float
    n_layers: int


class TransformerEncoder1D(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        conditioning_dim: Optional[int] = None,
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        n_heads: int = 8,
        n_emb: int = 128,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, seq_len)
        self.input_shape = input_shape
        self.conditioning_dim = conditioning_dim
        self.pooling_method = pooling_method
        self.n_emb = n_emb

        n_channels, seq_len = input_shape

        # Encoder
        self.encoder_input_emb = nn.Linear(self.encoder_input_dim, n_emb)
        self.encoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
        self.encoder_drop_emb = nn.Dropout(p=p_drop_emb)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_heads,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

        if conditioning_dim is not None:
            # Decoder
            self.decoder_input_emb = nn.Linear(n_channels, n_emb)
            self.decoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
            self.decoder_drop_emb = nn.Dropout(p=p_drop_emb)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_heads,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=n_layers
            )

        self.ln_f = nn.LayerNorm(n_emb)
        self.pool = CONV_1D_OUTPUT_TO_1D_MAP[self.pooling_method]()

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, n_channels, seq_len)

        assert (conditioning is not None and self.conditioning_dim is not None) or (
            conditioning is None and self.conditioning_dim is None
        )
        if conditioning is not None and self.conditioning_dim is not None:
            assert conditioning.shape == (batch_size, self.conditioning_dim)

            # Condition encoder
            # Need to repeat conditioning to match seq_len
            conditioning = conditioning.reshape(
                batch_size,
                1,
                self.conditioning_dim,
            ).repeat(1, seq_len, 1)
            assert conditioning.shape == (
                batch_size,
                seq_len,
                self.conditioning_dim,
            )
            conditioning = self._encoder(conditioning)
            assert conditioning.shape == (batch_size, seq_len, self.n_emb)

            # Decoder
            x = x.permute(0, 2, 1)
            assert x.shape == (batch_size, seq_len, n_channels)
            x = self._decoder(x, conditioning)
            assert x.shape == (batch_size, seq_len, self.n_emb)
        else:
            # Encoder
            x = x.permute(0, 2, 1)
            assert x.shape == (batch_size, seq_len, n_channels)
            x = self._encoder(x)
            assert x.shape == (batch_size, seq_len, self.n_emb)

        x = self.ln_f(x)
        assert x.shape == (batch_size, seq_len, self.n_emb)

        # Need to permute to (batch_size, n_channels, seq_len) for pooling
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.flatten(start_dim=1)

        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        _, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, self.encoder_input_dim)

        x = self.encoder_input_emb(x)
        x = self.encoder_pos_emb(x)
        x = self.encoder_drop_emb(x)
        x = self.transformer_encoder(x)

        return x

    def _decoder(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, n_channels)

        x = self.decoder_input_emb(x)
        x = self.decoder_pos_emb(x)
        x = self.decoder_drop_emb(x)
        x = self.transformer_decoder(tgt=x, memory=conditioning)

        return x

    @cached_property
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = (
            torch.randn(example_batch_size, self.conditioning_dim)
            if self.conditioning_dim is not None
            else None
        )
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]

    @property
    def encoder_input_dim(self) -> int:
        n_channels, _ = self.input_shape
        return (
            self.conditioning_dim if self.conditioning_dim is not None else n_channels
        )


### Attention Encoder Decoder ###
class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        pooling_method: ConvOutputTo1D = ConvOutputTo1D.FLATTEN,
        n_emb: int = 128,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()

        # input_shape: (n_channels, seq_len)
        self.input_shape = input_shape
        self.pooling_method = pooling_method
        self.n_emb = n_emb

        n_channels, seq_len = input_shape

        # Encoder
        self.encoder_input_emb = nn.Linear(n_channels, n_emb)
        self.encoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
        self.encoder_drop_emb = nn.Dropout(p=p_drop_emb)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=4,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

        # Decoder
        self.decoder_input_emb = nn.Linear(n_channels, n_emb)
        self.decoder_pos_emb = Summer(PositionalEncoding1D(channels=n_emb))
        self.decoder_drop_emb = nn.Dropout(p=p_drop_emb)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=4,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layers
        )

        self.ln_f = nn.LayerNorm(n_emb)
        self.pool = CONV_1D_OUTPUT_TO_1D_MAP[self.pooling_method]()

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = conditioning.shape[0]
        n_channels, seq_len = self.input_shape
        assert conditioning.shape == x.shape == (batch_size, n_channels, seq_len)

        conditioning = conditioning.permute(0, 2, 1)
        assert conditioning.shape == (batch_size, seq_len, n_channels)
        conditioning = self._encoder(conditioning)
        assert conditioning.shape == (batch_size, seq_len, self.n_emb)

        # Decoder
        x = x.permute(0, 2, 1)
        assert x.shape == (batch_size, seq_len, n_channels)
        x = self._decoder(x, conditioning=conditioning)
        assert x.shape == (batch_size, seq_len, self.n_emb)

        x = self.ln_f(x)
        assert x.shape == (batch_size, seq_len, self.n_emb)

        # Need to permute to (batch_size, n_channels, seq_len) for pooling
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.flatten(start_dim=1)

        assert len(x.shape) == 2 and x.shape[0] == batch_size

        return x

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, n_channels)

        x = self.encoder_input_emb(x)
        x = self.encoder_pos_emb(x)
        x = self.encoder_drop_emb(x)
        x = self.transformer_encoder(x)

        return x

    def _decoder(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_channels, seq_len = self.input_shape
        assert x.shape == (batch_size, seq_len, n_channels)

        x = self.decoder_input_emb(x)
        x = self.decoder_pos_emb(x)
        x = self.decoder_drop_emb(x)
        x = self.transformer_decoder(tgt=x, memory=conditioning)

        return x

    @cached_property
    def output_dim(self) -> int:
        # Compute output shape
        example_batch_size = 2
        example_input = torch.randn(example_batch_size, *self.input_shape)
        example_conditioning = torch.randn(example_batch_size, *self.input_shape)
        example_output = self(example_input, conditioning=example_conditioning)
        assert len(example_output.shape) == 2
        return example_output.shape[1]


def main() -> None:
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, n_fingers, seq_len, height, width = (
        2,
        2,
        10,
        40,
        40,
    )  # WARNING: Easy to OOM
    conditioning_dim = 11
    print(f"batch_size = {batch_size}")
    print(f"n_fingers = {n_fingers}")
    print(f"seq_len = {seq_len}")
    print(f"height = {height}")
    print(f"width = {width}")
    print(f"conditioning_dim = {conditioning_dim}")
    print()

    example_input = torch.randn(
        batch_size, n_fingers, seq_len, height, width, device=device
    )
    example_conditioning = torch.randn(
        batch_size, n_fingers, conditioning_dim, device=device
    )

    # Spatial softmax
    x = torch.randn(batch_size, seq_len, width, device=device)
    xx = torch.randn(batch_size, seq_len, height, width, device=device)
    actual_max_x = torch.argmax(x, dim=-1)
    actual_max_xx = []
    for i in range(batch_size):
        temp = torch.stack(
            [(xx[i, j] == torch.max(xx[i, j])).nonzero() for j in range(seq_len)], dim=0
        )
        actual_max_xx.append(temp)

    actual_max_xx = torch.stack(actual_max_xx, dim=0)
    spatial_softmax = SpatialSoftmax(temperature=0.001, output_variance=False)
    spatial_softmax_with_variance = SpatialSoftmax(
        temperature=0.001, output_variance=True
    )
    print("SpatialSoftmax")
    print("=" * 80)
    print(f"x.shape = {x.shape}")
    print(f"xx.shape = {xx.shape}")
    print(f"spatial_softmax(x).shape = {spatial_softmax(x).shape}")
    print(f"spatial_softmax(xx).shape = {spatial_softmax(xx).shape}")
    print(
        f"spatial_softmax_with_variance(x).shape = {spatial_softmax_with_variance(x).shape}"
    )
    print(
        f"spatial_softmax_with_variance(xx).shape = {spatial_softmax_with_variance(xx).shape}"
    )
    batch_idx_to_print = 0
    seq_idx_to_print = 0
    print(f"For batch_idx, seq_idx_to_print = {batch_idx_to_print}, {seq_idx_to_print}")
    print(
        f"spatial_softmax(x)[batch_idx_to_print, seq_idx_to_print] = {(spatial_softmax(x)[batch_idx_to_print, seq_idx_to_print] + 1) / 2 * width}"
    )
    print(
        f"spatial_softmax(xx)[batch_idx_to_print, seq_idx_to_print] = {(spatial_softmax(xx)[batch_idx_to_print, seq_idx_to_print] + 1) / 2 * width}"
    )
    print(
        f"actual_max_x[batch_idx_to_print, seq_idx_to_print] = {actual_max_x[batch_idx_to_print, seq_idx_to_print]}"
    )
    print(
        f"actual_max_xx[batch_idx_to_print, seq_idx_to_print] = {actual_max_xx[batch_idx_to_print, seq_idx_to_print]}"
    )
    print()


from diffusers.optimization import (
    Union,
    SchedulerType,
    Optional,
    Optimizer,
    TYPE_TO_SCHEDULER_FUNCTION,
)


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )


def set_seed(seed) -> None:
    import random
    import numpy as np
    import os

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)  # TODO: Is this slowing things down?

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Set random seed to {seed}")


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    set_seed(43)

    LAUNCH_WITH_IPDB = False
    if LAUNCH_WITH_IPDB:
        with launch_ipdb_on_exception():
            main()
    else:
        main()
