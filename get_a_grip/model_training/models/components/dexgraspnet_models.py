import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from functools import lru_cache
from nerf_grasping.models.tyler_new_models import (
    conv_encoder,
    PoolType,
    ConvOutputTo1D,
    mlp,
    ConvEncoder2D,
    ConvEncoder1D,
    ResnetType2d,
)

from nerf_grasping.models.preston_new_models import CNN2DFiLM, CNN1DFiLM, MLP, LSTMModel


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class CNN_3D_Model(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        conv_channels: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
        n_fingers: int,
        n_tasks: int = 1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.n_fingers = n_fingers
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=conv_channels,
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(
            example_batch_size, self.n_fingers, *self.input_shape
        )
        example_input = example_input.reshape(
            example_batch_size * self.n_fingers, *self.input_shape
        )
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert_equals(
            conv_output.shape,
            (
                example_batch_size * self.n_fingers,
                self.conv_output_dim,
            ),
        )

        self.mlp = mlp(
            num_inputs=self.conv_output_dim * self.n_fingers,
            num_outputs=self.n_classes * self.n_tasks,
            hidden_layers=mlp_hidden_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert_equals(
            x.shape,
            (
                batch_size,
                self.n_fingers,
                *self.input_shape,
            ),
        )

        # Put n_fingers into batch dim
        x = x.reshape(batch_size * self.n_fingers, *self.input_shape)

        x = self.conv(x)
        assert_equals(
            x.shape,
            (
                batch_size * self.n_fingers,
                self.conv_output_dim,
            ),
        )
        x = x.reshape(batch_size, self.n_fingers, self.conv_output_dim)
        x = x.reshape(batch_size, self.n_fingers * self.conv_output_dim)

        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


class CNN_3D_CNN_3D_Model(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        conv_channels: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
        global_input_shape: Tuple[int, int, int, int],
        global_conv_channels: Tuple[int, ...],
        n_fingers: int,
        n_tasks: int = 1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.global_input_shape = global_input_shape
        self.n_fingers = n_fingers
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        # Conv
        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=conv_channels,
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(
            example_batch_size, self.n_fingers, *self.input_shape
        )
        example_input = example_input.reshape(
            example_batch_size * self.n_fingers, *self.input_shape
        )
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert_equals(
            conv_output.shape,
            (
                example_batch_size * self.n_fingers,
                self.conv_output_dim,
            ),
        )

        # Global Conv
        self.global_conv = conv_encoder(
            input_shape=self.global_input_shape,
            conv_channels=global_conv_channels,
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get global conv output shape
        example_batch_size = 2
        example_input = torch.zeros(example_batch_size, *self.global_input_shape)
        global_conv_output = self.global_conv(example_input)
        self.global_conv_output_dim = global_conv_output.shape[-1]
        assert_equals(
            global_conv_output.shape,
            (
                example_batch_size,
                self.global_conv_output_dim,
            ),
        )

        # MLP
        self.mlp = mlp(
            num_inputs=self.conv_output_dim * self.n_fingers
            + self.global_conv_output_dim,
            num_outputs=self.n_classes * self.n_tasks,
            hidden_layers=mlp_hidden_layers,
        )

    def forward(self, x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert_equals(
            x.shape,
            (
                batch_size,
                self.n_fingers,
                *self.input_shape,
            ),
        )
        assert_equals(
            global_x.shape,
            (
                batch_size,
                *self.global_input_shape,
            ),
        )

        # Put n_fingers into batch dim
        x = x.reshape(batch_size * self.n_fingers, *self.input_shape)

        x = self.conv(x)
        assert_equals(
            x.shape,
            (
                batch_size * self.n_fingers,
                self.conv_output_dim,
            ),
        )
        x = x.reshape(batch_size, self.n_fingers, self.conv_output_dim)
        x = x.reshape(batch_size, self.n_fingers * self.conv_output_dim)

        global_x = self.global_conv(global_x)
        assert_equals(
            global_x.shape,
            (
                batch_size,
                self.global_conv_output_dim,
            ),
        )

        x = torch.cat([x, global_x], dim=1)
        assert_equals(
            x.shape,
            (
                batch_size,
                self.n_fingers * self.conv_output_dim + self.global_conv_output_dim,
            ),
        )

        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(self, x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        return self(x, global_x)


class CNN_3D_MLP_Model(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        conv_channels: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
        global_input_shape: Tuple[int],
        global_mlp_hidden_layers: Tuple[int, ...],
        n_fingers: int,
        n_tasks: int = 1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.global_input_shape = global_input_shape
        self.global_mlp_hidden_layers = global_mlp_hidden_layers
        self.n_fingers = n_fingers
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        # Conv
        self.conv = conv_encoder(
            input_shape=self.input_shape,
            conv_channels=conv_channels,
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )

        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(
            example_batch_size, self.n_fingers, *self.input_shape
        )
        example_input = example_input.reshape(
            example_batch_size * self.n_fingers, *self.input_shape
        )
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert_equals(
            conv_output.shape,
            (
                example_batch_size * self.n_fingers,
                self.conv_output_dim,
            ),
        )

        # Global MLP
        assert (
            len(self.global_input_shape) == 1
        ), f"self.global_input_shape: {self.global_input_shape}"
        assert (
            len(global_mlp_hidden_layers) > 0
        ), f"global_mlp_hidden_layers: {global_mlp_hidden_layers}"
        self.global_mlp = mlp(
            num_inputs=self.global_input_shape[0],
            num_outputs=global_mlp_hidden_layers[-1],
            hidden_layers=global_mlp_hidden_layers[:-1],
        )

        # MLP
        self.mlp = mlp(
            num_inputs=self.conv_output_dim * self.n_fingers
            + global_mlp_hidden_layers[-1],
            num_outputs=self.n_classes * self.n_tasks,
            hidden_layers=mlp_hidden_layers,
        )

    def forward(self, x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert_equals(
            x.shape,
            (
                batch_size,
                self.n_fingers,
                *self.input_shape,
            ),
        )
        assert_equals(
            global_x.shape,
            (
                batch_size,
                *self.global_input_shape,
            ),
        )

        # Put n_fingers into batch dim
        x = x.reshape(batch_size * self.n_fingers, *self.input_shape)

        x = self.conv(x)
        assert_equals(
            x.shape,
            (
                batch_size * self.n_fingers,
                self.conv_output_dim,
            ),
        )
        x = x.reshape(batch_size, self.n_fingers, self.conv_output_dim)
        x = x.reshape(batch_size, self.n_fingers * self.conv_output_dim)

        global_x = self.global_mlp(global_x)
        assert_equals(
            global_x.shape,
            (
                batch_size,
                self.global_mlp_hidden_layers[-1],
            ),
        )

        x = torch.cat([x, global_x], dim=1)
        assert_equals(
            x.shape,
            (
                batch_size,
                self.n_fingers * self.conv_output_dim
                + self.global_mlp_hidden_layers[-1],
            ),
        )

        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(self, x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        return self(x, global_x)


class CNN_2D_1D_Model(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        n_fingers: int,
        conditioning_dim: int,
        conv_2d_film_hidden_layers: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
        use_pretrained_2d: bool = True,
        resnet_type_2d: ResnetType2d = ResnetType2d.RESNET18,
        pooling_method_2d: ConvOutputTo1D = ConvOutputTo1D.AVG_POOL_SPATIAL,
        n_tasks: int = 1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.n_fingers = n_fingers
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.conditioning_dim = conditioning_dim

        n_pts_x, n_pts_y, n_pts_z = self.grid_shape
        seq_len = n_pts_z

        self.conv_2d = ConvEncoder2D(
            input_shape=(1, n_pts_x, n_pts_y),
            conditioning_dim=conditioning_dim,
            use_pretrained=use_pretrained_2d,
            pooling_method=pooling_method_2d,
            film_hidden_layers=conv_2d_film_hidden_layers,
            resnet_type=resnet_type_2d,
        )

        self.conv_1d = ConvEncoder1D(
            input_shape=(self.conv_2d.output_dim(), seq_len),
            conditioning_dim=conditioning_dim,
            pooling_method=ConvOutputTo1D.AVG_POOL_SPATIAL,
            # TODO: Config this
        )
        self.mlp = mlp(
            num_inputs=n_fingers * self.conv_1d.output_dim()
            + n_fingers * conditioning_dim,
            num_outputs=self.n_classes * self.n_tasks,
            hidden_layers=mlp_hidden_layers,
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_fingers = self.n_fingers
        n_pts_x, n_pts_y, n_pts_z = self.grid_shape
        seq_len = n_pts_z
        conditioning_dim = self.conditioning_dim

        # Check shapes
        assert_equals(x.shape, (batch_size, n_fingers, n_pts_x, n_pts_y, seq_len))
        assert_equals(conditioning.shape, (batch_size, n_fingers, conditioning_dim))

        # Batch n_fingers with batch_size
        # TODO: Shouldn't need to do this reshaping
        # but currently have asserts in ConvEncoder2D to sanity check
        x = x.reshape(batch_size * n_fingers, n_pts_x, n_pts_y, seq_len)
        conditioning = conditioning.reshape(batch_size * n_fingers, conditioning_dim)

        # Reorder so each image in sequence is processed independently
        x = x.permute(0, 3, 1, 2)
        assert_equals(x.shape, (batch_size * n_fingers, seq_len, n_pts_x, n_pts_y))

        # Conv 2D
        BATCHED_COMPUTATION = True  # TODO: See if OOM is an issue
        if BATCHED_COMPUTATION:
            x = x.reshape(batch_size * n_fingers * seq_len, 1, n_pts_x, n_pts_y)
            conditioning_repeated = conditioning.repeat_interleave(seq_len, dim=0)
            assert_equals(
                x.shape,
                (
                    batch_size * n_fingers * seq_len,
                    1,
                    n_pts_x,
                    n_pts_y,
                ),
            )
            assert_equals(
                conditioning_repeated.shape,
                (
                    batch_size * n_fingers * seq_len,
                    conditioning_dim,
                ),
            )
            x = self.conv_2d(x, conditioning=conditioning_repeated)
            assert_equals(
                x.shape,
                (
                    batch_size * n_fingers * seq_len,
                    self.conv_2d.output_dim(),
                ),
            )
            x = x.reshape(batch_size * n_fingers, seq_len, self.conv_2d.output_dim())
        else:
            output_list = []
            for i in range(seq_len):
                input = x[:, i : i + 1, :, :]
                assert_equals(
                    input.shape, (batch_size * n_fingers, 1, n_pts_x, n_pts_y)
                )
                assert_equals(
                    conditioning.shape, (batch_size * n_fingers, conditioning_dim)
                )
                output = self.conv_2d(input, conditioning=conditioning)
                assert_equals(
                    output.shape, (batch_size * n_fingers, self.conv_2d.output_dim())
                )
                output_list.append(output)
            x = torch.stack(output_list, dim=1)
        assert_equals(
            x.shape, (batch_size * n_fingers, seq_len, self.conv_2d.output_dim())
        )

        # Reorder so Conv 1D filters can process along sequence dim
        x = x.permute(0, 2, 1)
        assert_equals(
            x.shape, (batch_size * n_fingers, self.conv_2d.output_dim(), seq_len)
        )
        assert_equals(conditioning.shape, (batch_size * n_fingers, conditioning_dim))

        # Conv 1D
        x = self.conv_1d(x, conditioning=conditioning)
        assert_equals(x.shape, (batch_size * n_fingers, self.conv_1d.output_dim()))

        # Aggregate into one feature dimension
        x = x.reshape(batch_size, n_fingers * self.conv_1d.output_dim())

        # Concatenate with conditioning
        conditioning = conditioning.reshape(batch_size, n_fingers * conditioning_dim)
        x = torch.cat([x, conditioning], dim=1)
        assert_equals(
            x.shape,
            (
                batch_size,
                n_fingers * self.conv_1d.output_dim() + n_fingers * conditioning_dim,
            ),
        )

        # MLP
        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        return self(x, conditioning=conditioning)


class Simple_CNN_2D_1D_Model(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],  # n_x, n_y, n_z
        n_fingers: int,
        conditioning_dim: int = 7,
        n_tasks: int = 1,
        n_classes: int = 2,
        mlp_hidden_layers: List[int] = [32, 32],
        conv_2d_channels: List[int] = [32, 16, 8, 4],
        conv_1d_channels: List[int] = [4, 8],
        film_2d_hidden_layers: List[int] = [8, 8],
        film_1d_hidden_layers: List[int] = [8, 8],
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.n_fingers = n_fingers
        self.conditioning_dim = conditioning_dim
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        n_x, n_y, n_z = self.grid_shape

        self.cnn2d_film = CNN2DFiLM(
            input_shape=(n_x, n_y),
            conv_channels=conv_2d_channels,
            conditioning_dim=conditioning_dim,
            num_in_channels=1,
            film_hidden_layers=film_2d_hidden_layers,
        )

        self.flattened_2d_output_shape = (
            self.cnn2d_film.output_shape[0]
            * self.cnn2d_film.output_shape[1]
            * self.cnn2d_film.output_shape[2]
        )

        self.cnn1d_film = CNN1DFiLM(
            seq_len=n_z,
            conv_channels=conv_1d_channels,
            conditioning_dim=conditioning_dim,
            num_in_channels=self.flattened_2d_output_shape,
            film_hidden_layers=film_1d_hidden_layers,
        )

        self.flattened_1d_output_shape = (
            self.cnn1d_film.output_shape[0] * self.cnn1d_film.output_shape[1]
        )

        self.mlp = MLP(
            (self.flattened_1d_output_shape + self.conditioning_dim) * self.n_fingers,
            mlp_hidden_layers,
            n_classes * n_tasks,
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[
            0
        ]  # Hardcoding no leading batch dims on input -- probably good to check here.
        n_fingers = self.n_fingers
        n_x, n_y, n_z = self.grid_shape
        conditioning_dim = self.conditioning_dim

        # Check shapes
        assert_equals(x.shape, (batch_size, n_fingers, n_x, n_y, n_z))
        assert_equals(conditioning.shape, (batch_size, n_fingers, conditioning_dim))

        # Permute and expand stuff for correct 2D CNN batch shapes.
        x = x.permute(0, 1, 4, 2, 3)  # Put n_z as a batch dim.
        x = x.unsqueeze(-3)  # Add channel dim.
        conditioning_2d = conditioning.unsqueeze(2).expand(-1, -1, n_z, -1)

        # Forward 2D CNN pass.
        x = self.cnn2d_film(x, conditioning=conditioning_2d)
        assert_equals(
            x.shape,
            (batch_size, n_fingers, n_z, *self.cnn2d_film.output_shape),
        )

        x = x.flatten(-2, -1)  # flatten x/y

        # Flatten + permute stuff for correct 1D CNN batch shapes.
        x = x.flatten(-2, -1)  # Flatten 2DCNN channels + x/y dims.
        x = x.permute(0, 1, 3, 2)  # Put n_z as the sequence dim.
        assert_equals(
            x.shape,
            (batch_size, n_fingers, self.flattened_2d_output_shape, n_z),
        )

        # Forward 1D CNN pass.
        x = self.cnn1d_film(x, conditioning=conditioning)
        x = x.flatten(-2, -1)  # Flatten 1DCNN channels + z dim.
        assert_equals(
            x.shape,
            (batch_size, n_fingers, self.flattened_1d_output_shape),
        )

        # Add context one last time + flatten across fingers.
        x = torch.cat([x, conditioning], dim=-1)
        assert_equals(
            x.shape,
            (batch_size, n_fingers, self.flattened_1d_output_shape + conditioning_dim),
        )
        x = x.flatten(-2, -1)

        # Forward MLP pass.
        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        return self(x, conditioning=conditioning)


class Simple_CNN_1D_2D_Model(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],  # n_x, n_y, n_z
        n_fingers: int,
        conditioning_dim: int = 7,
        n_tasks: int = 1,
        n_classes: int = 2,
        mlp_hidden_layers: List[int] = [32, 32],
        conv_2d_channels: List[int] = [32, 16, 8, 4],
        conv_1d_channels: List[int] = [4, 8],
        film_2d_hidden_layers: List[int] = [8, 8],
        film_1d_hidden_layers: List[int] = [8, 8],
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.n_fingers = n_fingers
        self.conditioning_dim = conditioning_dim
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        n_x, n_y, n_z = self.grid_shape

        self.cnn2d_film = CNN2DFiLM(
            input_shape=(n_x, n_y),
            conv_channels=conv_2d_channels,
            conditioning_dim=conditioning_dim,
            num_in_channels=1,
            film_hidden_layers=film_2d_hidden_layers,
        )

        self.flattened_2d_output_shape = (
            self.cnn2d_film.output_shape[0]
            * self.cnn2d_film.output_shape[1]
            * self.cnn2d_film.output_shape[2]
        )

        self.cnn1d_film = CNN1DFiLM(
            seq_len=n_z,
            conv_channels=conv_1d_channels,
            conditioning_dim=conditioning_dim,
            num_in_channels=self.flattened_2d_output_shape,
            film_hidden_layers=film_1d_hidden_layers,
        )

        self.flattened_1d_output_shape = (
            self.cnn1d_film.output_shape[0] * self.cnn1d_film.output_shape[1]
        )

        self.mlp = MLP(
            (self.flattened_1d_output_shape + self.conditioning_dim) * self.n_fingers,
            mlp_hidden_layers,
            n_classes * n_tasks,
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[
            0
        ]  # Hardcoding no leading batch dims on input -- probably good to check here.
        n_fingers = self.n_fingers
        n_x, n_y, n_z = self.grid_shape
        conditioning_dim = self.conditioning_dim

        # Check shapes
        assert_equals(x.shape, (batch_size, n_fingers, n_x, n_y, n_z))
        assert_equals(conditioning.shape, (batch_size, n_fingers, conditioning_dim))

        # Permute and expand stuff for correct 2D CNN batch shapes.
        x = x.permute(0, 1, 4, 2, 3)  # Put n_z as a batch dim.
        x = x.unsqueeze(-3)  # Add channel dim.
        conditioning_2d = conditioning.unsqueeze(2).expand(-1, -1, n_z, -1)

        # Forward 2D CNN pass.
        x = self.cnn2d_film(x, conditioning=conditioning_2d)
        assert_equals(
            x.shape,
            (batch_size, n_fingers, n_z, *self.cnn2d_film.output_shape),
        )

        x = x.flatten(-2, -1)  # flatten x/y

        # Flatten + permute stuff for correct 1D CNN batch shapes.
        x = x.flatten(-2, -1)  # Flatten 2DCNN channels + x/y dims.
        x = x.permute(0, 1, 3, 2)  # Put n_z as the sequence dim.
        assert_equals(
            x.shape,
            (batch_size, n_fingers, self.flattened_2d_output_shape, n_z),
        )

        # Forward 1D CNN pass.
        x = self.cnn1d_film(x, conditioning=conditioning)
        x = x.flatten(-2, -1)  # Flatten 1DCNN channels + z dim.
        assert_equals(
            x.shape,
            (batch_size, n_fingers, self.flattened_1d_output_shape),
        )

        # Add context one last time + flatten across fingers.
        x = torch.cat([x, conditioning], dim=-1)
        assert_equals(
            x.shape,
            (batch_size, n_fingers, self.flattened_1d_output_shape + conditioning_dim),
        )
        x = x.flatten(-2, -1)

        # Forward MLP pass.
        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        return self(x, conditioning=conditioning)


class Simple_CNN_LSTM_Model(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],  # n_x, n_y, n_z
        n_fingers: int,
        conditioning_dim: int = 7,
        n_tasks: int = 1,
        n_classes: int = 2,
        mlp_hidden_layers: List[int] = [32, 32],
        conv_2d_channels: List[int] = [32, 16, 8, 4],
        film_2d_hidden_layers: List[int] = [8, 8],
        lstm_hidden_size: int = 32,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.n_fingers = n_fingers
        self.conditioning_dim = conditioning_dim
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        n_x, n_y, n_z = self.grid_shape

        # Will perform reduction to eliminate z dim after this.

        self.cnn2d_film = CNN2DFiLM(
            input_shape=(n_x, n_y),
            conv_channels=conv_2d_channels,
            conditioning_dim=conditioning_dim,
            num_in_channels=1,
            film_hidden_layers=film_2d_hidden_layers,
        )

        self.flattened_2d_output_shape = (
            self.cnn2d_film.output_shape[0]
            * self.cnn2d_film.output_shape[1]
            * self.cnn2d_film.output_shape[2]
        )

        self.lstm = LSTMModel(
            input_size=self.flattened_2d_output_shape,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
        )

        self.mlp = MLP(
            (self.lstm_hidden_size + self.conditioning_dim) * self.n_fingers,
            mlp_hidden_layers,
            n_classes * n_tasks,
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[
            0
        ]  # Hardcoding no leading batch dims on input -- probably good to check here.
        n_fingers = self.n_fingers
        n_x, n_y, n_z = self.grid_shape
        conditioning_dim = self.conditioning_dim

        # Check shapes
        assert_equals(x.shape, (batch_size, n_fingers, n_x, n_y, n_z))
        assert_equals(conditioning.shape, (batch_size, n_fingers, conditioning_dim))

        # Permute and expand stuff for correct 2D CNN batch shapes
        x = x.permute(0, 1, 4, 2, 3)  # Put n_z as a batch dim.
        x = x.unsqueeze(-3)  # Add channel dim.
        conditioning_2d = conditioning.unsqueeze(2).expand(-1, -1, n_z, -1)

        # Forward 2D CNN pass.
        x = self.cnn2d_film(x, conditioning=conditioning_2d)
        assert_equals(
            x.shape,
            (batch_size, n_fingers, n_z, *self.cnn2d_film.output_shape),
        )

        x = x.flatten(-3, -1)  # flatten x,y,channels
        assert_equals(
            x.shape,
            (batch_size, n_fingers, n_z, self.flattened_2d_output_shape),
        )

        # Flatten + permute stuff for correct LSTM batch shapes.
        x = x.reshape(
            -1, n_z, self.flattened_2d_output_shape
        )  # Flatten batch/num_fingers dims
        assert_equals(
            x.shape,
            (batch_size * n_fingers, n_z, self.flattened_2d_output_shape),
        )

        # Run LSTM over z dim.
        x = self.lstm(x)

        # Shape check
        assert_equals(
            x.shape,
            (
                batch_size * n_fingers,
                self.lstm_hidden_size,
            ),
        )

        # Reshape x and add conditioning for MLP.
        x = x.reshape(batch_size, n_fingers, self.lstm_hidden_size)
        x = torch.cat([x, conditioning], dim=-1)

        assert_equals(
            x.shape,
            (
                batch_size,
                n_fingers,
                self.lstm_hidden_size + conditioning_dim,
            ),
        )

        # Flatten num fingers.
        x = x.flatten(-2, -1)

        # Forward MLP pass.
        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        return self(x, conditioning=conditioning)


class DepthImage_CNN_2D_Model(nn.Module):
    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        n_fingers: int,
        conditioning_dim: int,
        conv_2d_film_hidden_layers: Tuple[int, ...],
        mlp_hidden_layers: Tuple[int, ...],
        use_pretrained_2d: bool = True,
        resnet_type_2d: ResnetType2d = ResnetType2d.RESNET18,
        pooling_method_2d: ConvOutputTo1D = ConvOutputTo1D.AVG_POOL_SPATIAL,
        n_tasks: int = 1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.n_fingers = n_fingers
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.conditioning_dim = conditioning_dim

        C, H, W = self.img_shape

        self.conv_2d = ConvEncoder2D(
            input_shape=(1, H, W),
            conditioning_dim=conditioning_dim,
            use_pretrained=use_pretrained_2d,
            pooling_method=pooling_method_2d,
            film_hidden_layers=conv_2d_film_hidden_layers,
            resnet_type=resnet_type_2d,
        )

        self.mlp = mlp(
            num_inputs=n_fingers * self.conv_2d.output_dim() * C
            + n_fingers * conditioning_dim,
            num_outputs=self.n_classes * self.n_tasks,
            hidden_layers=mlp_hidden_layers,
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_fingers = self.n_fingers
        C, H, W = self.img_shape
        conditioning_dim = self.conditioning_dim

        # Check shapes
        assert_equals(x.shape, (batch_size, n_fingers, C, H, W))
        assert_equals(conditioning.shape, (batch_size, n_fingers, conditioning_dim))

        # Batch n_fingers with batch_size
        # TODO: Shouldn't need to do this reshaping
        # but currently have asserts in ConvEncoder2D to sanity check
        x = x.reshape(batch_size * n_fingers * C, 1, H, W)
        conditioning = conditioning.reshape(batch_size * n_fingers, conditioning_dim)
        conditioning_repeated = conditioning.repeat_interleave(C, dim=0)
        assert_equals(
            conditioning_repeated.shape, (batch_size * n_fingers * C, conditioning_dim)
        )

        # Conv 2D
        x = self.conv_2d(x, conditioning=conditioning_repeated)
        assert_equals(x.shape, (batch_size * n_fingers * C, self.conv_2d.output_dim()))

        # Aggregate into one feature dimension
        x = x.reshape(batch_size, n_fingers * C * self.conv_2d.output_dim())

        # Concatenate with conditioning
        conditioning = conditioning.reshape(batch_size, n_fingers * conditioning_dim)
        x = torch.cat([x, conditioning], dim=1)
        assert_equals(
            x.shape,
            (
                batch_size,
                n_fingers * self.conv_2d.output_dim() * C
                + n_fingers * conditioning_dim,
            ),
        )

        # MLP
        all_logits = self.mlp(x)
        assert_equals(all_logits.shape, (batch_size, self.n_classes * self.n_tasks))
        all_logits = all_logits.reshape(batch_size, self.n_tasks, self.n_classes)
        return all_logits

    def get_all_logits(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        return self(x, conditioning=conditioning)


if __name__ == "__main__":
    batch_size = 2
    n_fingers = 2
    n_x, n_y, n_z = 32, 32, 32
    conditioning_dim = 7
    n_tasks = 1
    n_classes = 2
    print("=" * 80)
    print(f"batch_size: {batch_size}")
    print(f"n_fingers: {n_fingers}")
    print(f"n_x: {n_x}")
    print(f"n_y: {n_y}")
    print(f"n_z: {n_z}")
    print(f"conditioning_dim: {conditioning_dim}")
    print(f"n_tasks: {n_tasks}")
    print(f"n_classes: {n_classes}")
    print("=" * 80 + "\n")

    # 2D / 1D CNN.
    model = Simple_CNN_2D_1D_Model(
        grid_shape=(n_x, n_y, n_z),
        n_fingers=n_fingers,
        conditioning_dim=conditioning_dim,
        n_tasks=n_tasks,
        n_classes=n_classes,
        mlp_hidden_layers=[32, 32],
        conv_2d_channels=[32, 16, 8, 4],
        conv_1d_channels=[4, 8],
        film_2d_hidden_layers=[8, 8],
        film_1d_hidden_layers=[8, 8],
    )

    x = torch.zeros(batch_size, n_fingers, n_x, n_y, n_z)
    conditioning = torch.zeros(batch_size, n_fingers, conditioning_dim)

    all_logits = model.get_all_logits(x, conditioning)
    assert_equals(all_logits.shape, (batch_size, model.n_tasks, model.n_classes))
    print("For 2D / 1D CNN:")
    print(f"all_logits.shape: {all_logits.shape}" + "\n")

    # 1D / 2D CNN.
    model = Simple_CNN_1D_2D_Model(
        grid_shape=(32, 32, 32),
        n_fingers=2,
        conditioning_dim=7,
        n_tasks=n_tasks,
        n_classes=n_classes,
        mlp_hidden_layers=[32, 32],
        conv_2d_channels=[32, 16, 8, 4],
        conv_1d_channels=[4, 8],
        film_2d_hidden_layers=[8, 8],
        film_1d_hidden_layers=[8, 8],
    )
    all_logits = model.get_all_logits(x, conditioning)
    assert_equals(all_logits.shape, (batch_size, model.n_tasks, model.n_classes))
    print("For 1D / 2D CNN:")
    print(f"all_logits.shape: {all_logits.shape}" + "\n")

    # LSTM.
    model = Simple_CNN_LSTM_Model(
        grid_shape=(32, 32, 32),
        n_fingers=2,
        conditioning_dim=7,
        n_tasks=n_tasks,
        n_classes=n_classes,
        mlp_hidden_layers=[32, 32],
        conv_2d_channels=[32, 16, 8, 4],
        film_2d_hidden_layers=[8, 8],
        lstm_hidden_size=32,
        num_lstm_layers=1,
    )
    all_logits = model.get_all_logits(x, conditioning)
    assert_equals(all_logits.shape, (batch_size, model.n_tasks, model.n_classes))
    print("For LSTM:")
    print(f"all_logits.shape: {all_logits.shape}" + "\n")
