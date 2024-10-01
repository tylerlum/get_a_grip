from typing import Tuple

import torch
import torch.nn as nn

from get_a_grip.model_training.models.components.layers import (
    ConvOutputTo1D,
    PoolType,
    conv_encoder,
    mlp,
)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class CNN3D_Model(nn.Module):
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


class CNN3D_CNN3D_Model(nn.Module):
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


if __name__ == "__main__":
    batch_size = 2
    n_fingers = 2
    n_x, n_y, n_z = 32, 32, 32
    n_tasks = 1
    n_classes = 2
    print("=" * 80)
    print(f"batch_size: {batch_size}")
    print(f"n_fingers: {n_fingers}")
    print(f"n_x: {n_x}")
    print(f"n_y: {n_y}")
    print(f"n_z: {n_z}")
    print(f"n_tasks: {n_tasks}")
    print(f"n_classes: {n_classes}")
    print("=" * 80 + "\n")

    x = torch.zeros(batch_size, n_fingers, n_x, n_y, n_z)

    # CNN 3D.
    model = CNN3D_Model(
        input_shape=(n_fingers, n_x, n_y, n_z),
        conv_channels=(32, 16, 8, 4),
        mlp_hidden_layers=(32, 32),
        n_fingers=2,
        n_tasks=n_tasks,
        n_classes=n_classes,
    )
    all_logits = model.get_all_logits(x)
    assert_equals(all_logits.shape, (batch_size, model.n_tasks, model.n_classes))
    print("For CNN 3D:")
    print(f"all_logits.shape: {all_logits.shape}" + "\n")
