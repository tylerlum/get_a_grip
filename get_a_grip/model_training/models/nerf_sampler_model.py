from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention

from get_a_grip.model_training.models.components.fc_resblock import FCResBlock
from get_a_grip.model_training.models.components.layers import (
    ConvOutputTo1D,
    PoolType,
    conv_encoder,
)


class NerfSamplerModel(nn.Module):
    def __init__(
        self,
        global_grid_shape: Tuple[int, int, int, int],
        grasp_dim: int,
        d_model: int = 128,
        virtual_seq_len: int = 4,
        conv_channels: Tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.global_grid_shape = global_grid_shape
        assert (
            len(global_grid_shape) == 4
        ), f"Expected 4D shape, got {global_grid_shape}"
        self.grasp_dim = grasp_dim
        self.d_model = d_model
        self.virtual_seq_len = virtual_seq_len
        S = virtual_seq_len

        # Grasp self attention
        self.resblock = FCResBlock(grasp_dim + 1, d_model * S)
        self.sa_fc_query = nn.Linear(d_model, d_model)
        self.sa_fc_key = nn.Linear(d_model, d_model)
        self.sa_fc_value = nn.Linear(d_model, d_model)
        self.self_attention = MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=False
        )

        # Conv encode nerf grid
        self.conv = conv_encoder(
            input_shape=global_grid_shape,
            conv_channels=conv_channels,
            pool_type=PoolType.MAX,
            dropout_prob=0.1,
            conv_output_to_1d=ConvOutputTo1D.AVG_POOL_SPATIAL,
        )
        # Get conv output shape
        example_batch_size = 2
        example_input = torch.zeros(example_batch_size, *global_grid_shape)
        example_input = example_input.reshape(example_batch_size, *global_grid_shape)
        conv_output = self.conv(example_input)
        self.conv_output_dim = conv_output.shape[-1]
        assert conv_output.shape == (example_batch_size, self.conv_output_dim)

        self.ca_fc_key = nn.Linear(self.conv_output_dim, d_model * S)
        self.ca_fc_value = nn.Linear(self.conv_output_dim, d_model * S)
        self.cross_attention = MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=False
        )

        self.fc_out = nn.Linear(d_model * S, grasp_dim)

        # HACK: For performance, we can store the conv output and reuse it if we can assume that the conv input is the same
        self._HACK_MODE_FOR_PERFORMANCE: bool = False

    def forward(
        self, f_O: torch.Tensor, g_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        B = f_O.shape[0]
        S = self.virtual_seq_len
        assert f_O.shape == (
            B,
            *self.global_grid_shape,
        ), f"Expected shape ({B}, {self.global_grid_shape}), got {f_O.shape}"
        assert g_t.shape == (
            B,
            self.grasp_dim,
        ), f"Expected shape ({B}, {self.grasp_dim}), got {g_t.shape}"
        assert t.shape == (
            B,
            1,
        ), f"Expected shape ({B}, 1), got {t.shape}"

        # Grasp self attention
        x = torch.cat([g_t, t], dim=-1)
        x = self.resblock(x).reshape(B, S, self.d_model).permute(1, 0, 2)
        sa_query = self.sa_fc_query(x)
        sa_key = self.sa_fc_key(x)
        sa_value = self.sa_fc_value(x)
        ca_query, _ = self.self_attention(key=sa_key, value=sa_value, query=sa_query)
        assert ca_query.shape == (
            S,
            B,
            self.d_model,
        ), f"Expected shape ({S}, {B}, {self.d_model}), got {ca_query.shape}"

        # Grasp-BPS cross attention
        if self._HACK_MODE_FOR_PERFORMANCE:
            if not hasattr(self, "_HACK_STORED_f_O"):
                f_O = self.conv(f_O)
                self._HACK_STORED_f_O = f_O
            else:
                f_O = self._HACK_STORED_f_O
                assert f_O.shape[0] <= B, f"Expected batch size <= {B}, got {f_O}"
                f_O = f_O[:B]

            # Print only a max of 3 times
            if not hasattr(self, "_HACK_PRINT_COUNT"):
                self._HACK_PRINT_COUNT = 0
            if self._HACK_PRINT_COUNT < 3:
                print("_HACK_MODE_FOR_PERFORMANCE")
                self._HACK_PRINT_COUNT += 1
        else:
            f_O = self.conv(f_O)

        assert f_O.shape == (
            B,
            self.conv_output_dim,
        ), f"Expected shape ({B}, {self.conv_output_dim}), got {f_O.shape}"

        ca_key = self.ca_fc_key(f_O).reshape(B, S, self.d_model).permute(1, 0, 2)
        ca_value = self.ca_fc_value(f_O).reshape(B, S, self.d_model).permute(1, 0, 2)
        eps, _ = self.cross_attention(key=ca_key, value=ca_value, query=ca_query)
        assert eps.shape == (
            S,
            B,
            self.d_model,
        ), f"Expected shape ({S}, {B}, {self.d_model}), got {eps.shape}"

        # Output
        eps = eps.permute(1, 0, 2).reshape(B, self.d_model * S)
        eps = self.fc_out(eps)
        assert eps.shape == (
            B,
            self.grasp_dim,
        ), f"Expected shape ({B}, {self.grasp_dim}), got {eps.shape}"

        return eps


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing NerfSampler...")
    print("-" * 80)

    nerf_sampler = NerfSamplerModel(
        global_grid_shape=(4, 30, 30, 30),
        grasp_dim=37,
        d_model=128,
        virtual_seq_len=8,
        conv_channels=(32, 64, 128),
    ).to(device)

    batch_size = 2
    f_O = torch.rand(batch_size, 4, 30, 30, 30, device=device)
    g_t = torch.rand(batch_size, 37, device=device)
    t = torch.rand(batch_size, 1, device=device)
    output = nerf_sampler(f_O=f_O, g_t=g_t, t=t)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
