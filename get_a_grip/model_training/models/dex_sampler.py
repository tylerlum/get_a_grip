import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention

from get_a_grip.model_training.models.components.fc_resblock import FCResBlock
from get_a_grip.model_training.models.components.layers import (
    ConvOutputTo1D,
    PoolType,
    conv_encoder,
)


class ResBlock(nn.Module):
    """A residual block that can optionally change the number of channels.

    Haofei said their ResBlock impl was similar to this:
    github.com/scenediffuser/Scene-Diffuser/blob/4a62ca30a4b37bb6d7b538e512905c570c4ded7c/models/model/utils.py#L32
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels

        self.in_layers = nn.Sequential(
            nn.LayerNorm((self.in_channels, 1)),
            nn.SiLU(),
            nn.Conv1d(self.in_channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(self.emb_channels, self.out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm((self.out_channels, 1)),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out.unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


# UPDATE: this sucks, tried it and does worse than the one we made by guessing
# class DexSampler(nn.Module):
#     """The DexSampler from the DexDiffuser paper.

#     See: https://arxiv.org/pdf/2402.02989.
#     """

#     def __init__(self, dim_grasp: int = 37) -> None:
#         """Initialize the sampler."""
#         super().__init__()

#         # query path
#         self.res_block = ResBlock(
#             in_channels=dim_grasp,  # grasp features
#             emb_channels=1,  # time
#             dropout=0.1,
#             out_channels=512,  # we increase the channel dims to 512
#         )
#         self.sa = MultiheadAttention(
#             embed_dim=512,
#             num_heads=1,
#             batch_first=True,
#         )

#         # key/value path
#         self.fc_key = nn.Linear(512, 512)
#         self.fc_value = nn.Linear(512, 512)

#         # output path
#         self.ca = MultiheadAttention(
#             embed_dim=512,
#             num_heads=1,
#             batch_first=True,
#         )
#         self.fc_output = nn.Linear(512, dim_grasp)  # output the noise for the diffusion model

#     def forward(self, f_O: torch.Tensor, g_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         """Forward pass through the sampler.

#         :param f_O: the basis point set. Shape: (B, 4096).
#         :param g_t: the grasp features. Shape: (B, 37).
#         :param t: the timestep. Shape: (B, 1).
#         :return: the noise for the diffusion model. Shape: (B, 37).
#         """
#         # key/value path
#         f_O_seq = f_O.reshape((*f_O.shape[:-1], 8, 512))  # (B, 4096) -> (B, 8, 512)
#         x_k = self.fc_key(f_O_seq)  # (B, 8, 512)
#         x_v = self.fc_value(f_O_seq)  # (B, 8, 512)

#         # query path
#         x_q = self.res_block(g_t.unsqueeze(-1), t)  # (B, 512, 1)
#         x_q = x_q.transpose(-1, -2)  # (B, 1, 512)

#         # output path
#         x, _ = self.sa(query=x_q, key=x_k, value=x_v)  # (B, 1, 512)
#         x = x.squeeze(-2)  # (B, 512)
#         eps = self.fc_output(x)  # (B, 37)  # output noise
#         return eps


class DexSampler(nn.Module):
    """DexDiffuser: https://arxiv.org/pdf/2402.02989
    It takes three inputs: fO, gt, and t and outputs εˆt for grasp denoising.
    Its input fO is processed into a key-value pair while gt at time t is processed into a query using a self-attention block.
    The key-value-query triplet is then embedded using a crossattention block and used to compute ε
    """

    def __init__(
        self, n_pts: int, grasp_dim: int, d_model: int, virtual_seq_len: int
    ) -> None:
        """Attention needs a seq_len dimension, but the input doesn't have it, so we create a virtual seq_len dimension"""
        super().__init__()
        self.n_pts = n_pts
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

        # Grasp-BPS cross attention
        self.ca_fc_key = nn.Linear(n_pts, d_model * S)
        self.ca_fc_value = nn.Linear(n_pts, d_model * S)
        self.cross_attention = MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=False
        )

        self.fc_out = nn.Linear(d_model * S, grasp_dim)

    def forward(
        self, f_O: torch.Tensor, g_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        B = f_O.shape[0]
        S = self.virtual_seq_len
        assert f_O.shape == (
            B,
            self.n_pts,
        ), f"Expected shape ({B}, {self.n_pts}), got {f_O.shape}"
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


class NerfSampler(nn.Module):
    """ """

    def __init__(
        self,
        global_grid_shape: tuple[int, int, int, int],
        grasp_dim: int,
        d_model: int,
        virtual_seq_len: int,
        conv_channels: tuple[int, ...],
        HACK_MODE_FOR_PERFORMANCE: bool = False,
    ) -> None:
        super().__init__()
        self.global_grid_shape = global_grid_shape
        assert (
            len(global_grid_shape) == 4
        ), f"Expected 4D shape, got {global_grid_shape}"
        self.grasp_dim = grasp_dim
        self.d_model = d_model
        self.virtual_seq_len = virtual_seq_len
        self.HACK_MODE_FOR_PERFORMANCE = HACK_MODE_FOR_PERFORMANCE
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
        if self.HACK_MODE_FOR_PERFORMANCE:
            if not hasattr(self, "HACK_STORED_f_O"):
                f_O = self.conv(f_O)
                self.HACK_STORED_f_O = f_O
            else:
                f_O = self.HACK_STORED_f_O
                assert f_O.shape[0] <= B, f"Expected batch size <= {B}, got {f_O}"
                f_O = f_O[:B]

            # Print only a max of 10 times
            if not hasattr(self, "HACK_PRINT_COUNT"):
                self.HACK_PRINT_COUNT = 0
            if self.HACK_PRINT_COUNT < 10:
                print("HACK_MODE_FOR_PERFORMANCE")
                self.HACK_PRINT_COUNT += 1
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
    print("Testing DexSampler...")
    print("-" * 80)
    dex_sampler = DexSampler(
        n_pts=4096, grasp_dim=37, d_model=128, virtual_seq_len=4
    ).to(device)
    batch_size = 2
    f_O = torch.rand(batch_size, 4096).to(device)
    g_t = torch.rand(batch_size, 37).to(device)
    t = torch.rand(batch_size, 1).to(device)
    output = dex_sampler(f_O=f_O, g_t=g_t, t=t)
    print(f"Output shape: {output.shape}")

    print("\n" + "-" * 80)
    print("Testing NerfSampler...")
    print("-" * 80)

    nerf_sampler = NerfSampler(
        global_grid_shape=(4, 30, 30, 30),
        grasp_dim=37,
        d_model=128,
        virtual_seq_len=8,
        conv_channels=(32, 64, 128),
    ).to(device)

    batch_size = 2
    f_O = torch.rand(batch_size, 4, 30, 30, 30).to(device)
    g_t = torch.rand(batch_size, 37).to(device)
    t = torch.rand(batch_size, 1).to(device)
    output = nerf_sampler(f_O=f_O, g_t=g_t, t=t)
    print(f"Output shape: {output.shape}")

    # dex_sampler = DexSampler(
    #     n_pts=4096, grasp_dim=3 + 6 + 16 + 3 * 4, d_model=128, virtual_seq_len=4
    # ).to(device)

    # batch_size = 2
    # f_O = torch.rand(batch_size, 4096).to(device)
    # g_t = torch.rand(batch_size, 3 + 6 + 16 + 3 * 4).to(device)
    # t = torch.rand(batch_size, 1).to(device)

    # output = dex_sampler(f_O=f_O, g_t=g_t, t=t)

    # assert output.shape == (
    #     batch_size,
    #     3 + 6 + 16 + 3 * 4,
    # ), f"Expected shape ({batch_size}, 3 + 6 + 16 + 3 * 4), got {output.shape}"
    # print(f"Output shape: {output.shape}")
    # print(f"Output: {output}")


if __name__ == "__main__":
    main()
