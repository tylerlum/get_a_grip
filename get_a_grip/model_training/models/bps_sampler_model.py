import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention

from get_a_grip.model_training.models.components.fc_resblock import FCResBlock


class BpsSamplerModel(nn.Module):
    """Motivated by DexDiffuser: https://arxiv.org/pdf/2402.02989
    It takes three inputs: fO, gt, and t and outputs εˆt for grasp denoising.
    Its input fO is processed into a key-value pair while gt at time t is processed into a query using a self-attention block.
    The key-value-query triplet is then embedded using a crossattention block and used to compute ε
    """

    def __init__(
        self, n_pts: int, grasp_dim: int, d_model: int = 128, virtual_seq_len: int = 4
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing BpsSampler...")
    print("-" * 80)
    bps_sampler = BpsSamplerModel(
        n_pts=4096, grasp_dim=37, d_model=128, virtual_seq_len=4
    ).to(device)
    batch_size = 2
    f_O = torch.rand(batch_size, 4096, device=device)
    g_t = torch.rand(batch_size, 37, device=device)
    t = torch.rand(batch_size, 1, device=device)
    output = bps_sampler(f_O=f_O, g_t=g_t, t=t)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
