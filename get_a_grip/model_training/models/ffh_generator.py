from typing import Tuple

import torch
import torch.nn as nn

from get_a_grip.model_training.models.components.fc_resblock import FCResBlock

DEFAULT_LATENT_DIM = 5  # From FFHNet paper


class Encoder(nn.Module):
    """FFHNet: https://ieeexplore.ieee.org/document/9811666
    Given an encoded object observation xb∈R4096, joint configuration θ∈R15, palm rotation R∈R3×3 and translation t∈R3,
    the encoder maps the distribution of grasps for an object observation into a latent space following a univariate gaussian distribution.
    We found two FC Resblocks for the FFHGenerator achieved the best performance achieved the best performance
    The inputs of the encoder (xb,θ,R,t) and the conditional input xb are also pre-processed by BN.
    """

    def __init__(self, n_pts: int, grasp_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.n_pts = n_pts
        self.grasp_dim = grasp_dim
        self.latent_dim = latent_dim

        self.n_inputs = n_pts + grasp_dim
        n_hidden = self.n_inputs  # Must be same for skip connections
        self.bn = nn.BatchNorm1d(self.n_inputs)
        self.fc_resblock_1 = FCResBlock(
            in_features=self.n_inputs, out_features=n_hidden
        )
        self.fc_resblock_2 = FCResBlock(in_features=n_hidden, out_features=n_hidden)

        self.fc_mu = nn.Linear(n_hidden, latent_dim)
        self.fc_sigma = nn.Linear(n_hidden, latent_dim)

    def forward(
        self, f_O: torch.Tensor, g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = f_O.shape[0]
        assert f_O.shape == (
            B,
            self.n_pts,
        ), f"Expected shape ({B}, {self.n_pts}), got {f_O.shape}"
        assert g.shape == (
            B,
            self.grasp_dim,
        ), f"Expected shape ({B}, {self.grasp_dim}), got {g.shape}"

        # Concat and batch norm
        x = torch.cat([f_O, g], dim=1)
        x = self.bn(x)

        # Resblocks
        x = self.fc_resblock_1(x) + x
        x = self.fc_resblock_2(x) + x

        # Output
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    """FFHNet: https://ieeexplore.ieee.org/document/9811666
    During inference, samples from the latent space can be decoded into grasps.
    We found two FC Resblocks for the FFHGenerator achieved the best performance achieved the best performance
    The inputs of the encoder (xb,θ,R,t) and the conditional input xb are also pre-processed by BN.
    """

    def __init__(self, n_pts: int, grasp_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.n_pts = n_pts
        self.grasp_dim = grasp_dim
        self.latent_dim = latent_dim

        self.n_inputs = n_pts + latent_dim
        hidden_dim = self.n_inputs  # Must be same for skip connections

        self.bn = nn.BatchNorm1d(self.n_pts)
        self.fc_resblock_1 = FCResBlock(
            in_features=self.n_inputs, out_features=hidden_dim
        )
        self.fc_resblock_2 = FCResBlock(in_features=hidden_dim, out_features=hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, grasp_dim)

    def forward(self, f_O: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = f_O.shape[0]
        assert f_O.shape == (
            B,
            self.n_pts,
        ), f"Expected shape ({B}, {self.n_pts}), got {f_O.shape}"
        assert z.shape == (
            B,
            self.latent_dim,
        ), f"Expected shape ({B}, {self.latent_dim}), got {z.shape}"

        # Concat and batch norm
        f_O = self.bn(f_O)
        x = torch.cat([f_O, z], dim=1)

        # Resblocks
        x = self.fc_resblock_1(x) + x
        x = self.fc_resblock_2(x) + x

        # Output
        x = self.fc_out(x)
        assert x.shape == (
            B,
            self.grasp_dim,
        ), f"Expected shape ({B}, {self.grasp_dim}), got {x.shape}"
        return x


class FFHGenerator(nn.Module):
    def __init__(self, n_pts: int, grasp_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.n_pts = n_pts
        self.grasp_dim = grasp_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(n_pts=n_pts, grasp_dim=grasp_dim, latent_dim=latent_dim)
        self.decoder = Decoder(n_pts=n_pts, grasp_dim=grasp_dim, latent_dim=latent_dim)

    def forward(
        self, f_O: torch.Tensor, g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encoder(f_O=f_O, g=g)
        eps = torch.randn_like(sigma, device=sigma.device)
        z = mu + sigma * eps
        g_hat = self.decoder(f_O=f_O, z=z)
        return mu, sigma, z, g_hat

    def encode(
        self, f_O: torch.Tensor, g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(f_O=f_O, g=g)

    def decode(self, f_O: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(f_O=f_O, z=z)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing FFHGenerator...")
    print("-" * 80)
    ffh_generator = FFHGenerator(
        n_pts=4096,
        grasp_dim=3 + 6 + 16,
        latent_dim=DEFAULT_LATENT_DIM,
    ).to(device)

    batch_size = 2
    f_O = torch.rand(batch_size, 4096).to(device)
    g = torch.rand(batch_size, 3 + 6 + 16).to(device)

    mu, sigma, z, g_hat = ffh_generator(f_O=f_O, g=g)

    assert mu.shape == (
        batch_size,
        DEFAULT_LATENT_DIM,
    ), f"Expected shape ({batch_size}, {DEFAULT_LATENT_DIM}), got {mu.shape}"
    assert sigma.shape == (
        batch_size,
        DEFAULT_LATENT_DIM,
    ), f"Expected shape ({batch_size}, {DEFAULT_LATENT_DIM}), got {sigma.shape}"
    assert z.shape == (
        batch_size,
        DEFAULT_LATENT_DIM,
    ), f"Expected shape ({batch_size}, {DEFAULT_LATENT_DIM}), got {z.shape}"
    assert g_hat.shape == (
        batch_size,
        3 + 6 + 16,
    ), f"Expected shape ({batch_size}, {3 + 6 + 16}), got {g_hat.shape}"
    print(
        f"mu.shape: {mu.shape}, sigma.shape: {sigma.shape}, z.shape: {z.shape}, g_hat.shape: {g_hat.shape}"
    )
    print(f"mu: {mu}")
    print(f"sigma: {sigma}")
    print(f"z: {z}")
    print(f"g_hat: {g_hat}")


if __name__ == "__main__":
    main()
