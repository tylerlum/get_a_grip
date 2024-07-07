from typing import Optional

import torch
import torch.nn as nn

from get_a_grip.model_training.models.components.fc_resblock import FCResBlock


class BpsEvaluatorModel(nn.Module):
    """Motivated by the DexDiffuser evaluator module.

    Adapted for use in our repo.

    See: https://github.com/qianbot/FFHNet/blob/4aa38dd6bd59bcf4b794ca872f409844579afa9f/FFHNet/models/networks.py#L243
    """

    def __init__(
        self,
        in_grasp: int,
        n_neurons: int = 2048,
        in_bps: int = 4096,
        cov_mcmc: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_bps + in_grasp)
        self.rb1 = FCResBlock(in_bps + in_grasp, n_neurons)
        self.rb2 = FCResBlock(in_bps + in_grasp + n_neurons, n_neurons)
        self.rb3 = FCResBlock(in_bps + in_grasp + n_neurons, n_neurons)
        self.out_success = nn.Linear(n_neurons, 3)
        self.dout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cov_mcmc is None:
            self.cov_mcmc = torch.diag(
                torch.tensor(
                    [0.005**2] * 3  # translations
                    + [0.025**2] * 6  # x and y axes
                    + [0.025**2] * 16  # joint angles
                    + [0.025**2] * 12,  # grasp directions
                    device=self.device,
                )
            )
        else:
            self.cov_mcmc = cov_mcmc

    def forward(self, f_O: torch.Tensor, g_O: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            f_O: The basis point set. Shape: (B, dim_BPS)
            g_O: The grasp features. Shape: (B, dim_grasp)
                We have dim_grasp = 3 + 6 + 16 + 3 * 4 = 37.
                The 6 rotation dims are the first two cols of the rot matrix.
                The extra 12 inputs are the grasp directions, which we provide to all.

        Returns:
            ys: The three labels for the grasp: y_coll, y_pick, y_eval.
        """
        X = torch.cat([f_O, g_O], dim=-1)

        X0 = self.bn1(X)
        X = self.rb1(X0)
        X = self.dout(X)
        X = self.rb2(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.rb3(torch.cat([X, X0], dim=1), final_nl=False)
        X = self.dout(X)
        X = self.out_success(X)
        p_success = self.sigmoid(X)
        return p_success


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing BpsEvaluator...")
    print("-" * 80)
    bps_evaluator = BpsEvaluatorModel(in_grasp=3 + 6 + 16 + 12, in_bps=4096).to(device)

    batch_size = 2
    f_O = torch.rand(batch_size, 4096, device=device)
    g_O = torch.rand(batch_size, 3 + 6 + 16 + 12, device=device)

    labels = bps_evaluator(f_O=f_O, g_O=g_O)
    PGS_0 = labels[..., -1]
    g_new = bps_evaluator.refine(f_O=f_O, g_O=g_O, num_steps=100, stage="all")
    PGS_new = bps_evaluator(f_O=f_O, g_O=g_new)[..., -1]
    print(f"PGS_0: {PGS_0}")
    print(f"PGS_new: {PGS_new}")
    breakpoint()

    assert labels.shape == (
        batch_size,
        3,
    ), f"Expected shape ({batch_size}, 3), got {labels.shape}"
    print(f"Output shape: {labels.shape}")
    print(f"Output: {labels}")


if __name__ == "__main__":
    main()
