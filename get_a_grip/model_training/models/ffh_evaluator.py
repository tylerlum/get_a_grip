import torch

from get_a_grip.model_training.models.dex_evaluator import DexEvaluator


class FFHEvaluator(DexEvaluator):
    """Identical to DexEvaluator, but with a different name."""

    pass


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "-" * 80)
    print("Testing FFHEvaluator...")
    print("-" * 80)
    ffh_evaluator = FFHEvaluator(n_pts=4096, grasp_dim=3 + 6 + 16).to(device)

    batch_size = 2
    f_O = torch.rand(batch_size, 4096, device=device)
    g_0 = torch.rand(batch_size, 3 + 6 + 16, device=device)

    output = ffh_evaluator(f_O=f_O, g_0=g_0)

    assert output.shape == (
        batch_size,
        1,
    ), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
