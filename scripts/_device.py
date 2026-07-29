"""Device selection shared by scripts that train the MNIST MLP.

Synthetic-quadratic scripts (measure_rho's sibling gate/gain sweeps on
12-dimensional problems) are left on CPU: moving a handful of 12x12 tensors to
the GPU adds transfer overhead with no compute to amortize it against. This
module is for the scripts that actually build and train the 76k-parameter MLP.
"""

import torch


def add_device_arg(ap):
    ap.add_argument(
        "--device", type=str, default="auto",
        help="'auto' (default): cuda if available, else cpu. Or force 'cpu'/'cuda'.",
    )
    return ap


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False"
        )
    return torch.device(name)
