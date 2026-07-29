"""CIFAR-10 benchmark model for the autoresearch harness.

Companion to sanity_mnist.py's build_model(), same shape of contract
(no-arg factory returning an untrained nn.Sequential), but exercises the
part of a candidate optimizer's state machine that MNIST's pure-Linear MLP
never does: 4D conv weight tensors. A candidate that only special-cases
p.dim() == 2 silently falls back to its 1D/element-wise path for every
conv layer -- this benchmark is what would catch that.

Small on purpose: this has to train something meaningful within a
CPU-bound fixed wall-clock budget, same constraint sanity_mnist.py's
76k-param MLP was sized for.
"""

import torch


def build_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # 32x32 -> 16x16
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # 16x16 -> 8x8
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 8 * 8, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    )
