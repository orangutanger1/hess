"""CPU smoke run of DSN on MNIST. Not a test: this downloads data.

Usage:  python scripts/sanity_mnist.py [--steps 200] [--optimizer dsn|adamw]
"""

import argparse
import time

import torch
from torchvision import datasets, transforms

from dsn import DSN
from dsn.subspace import KrylovBuilder


def build_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 96),
        torch.nn.Tanh(),
        torch.nn.Linear(96, 10),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--optimizer", choices=["dsn", "adamw"], default="dsn")
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    torch.manual_seed(0)
    ds = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = build_model()
    if args.optimizer == "dsn":
        opt = DSN(model.parameters(), lr=args.lr, weight_decay=0.0,
                  builder=KrylovBuilder(k_max=8, m_recycle=4, tau=0.1))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    t0 = time.time()
    it = iter(loader)
    for step in range(args.steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        def closure():
            return torch.nn.functional.cross_entropy(model(x), y)

        if args.optimizer == "dsn":
            loss = opt.step(closure)
            t = opt.telemetry
            extra = (f" k={t.k} hvp={t.n_hvp} res={t.rel_residual:.3f} "
                     f"reuse={t.reuse_frac:.2f} fb={t.n_fallback}")
        else:
            opt.zero_grad()
            loss = closure()
            loss.backward()
            opt.step()
            extra = ""

        if step % 20 == 0:
            print(f"step {step:4d} loss {float(loss):.4f}{extra}")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
