import pytest
import torch


@pytest.fixture(autouse=True)
def _float64_and_seed():
    """All numerical tests run in float64 with a fixed seed.

    Krylov and eigendecomposition assertions in float32 are flaky at the
    tolerances that actually catch bugs.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(prev)
