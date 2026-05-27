import pytest
import torch

from pplx_garden.utils.torch import has_cuda, has_tp

mark_ci_2gpu = pytest.mark.ci_2gpu
mark_ci_4gpu = pytest.mark.ci_4gpu

mark_cuda = pytest.mark.cuda
mark_distributed = pytest.mark.distributed
mark_fabric = pytest.mark.fabric
mark_kernel = pytest.mark.kernel
mark_perf = pytest.mark.perf


def mark_tp(n: int) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not has_tp(n), reason=f"requires {n} GPUs")


gpu_only = pytest.mark.skipif(not has_cuda(), reason="test requires CUDA")
cpu_only = pytest.mark.cpu_only

all_devices = pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cuda"), marks=gpu_only, id="cuda"),
        pytest.param(torch.device("cpu"), id="cpu"),
    ],
)
