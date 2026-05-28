# ruff: noqa: E402

import os
from urllib.parse import urlparse

import pytest
import torch

from pplx_garden.distributed import ParallelLaunch
from tests.markers import (
    gpu_only,
    mark_cuda,
    mark_distributed,
    mark_fabric,
    mark_kernel,
)
from tests.p2p_all_to_all.test_p2p_all_to_all import (
    _Config,
    _test_p2p_all_to_all_worker,
)


def _node_rank() -> int:
    if "PPLX_NODE_RANK" in os.environ:
        return int(os.environ["PPLX_NODE_RANK"])
    return int(os.environ["SLURM_NODEID"])


def _nets_per_gpu() -> int:
    return int(os.environ.get("PPLX_NETS_PER_GPU", "1"))


def _init_method(port_offset: int) -> str:
    if "PPLX_INIT_HOST" in os.environ:
        port = int(os.environ.get("PPLX_INIT_PORT", "29500")) + port_offset
        return f"tcp://{os.environ['PPLX_INIT_HOST']}:{port}"

    init_method = os.environ["PPLX_INIT_METHOD"]
    parsed = urlparse(init_method)
    if parsed.scheme != "tcp":
        return init_method

    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Invalid tcp init method: {init_method}")
    return f"tcp://{parsed.hostname}:{parsed.port + port_offset}"


@mark_fabric
@mark_kernel
@mark_cuda
@mark_distributed
@gpu_only
@pytest.mark.multinode
@pytest.mark.parametrize(
    ("config", "port_offset"),
    [
        pytest.param(
            _Config(
                world_size=8,
                dp_size=1,
                nets_per_gpu=_nets_per_gpu(),
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=4,
            ),
            0,
            id="EP8-TP1-BF16",
        ),
        pytest.param(
            _Config(
                world_size=8,
                dp_size=1,
                nets_per_gpu=_nets_per_gpu(),
                max_num_tokens=256,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=56,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.float8_e4m3fn,
                out_dtype=torch.bfloat16,
                scale_dtype=torch.float32,
                expert_padding=1,
                nvlink_group=4,
            ),
            1,
            id="EP8-TP1-FP8",
        ),
        pytest.param(
            _Config(
                world_size=8,
                dp_size=2,
                nets_per_gpu=_nets_per_gpu(),
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=4,
            ),
            2,
            id="EP8-TP2-BF16",
        ),
        pytest.param(
            _Config(
                world_size=8,
                dp_size=4,
                nets_per_gpu=_nets_per_gpu(),
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=4,
            ),
            3,
            id="EP8-TP4-BF16",
        ),
    ],
)
def test_p2p_all_to_all_multinode(config: _Config, port_offset: int) -> None:
    ParallelLaunch(
        world_size=config.world_size,
        dp_size=config.dp_size,
        init_method=_init_method(port_offset),
        node_rank=_node_rank(),
    ).run(
        _test_p2p_all_to_all_worker,
        config,
    )
