from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from pplx_garden.distributed import ParallelGroup, ParallelLaunch
from pplx_garden.kernels.p2p_all_to_all import P2PAllToAll
from pplx_garden.utils import logging_utils
from pplx_garden.utils.math import round_up
from pplx_garden.utils.torch import has_tp
from tests.fabric import get_nets_per_gpu
from tests.markers import gpu_only, mark_ci_2gpu, mark_ci_4gpu, mark_fabric, mark_kernel
from tests.p2p_all_to_all.data import RankTestData

logger = logging_utils.get_logger(__name__)


def require_nets_per_gpu(n: int) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        get_nets_per_gpu() < n,
        reason=f"requires {n} NICs per GPU, got {get_nets_per_gpu()}",
    )


@dataclass
class _Config:
    world_size: int
    dp_size: int
    nets_per_gpu: int
    max_num_tokens: int
    num_experts: int
    hidden_dim: int
    hidden_dim_scale: Optional[int]
    max_private_tokens: Optional[int]
    num_experts_per_token: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    scale_dtype: Optional[torch.dtype]
    expert_padding: int
    nvlink_group: Optional[int]


def _act(x: torch.Tensor, x_scale: Optional[torch.Tensor]) -> torch.Tensor:
    if x_scale is None:
        return x * 2

    _, hidden_dim = x.shape
    _, hidden_dim_scale = x_scale.shape
    return x.to(torch.float32) * x_scale.repeat(1, hidden_dim // hidden_dim_scale) * 2


def _generator(device: torch.device, rank: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(rank)
    return generator


def _test_p2p_all_to_all_worker(
    device: torch.device,
    tp_group: Optional[ParallelGroup],
    global_group: Optional[ParallelGroup],
    config: _Config,
) -> None:
    assert tp_group is not None
    assert global_group is not None

    dp_rank = global_group.rank // tp_group.size
    num_dp_groups = global_group.size // tp_group.size

    max_num_tokens = config.max_num_tokens
    num_experts = config.num_experts
    hidden_dim = config.hidden_dim
    hidden_dim_scale = config.hidden_dim_scale
    num_experts_per_token = config.num_experts_per_token
    in_dtype = config.in_dtype
    out_dtype = config.out_dtype
    scale_dtype = config.scale_dtype

    num_local_experts = num_experts // global_group.size
    first_expert = global_group.rank * num_local_experts
    last_expert = min(first_expert + num_local_experts, num_experts)

    max_recv_tokens = max_num_tokens * num_local_experts * num_dp_groups

    # Set up dummy test data.
    rank_data = [
        RankTestData.create(
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            max_num_tokens=max_num_tokens,
            hidden_dim=hidden_dim,
            hidden_dim_scale=hidden_dim_scale,
            in_dtype=in_dtype,
            scale_dtype=scale_dtype,
            generator=_generator(device, rank),
            device=device,
        )
        for rank in range(num_dp_groups)
    ]
    local_rank = rank_data[dp_rank]
    ref_out_tokens = _act(local_rank.dp_x, local_rank.dp_x_scale).to(out_dtype)

    node_group: Optional[ParallelGroup]
    if config.nvlink_group is not None:
        assert config.nvlink_group > 0
        assert global_group.size % config.nvlink_group == 0
        node_group = global_group.slice_by_count(
            global_group.size // config.nvlink_group,
        )
    else:
        node_group = None

    # Instantiate the all-to-all kernel.
    all_to_all = P2PAllToAll(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        expert_padding=config.expert_padding,
        hidden_dim=hidden_dim,
        hidden_dim_scale=hidden_dim_scale,
        max_private_tokens=config.max_private_tokens,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scale_dtype=scale_dtype,
        num_experts_per_token=num_experts_per_token,
        nets_per_gpu=config.nets_per_gpu,
        device=device,
        dp_group=tp_group,
        node_group=node_group,
        global_group=global_group,
    )

    try:
        expected_num_tokens = torch.sum(
            torch.stack(
                [data.expected_num_tokens for data in rank_data],
                dim=0,
            ),
            dim=0,
            dtype=torch.int32,
        ).to("cpu")

        # Dispatch.
        expert_num_tokens = torch.empty(
            (num_local_experts,),
            dtype=torch.int32,
            device=device,
        )
        out_expert_x = torch.empty(
            (max_recv_tokens, hidden_dim),
            dtype=in_dtype,
            device=device,
        )
        out_tokens = torch.empty(
            (max_num_tokens, hidden_dim),
            dtype=out_dtype,
            device=device,
        )

        if hidden_dim_scale is not None or scale_dtype is not None:
            assert scale_dtype is not None
            assert hidden_dim_scale is not None
            out_expert_x_scale = torch.empty(
                (max_recv_tokens, hidden_dim_scale),
                dtype=scale_dtype,
                device=device,
            )
        else:
            out_expert_x_scale = None

        # Test run.
        all_to_all.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=local_rank.dp_x,
            dp_x_scale=local_rank.dp_x_scale,
            indices=local_rank.indices,
            weights=local_rank.weights,
            bound_m=None,
        )
        expert_y = _act(out_expert_x, out_expert_x_scale).to(out_dtype)
        all_to_all.combine(
            out_tokens=out_tokens,
            indices=local_rank.indices,
            weights=local_rank.weights,
            expert_y=expert_y,
            bound_m=local_rank.bound_m,
        )
        torch.cuda.synchronize()

        # Verify the token counts.
        expected_local_tokens = expected_num_tokens[first_expert:last_expert]
        torch.testing.assert_close(expected_local_tokens, expert_num_tokens.to("cpu"))

        # Verify the tokens.
        def hash_token(x: torch.Tensor) -> str:
            return ",".join(f"{v:.2f}" for v in x.tolist())

        tokens_on_rank = set()
        index = 0
        for n in expected_local_tokens.tolist():
            for token in out_expert_x[index : index + n]:
                tokens_on_rank.add(hash_token(token))

            index = round_up(index + n, config.expert_padding)
    except Exception:
        logger.exception("All-to-all failed")
        raise
    finally:
        logger.info("Stopping all-to-all")
        all_to_all.destroy()

    # Verify the tokens on the rank.
    num_missing = 0
    for i, (token, routes) in enumerate(
        zip(list(local_rank.dp_x), local_rank.indices.tolist())
    ):
        if not any(first_expert <= route < last_expert for route in routes):
            continue
        key = hash_token(token)
        if key not in tokens_on_rank:
            num_missing += 1
            logger.error(
                "Token %i: %s not found in output on rank %i (routed to %s)",
                i,
                key,
                dp_rank,
                ", ".join(str(route) for route in routes),
            )
    assert num_missing == 0, f"Missing {num_missing} tokens on rank {dp_rank}"

    # Verify the combine output.
    torch.testing.assert_close(out_tokens, ref_out_tokens)


@mark_fabric
@mark_kernel
@gpu_only
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                num_experts_per_token=2,
                max_private_tokens=None,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP32",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                num_experts_per_token=2,
                max_private_tokens=8,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP32-MIXED",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                max_private_tokens=8,
                num_experts_per_token=2,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=2,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP32-NVL",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-BF16",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=8,
                num_experts=16,
                hidden_dim=16,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=16,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-BF16-PADDED",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=2,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=16,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=torch.float32,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP8",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=128,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-NIC1-FP32",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=2,
                max_num_tokens=128,
                num_experts=256,
                hidden_dim=7168,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
                require_nets_per_gpu(2),
            ],
            id="TP4-NIC2-BF16",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=1,
                max_num_tokens=1,
                num_experts=4,
                hidden_dim=8,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=1,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-DP2-NIC1-BF16",
        ),
        pytest.param(
            _Config(
                world_size=8,
                dp_size=1,
                nets_per_gpu=get_nets_per_gpu(),
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
                nvlink_group=None,
            ),
            marks=[pytest.mark.skipif(not has_tp(8), reason="Requires 8 devices")],
            id="TP8-BF16",
        ),
        pytest.param(
            _Config(
                world_size=8,
                dp_size=1,
                nets_per_gpu=get_nets_per_gpu(),
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
                nvlink_group=None,
            ),
            marks=[pytest.mark.skipif(not has_tp(8), reason="Requires 8 devices")],
            id="TP8-FP8",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=56,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.float8_e4m3fn,
                out_dtype=torch.bfloat16,
                scale_dtype=torch.float32,
                expert_padding=1,
                nvlink_group=2,
            ),
            marks=[pytest.mark.skipif(not has_tp(8), reason="Requires 8 devices")],
            id="TP4-FP8-NVL2",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=8,
                num_experts=4,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=4,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=2,
            ),
            marks=[pytest.mark.skipif(not has_tp(8), reason="Requires 8 devices")],
            id="TP4-DP2-NVL2",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=1,
                num_experts=2,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=32,
                num_experts_per_token=1,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices")],
            id="TP2-EMPTY",
        ),
    ],
)
def test_p2p_all_to_all(config: _Config) -> None:
    ParallelLaunch(world_size=config.world_size, dp_size=config.dp_size).run(
        _test_p2p_all_to_all_worker,
        config,
    )
