# ruff: noqa: B023

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from pplx_garden.distributed import ParallelGroup, ParallelLaunch
from pplx_garden.kernels.p2p_all_to_all import P2PAllToAll
from pplx_garden.utils import logging_utils
from pplx_garden.utils.math import Statistics, round_up
from pplx_garden.utils.torch import profile_range, str_to_dtype
from tests.p2p_all_to_all.data import RankTestData

logger = logging_utils.get_logger("bench_all_to_all")


def rand_topk_idx(
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    scores = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    scores = scores.abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    return topk_idx.to(torch.uint32)


def act(x: torch.Tensor, x_scale: Optional[torch.Tensor]) -> torch.Tensor:
    if x_scale is None:
        return x * 2

    b, h = x.shape
    _, hs = x_scale.shape
    x_reshaped = x.view(b, hs, h // hs)
    result = x_reshaped.to(torch.float32)
    result *= x_scale.view(b, hs, 1)
    result *= 2
    return result.view(b, h)


def make_rng(device: torch.device, rank: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(rank + 123)
    return generator


@dataclass(slots=True)
class AllToAllConfig:
    nets_per_gpu: int
    max_num_tokens: int
    max_private_tokens: int
    num_experts: int
    hidden_dim: int
    hidden_dim_scale: Optional[int]
    num_experts_per_token: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    scale_dtype: Optional[torch.dtype]
    nvlink: Optional[int]

    @property
    def dispatch_bytes(self) -> int:
        dispatch_token_dim = self.hidden_dim * self.in_dtype.itemsize
        if self.hidden_dim_scale is not None:
            assert self.scale_dtype is not None
            dispatch_token_dim += self.hidden_dim_scale * self.scale_dtype.itemsize
        return self.max_num_tokens * self.num_experts_per_token * dispatch_token_dim

    @property
    def combine_bytes(self) -> int:
        combine_token_dim = self.hidden_dim * self.out_dtype.itemsize
        return self.max_num_tokens * self.num_experts_per_token * combine_token_dim


class AllToAllResource:
    def __init__(
        self,
        device: torch.device,
        dp_group: ParallelGroup,
        global_group: ParallelGroup,
        cfg: AllToAllConfig,
    ) -> None:
        self.device = device
        self.dp_group = dp_group
        self.global_group = global_group
        self.cfg = cfg

        self.expert_padding = expert_padding = 1
        self.dp_rank = global_group.rank // dp_group.size
        self.num_dp_groups = num_dp_groups = global_group.size // dp_group.size
        self.num_local_experts = num_local_experts = (
            cfg.num_experts // global_group.size
        )
        self.num_tokens = num_tokens = num_dp_groups * cfg.max_num_tokens
        max_recv_tokens = round_up(
            max(
                min(
                    num_tokens * cfg.num_experts_per_token
                    + num_local_experts * (expert_padding - 1),
                    num_tokens * num_local_experts,
                ),
                num_local_experts * expert_padding,
            ),
            expert_padding,
        )

        node_group: Optional[ParallelGroup] = None
        if cfg.nvlink is not None:
            assert 0 < cfg.nvlink <= min(8, global_group.size, 8)
            node_group = global_group.slice_by_count(global_group.size // cfg.nvlink)

        # Instantiate the all-to-all kernel.
        self.all_to_all = P2PAllToAll(
            max_num_tokens=cfg.max_num_tokens,
            num_experts=cfg.num_experts,
            expert_padding=expert_padding,
            hidden_dim=cfg.hidden_dim,
            hidden_dim_scale=cfg.hidden_dim_scale,
            max_private_tokens=cfg.max_private_tokens,
            in_dtype=cfg.in_dtype,
            out_dtype=cfg.out_dtype,
            scale_dtype=cfg.scale_dtype,
            num_experts_per_token=cfg.num_experts_per_token,
            nets_per_gpu=cfg.nets_per_gpu,
            device=device,
            dp_group=dp_group,
            node_group=node_group,
            global_group=global_group,
        )

        # Allocate buffers.
        self.expert_num_tokens = torch.empty(
            (num_local_experts,),
            dtype=torch.int32,
            device=device,
        )
        self.out_expert_x = torch.empty(
            (max_recv_tokens, cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device,
        )
        self.out_tokens = torch.empty(
            (cfg.max_num_tokens, cfg.hidden_dim),
            dtype=cfg.out_dtype,
            device=device,
        )
        self.expert_y = torch.empty(
            (max_recv_tokens, cfg.hidden_dim),
            dtype=cfg.out_dtype,
            device=device,
        )
        self.out_expert_x_scale: torch.Tensor | None = None
        if cfg.hidden_dim_scale is not None or cfg.scale_dtype is not None:
            assert cfg.scale_dtype is not None
            assert cfg.hidden_dim_scale is not None
            self.out_expert_x_scale = torch.empty(
                (max_recv_tokens, cfg.hidden_dim_scale),
                dtype=cfg.scale_dtype,
                device=device,
            )

    def create_rank_data(self, dp_rank: int) -> RankTestData:
        return RankTestData.create(
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
            max_num_tokens=self.cfg.max_num_tokens,
            hidden_dim=self.cfg.hidden_dim,
            hidden_dim_scale=self.cfg.hidden_dim_scale,
            in_dtype=self.cfg.in_dtype,
            scale_dtype=self.cfg.scale_dtype,
            generator=make_rng(self.device, dp_rank),
            device=self.device,
        )


def correctness_check(r: AllToAllResource) -> None:
    expected_num_tokens_list = [
        RankTestData.rand_indices_and_count(
            num_experts=r.cfg.num_experts,
            num_experts_per_token=r.cfg.num_experts_per_token,
            max_num_tokens=r.cfg.max_num_tokens,
            generator=make_rng(r.device, dp_rank),
            device=r.device,
        )[1]
        for dp_rank in range(r.num_dp_groups)
    ]
    expected_num_tokens = torch.sum(
        torch.stack(expected_num_tokens_list, dim=0),
        dim=0,
        dtype=torch.int32,
    ).to("cpu")

    local_rank = r.create_rank_data(r.dp_rank)
    ref_out_tokens = act(local_rank.dp_x, local_rank.dp_x_scale).to(r.cfg.out_dtype)

    # Test run.
    r.all_to_all.dispatch(
        out_expert_num_tokens=r.expert_num_tokens,
        out_expert_x=r.out_expert_x,
        out_expert_x_scale=r.out_expert_x_scale,
        dp_x=local_rank.dp_x,
        dp_x_scale=local_rank.dp_x_scale,
        indices=local_rank.indices,
        weights=local_rank.weights,
    )
    expert_y = act(r.out_expert_x, r.out_expert_x_scale).to(r.cfg.out_dtype)
    r.all_to_all.combine(
        out_tokens=r.out_tokens,
        indices=local_rank.indices,
        weights=local_rank.weights,
        expert_y=expert_y,
        bound_m=local_rank.bound_m,
    )
    torch.cuda.synchronize()

    # Verify the token counts.
    first_expert = r.global_group.rank * r.num_local_experts
    last_expert = min(first_expert + r.num_local_experts, r.cfg.num_experts)
    expected_local_tokens = expected_num_tokens[first_expert:last_expert]
    torch.testing.assert_close(expected_local_tokens, r.expert_num_tokens.to("cpu"))

    # Verify the tokens.
    def hash_token(x: torch.Tensor) -> str:
        return ",".join(f"{v:.2f}" for v in x.tolist())

    tokens_on_rank = set()
    index = 0
    for n in expected_local_tokens.tolist():
        for token in r.out_expert_x[index : index + n]:
            tokens_on_rank.add(hash_token(token))

        index = round_up(index + n, r.expert_padding)

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
                r.dp_rank,
                ", ".join(str(route) for route in routes),
            )
    assert num_missing == 0, f"Missing {num_missing} tokens on rank {r.dp_rank}"

    # Verify the combine output.
    torch.testing.assert_close(r.out_tokens, ref_out_tokens)


def benchmark(
    r: AllToAllResource, num_warmup: int, num_repeats: int, output: Path, verbose: bool = False
) -> None:
    logger.info("Starting benchmark setup")
    local_rank = r.create_rank_data(r.dp_rank)
    rng = make_rng(r.device, r.dp_rank)

    def dispatch() -> None:
        r.all_to_all.dispatch(
            out_expert_num_tokens=r.expert_num_tokens,
            out_expert_x=r.out_expert_x,
            out_expert_x_scale=r.out_expert_x_scale,
            dp_x=local_rank.dp_x,
            dp_x_scale=local_rank.dp_x_scale,
            indices=local_rank.indices,
            weights=local_rank.weights,
            bound_m=local_rank.bound_m,
            do_send=True,
            do_recv=True,
        )

    def combine() -> None:
        r.all_to_all.combine(
            out_tokens=r.out_tokens,
            indices=local_rank.indices,
            weights=local_rank.weights,
            expert_y=r.expert_y,
            bound_m=local_rank.bound_m,
            do_send=True,
            do_recv=True,
        )

    # Combined Benchmark Events
    dispatch_events = []
    combine_events = []
    for _ in range(num_warmup + num_repeats):
        d_start = torch.cuda.Event(enable_timing=True)
        d_end = torch.cuda.Event(enable_timing=True)
        dispatch_events.append((d_start, d_end))

        c_start = torch.cuda.Event(enable_timing=True)
        c_end = torch.cuda.Event(enable_timing=True)
        combine_events.append((c_start, c_end))

    logger.info("[Rank %d] Starting lockstep warmup (%d iterations)", r.global_group.rank, num_warmup)
    last_report_time = time.time()
    for i in range(num_warmup + num_repeats):
        if i == num_warmup:
            if r.global_group.rank == 0:
                logger.info("Completed warmup. Starting benchmark (%d iterations)", num_repeats)
            torch.cuda.profiler.start()

        # Update indices
        if verbose:
            logger.info("[Rank %d] Iteration %d - Generating indices...", r.global_group.rank, i + 1)
        local_rank.indices = rand_topk_idx(
            r.cfg.max_num_tokens,
            r.cfg.num_experts,
            r.cfg.num_experts_per_token,
            rng,
            r.device,
        )

        # Time Dispatch
        if verbose:
            logger.info("[Rank %d] Iteration %d - Starting dispatch...", r.global_group.rank, i + 1)
        d_start, d_end = dispatch_events[i]
        d_start.record()
        dispatch()
        d_end.record()
        if verbose:
            logger.info("[Rank %d] Iteration %d - Dispatch completed", r.global_group.rank, i + 1)

        # Time Combine
        if verbose:
            logger.info("[Rank %d] Iteration %d - Starting combine...", r.global_group.rank, i + 1)
        c_start, c_end = combine_events[i]
        c_start.record()
        combine()
        c_end.record()
        if verbose:
            logger.info("[Rank %d] Iteration %d - Combine completed", r.global_group.rank, i + 1)

        if verbose:
            logger.info("[Rank %d] Iteration %d - Synchronizing CUDA...", r.global_group.rank, i + 1)
            torch.cuda.synchronize()
            logger.info("[Rank %d] Iteration %d - Synchronized successfully", r.global_group.rank, i + 1)
            logger.info("[Rank %d] Iteration %d - Entering global process group barrier...", r.global_group.rank, i + 1)
            r.global_group.barrier()
            logger.info("[Rank %d] Iteration %d - Passed global barrier", r.global_group.rank, i + 1)
            if (i + 1) % 10 == 0 or i < 5:
                logger.info("[Rank %d] Iteration %d/%d completed", r.global_group.rank, i + 1, num_warmup + num_repeats)

        if (i + 1) % 100 == 0 or i + 1 == num_warmup + num_repeats:
            now = time.time()
            if now - last_report_time > 1 or i + 1 == num_warmup + num_repeats:
                if r.global_group.rank == 0:
                    logger.info("Iteration %i/%i", i + 1, num_warmup + num_repeats)
                last_report_time = now

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    logger.info("[Rank %d] Completed benchmark iterations", r.global_group.rank)

    dispatch_times: list[float] = []
    for start, end in dispatch_events[num_warmup:]:
        dispatch_times.append(start.elapsed_time(end) * 1000)

    combine_times: list[float] = []
    for start, end in combine_events[num_warmup:]:
        combine_times.append(start.elapsed_time(end) * 1000)

    # All-gather results from all ranks
    dispatch_times = sum(r.global_group.all_gather_object(dispatch_times), [])
    combine_times = sum(r.global_group.all_gather_object(combine_times), [])

    # Report the results
    if r.global_group.rank == 0:
        stat_dispatch = Statistics.create(dispatch_times)
        stat_combine = Statistics.create(combine_times)

        dispatch_bandwidth = r.cfg.dispatch_bytes / stat_dispatch.p50 * 1e-3
        combine_bandwidth = r.cfg.combine_bytes / stat_combine.p50 * 1e-3

        logger.info(
            "Dispatch time: %s, %.1f GB/s",
            stat_dispatch,
            dispatch_bandwidth,
        )
        logger.info(
            "Combine time: %s, %.1f GB/s",
            stat_combine,
            combine_bandwidth,
        )

        data = {
            "dispatch": asdict(stat_dispatch),
            "combine": asdict(stat_combine),
        }

        with output.open("w") as f:
            f.write(json.dumps(data))


def _worker(
    device: torch.device,
    dp_group: Optional[ParallelGroup],
    global_group: Optional[ParallelGroup],
    config: AllToAllConfig,
    num_warmup: int,
    num_repeats: int,
    output: Path,
    check: bool,
    verbose: bool = False,
) -> None:
    """Benchmark worker process."""

    assert dp_group is not None
    assert global_group is not None

    logging_utils.setup(level="INFO")

    import os
    env_verbose = "PPLX_GARDEN_DEBUG" in os.environ or "PPLX_DEBUG" in os.environ
    verbose = verbose or env_verbose

    r = AllToAllResource(device, dp_group, global_group, config)

    try:
        # Correctness check
        if check:
            logger.info("Checking correctness...")
            ex: Exception | None = None
            try:
                correctness_check(r)
                ok = True
            except Exception as e:  #  noqa: BLE001
                ex = e
                ok = False
            global_ok = r.global_group.all_gather_object(ok)
            bad_rank = next((i for i, ok in enumerate(global_ok) if not ok), None)
            if bad_rank is not None:
                raise RuntimeError(
                    f"Correctness check failed on rank {bad_rank}"
                ) from ex
            logger.info("Correctness check passed")
        else:
            logger.info("Skipping correctness check")

        # Benchmark.
        benchmark(r, num_warmup, num_repeats, output, verbose=verbose)
    finally:
        global_group.barrier()
        r.all_to_all.destroy()


def main() -> None:
    """Benchmark entry point."""

    parser = argparse.ArgumentParser(description="All-to-All Benchmark")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--init-method",
        type=str,
        default=None,
    )
    parser.add_argument("--dp-size", type=int, default=1)  # TODO: rename to tp_size
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-warmup", type=int, default=10000)
    parser.add_argument("--num-repeats", type=int, default=10000)
    parser.add_argument("--nets-per-gpu", type=int, default=2)
    parser.add_argument("--max-num-tokens", type=int, default=128)
    parser.add_argument("--max-private-tokens", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--hidden-dim-scale", type=int, default=56)
    parser.add_argument("--num-experts-per-token", type=int, default=8)
    parser.add_argument("--in-dtype", type=str_to_dtype, default=torch.float8_e4m3fn)
    parser.add_argument("--out-dtype", type=str_to_dtype, default=torch.bfloat16)
    parser.add_argument("--scale-dtype", type=str_to_dtype, default=torch.float32)
    parser.add_argument("--nvlink", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("/dev/stdout"))
    parser.add_argument(
        "--check", type=bool, default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging and syncs"
    )
    args = parser.parse_args()

    config = AllToAllConfig(
        nets_per_gpu=args.nets_per_gpu,
        max_num_tokens=args.max_num_tokens,
        max_private_tokens=args.max_private_tokens,
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        hidden_dim_scale=args.hidden_dim_scale,
        num_experts_per_token=args.num_experts_per_token,
        in_dtype=args.in_dtype,
        out_dtype=args.out_dtype,
        scale_dtype=args.scale_dtype,
        nvlink=args.nvlink,
    )

    ParallelLaunch(
        world_size=args.world_size,
        init_method=args.init_method,
        dp_size=args.dp_size,
        node_rank=args.node_rank,
    ).run(
        _worker,
        config,
        args.num_warmup,
        args.num_repeats,
        args.output,
        args.check,
        args.verbose,
    )


if __name__ == "__main__":
    logger.info("Launching benchmark script")
    main()
