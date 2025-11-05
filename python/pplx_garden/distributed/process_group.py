import dataclasses
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Concatenate, Optional, ParamSpec, TypeVar

import torch
import torch.distributed
from torch.multiprocessing import (
    ProcessContext,  # pyright: ignore[reportPrivateImportUsage]
    spawn,  # pyright: ignore[reportPrivateImportUsage]
)

from pplx_garden.distributed.parallel_group import ParallelGroup
from pplx_garden.utils import logging_utils

logger = logging_utils.get_logger(__name__)


class ProcessGroup:
    """A wrapper around torch process groups."""

    def __init__(
        self,
        *,
        init_method: str,
        node_rank: int,
        local_rank: int,
        global_rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        assert 0 <= global_rank < world_size
        self._node_rank = node_rank
        self._local_rank = local_rank
        self._global_rank = global_rank
        self._world_size = world_size
        self._device = device

        if torch.distributed.is_initialized():
            self._should_destroy = False
            assert world_size == torch.distributed.get_world_size()
        else:
            self._should_destroy = True

            logger.info(
                "[rank=%d] Initializing global process group. "
                "device=%s, init_method=%s, world_size=%d",
                global_rank,
                device,
                init_method,
                world_size,
            )

            torch.cuda.set_device(device)
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                init_method=init_method,
                world_size=world_size,
                rank=global_rank,
                device_id=device,
            )

            torch.distributed.barrier()
            torch.cuda.synchronize()

            logger.info(
                "[rank=%d] Initialized global process group.", self._global_rank
            )

    def destroy(self) -> None:
        """Destroys the global torch process group."""

        if not self._should_destroy:
            return

        torch.cuda.synchronize()
        torch.distributed.barrier()

        torch.distributed.destroy_process_group()
        logger.info("[rank=%d] Destroyed global process group.", self._global_rank)
        self._should_destroy = False

    def create_group(self) -> ParallelGroup:
        from .torch_group import TorchParallelGroup

        return TorchParallelGroup(
            self._device,
            self._node_rank,
            self._local_rank,
            self._global_rank,
            list(range(self._world_size)),
        )


P = ParamSpec("P")
R = TypeVar("R")


@dataclasses.dataclass
class ParallelLaunch:
    world_size: int = 1
    init_method: Optional[str] = None
    devices: Optional[list[int]] = None
    dp_size: Optional[int] = None
    dp_lens: Optional[list[int]] = None
    node_rank: int = 0
    tmpdir_base: Optional[Path] = None

    def run(
        self,
        worker: Callable[
            Concatenate[
                torch.device, Optional[ParallelGroup], Optional[ParallelGroup], P
            ],
            R,
        ],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[R]:
        """
        Launch a parallel job, creating the processes and the ParallelGroup.
        Rank 0 runs on the main process, while the rest run on separate processes.
        Returns a list of return values from each local process.

        Worker signature:
            def worker(device, dp_group, global_group, *args, **kwargs) -> R

        When `dp_size` is set, `dp_group` is a slice of `global_group` that contains `dp_size` ranks.

        Use `devices` to specify the GPUs to use. If `devices` is not specified, use all available GPUs.

        To use with multiple nodes, set `init_method` and `node_rank` accordingly.
        """
        return _parallel_launch(self, worker, *args, **kwargs)


def _parallel_worker(
    _torch_rank: int,
    config: ParallelLaunch,
    init_method: str,
    devices: list[int],
    tmpdir: Path,
    setup_logging_level: int | str | None,
    node_rank: int,
    local_rank: int,
    global_rank: int,
    worker: Callable[
        Concatenate[torch.device, Optional[ParallelGroup], Optional[ParallelGroup], P],
        R,
    ],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    assert config.world_size > 1
    if setup_logging_level is not None:
        logging_utils.setup(level=setup_logging_level)

    device = torch.device(devices[local_rank])

    process_group = ProcessGroup(
        init_method=init_method,
        node_rank=node_rank,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=config.world_size,
        device=device,
    )
    global_group = process_group.create_group()
    assert global_group.rank == global_rank

    dp_group = global_group
    if config.dp_size is not None and config.dp_size != config.world_size:
        dp_group = global_group.slice_by_count(config.world_size // config.dp_size)
    elif config.dp_lens is not None:
        dp_group = global_group.slice_by_lens(config.dp_lens)

    try:
        ret = worker(device, dp_group, global_group, *args, **kwargs)
        with open(tmpdir / f"local_rank_{local_rank}.pkl", "wb") as f:
            pickle.dump(ret, f)
    except Exception:
        logger.exception("[rank=%d] Error in parallel group", global_rank)
        raise
    finally:
        if dp_group is not global_group:
            dp_group.destroy()
        global_group.destroy()
        process_group.destroy()


@contextmanager
def _init_context(tmp_dir: Path, init_method: Optional[str]) -> Iterator[str]:
    if init_method is not None:
        yield init_method
        return

    some_file = tmp_dir / "pplx_garden_parallel_init"
    some_file.touch(exist_ok=True)

    try:
        yield f"file://{some_file}"
    finally:
        some_file.unlink(missing_ok=True)


def _parallel_launch(
    config: ParallelLaunch,
    worker: Callable[
        Concatenate[torch.device, Optional[ParallelGroup], Optional[ParallelGroup], P],
        R,
    ],
    *args: P.args,
    **kwargs: P.kwargs,
) -> list[R]:
    if kwargs:
        raise ValueError(
            "Keyword arguments are not supported by torch.multiprocessing.spawn()"
        )
    if config.dp_size is not None and config.dp_lens is not None:
        raise ValueError("Cannot specify both dp_size and dp_lens")

    devices = config.devices
    if devices is None:
        devices = list(range(min(torch.cuda.device_count(), config.world_size)))
    assert config.world_size % len(devices) == 0
    ranks_per_node = len(devices)
    num_ranks = config.world_size // ranks_per_node
    assert 0 <= config.node_rank < num_ranks
    if config.dp_size is not None:
        assert config.dp_lens is None
        assert config.world_size % config.dp_size == 0
        assert len(devices) % config.dp_size == 0
    if config.dp_lens is not None:
        assert config.dp_size is None
        assert sum(config.dp_lens) == config.world_size

    if config.world_size == 1:
        # No TP, call the worker directly.
        ret = worker(
            torch.device(devices[0]),
            None,
            None,
            *args,
            **kwargs,
        )
        return [ret]

    setup_logging_level = logger.getEffectiveLevel() if logger.hasHandlers() else None

    with TemporaryDirectory(dir=config.tmpdir_base) as tmpdir:
        tmp_path = Path(tmpdir)
        with _init_context(tmp_path, config.init_method) as init_method:
            workers: list[ProcessContext] = []
            local_ranks = range(len(devices))
            for local_rank in local_ranks:
                p = spawn(
                    _parallel_worker,
                    (
                        config,
                        init_method,
                        devices,
                        tmp_path,
                        setup_logging_level,
                        config.node_rank,
                        local_rank,
                        config.node_rank * ranks_per_node + local_rank,
                        worker,
                    )
                    + args,
                    join=False,
                )
                assert p is not None
                workers.append(p)

            for p in workers:
                p.join()

            rets = []
            for local_rank in local_ranks:
                with open(Path(tmpdir) / f"local_rank_{local_rank}.pkl", "rb") as f:
                    rets.append(pickle.load(f))
            return rets
