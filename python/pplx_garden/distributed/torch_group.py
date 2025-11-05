from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar, cast

import torch
import torch.distributed
from torch.distributed import GroupMember, ProcessGroup, ReduceOp, Work
from typing_extensions import override

from pplx_garden.distributed.distributed_ops import Reducer
from pplx_garden.distributed.nccl_all_reduce import NcclAllReduce
from pplx_garden.distributed.parallel_group import ParallelGroup
from pplx_garden.utils.logging_utils import get_logger
from pplx_garden.utils.torch import profile_range

logger = get_logger(__name__)


T = TypeVar("T")


class TorchParallelGroup(ParallelGroup):
    """
    Wrapper around a torch.distributed process group for this rank's
    tensor or pipeline parallelism group.

    The global process group must be initialized before creating an instance.
    See `init_global_process_group()` and `destroy_global_process_group()`.

    Call `destroy` when finished using this instance.
    """

    def __init__(
        self,
        device: torch.device,
        node_rank: int,
        local_rank: int,
        global_rank: int,
        ranks: list[int],
    ) -> None:
        assert torch.distributed.is_initialized(), (
            "The process group must be initialized before the parallel group"
        )

        self._device = device
        self._node_rank = node_rank
        self._local_rank = local_rank
        self._global_rank = global_rank

        self._ranks = ranks
        self._size = len(self._ranks)

        # The rank is an index within the local parallel group.
        assert self._global_rank in self._ranks
        self._rank = self._ranks.index(self._global_rank)
        assert self._rank >= 0

        # Create new groups
        new_groups = self._create_new_groups(self._ranks)
        assert new_groups is not None
        self._device_group, self._cmd_group = new_groups

        # Instantiate the all-reduce implementation.
        self._reducer = NcclAllReduce(group=self._device_group)

    @staticmethod
    def _create_new_groups(
        ranks: list[int],
    ) -> tuple[ProcessGroup, ProcessGroup] | None:
        assert torch.distributed.is_initialized()
        device_group = torch.distributed.new_group(ranks=ranks, backend="nccl")
        cmd_group = torch.distributed.new_group(ranks=ranks, backend="gloo")

        # new_group() returns NON_GROUP_MEMBER when the current rank is not in the subgroup.
        num_non_group = 0
        num_non_group += int(device_group == GroupMember.NON_GROUP_MEMBER)
        num_non_group += int(cmd_group == GroupMember.NON_GROUP_MEMBER)
        assert num_non_group in [0, 2]
        if num_non_group != 0:
            return None

        return (
            cast(ProcessGroup, device_group),
            cast(ProcessGroup, cmd_group),
        )

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def rank(self) -> int:
        """Local index within the parallel group."""
        return self._rank

    @property
    @override
    def node_rank(self) -> int:
        """The rank of the node within the parallel group."""
        return self._node_rank

    @property
    @override
    def global_rank(self) -> int:
        """Global index within the global group."""
        return self._global_rank

    @property
    @override
    def local_rank(self) -> int:
        """Local index within the current node."""
        return self._local_rank

    @property
    @override
    def size(self) -> int:
        """The size of the parallel group."""
        return self._size

    @property
    @override
    def is_inter_node(self) -> bool:
        """Returns true if the group spans multiple nodes."""
        return self._size > 8

    @override
    @profile_range("reducer")
    def reducer(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Reducer:
        return self._reducer.reducer(shape, dtype, op)

    @override
    @profile_range("all_reduce")
    def all_reduce(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> torch.Tensor:
        return self._reducer.all_reduce(x, op)

    @override
    @profile_range("all_reduce_cpu_async")
    def all_reduce_cpu_async(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Work:
        assert x.device.type == "cpu"
        work = torch.distributed.all_reduce(
            x,
            op=op,
            group=self._cmd_group,
            async_op=True,
        )
        return cast(Work, work)

    @override
    @profile_range("all_gather")
    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Implementation adapted from vLLM
        assert -x.dim() <= dim < x.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {x.size()}"
        )
        assert x.device == self._device

        if dim < 0:
            dim += x.dim()

        input_size = x.size()
        output_tensor = torch.empty(
            (self._size,) + input_size,
            dtype=x.dtype,
            device=x.device,
        )
        torch.distributed.all_gather_into_tensor(
            output_tensor,
            x,
            group=self._device_group,
        )

        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim] + (self._size * input_size[dim],) + input_size[dim + 1 :]
        )

    @override
    def all_gather_object(self, obj: T) -> list[T]:
        object_list = [None] * self._size
        torch.distributed.all_gather_object(object_list, obj, group=self._cmd_group)
        return cast(list[T], object_list)

    @override
    def broadcast_object(self, obj: T | None, root: int) -> T:
        assert 0 <= root < self._size, (
            f"Invalid rank {root} for group of size {self._size}"
        )
        objs = [obj]
        torch.distributed.broadcast_object_list(
            objs,
            src=self._ranks[root],
            group=self._cmd_group,
        )
        return cast(T, objs[0])

    @override
    def broadcast_cpu_tensor_async(self, tensor: torch.Tensor, root: int) -> Work:
        assert tensor.device.type == "cpu"
        work = torch.distributed.broadcast(
            tensor,
            src=self._ranks[root],
            group=self._cmd_group,
            async_op=True,
        )
        return cast(Work, work)

    @override
    def broadcast(self, tensor: torch.Tensor, root: int) -> torch.Tensor:
        torch.distributed.broadcast(
            tensor,
            src=self._ranks[root],
            group=self._device_group,
        )
        return tensor

    @override
    def all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        m, *_ = tensor.shape
        if m != self._size:
            msg = f"Expected leading dim {m} to match group size {self._size}"
            raise ValueError(msg)

        output = torch.empty_like(tensor)
        torch.distributed.all_to_all_single(output, tensor, group=self._device_group)
        return output

    @override
    def barrier(self) -> None:
        torch.distributed.barrier(
            group=self._device_group,
            device_ids=[self._device.index],
        )

    @override
    @contextmanager
    def capture(self) -> Iterator[None]:
        with self._reducer.capture():
            yield

    def destroy(self) -> None:
        self._reducer.destroy()
        torch.distributed.destroy_process_group(self._cmd_group)
        torch.distributed.destroy_process_group(self._device_group)

    @override
    def slice_by_count(self, slice_count: int) -> ParallelGroup:
        assert self._size % slice_count == 0
        slice_size = self._size // slice_count
        return self.slice_by_lens([slice_size] * slice_count)

    @override
    def slice_by_lens(self, slice_lens: list[int]) -> ParallelGroup:
        # new_group() requires all ranks to call it with the same ranks,
        # even if the current rank is not in the subgroup.
        # And this needs to happen in the same order.
        # So we need to loop over all subgroups.

        assert sum(slice_lens) == self._size
        slice_ranks: list[list[int]] = []
        cumsum = 0
        for sl in slice_lens:
            slice_ranks.append(self._ranks[cumsum : cumsum + sl])
            cumsum += sl

        ret: TorchParallelGroup | None = None
        for ranks in slice_ranks:
            if self._global_rank in ranks:
                ret = TorchParallelGroup(
                    self._device,
                    self._node_rank,
                    self._local_rank,
                    self._global_rank,
                    ranks,
                )
            else:
                new_groups = self._create_new_groups(ranks)
                assert new_groups is None
        assert ret is not None

        # A barrier on the new group is required
        torch.distributed.barrier(
            group=ret._device_group,
            device_ids=[self._device.index],
        )
        return ret

    def _slice_ranks(self, slice_rank: int, slice_count: int) -> list[int]:
        """Slice the ranks assigned to this group."""

        assert 0 < slice_count <= len(self._ranks)
        assert 0 <= slice_rank < slice_count
        assert len(self._ranks) % slice_count == 0
        slice_size = len(self._ranks) // slice_count

        return self._ranks[slice_rank * slice_size : (slice_rank + 1) * slice_size]
