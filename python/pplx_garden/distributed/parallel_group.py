from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch
from torch.distributed import ReduceOp, Work

from pplx_garden.distributed.distributed_ops import Reducer
from pplx_garden.utils import logging_utils

logger = logging_utils.get_logger(__name__)


T = TypeVar("T")


class ParallelGroup(ABC):
    """Abstract base for parallel configurations."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device assigned to the current rank."""
        ...

    @property
    @abstractmethod
    def rank(self) -> int:
        """Current rank within the current group."""
        ...

    @property
    @abstractmethod
    def global_rank(self) -> int:
        """Current rank within the global group."""
        ...

    @property
    @abstractmethod
    def node_rank(self) -> int:
        """The rank of the node."""
        ...

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """The rank within the current node."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """The size of the parallel group."""
        ...

    @property
    @abstractmethod
    def is_inter_node(self) -> bool:
        """Returns true of the group spans multiple nodes."""
        ...

    @abstractmethod
    def broadcast_object(self, obj: T | None, root: int) -> T:
        """Broadcast an object across the CPU interconnect."""
        ...

    @abstractmethod
    def broadcast_cpu_tensor_async(self, tensor: torch.Tensor, root: int) -> Work:
        """Broadcast a CPU tensor across the CPU interconnect."""
        ...

    @abstractmethod
    def reducer(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Reducer: ...

    @abstractmethod
    def all_reduce(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> torch.Tensor: ...

    @abstractmethod
    def all_reduce_cpu_async(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Work: ...

    def all_reduce_cpu(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> None:
        self.all_reduce_cpu_async(x, op).wait()

    @abstractmethod
    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor: ...

    @abstractmethod
    def all_gather_object(self, obj: T) -> list[T]: ...

    @abstractmethod
    def broadcast(self, tensor: torch.Tensor, root: int) -> torch.Tensor: ...

    @abstractmethod
    def all_to_all(self, tensor: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def barrier(self) -> None: ...

    @abstractmethod
    @contextmanager
    def capture(self) -> Iterator[None]: ...

    @abstractmethod
    def destroy(self) -> None: ...

    @abstractmethod
    def slice_by_count(self, slice_count: int) -> "ParallelGroup":
        """
        Slice the group into equal-sized `slice_count` subgroups.
        Return the subgroup that the current rank belongs to.
        """
        ...

    @abstractmethod
    def slice_by_lens(self, slice_lens: list[int]) -> "ParallelGroup":
        """
        Slice the group into subgroups of the given lengths.
        Return the subgroup that the current rank belongs to.
        Require: sum(slice_lens) == self.size
        """
        ...

    @property
    def has_nvshmem(self) -> bool:
        return False
