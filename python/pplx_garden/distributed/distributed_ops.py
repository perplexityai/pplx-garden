"""Definition of interfaces for distributed operation wrappers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import torch
from torch.distributed import ReduceOp


class Reducer(ABC):
    """Wrapper around a contextual reducer."""

    @property
    @abstractmethod
    def input(self) -> Optional[torch.Tensor]:
        """Pre-allocated input tensor to be reduced."""

    @abstractmethod
    def reduce(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the reduction on the pre-allocated input."""


class ReducerBuilder(ABC):
    """Interface for all-reduce operations."""

    @abstractmethod
    def reducer(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Reducer:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass

    @abstractmethod
    def all_reduce(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> torch.Tensor:
        pass

    @contextmanager
    @abstractmethod
    def capture(self) -> Iterator[None]:
        yield
