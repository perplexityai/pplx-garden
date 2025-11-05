from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import torch
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import override

from pplx_garden.distributed.distributed_ops import Reducer, ReducerBuilder


class NcclReducer(Reducer):
    def __init__(
        self,
        group: ProcessGroup,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> None:
        self._group = group
        self._op = op

    @property
    @override
    def input(self) -> Optional[torch.Tensor]:
        return None

    def reduce(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = x
        elif out is not x:
            out.copy_(x)
        torch.distributed.all_reduce(out, op=self._op, group=self._group)
        return out


class NcclAllReduce(ReducerBuilder):
    def __init__(self, group: ProcessGroup) -> None:
        self._group = group

    @override
    def reducer(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> Reducer:
        return NcclReducer(group=self._group, op=op)

    @override
    def destroy(self) -> None:
        pass

    @override
    def all_reduce(
        self,
        x: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
    ) -> torch.Tensor:
        torch.distributed.all_reduce(x, op=op, group=self._group)
        return x

    @contextmanager
    @override
    def capture(self) -> Iterator[None]:
        yield
