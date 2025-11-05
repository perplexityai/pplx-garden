# ruff: noqa: A002

from collections.abc import Callable, Sequence
from typing import Any

import torch

class DomainInfo:
    @property
    def name(self) -> str: ...
    @property
    def link_speed(self) -> int: ...

class DomainAddress:
    @classmethod
    def from_bytes(cls, bytes: bytes) -> DomainAddress: ...
    def as_bytes(self) -> bytes: ...
    @classmethod
    def from_str(cls, s: str) -> DomainAddress: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class TopologyGroup:
    @property
    def cuda_device(self) -> int: ...
    @property
    def numa(self) -> int: ...
    @property
    def domains(self) -> list[DomainInfo]: ...
    @property
    def cpus(self) -> list[int]: ...

class MemoryRegionHandle:
    def capsule(self) -> Any: ...
    def debug_str(self) -> str: ...

class MemoryRegionDescriptor:
    @classmethod
    def from_bytes(cls, bytes_: bytes) -> MemoryRegionDescriptor: ...
    def as_bytes(self) -> bytes: ...
    def debug_str(self) -> str: ...

class PageIndices:
    def __init__(self, indices: Sequence[int]) -> None: ...

class UvmWatcher:
    @property
    def ptr(self) -> int: ...

class TransferEngineBuilder:
    def add_gpu_domains(
        self,
        cuda_device: int,
        domains: list[DomainInfo],
        pin_worker_cpu: int,
        pin_uvm_cpu: int,
    ) -> TransferEngineBuilder: ...
    def build(self) -> TransferEngine: ...

class FabricEngine: ...

class TransferEngine:
    def __init__(self, nets_per_gpu: int, cuda_devices: list[int]) -> None: ...
    @staticmethod
    def detect_topology() -> list[TopologyGroup]: ...
    @staticmethod
    def builder() -> TransferEngineBuilder: ...
    @property
    def main_address(self) -> DomainAddress: ...
    @property
    def num_domains(self) -> int: ...
    @property
    def aggregated_link_speed(self) -> int: ...
    @property
    def nets_per_gpu(self) -> int: ...
    @property
    def fabric_engine(self) -> FabricEngine: ...
    def stop(self) -> None: ...
    def register_tensor(
        self,
        tensor: torch.Tensor,
    ) -> tuple[MemoryRegionHandle, MemoryRegionDescriptor]: ...
    def register_memory(
        self,
        ptr: int,
        len: int,
        device: torch.device,
    ) -> tuple[MemoryRegionHandle, MemoryRegionDescriptor]: ...
    def unregister_memory(self, ptr: int) -> None: ...
    def alloc_scalar_watcher(
        self,
        callback: Callable[[int, int], bool],
    ) -> UvmWatcher:
        """
        Allocates a watcher for a scalar value on Unified Memory.
        The returned watcher has a pointer to a 64-bit value.
        The value is initialized to 0.
        Callback: (old_value, new_value) -> continue_watch
        """

    def set_imm_callback(self, callback: Callable[[int], None]) -> None:
        """
        Sets a callback when receiving an immediate number that is not used as a counter.
        Callback signature: (imm: int) -> None
        """
    def set_imm_count_expected(
        self,
        imm: int,
        expected_count: int,
        on_reached: Callable[[], None],
    ) -> tuple[int, int] | None:
        """
        Use imm as a counter. Set the expected count.
        Once the expected count is reached, the callback will be called.
        Then, the imm is no longer used as a counter.

        If the imm was not previously used as a counter, return None.
        Otherwise, return the previous counter and the previous expected count.
        The previous counter and callback will be discarded.
        """
    def remove_imm_count(self, imm: int) -> tuple[int, int] | None:
        """
        If imm is not used as a counter, return None.
        Otherwise, return the previous counter and the previous expected count.
        The previous counter and callback will be discarded.

        Normally you don't need to call this function because set_imm_count_expected
        removes the counter after reaching the expected count.

        This function is useful if you know that the count will not reach the
        expected count, for example, when a transfer is cancelled.
        """

    def submit_bouncing_recvs(
        self,
        count: int,
        len: int,
        on_recv: Callable[[bytes], None],
        on_error: Callable[[str], None],
    ) -> None: ...
    def submit_send(
        self,
        addr: DomainAddress,
        data: bytes,
        on_done: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None: ...
    def submit_imm(
        self,
        imm_data: int,
        dst_mr: MemoryRegionDescriptor,
        on_done: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None: ...
    def submit_write(
        self,
        src_mr: MemoryRegionHandle,
        offset: int,
        length: int,
        imm_data: int | None,
        dst_mr: MemoryRegionDescriptor,
        dst_offset: int,
        on_done: Callable[[], None],
        on_error: Callable[[str], None],
        num_shards: int | None = None,
    ) -> None:
        """
        Args:
            num_shards: If None, shard the transfer across all domains. \
                Otherwise, shard the transfer across the specified number of domains.
        """
    def submit_paged_writes(
        self,
        length: int,
        src_mr: MemoryRegionHandle,
        src_page_indices: PageIndices,
        src_stride: int,
        src_offset: int,
        dst_mr: MemoryRegionDescriptor,
        dst_page_indices: PageIndices,
        dst_stride: int,
        dst_offset: int,
        imm_data: int | None,
        on_done: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None: ...
