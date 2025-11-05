# ruff: noqa: A002

import torch

from pplx_garden.fabric_lib import (
    DomainAddress,
    MemoryRegionDescriptor,
    MemoryRegionHandle,
    TransferEngine,
)

class AllToAllContext:
    @classmethod
    def create(
        cls,
        hidden_dim: int,
        hidden_dim_scale: int | None,
        in_elemsize: int,
        out_elemsize: int,
        out_dtype: torch.dtype,
        scale_elemsize: int | None,
        max_num_tokens: int,
        max_recv_tokens: int,
        max_private_tokens: int,
        num_experts: int,
        expert_padding: int,
        num_experts_per_token: int,
        rank: int,
        dp_size: int,
        node_size: int,
        world_size: int,
        num_routed_ptr: int,
        num_routed_mr: MemoryRegionHandle,
        send_buffer_ptr: int,
        send_buffer_mr: MemoryRegionHandle,
        recv_buffer_ptr: int,
        recv_buffer_mr: MemoryRegionHandle,
        sync_ptrs: list[int],
        send_ptrs: list[int],
        recv_ptrs: list[int],
        device: int,
        imm_base: int,
        ranks: list[
            tuple[
                DomainAddress,
                MemoryRegionDescriptor,
                MemoryRegionDescriptor,
            ]
        ],
        transfer_engine: TransferEngine,
        worker_cpu: int | None,
    ) -> None: ...
    def dispatch_send(
        self,
        num_tokens: int,
        x_ptr: int,
        x_stride: int,
        x_scale_ptr: int | None,
        x_scale_stride_elem: int | None,
        x_scale_stride_token: int | None,
        indices_ptr: int,
        indices_stride: int,
        weights_ptr: int,
        weights_stride: int,
        bound_m_ptr: int | None,
        stream: int,
    ) -> None: ...
    def dispatch_recv(
        self,
        out_num_tokens_ptr: int,
        out_x_ptr: int,
        out_x_stride: int,
        out_x_scale_ptr: int | None,
        out_x_scale_stride_elem: int | None,
        out_x_scale_stride_token: int | None,
        stream: int,
    ) -> None: ...
    def combine_send(
        self,
        expert_x_ptr: int,
        expert_x_stride: int,
        stream: int,
    ) -> None: ...
    def combine_recv(
        self,
        num_tokens: int,
        num_recv_tokens: int,
        expert_y_dtype: torch.dtype,
        out_tokens_ptr: int,
        out_tokens_stride: int,
        indices_ptr: int,
        indices_stride: int,
        weights_ptr: int,
        weights_stride: int,
        bound_m_ptr: int | None,
        accumulate: bool,
        stream: int,
    ) -> None: ...
