import pickle
import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch
from typing_extensions import override

from pplx_garden.distributed import ParallelGroup
from pplx_garden.fabric_lib import (
    DomainAddress,
    MemoryRegionDescriptor,
    TransferEngine,
)
from pplx_garden.kernels.all_to_all import AllToAllKernel
from pplx_garden.native.cumem import (
    CUMemAllocHandle,
    CUMemExportHandle,
    CUMemHandleKind,
    CUMemMapping,
)
from pplx_garden.native.p2p_all_to_all import AllToAllContext
from pplx_garden.utils import logging_utils
from pplx_garden.utils.math import ceil_div, round_up

logger = logging_utils.get_logger(__name__)

_PAGE_SIZE = 4096


@dataclass
class _RdmaRankData:
    address: bytes
    num_routed_desc: bytes
    recv_buffer_descs: list[bytes]


@dataclass
class _NVLRankData:
    sync_fds: list[CUMemExportHandle]
    send_fds: list[CUMemExportHandle]
    recv_fds: list[CUMemExportHandle]


@dataclass
class _NVLRankMapping:
    sync_mapping: CUMemMapping
    send_mapping: CUMemMapping
    recv_mapping: CUMemMapping


@dataclass
class P2PDispatchHandle:
    kernel: "P2PAllToAll"
    out_expert_num_tokens: torch.Tensor
    out_expert_x: torch.Tensor
    out_expert_x_scale: Optional[torch.Tensor]
    dp_x: torch.Tensor
    dp_x_scale: Optional[torch.Tensor]
    indices: torch.Tensor
    weights: torch.Tensor
    bound_m: Optional[torch.Tensor]
    send_done_event: torch.cuda.Event
    slot: int
    recv_done: bool = False

    def recv(self) -> None:
        if self.recv_done:
            return
        torch.cuda.current_stream(self.dp_x.device).wait_event(self.send_done_event)
        self.kernel.dispatch(
            out_expert_num_tokens=self.out_expert_num_tokens,
            out_expert_x=self.out_expert_x,
            out_expert_x_scale=self.out_expert_x_scale,
            dp_x=self.dp_x,
            dp_x_scale=self.dp_x_scale,
            indices=self.indices,
            weights=self.weights,
            bound_m=self.bound_m,
            slot=self.slot,
            do_send=False,
            do_recv=True,
        )
        self.recv_done = True


@dataclass
class P2PCombineHandle:
    kernel: "P2PAllToAll"
    dispatch_handle: P2PDispatchHandle
    out_tokens: torch.Tensor
    expert_y: torch.Tensor
    bound_m: Optional[torch.Tensor]
    accumulate: bool
    send_done_event: torch.cuda.Event
    slot: int
    recv_done: bool = False

    def recv(self) -> None:
        if self.recv_done:
            return
        torch.cuda.current_stream(self.out_tokens.device).wait_event(
            self.send_done_event
        )
        self.kernel.combine(
            out_tokens=self.out_tokens,
            indices=self.dispatch_handle.indices,
            weights=self.dispatch_handle.weights,
            expert_y=self.expert_y,
            bound_m=self.bound_m,
            slot=self.slot,
            do_send=False,
            do_recv=True,
            accumulate=self.accumulate,
        )
        self.recv_done = True
        self.dispatch_handle.kernel._release_slot(self.slot)


class P2PAllToAll(AllToAllKernel):
    def __init__(
        self,
        *,
        max_num_tokens: int,
        num_experts: int,
        expert_padding: int,
        hidden_dim: int,
        hidden_dim_scale: Optional[int],
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
        scale_dtype: Optional[torch.dtype],
        num_experts_per_token: int,
        nets_per_gpu: int,
        max_private_tokens: Optional[int],
        device: torch.device,
        dp_group: Optional[ParallelGroup],
        node_group: Optional[ParallelGroup],
        global_group: ParallelGroup,
        imm_base: int = 0x80000000,
        transfer_engine: Optional[TransferEngine] = None,
        worker_cpu: Optional[int] = None,
        max_tokens_per_expert: Optional[int] = None,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._hidden_dim_scale = hidden_dim_scale
        self._num_experts_per_token = num_experts_per_token
        self._in_dtype = in_dtype
        self._out_dtype = out_dtype
        self._scale_dtype = scale_dtype
        self._device = device
        self._global_group = global_group
        self._max_tokens_per_expert = max_tokens_per_expert
        self._handle_kind = CUMemHandleKind.FileDescriptor
        self._slot_lock = threading.Condition()
        self._free_slots = [0, 1]

        # Determine the number of local experts.
        self._node_group: Optional[ParallelGroup]
        if dp_group is not None:
            self._node_group = node_group or dp_group
            self._dp_size = dp_group.size
        else:
            self._node_group = node_group
            self._dp_size = 1

        rank = global_group.rank
        world_size = global_group.size
        num_dp_groups = world_size // self._dp_size
        self._num_local_experts = ceil_div(num_experts, num_dp_groups)

        # Determine the size of the recv buffers.
        avg_tokens_per_expert = int(
            ceil_div(max_num_tokens * num_experts_per_token, num_experts) * 1.2
        )

        if max_private_tokens is None:
            max_private_tokens = avg_tokens_per_expert * self._num_local_experts
        assert max_private_tokens >= 0

        num_tokens = max_num_tokens * num_dp_groups
        max_recv_tokens = max_private_tokens * num_dp_groups + round_up(
            max(
                min(
                    num_tokens * num_experts_per_token
                    + self._num_local_experts * (expert_padding - 1),
                    num_tokens * self._num_local_experts,
                ),
                self._num_local_experts * expert_padding,
            ),
            expert_padding,
        )

        self._transfer_engine: Optional[TransferEngine] = None
        self._all_to_alls: list[AllToAllContext] = []

        # Detect topology and identify NICs and CPUs.
        system_topo = TransferEngine.detect_topology()

        device_groups = [
            group for group in system_topo if group.cuda_device == device.index
        ]
        if len(device_groups) == 1:
            group = device_groups[0]
        elif device.index is not None and 0 <= device.index < len(system_topo):
            group = system_topo[device.index]
            logger.warning(
                "Falling back to topology group %s for cuda:%s; detected CUDA groups are %s",
                device.index,
                device.index,
                [group.cuda_device for group in system_topo],
            )
        else:
            msg = (
                f"Cannot identify topology group for cuda:{device.index}; "
                f"detected CUDA groups are {[group.cuda_device for group in system_topo]}"
            )
            raise RuntimeError(msg)

        if len(group.cpus) < 2:
            msg = f"Not enough CPUs in device group for cuda:{device.index}"
            raise RuntimeError(msg)

        default_worker_cpu, domain_cpu, uvm_cpu, *_ = group.cpus
        if worker_cpu is None:
            worker_cpu = default_worker_cpu
        domains = group.domains[:nets_per_gpu]

        # Build or reuse the transfer engine. Multiple live P2PAllToAll contexts
        # should not each spawn their own fabric workers for the same GPU/NIC
        # resources; that path is extremely slow on the CXI backend.
        self._owns_transfer_engine = transfer_engine is None
        if transfer_engine is None:
            builder = TransferEngine.builder()
            builder.add_gpu_domains(group.cuda_device, domains, domain_cpu, uvm_cpu)
            transfer_engine = builder.build()
        self._transfer_engine = transfer_engine

        # Allocate and register a buffer for per-expert routed counts on the host.
        self._num_routed_buffer = torch.empty(
            (
                round_up(
                    num_dp_groups * num_experts * torch.uint32.itemsize,
                    _PAGE_SIZE,
                ),
            ),
            dtype=torch.uint8,
            pin_memory=True,
        ).view(torch.uint32)
        num_routed_mr, num_routed_desc = self._transfer_engine.register_tensor(
            self._num_routed_buffer
        )

        num_slots = len(self._free_slots)

        # Allocate a a buffer to send from.
        token_dim_dispatch = round_up(hidden_dim * in_dtype.itemsize, 16) + 16
        if hidden_dim_scale is not None or scale_dtype is not None:
            assert scale_dtype is not None
            assert hidden_dim_scale is not None
            token_dim_dispatch += round_up(hidden_dim_scale * scale_dtype.itemsize, 16)

            # TODO: support other scale dtypes
            assert scale_dtype == torch.float32

        token_dim_combine = round_up(hidden_dim * out_dtype.itemsize, 16)
        token_dim = max(token_dim_dispatch, token_dim_combine)

        send_buffer_bytes = round_up(max_recv_tokens * token_dim, _PAGE_SIZE)
        recv_buffer_bytes = round_up(max_recv_tokens * token_dim, _PAGE_SIZE)

        self._send_buffer_handles: list[CUMemAllocHandle] = []
        self._send_buffer_mappings: list[CUMemMapping] = []
        send_buffer_mrs: list[MemoryRegionHandle] = []
        send_buffer_descs: list[MemoryRegionDescriptor] = []
        self._recv_buffer_handles: list[CUMemAllocHandle] = []
        self._recv_buffer_mappings: list[CUMemMapping] = []
        recv_buffer_mrs: list[MemoryRegionHandle] = []
        recv_buffer_descs: list[MemoryRegionDescriptor] = []

        for _ in range(num_slots):
            send_buffer_handle = CUMemAllocHandle(
                send_buffer_bytes,
                self._device,
                self._handle_kind,
            )
            send_buffer_mapping = send_buffer_handle.map(self._device)
            send_buffer_mr, send_buffer_desc = self._transfer_engine.register_tensor(
                send_buffer_mapping.to_tensor(
                    (send_buffer_bytes,),
                    torch.uint8,
                )
            )
            self._send_buffer_handles.append(send_buffer_handle)
            self._send_buffer_mappings.append(send_buffer_mapping)
            send_buffer_mrs.append(send_buffer_mr)
            send_buffer_descs.append(send_buffer_desc)

            recv_buffer_handle = CUMemAllocHandle(
                recv_buffer_bytes,
                self._device,
                self._handle_kind,
            )
            recv_buffer_mapping = recv_buffer_handle.map(self._device)
            recv_buffer_mr, recv_buffer_desc = self._transfer_engine.register_tensor(
                recv_buffer_mapping.to_tensor(
                    (recv_buffer_bytes,),
                    torch.uint8,
                )
            )
            self._recv_buffer_handles.append(recv_buffer_handle)
            self._recv_buffer_mappings.append(recv_buffer_mapping)
            recv_buffer_mrs.append(recv_buffer_mr)
            recv_buffer_descs.append(recv_buffer_desc)

        # Exchange NVLink buffers.
        self._nvl_mappings: list[list[_NVLRankMapping]] = []
        sync_ptrs_by_slot: list[list[int]] = [[] for _ in range(num_slots)]
        send_ptrs_by_slot: list[list[int]] = [[] for _ in range(num_slots)]
        recv_ptrs_by_slot: list[list[int]] = [[] for _ in range(num_slots)]
        if self._node_group is not None:
            logger.info(
                "Setting up RDMA (%d) + NVLink (%d)",
                global_group.size,
                self._node_group.size,
            )
            self._sync_buffer_handles: list[CUMemAllocHandle] = []
            sync_mappings: list[CUMemMapping] = []
            for _ in range(num_slots):
                sync_buffer_handle = CUMemAllocHandle(
                    torch.uint32.itemsize * self._node_group.size * 2,
                    self._device,
                    self._handle_kind,
                )
                sync_mapping = sync_buffer_handle.map(self._device)
                sync_mapping.to_tensor(
                    (self._node_group.size * 2,),
                    torch.uint32,
                ).fill_(0)
                self._sync_buffer_handles.append(sync_buffer_handle)
                sync_mappings.append(sync_mapping)

            local_handle = _NVLRankData(
                sync_fds=[h.export() for h in self._sync_buffer_handles],
                send_fds=[h.export() for h in self._send_buffer_handles],
                recv_fds=[h.export() for h in self._recv_buffer_handles],
            )
            handles = self._node_group.all_gather_object(pickle.dumps(local_handle))

            self._nvl_mappings = [[] for _ in range(num_slots)]
            for slot in range(num_slots):
                for peer, h in enumerate(handles):
                    if peer == self._node_group.rank:
                        self._nvl_mappings[slot].append(
                            _NVLRankMapping(
                                sync_mapping=sync_mappings[slot],
                                send_mapping=self._send_buffer_mappings[slot],
                                recv_mapping=self._recv_buffer_mappings[slot],
                            )
                        )
                    else:
                        assert h is not None
                        peer_data = pickle.loads(h)
                        assert isinstance(peer_data, _NVLRankData)
                        self._nvl_mappings[slot].append(
                            _NVLRankMapping(
                                sync_mapping=peer_data.sync_fds[slot]
                                .bind()
                                .map(self._device),
                                send_mapping=peer_data.send_fds[slot]
                                .bind()
                                .map(self._device),
                                recv_mapping=peer_data.recv_fds[slot]
                                .bind()
                                .map(self._device),
                            )
                        )
                        del peer_data

            self._node_group.barrier()
            del local_handle

            node_size = self._node_group.size
            for slot in range(num_slots):
                for i in range(node_size):
                    recv_ptrs_by_slot[slot].append(
                        self._nvl_mappings[slot][i].recv_mapping.data_ptr()
                    )
                    send_ptrs_by_slot[slot].append(
                        self._nvl_mappings[slot][i].send_mapping.data_ptr()
                    )
                    sync_ptrs_by_slot[slot].append(
                        self._nvl_mappings[slot][i].sync_mapping.data_ptr()
                    )
        else:
            logger.info("Setting up RDMA (%d)", global_group.size)
            node_size = 1

        # Collect the metadata associated with all ranks.
        gathered_rank_data = global_group.all_gather_object(
            _RdmaRankData(
                address=self._transfer_engine.main_address.as_bytes(),
                num_routed_desc=num_routed_desc.as_bytes(),
                recv_buffer_descs=[desc.as_bytes() for desc in recv_buffer_descs],
            )
        )
        ranks_by_slot = [
            [
                (
                    DomainAddress.from_bytes(data.address),
                    MemoryRegionDescriptor.from_bytes(data.num_routed_desc),
                    MemoryRegionDescriptor.from_bytes(data.recv_buffer_descs[slot]),
                )
                for data in gathered_rank_data
            ]
            for slot in range(num_slots)
        ]

        # Set up the all-to-all context.
        for slot in range(num_slots):
            self._all_to_alls.append(
                AllToAllContext.create(
                    hidden_dim=hidden_dim,
                    hidden_dim_scale=hidden_dim_scale,
                    in_elemsize=in_dtype.itemsize,
                    out_elemsize=out_dtype.itemsize,
                    out_dtype=out_dtype,
                    scale_elemsize=scale_dtype.itemsize if scale_dtype else None,
                    max_num_tokens=max_num_tokens,
                    max_recv_tokens=max_recv_tokens,
                    max_tokens_per_expert=max_tokens_per_expert,
                    max_private_tokens=max_private_tokens,
                    num_experts=num_experts,
                    expert_padding=expert_padding,
                    num_experts_per_token=num_experts_per_token,
                    rank=rank,
                    dp_size=self._dp_size,
                    node_size=node_size,
                    world_size=world_size,
                    num_routed_ptr=self._num_routed_buffer.data_ptr(),
                    num_routed_mr=num_routed_mr,
                    send_buffer_ptr=self._send_buffer_mappings[slot].data_ptr(),
                    send_buffer_mr=send_buffer_mrs[slot],
                    recv_buffer_ptr=self._recv_buffer_mappings[slot].data_ptr(),
                    recv_buffer_mr=recv_buffer_mrs[slot],
                    sync_ptrs=sync_ptrs_by_slot[slot],
                    send_ptrs=send_ptrs_by_slot[slot],
                    recv_ptrs=recv_ptrs_by_slot[slot],
                    device=device.index,
                    imm_base=imm_base + slot * 5,
                    ranks=ranks_by_slot[slot],
                    transfer_engine=self._transfer_engine,
                    worker_cpu=worker_cpu,
                    num_slots=1,
                )
            )

        # Ensure that all ranks start the workers threads and registered imm callbacks.
        global_group.barrier()

    def _acquire_slot(self) -> int:
        with self._slot_lock:
            while not self._free_slots:
                self._slot_lock.wait()
            return self._free_slots.pop(0)

    def _release_slot(self, slot: int) -> None:
        with self._slot_lock:
            if slot not in self._free_slots:
                self._free_slots.append(slot)
                self._free_slots.sort()
                self._slot_lock.notify()

    def _ctx(self, slot: int) -> AllToAllContext:
        return self._all_to_alls[slot]

    @override
    def dispatch(
        self,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: Optional[torch.Tensor],
        dp_x: torch.Tensor,
        dp_x_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        weights: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
        slot: int = 0,
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        assert self._all_to_alls
        assert do_send or do_recv
        all_to_all = self._ctx(slot)

        num_tokens, _ = dp_x.shape

        # Verify the output count buffer.
        assert out_expert_num_tokens.shape == (self._num_local_experts,)
        assert out_expert_num_tokens.stride(0) == 1
        assert out_expert_num_tokens.dtype == torch.int32
        out_expert_num_tokens_ptr = out_expert_num_tokens.data_ptr()

        # Verify the output token buffer.
        out_expert_x = self._flatten_batched_experts(
            out_expert_x,
            dtype=self._in_dtype,
            hidden_dim=self._hidden_dim,
            name="out_expert_x",
        )
        out_x_ptr = out_expert_x.data_ptr()
        out_x_stride = out_expert_x.stride(0) * out_expert_x.dtype.itemsize

        # Verify the output scale buffer.
        out_x_scale_ptr: Optional[int]
        out_x_scale_stride_elem: Optional[int]
        out_x_scale_stride_token: Optional[int]
        if out_expert_x_scale is not None:
            assert self._scale_dtype is not None
            assert self._hidden_dim_scale is not None
            out_expert_x_scale = self._flatten_batched_experts(
                out_expert_x_scale,
                dtype=self._scale_dtype,
                hidden_dim=self._hidden_dim_scale,
                name="out_expert_x_scale",
            )
            assert out_expert_x_scale.dtype == self._scale_dtype
            out_x_scale_ptr = out_expert_x_scale.data_ptr()
            out_x_scale_stride_elem = out_expert_x_scale.stride(1)
            out_x_scale_stride_token = out_expert_x_scale.stride(0)
        else:
            out_x_scale_ptr = None
            out_x_scale_stride_elem = None
            out_x_scale_stride_token = None

        # Verify the input tokens.
        assert dp_x.shape == (num_tokens, self._hidden_dim)
        assert dp_x.stride(1) == 1
        assert dp_x.dtype == self._in_dtype
        x_ptr = dp_x.data_ptr()
        x_stride = dp_x.stride(0)

        # Verify the input scales.
        x_scale_ptr: Optional[int]
        x_scale_stride_elem: Optional[int]
        x_scale_stride_token: Optional[int]
        if dp_x_scale is not None:
            assert self._scale_dtype is not None
            assert self._hidden_dim_scale is not None
            assert out_expert_x_scale is not None
            assert dp_x_scale.dtype == self._scale_dtype
            x_scale_ptr = dp_x_scale.data_ptr()
            x_scale_stride_elem = dp_x_scale.stride(1)
            x_scale_stride_token = dp_x_scale.stride(0)
        else:
            assert self._scale_dtype is None
            assert self._hidden_dim_scale is None
            x_scale_ptr = None
            x_scale_stride_elem = None
            x_scale_stride_token = None

        # Verify the indices.
        assert indices.shape == (num_tokens, self._num_experts_per_token)
        assert indices.stride(1) == 1
        assert indices.dtype == torch.uint32
        indices_ptr = indices.data_ptr()
        indices_stride = indices.stride(0)

        # Verify the weights.
        assert weights.shape == (num_tokens, self._num_experts_per_token)
        assert weights.stride(1) == 1
        assert weights.dtype == torch.float32
        weights_ptr = weights.data_ptr()
        weights_stride = weights.stride(0)

        # Verify the dynamic `m` bound.
        bound_m_ptr: Optional[int]
        if bound_m is not None:
            assert bound_m.numel() == 1
            assert bound_m.dtype == torch.int32
            bound_m_ptr = bound_m.data_ptr()
        else:
            bound_m_ptr = None

        stream = torch.cuda.current_stream().cuda_stream

        if do_send:
            all_to_all.dispatch_send(
                slot=0,
                num_tokens=num_tokens,
                x_ptr=x_ptr,
                x_stride=x_stride * self._in_dtype.itemsize,
                x_scale_ptr=x_scale_ptr,
                x_scale_stride_elem=x_scale_stride_elem,
                x_scale_stride_token=x_scale_stride_token,
                indices_ptr=indices_ptr,
                indices_stride=indices_stride,
                weights_ptr=weights_ptr,
                weights_stride=weights_stride,
                bound_m_ptr=bound_m_ptr,
                stream=stream,
            )

        if do_recv:
            all_to_all.dispatch_recv(
                slot=0,
                out_num_tokens_ptr=out_expert_num_tokens_ptr,
                out_x_ptr=out_x_ptr,
                out_x_stride=out_x_stride,
                out_x_scale_ptr=out_x_scale_ptr,
                out_x_scale_stride_elem=out_x_scale_stride_elem,
                out_x_scale_stride_token=out_x_scale_stride_token,
                stream=stream,
            )

    def _flatten_batched_experts(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        hidden_dim: int,
        name: str,
    ) -> torch.Tensor:
        if tensor.ndim == 2:
            num_expert_tokens, _ = tensor.shape
            assert tensor.shape == (num_expert_tokens, hidden_dim)
            assert tensor.stride(1) == 1
            assert tensor.dtype == dtype
            return tensor

        assert self._max_tokens_per_expert is not None
        assert tensor.ndim == 3
        assert tensor.shape == (
            self._num_local_experts,
            self._max_tokens_per_expert,
            hidden_dim,
        ), f"{name} has unexpected shape {tuple(tensor.shape)}"
        assert tensor.stride(2) == 1
        assert tensor.dtype == dtype
        return tensor.reshape(-1, hidden_dim)

    def dispatch_async(
        self,
        *,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: Optional[torch.Tensor],
        dp_x: torch.Tensor,
        dp_x_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        weights: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
    ) -> P2PDispatchHandle:
        slot = self._acquire_slot()
        self.dispatch(
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            weights=weights,
            bound_m=bound_m,
            slot=slot,
            do_send=True,
            do_recv=False,
        )
        send_done_event = torch.cuda.Event()
        send_done_event.record(torch.cuda.current_stream(dp_x.device))
        return P2PDispatchHandle(
            kernel=self,
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            weights=weights,
            bound_m=bound_m,
            send_done_event=send_done_event,
            slot=slot,
        )

    @override
    def combine(
        self,
        out_tokens: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
        slot: int = 0,
        do_send: bool = True,
        do_recv: bool = True,
        accumulate: bool = False,
    ) -> None:
        assert self._all_to_alls
        assert do_send or do_recv
        all_to_all = self._ctx(slot)

        # TODO: accumulate with TP across NVLink
        assert not accumulate or self._dp_size == 1

        num_tokens, _ = indices.shape
        expert_y = self._flatten_batched_experts(
            expert_y,
            dtype=expert_y.dtype,
            hidden_dim=self._hidden_dim,
            name="expert_y",
        )
        num_recv_tokens, _ = expert_y.shape

        assert out_tokens.shape == (num_tokens, self._hidden_dim)
        assert out_tokens.dtype == self._out_dtype
        assert out_tokens.stride(1) == 1
        out_tokens_ptr = out_tokens.data_ptr()
        out_tokens_stride = out_tokens.stride(0)

        assert indices.shape == (num_tokens, self._num_experts_per_token)
        assert indices.stride(1) == 1
        assert indices.dtype == torch.uint32
        indices_ptr = indices.data_ptr()
        indices_stride = indices.stride(0)

        assert weights.shape == (num_tokens, self._num_experts_per_token)
        assert weights.stride(1) == 1
        assert weights.dtype == torch.float32
        weights_ptr = weights.data_ptr()
        weights_stride = weights.stride(0)

        assert expert_y.shape == (num_recv_tokens, self._hidden_dim)
        expert_y_ptr = expert_y.data_ptr()
        expert_y_stride = expert_y.stride(0) * expert_y.dtype.itemsize

        bound_m_ptr: Optional[int]
        if bound_m is not None:
            assert bound_m.numel() == 1
            assert bound_m.dtype == torch.int32
            bound_m_ptr = bound_m.data_ptr()
        else:
            bound_m_ptr = None

        stream = torch.cuda.current_stream().cuda_stream

        if do_send:
            all_to_all.combine_send(
                slot=0,
                expert_x_ptr=expert_y_ptr,
                expert_x_stride=expert_y_stride,
                stream=stream,
            )

        if do_recv:
            all_to_all.combine_recv(
                slot=0,
                num_tokens=num_tokens,
                num_recv_tokens=num_recv_tokens,
                expert_y_dtype=expert_y.dtype,
                out_tokens_ptr=out_tokens_ptr,
                out_tokens_stride=out_tokens_stride,
                indices_ptr=indices_ptr,
                indices_stride=indices_stride,
                weights_ptr=weights_ptr,
                weights_stride=weights_stride,
                bound_m_ptr=bound_m_ptr,
                accumulate=accumulate,
                stream=stream,
            )

    def combine_async(
        self,
        *,
        out_tokens: torch.Tensor,
        dispatch_handle: P2PDispatchHandle,
        expert_y: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
        accumulate: bool = False,
    ) -> P2PCombineHandle:
        self.combine(
            out_tokens=out_tokens,
            indices=dispatch_handle.indices,
            weights=dispatch_handle.weights,
            expert_y=expert_y,
            bound_m=bound_m,
            slot=dispatch_handle.slot,
            do_send=True,
            do_recv=False,
            accumulate=accumulate,
        )
        send_done_event = torch.cuda.Event()
        send_done_event.record(torch.cuda.current_stream(out_tokens.device))
        return P2PCombineHandle(
            kernel=self,
            dispatch_handle=dispatch_handle,
            out_tokens=out_tokens,
            expert_y=expert_y,
            bound_m=bound_m,
            accumulate=accumulate,
            send_done_event=send_done_event,
            slot=dispatch_handle.slot,
        )

    def get_perf_stats(self) -> dict[str, Any]:
        if not self._all_to_alls:
            return {}
        stats = self._all_to_alls[0].get_perf_stats()
        for ctx in self._all_to_alls[1:]:
            slot_stats = ctx.get_perf_stats()
            for key, value in slot_stats.items():
                if isinstance(value, list):
                    stats[key] = [a + b for a, b in zip(stats[key], value)]
                else:
                    stats[key] += value
        return stats

    @override
    def destroy(self) -> None:
        """Clean up the all-to-all context."""

        # Stop the a2a engine, ensuring all RDMA transfers complete.
        self._global_group.barrier()
        self._all_to_alls.clear()

        # Stop the transfer engine once no rank is active.
        self._global_group.barrier()
        if self._transfer_engine is not None and self._owns_transfer_engine:
            self._transfer_engine.stop()
            del self._transfer_engine
            self._transfer_engine = None
