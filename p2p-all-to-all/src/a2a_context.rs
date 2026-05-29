use std::{
    ffi::c_void,
    ptr::null_mut,
    sync::{Arc, atomic::Ordering},
    thread::JoinHandle,
};

use anyhow::{Result, anyhow};
use cuda_lib::{
    CudaDeviceMemory, cuda_check,
    gdr::{GdrCopyContext, GdrFlag},
    rt::{CudartError, cudaGetNumSMs},
};
use fabric_lib::{TransferEngine, api::MemoryRegionHandle};
use thread_lib::pin_cpu;
use torch_lib::ScalarType;

use crate::{a2a_handles::AllToAllRankHandle, a2a_worker::WorkerState};

// Collects the private workspace buffers used by dispatch and combine.
struct DeviceWorkspace {
    /// The offset of each expert within the contiguous token buffer.
    expert_offsets: CudaDeviceMemory,
    /// The offset of the token within the expert group.
    token_offset: CudaDeviceMemory,
    /// Counter for the number of tokens sent during combine.
    token_counter: CudaDeviceMemory,
    /// Counter for per-grid synchronization.
    grid_counter: CudaDeviceMemory,
    /// Counter for synchronization barriers across NVLink.
    sync_counter: CudaDeviceMemory,
    /// Device-side sync pointers.
    sync_ptrs: Option<CudaDeviceMemory>,
    /// Device-side send pointers.
    send_ptrs: Option<CudaDeviceMemory>,
    /// Device-side recv pointers.
    recv_ptrs: Option<CudaDeviceMemory>,
}

impl DeviceWorkspace {
    pub fn new(
        num_experts: usize,
        max_num_tokens: usize,
        num_experts_per_token: usize,
        host_sync_ptrs: &[u64],
        host_send_ptrs: &[u64],
        host_recv_ptrs: &[u64],
    ) -> Result<Self, CudartError> {
        let expert_offsets =
            CudaDeviceMemory::device(num_experts * std::mem::size_of::<u32>())?;
        expert_offsets.zero();

        let token_offset = CudaDeviceMemory::device(
            max_num_tokens * num_experts_per_token * std::mem::size_of::<u32>(),
        )?;

        let token_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        token_counter.zero();
        let sync_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        sync_counter.zero();
        let grid_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        grid_counter.zero();

        let sync_ptrs = if host_sync_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_sync_ptrs)?)
        };
        let send_ptrs = if host_send_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_send_ptrs)?)
        };
        let recv_ptrs = if host_recv_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_recv_ptrs)?)
        };

        Ok(Self {
            expert_offsets,
            token_offset,
            token_counter,
            grid_counter,
            sync_counter,
            sync_ptrs,
            send_ptrs,
            recv_ptrs,
        })
    }

    fn get_sync_ptr(&mut self) -> *mut *mut u32 {
        self.sync_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }

    fn get_recv_ptr(&mut self) -> *mut *mut c_void {
        self.recv_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }

    fn get_send_ptr(&mut self) -> *mut *mut c_void {
        self.send_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }
}

#[allow(dead_code)]
pub struct AllToAllContext {
    hidden_dim: usize,
    hidden_dim_scale: usize,
    in_elemsize: usize,
    out_elemsize: usize,
    out_dtype: ScalarType,
    scale_elemsize: usize,
    num_experts: usize,
    max_num_tokens: usize,
    num_experts_per_token: usize,
    max_private_tokens: usize,
    rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    device: u8,
    workspaces: Vec<DeviceWorkspace>,
    workers: Vec<Arc<WorkerState>>,
    threads: Vec<JoinHandle<()>>,
    num_blocks: usize,
}

impl AllToAllContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_dim: usize,
        hidden_dim_scale: usize,
        in_elemsize: usize,
        out_elemsize: usize,
        out_dtype: ScalarType,
        scale_elemsize: usize,
        max_num_tokens: usize,
        max_recv_tokens: usize,
        max_tokens_per_expert: usize,
        max_private_tokens: usize,
        num_experts: usize,
        expert_padding: usize,
        num_experts_per_token: usize,
        rank: usize,
        dp_size: usize,
        node_size: usize,
        world_size: usize,
        num_routed_ptr: *mut u32,
        num_routed_mr: MemoryRegionHandle,
        send_buffer_ptr: *mut c_void,
        send_buffer_mr: MemoryRegionHandle,
        recv_buffer_ptr: *mut c_void,
        recv_buffer_mr: MemoryRegionHandle,
        sync_ptrs: Vec<u64>,
        send_ptrs: Vec<u64>,
        recv_ptrs: Vec<u64>,
        device: u8,
        imm_base: u32,
        rank_handles: Vec<AllToAllRankHandle>,
        transfer_engine: Arc<TransferEngine>,
        worker_cpu: Option<u16>,
        num_slots: usize,
    ) -> Result<Self> {
        // Start the all-to-all worker thread.
        for (i, peer) in rank_handles.iter().enumerate() {
            tracing::info!("Rank#{} Peer#{}: {}", rank, i, peer.address);
        }

        let num_slots = num_slots.max(1);
        let tx_ready_context = GdrCopyContext::new()?;
        let tx_ready = Arc::new(GdrFlag::new(&tx_ready_context)?);
        tx_ready.set(true);

        let mut workers = Vec::with_capacity(num_slots);
        let mut threads = Vec::with_capacity(num_slots);
        let mut workspaces = Vec::with_capacity(num_slots);

        for slot_idx in 0..num_slots {
            let slot_imm_base = imm_base + (slot_idx as u32) * 5;
            let worker: Arc<WorkerState> = Arc::new(WorkerState::new(
                hidden_dim,
                hidden_dim_scale,
                in_elemsize,
                out_elemsize,
                scale_elemsize,
                max_num_tokens,
                max_recv_tokens,
                max_tokens_per_expert,
                max_private_tokens,
                num_experts,
                expert_padding,
                num_experts_per_token,
                rank,
                dp_size,
                node_size,
                world_size,
                num_routed_ptr,
                num_routed_mr,
                send_buffer_ptr,
                send_buffer_mr,
                recv_buffer_ptr,
                recv_buffer_mr,
                device,
                slot_imm_base,
                rank_handles.clone(),
                transfer_engine.clone(),
                tx_ready.clone(),
            )?);

            let workspace = DeviceWorkspace::new(
                num_experts,
                max_num_tokens,
                num_experts_per_token,
                &sync_ptrs,
                &send_ptrs,
                &recv_ptrs,
            )?;

            // Create the worker thread.
            let (init_tx, init_rx) = oneshot::channel();
            let thread_worker = worker.clone();
            let thread = std::thread::Builder::new()
                .name(format!("p2p_all_to_all Worker[{slot_idx}]"))
                .spawn(move || {
                    // Pin to the desired CPU.
                    tracing::info!(
                        "Running worker slot {} for cuda:{}",
                        slot_idx,
                        device
                    );
                    if let Some(cpu) = worker_cpu {
                        if let Err(e) = pin_cpu(cpu.into()) {
                            tracing::info!("Failed to pin CPU {}: {:?}", cpu, e);
                        }
                        tracing::info!(
                            "Pinned worker slot {} for cuda:{} to CPU {}",
                            slot_idx,
                            device,
                            cpu
                        );
                    }

                    // Block until the worker is fully initialized.
                    if init_tx.send(()).is_err() {
                        panic!("Failed to send initialization signal");
                    } else {
                        tracing::info!(
                            "Initialized worker slot {} for cuda:{}",
                            slot_idx,
                            device
                        );
                    }

                    // Main loop.
                    thread_worker.main_loop();
                    tracing::info!(
                        "Stopping worker slot {} for cuda:{}",
                        slot_idx,
                        device
                    );
                })
                .expect("Failed to spawn p2p_all_to_all Worker thread");
            init_rx.recv()?;

            workers.push(worker);
            threads.push(thread);
            workspaces.push(workspace);
        }

        let num_blocks = cudaGetNumSMs(device)?;

        // Build the context.
        Ok(Self {
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            out_elemsize,
            out_dtype,
            scale_elemsize,
            num_experts,
            max_num_tokens,
            num_experts_per_token,
            max_private_tokens,
            rank,
            dp_size,
            node_size,
            world_size,
            device,
            workspaces,
            workers,
            threads,
            num_blocks,
        })
    }

    pub fn destroy(&mut self) -> Result<()> {
        // Stop all work on the worker thread.
        tracing::info!("Stopping worker thread for cuda:{}", self.device);

        for worker in &self.workers {
            worker.stop();
        }
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                return Err(anyhow!("Failed to join thread"));
            }
        }
        Ok(())
    }

    fn worker(&self, slot: usize) -> Result<&Arc<WorkerState>> {
        self.workers
            .get(slot)
            .ok_or_else(|| anyhow!("Invalid all-to-all slot {}", slot))
    }

    fn workspace_mut(&mut self, slot: usize) -> Result<&mut DeviceWorkspace> {
        self.workspaces
            .get_mut(slot)
            .ok_or_else(|| anyhow!("Invalid all-to-all slot {}", slot))
    }

    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn dispatch_send(
        &mut self,
        slot: usize,
        num_tokens: usize,
        x_ptr: *const c_void,
        x_stride: usize,
        x_scale_ptr: *const c_void,
        x_scale_stride_elem: usize,
        x_scale_stride_token: usize,
        indices: *const i32,
        indices_stride: usize,
        weights: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        stream: u64,
    ) -> Result<()> {
        if num_tokens > self.max_num_tokens {
            return Err(anyhow!("Number of tokens exceeds maximum allowed"));
        }
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let hidden_dim_scale = self.hidden_dim_scale;
        let num_experts = self.num_experts;
        let num_experts_per_token = self.num_experts_per_token;
        let max_private_tokens = self.max_private_tokens;
        let rank = self.rank;
        let dp_size = self.dp_size;
        let node_size = self.node_size;
        let world_size = self.world_size;
        let in_elemsize = self.in_elemsize;
        let scale_elemsize = self.scale_elemsize;
        let worker = self.worker(slot)?.clone();
        let workspace = self.workspace_mut(slot)?;

        cuda_check!(a2a_kernels::a2a_dispatch_send(
            num_blocks,
            hidden_dim,
            hidden_dim_scale,
            num_experts,
            num_experts_per_token,
            max_private_tokens,
            rank,
            dp_size,
            node_size,
            world_size,
            num_tokens,
            bound_m_ptr,
            x_ptr as *const u8,
            in_elemsize,
            x_stride,
            x_scale_ptr as *const u8,
            scale_elemsize,
            x_scale_stride_elem,
            x_scale_stride_token,
            indices,
            indices_stride,
            weights,
            weights_stride,
            workspace.token_offset.get_mut_ptr(),
            worker.buffers.num_routed_ptr,
            workspace.expert_offsets.get_mut_ptr(),
            worker.slot.dispatch_route_done.get_device_ptr(),
            worker.slot.dispatch_send_done.get_device_ptr(),
            worker.tx_ready.get_device_ptr(),
            worker.buffers.send_buffer_ptr as *mut u8,
            workspace.grid_counter.get_mut_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            workspace.get_recv_ptr() as *mut *mut u8,
            stream,
        ))?;

        if worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn dispatch_recv(
        &mut self,
        slot: usize,
        out_num_tokens_ptr: *mut i32,
        out_x_ptr: *mut c_void,
        out_x_stride: usize,
        out_x_scale_ptr: *mut c_void,
        out_x_scale_stride_elem: usize,
        out_x_scale_stride_token: usize,
        stream: u64,
    ) -> Result<()> {
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let hidden_dim_scale = self.hidden_dim_scale;
        let in_elemsize = self.in_elemsize;
        let scale_elemsize = self.scale_elemsize;
        let num_experts = self.num_experts;
        let rank = self.rank;
        let dp_size = self.dp_size;
        let node_size = self.node_size;
        let world_size = self.world_size;
        let worker = self.worker(slot)?.clone();
        let workspace = self.workspace_mut(slot)?;

        cuda_check!(a2a_kernels::a2a_dispatch_recv(
            num_blocks,
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            scale_elemsize,
            num_experts,
            rank,
            dp_size,
            node_size,
            world_size,
            out_num_tokens_ptr,
            out_x_ptr as *mut u8,
            out_x_stride,
            out_x_scale_ptr as *mut u8,
            out_x_scale_stride_elem,
            out_x_scale_stride_token,
            worker.slot.tokens_per_expert.get_device_ptr(),
            worker.buffers.send_buffer_ptr as *mut u8,
            worker.buffers.recv_buffer_ptr as *mut u8,
            worker.slot.source_rank.get_device_ptr(),
            worker.slot.source_dispatch_offset.get_device_ptr(),
            worker.slot.padded_index.get_device_ptr(),
            worker.buffers.num_routed_ptr,
            worker.slot.num_recv_tokens.get_device_ptr(),
            worker.slot.num_recv_tokens_flag.get_device_ptr(),
            worker.slot.dispatch_recv_flag.get_device_ptr(),
            worker.slot.dispatch_recv_done.get_device_ptr(),
            workspace.grid_counter.get_mut_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            workspace.get_send_ptr() as *mut *mut u8,
            stream,
        ))?;

        if worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }

        Ok(())
    }

    #[allow(
        unused_variables,
        clippy::too_many_arguments,
        clippy::not_unsafe_ptr_arg_deref
    )]
    pub fn combine_send(
        &mut self,
        slot: usize,
        expert_x_ptr: *const c_void,
        expert_x_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let out_elemsize = self.out_elemsize;
        let rank = self.rank;
        let node_size = self.node_size;
        let dp_size = self.dp_size;
        let worker = self.worker(slot)?.clone();
        let workspace = self.workspace_mut(slot)?;

        cuda_check!(a2a_kernels::a2a_combine_send(
            num_blocks,
            hidden_dim,
            out_elemsize,
            rank,
            node_size,
            dp_size,
            expert_x_ptr as *const u8,
            expert_x_stride,
            worker.tx_ready.get_device_ptr(),
            worker.buffers.send_buffer_ptr as *mut u8,
            worker.buffers.recv_buffer_ptr as *mut u8,
            worker.slot.source_rank.get_device_ptr(),
            worker.slot.combine_send_offset.get_device_ptr(),
            worker.slot.padded_index.get_device_ptr(),
            worker.slot.num_recv_tokens.get_device_ptr(),
            worker.slot.combine_send_done.get_device_ptr(),
            workspace.token_counter.get_mut_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            workspace.get_recv_ptr() as *mut *mut u8,
            stream,
        ))?;

        if worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }

        Ok(())
    }

    #[allow(
        unused_variables,
        clippy::too_many_arguments,
        clippy::not_unsafe_ptr_arg_deref
    )]
    pub fn combine_recv(
        &mut self,
        slot: usize,
        num_tokens: usize,
        num_recv_tokens: usize,
        expert_y_dtype: ScalarType,
        out_tokens_ptr: *mut c_void,
        out_tokens_stride: usize,
        indices_ptr: *const i32,
        indices_stride: usize,
        weights_ptr: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        accumulate: bool,
        stream: u64,
    ) -> Result<()> {
        let num_blocks = self.num_blocks;
        let hidden_dim = self.hidden_dim;
        let out_elemsize = self.out_elemsize;
        let out_dtype = self.out_dtype;
        let num_experts = self.num_experts;
        let num_experts_per_token = self.num_experts_per_token;
        let rank = self.rank;
        let node_size = self.node_size;
        let world_size = self.world_size;
        let worker = self.worker(slot)?.clone();
        let workspace = self.workspace_mut(slot)?;

        cuda_check!(a2a_kernels::a2a_combine_recv(
            num_blocks,
            hidden_dim,
            out_elemsize,
            expert_y_dtype,
            out_dtype,
            num_experts,
            num_experts_per_token,
            rank,
            node_size,
            world_size,
            num_tokens,
            bound_m_ptr,
            indices_ptr,
            indices_stride,
            weights_ptr,
            weights_stride,
            out_tokens_ptr as *mut u8,
            out_tokens_stride,
            accumulate,
            worker.buffers.recv_buffer_ptr as *mut u8,
            workspace.token_offset.get_mut_ptr(),
            workspace.expert_offsets.get_mut_ptr(),
            worker.slot.combine_recv_flag.get_device_ptr(),
            worker.slot.combine_recv_done.get_device_ptr(),
            workspace.sync_counter.get_mut_ptr(),
            workspace.get_sync_ptr(),
            stream,
        ))?;

        if worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }

        Ok(())
    }

    pub fn get_perf_stats(&self) -> AllToAllPerfStats {
        let mut stats = AllToAllPerfStats {
            local_dispatch_bytes: 0,
            nvlink_dispatch_bytes: 0,
            network_dispatch_bytes: 0,
            local_combine_bytes: 0,
            nvlink_combine_bytes: 0,
            network_combine_bytes: 0,
            peer_dispatch_bytes: vec![0; self.world_size],
            peer_combine_bytes: vec![0; self.world_size],
        };
        for worker in &self.workers {
            stats.local_dispatch_bytes +=
                worker.accumulated_local_dispatch_bytes.load(Ordering::Relaxed);
            stats.nvlink_dispatch_bytes +=
                worker.accumulated_nvlink_dispatch_bytes.load(Ordering::Relaxed);
            stats.network_dispatch_bytes +=
                worker.accumulated_network_dispatch_bytes.load(Ordering::Relaxed);
            stats.local_combine_bytes +=
                worker.accumulated_local_combine_bytes.load(Ordering::Relaxed);
            stats.nvlink_combine_bytes +=
                worker.accumulated_nvlink_combine_bytes.load(Ordering::Relaxed);
            stats.network_combine_bytes +=
                worker.accumulated_network_combine_bytes.load(Ordering::Relaxed);
            for (dst, value) in stats
                .peer_dispatch_bytes
                .iter_mut()
                .zip(worker.peer_dispatch_bytes.iter())
            {
                *dst += value.load(Ordering::Relaxed);
            }
            for (dst, value) in stats
                .peer_combine_bytes
                .iter_mut()
                .zip(worker.peer_combine_bytes.iter())
            {
                *dst += value.load(Ordering::Relaxed);
            }
        }
        stats
    }
}

pub struct AllToAllPerfStats {
    pub local_dispatch_bytes: u64,
    pub nvlink_dispatch_bytes: u64,
    pub network_dispatch_bytes: u64,
    pub local_combine_bytes: u64,
    pub nvlink_combine_bytes: u64,
    pub network_combine_bytes: u64,
    pub peer_dispatch_bytes: Vec<u64>,
    pub peer_combine_bytes: Vec<u64>,
}

impl Drop for AllToAllContext {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}
