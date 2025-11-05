use std::{
    ffi::c_void,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering},
    },
};

use anyhow::{Context, Result};
use cuda_lib::{
    CudaDeviceId, Device,
    gdr::{GdrCopyContext, GdrFlag, GdrVec},
};
use fabric_lib::{
    RdmaEngine, TransferEngine,
    api::{
        BarrierTransferRequest, DomainAddress, DomainGroupRouting, GdrCounter,
        GroupTransferRouting, ImmCounter, MemoryRegionDescriptor, MemoryRegionHandle,
        ScatterTarget, ScatterTransferRequest, TransferRequest,
    },
};
use nvtx::{range_end, range_start};

use crate::a2a_handles::AllToAllRankHandle;

pub(crate) struct WorkerBuffers {
    pub(crate) num_routed_ptr: *mut u32,
    pub(crate) send_buffer_ptr: *mut c_void,
    pub(crate) recv_buffer_ptr: *mut c_void,
}

unsafe impl Send for WorkerBuffers {}
unsafe impl Sync for WorkerBuffers {}

#[allow(dead_code)]
pub(crate) struct WorkerState {
    transfer_engine: Arc<TransferEngine>,
    max_num_tokens: usize,
    max_recv_tokens: usize,
    max_private_tokens: usize,
    hidden_dim: usize,
    hidden_dim_scale: usize,
    in_elemsize: usize,
    out_elemsize: usize,
    scale_elemsize: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    expert_padding: usize,
    rank: usize,
    dp_rank: usize,
    dp_group: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    rank_handles: Vec<AllToAllRankHandle>,
    stop_flag: AtomicBool,
    device: u8,
    dispatch_imm: u32,
    combine_imm: u32,
    route_imm: u32,
    dispatch_barrier_imm: u32,
    combine_barrier_imm: u32,
    num_routed_mr: MemoryRegionHandle,
    send_buffer_mr: MemoryRegionHandle,
    recv_buffer_mr: MemoryRegionHandle,
    pub(crate) buffers: WorkerBuffers,
    pub(crate) dispatch_route_done: GdrFlag,
    pub(crate) dispatch_send_done: GdrFlag,
    pub(crate) dispatch_recv_done: GdrFlag,
    pub(crate) combine_send_done: GdrFlag,
    pub(crate) combine_recv_done: GdrFlag,
    pub(crate) tokens_per_expert: GdrVec<u32>,
    pub(crate) source_dispatch_offset: GdrVec<u32>,
    pub(crate) combine_send_offset: GdrVec<u32>,
    pub(crate) source_rank: GdrVec<u32>,
    pub(crate) padded_index: GdrVec<u32>,
    pub(crate) num_recv_tokens: GdrVec<u32>,
    pub(crate) num_recv_tokens_flag: GdrFlag,
    pub(crate) dispatch_recv_flag: Arc<GdrFlag>,
    pub(crate) combine_recv_flag: Arc<GdrFlag>,
    pub(crate) tx_ready: GdrFlag,
    route_counter: ImmCounter,
    dispatch_counter: GdrCounter,
    combine_counter: ImmCounter,
    dispatch_barrier_counter: ImmCounter,
    combine_barrier_counter: ImmCounter,
    tx_counter: Arc<AtomicI64>,
    err_counter: Arc<AtomicI64>,
    route_write_op: TransferRequest,
    dispatch_barrier_write_op: TransferRequest,
    combine_barrier_write_op: TransferRequest,
}

#[derive(Debug)]
struct RoutingInfo {
    num_recv_tx: u32,
    dispatch_ranges: Arc<Vec<ScatterTarget>>,
    combine_ranges: Arc<Vec<ScatterTarget>>,
}

impl WorkerState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_dim: usize,
        hidden_dim_scale: usize,
        in_elemsize: usize,
        out_elemsize: usize,
        scale_elemsize: usize,
        max_num_tokens: usize,
        max_recv_tokens: usize,
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
        device: u8,
        imm_base: u32,
        rank_handles: Vec<AllToAllRankHandle>,
        transfer_engine: Arc<TransferEngine>,
    ) -> Result<Self> {
        let dp_rank = rank % dp_size;
        let dp_group = rank / dp_size;
        let num_local_experts = num_experts.div_ceil(world_size);

        let gdr_context = GdrCopyContext::new()?;

        let dispatch_route_done = GdrFlag::new(&gdr_context)?;
        let dispatch_send_done = GdrFlag::new(&gdr_context)?;
        let dispatch_recv_done = GdrFlag::new(&gdr_context)?;
        let combine_send_done = GdrFlag::new(&gdr_context)?;
        let combine_recv_done = GdrFlag::new(&gdr_context)?;
        let num_recv_tokens_flag = GdrFlag::new(&gdr_context)?;
        let tx_ready = GdrFlag::new(&gdr_context)?;
        let dispatch_recv_flag = Arc::new(GdrFlag::new(&gdr_context)?);
        let combine_recv_flag = Arc::new(GdrFlag::new(&gdr_context)?);

        let tokens_per_expert = GdrVec::new(&gdr_context, num_local_experts)?;
        let source_rank = GdrVec::new(&gdr_context, max_recv_tokens)?;
        let source_dispatch_offset = GdrVec::new(&gdr_context, max_recv_tokens)?;
        let combine_send_offset = GdrVec::new(&gdr_context, max_recv_tokens)?;
        let padded_index = GdrVec::new(&gdr_context, max_recv_tokens)?;
        let num_recv_tokens = GdrVec::new(&gdr_context, 3)?;

        num_recv_tokens.copy(&[0u32, 0u32, 0u32]);
        num_recv_tokens_flag.set(false);
        dispatch_recv_flag.set(false);
        combine_recv_flag.set(false);
        tx_ready.set(true);

        // Set up the immediate counters.
        let route_imm = imm_base;
        let route_counter = transfer_engine.get_imm_counter(route_imm);
        let dispatch_imm = imm_base + 1;
        let dispatch_counter =
            transfer_engine.get_gdr_counter(dispatch_imm, dispatch_recv_flag.clone());
        let combine_imm = imm_base + 2;
        let combine_counter = transfer_engine.get_imm_counter(combine_imm);
        let dispatch_barrier_imm = imm_base + 3;
        let dispatch_barrier_counter =
            transfer_engine.get_imm_counter(dispatch_barrier_imm);
        let combine_barrier_imm = imm_base + 4;
        let combine_barrier_counter =
            transfer_engine.get_imm_counter(combine_barrier_imm);

        // Prepare the re-usable command to send out routing info.
        let route_write_op = {
            // Send the expert counts, plus one, over to all peers.
            let mut dsts = Vec::with_capacity(world_size - 1);
            let mut addrs = Vec::with_capacity(world_size - 1);
            for i in 1..world_size {
                // Do not transfer to self.
                let peer_rank = (rank + i) % world_size;
                let peer_group = peer_rank / dp_size;
                if peer_group == dp_group || peer_rank % dp_size != dp_rank {
                    continue;
                }

                let dst_mr = rank_handles[peer_rank].num_routed_desc.clone();
                let length: u64 = (num_experts * std::mem::size_of::<u32>()) as u64;
                let offset: u64 = (dp_group as u64) * length;
                addrs.push(extract_addrs(&dst_mr));
                dsts.push(ScatterTarget {
                    length,
                    src_offset: offset,
                    dst_offset: offset,
                    dst_mr,
                })
            }
            let dst_handle = transfer_engine
                .add_peer_group(addrs, Device::Cuda(CudaDeviceId(device)))
                .context("Failed to add_peer_group for route_write_op")?;

            TransferRequest::Scatter(ScatterTransferRequest {
                src_mr: num_routed_mr,
                dst_handle: Some(dst_handle),
                dsts: Arc::new(dsts),
                imm_data: Some(route_imm),
                domain: GroupTransferRouting::AllDomainsShardPeers,
            })
        };

        let (dispatch_barrier_write_op, combine_barrier_write_op) = {
            // Send an immedate to all peer ranks.
            let mut dst_mrs = Vec::with_capacity(world_size - 1);
            for i in 1..world_size {
                let peer_rank = (rank + i) % world_size;
                if peer_rank == rank {
                    continue;
                }
                dst_mrs.push(rank_handles[peer_rank].recv_buffer_desc.clone());
            }
            let dispatch = TransferRequest::Barrier(BarrierTransferRequest {
                imm_data: dispatch_barrier_imm,
                dst_mrs: dst_mrs.clone(),
                domain: DomainGroupRouting::Pinned { domain_idx: 0 },
            });
            let combine = TransferRequest::Barrier(BarrierTransferRequest {
                imm_data: combine_barrier_imm,
                dst_mrs,
                domain: DomainGroupRouting::Pinned { domain_idx: 0 },
            });
            (dispatch, combine)
        };

        Ok(WorkerState {
            transfer_engine: transfer_engine.clone(),
            max_num_tokens,
            max_recv_tokens,
            max_private_tokens,
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            out_elemsize,
            scale_elemsize,
            num_experts,
            num_experts_per_token,
            expert_padding,
            rank,
            dp_rank,
            dp_group,
            dp_size,
            node_size,
            world_size,
            rank_handles,
            stop_flag: AtomicBool::new(false),
            device,
            dispatch_imm,
            combine_imm,
            route_imm,
            dispatch_barrier_imm,
            combine_barrier_imm,
            buffers: WorkerBuffers { num_routed_ptr, send_buffer_ptr, recv_buffer_ptr },
            dispatch_route_done,
            dispatch_send_done,
            dispatch_recv_done,
            combine_send_done,
            combine_recv_done,
            num_routed_mr,
            send_buffer_mr,
            recv_buffer_mr,
            tokens_per_expert,
            source_rank,
            source_dispatch_offset,
            combine_send_offset,
            padded_index,
            num_recv_tokens,
            num_recv_tokens_flag,
            dispatch_recv_flag,
            combine_recv_flag,
            tx_ready,
            route_counter,
            dispatch_counter,
            combine_counter,
            dispatch_barrier_counter,
            combine_barrier_counter,
            tx_counter: Arc::new(AtomicI64::new(0)),
            err_counter: Arc::new(AtomicI64::new(0)),
            route_write_op,
            dispatch_barrier_write_op,
            combine_barrier_write_op,
        })
    }

    fn is_running(&self) -> bool {
        !self.stop_flag.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        self.dispatch_route_done.set(true);
        self.dispatch_send_done.set(true);
        self.dispatch_recv_done.set(true);
        self.combine_send_done.set(true);
        self.combine_recv_done.set(true);
    }

    fn get_num_routed(&self, dp_group: usize, expert: usize) -> u32 {
        assert!(dp_group < self.world_size / self.dp_size);
        assert!(expert < self.num_experts);
        unsafe {
            AtomicU32::from_ptr(
                self.buffers.num_routed_ptr.add(dp_group * self.num_experts + expert),
            )
        }
        .load(Ordering::Relaxed)
    }

    fn get_dispatch_token_dim(&self) -> usize {
        let token_dim = (self.hidden_dim * self.in_elemsize).div_ceil(16) * 16;
        let scale_dim = (self.hidden_dim_scale * self.scale_elemsize).div_ceil(16) * 16;
        token_dim + scale_dim + 16
    }

    fn get_combine_token_dim(&self) -> usize {
        (self.hidden_dim * self.out_elemsize).div_ceil(16) * 16
    }

    pub fn main_loop(&self) {
        // Worker thread main loop.
        while self.is_running() {
            let step_dispatch_range = range_start!("p2p_all_to_all");
            self.step();
            range_end!(step_dispatch_range);
        }
    }

    pub fn failed(&self) -> bool {
        self.err_counter.load(Ordering::Relaxed) != 0
    }

    fn step(&self) {
        // Wait for the device to copy the routing info to the host.
        self.dispatch_route_done.wait();
        if !self.is_running() {
            return;
        }

        // Start exchanging routing info.
        self.transfer_engine
            .submit_transfer_atomic(
                self.route_write_op.clone(),
                self.tx_counter.clone(),
                self.err_counter.clone(),
            )
            .unwrap();

        // Wait for the dispatch kernel to copy tokens into send buffers.
        self.dispatch_send_done.wait();
        self.tx_ready.set(false);
        if !self.is_running() {
            return;
        }

        // Trigger transfers into private recv buffers.
        let num_private_ranges = self.dispatch_initial_routes();

        // Wait for the routing information to arrive and aggregate it.
        let num_dp_groups = (self.world_size / self.dp_size) as u32;
        self.route_counter.wait(num_dp_groups - 1);
        let route = self.process_routing_info();

        // Register a callback to wait for the expected number of immediates.
        let num_shards = self.transfer_engine.nets_per_gpu().get() as u32;
        let num_dispatch_tx = 1
            + if num_private_ranges == 0 { 0 } else { 1 }
            + if route.dispatch_ranges.is_empty() { 0 } else { 1 };
        let num_combine_tx = 1 + if self.world_size > self.node_size { 1 } else { 0 };
        let num_combine_imm = (self.world_size - self.node_size) as u32 * num_shards;

        // Dispatch stage.
        {
            let dispatch_range = range_start!("dispatch");

            if !route.dispatch_ranges.is_empty() {
                self.transfer_engine
                    .submit_transfer_atomic(
                        TransferRequest::Scatter(ScatterTransferRequest {
                            src_mr: self.send_buffer_mr,
                            dst_handle: None,
                            dsts: route.dispatch_ranges,
                            imm_data: Some(self.dispatch_imm),
                            domain: GroupTransferRouting::AllDomainsShardBytes,
                        }),
                        self.tx_counter.clone(),
                        self.err_counter.clone(),
                    )
                    .unwrap();
            }

            // Wait for the dispatch counter to settle and signal the kernel.
            self.dispatch_counter.wait(route.num_recv_tx);

            // Wait for the dispatch kernel to complete. It is triggered once
            // the immediate counter reaches zero by setting the dispatch recv flag.
            self.dispatch_recv_done.wait();

            range_end!(dispatch_range);
        }
        self.barrier(
            self.dispatch_barrier_write_op.clone(),
            &self.dispatch_barrier_counter,
            num_dispatch_tx,
        );

        // Combine stage.
        {
            self.combine_send_done.wait();
            if !self.is_running() {
                return;
            }

            // Sent the tokens.
            let combine_range = range_start!("combine");

            if !route.combine_ranges.is_empty() {
                self.transfer_engine
                    .submit_transfer_atomic(
                        TransferRequest::Scatter(ScatterTransferRequest {
                            src_mr: self.send_buffer_mr,
                            dst_handle: None,
                            dsts: route.combine_ranges,
                            imm_data: Some(self.combine_imm),
                            domain: GroupTransferRouting::AllDomainsShardBytes,
                        }),
                        self.tx_counter.clone(),
                        self.err_counter.clone(),
                    )
                    .unwrap();
            }

            // Wait for all remote writes to complete.
            self.combine_counter.wait(num_combine_imm);
            self.combine_recv_flag.set(true);

            // Let the recv phase output the tokens and proceed forward.
            self.combine_recv_done.wait();

            range_end!(combine_range);
        }

        self.barrier(
            self.combine_barrier_write_op.clone(),
            &self.combine_barrier_counter,
            num_combine_tx,
        );
    }

    fn dispatch_initial_routes(&self) -> usize {
        let mut dsts: Vec<ScatterTarget> = Vec::with_capacity(self.world_size - 1);

        let experts_per_rank = self.num_experts.div_ceil(self.world_size);
        let token_dim = self.get_dispatch_token_dim();
        let rank_node = self.rank / self.node_size;

        let mut tokens_per_rank = vec![0; self.world_size];
        let mut rank_offset = vec![0; self.world_size];
        let mut offset = 0;
        for peer_rank in 0..self.world_size {
            let first_expert = peer_rank * experts_per_rank;
            let last_expert =
                ((peer_rank + 1) * experts_per_rank).min(self.num_experts);

            let mut tokens_on_rank = 0;
            for expert in first_expert..last_expert {
                let num_routed = self.get_num_routed(self.dp_group, expert);
                tokens_on_rank += num_routed as usize;
            }
            tokens_per_rank[peer_rank] = tokens_on_rank;
            rank_offset[peer_rank] = offset;
            offset += tokens_on_rank
        }

        for peer_node in 1..(self.world_size / self.node_size) {
            for index in 0..self.node_size {
                let peer_rank = ((rank_node + peer_node) * self.node_size + index)
                    % self.world_size;
                let num_tokens =
                    tokens_per_rank[peer_rank].min(self.max_private_tokens);

                let dst_mr = self.rank_handles[peer_rank].recv_buffer_desc.clone();
                let src_offset = (token_dim * rank_offset[peer_rank]) as u64;
                let dst_offset =
                    (token_dim * (self.dp_group * self.max_private_tokens)) as u64;

                dsts.push(ScatterTarget {
                    length: (token_dim * num_tokens) as u64,
                    src_offset,
                    dst_offset,
                    dst_mr,
                });
            }
        }

        if dsts.is_empty() {
            return 0;
        }

        let num_ranges = dsts.len();
        self.transfer_engine
            .submit_transfer_atomic(
                TransferRequest::Scatter(ScatterTransferRequest {
                    src_mr: self.send_buffer_mr,
                    dst_handle: None,
                    dsts: Arc::new(dsts),
                    imm_data: Some(self.dispatch_imm),
                    domain: GroupTransferRouting::AllDomainsShardPeers,
                }),
                self.tx_counter.clone(),
                self.err_counter.clone(),
            )
            .unwrap();

        num_ranges
    }

    #[allow(clippy::needless_range_loop)]
    fn process_routing_info(&self) -> RoutingInfo {
        // Determine counts on the current rank.
        let process_routing_info_range = range_start!("process_routing_info");

        let num_dp_groups = self.world_size / self.dp_size;
        let experts_per_rank = self.num_experts.div_ceil(self.world_size);
        let first_local_expert = self.rank * experts_per_rank;
        let last_local_expert =
            (first_local_expert + experts_per_rank).min(self.num_experts);
        let num_local_experts = last_local_expert - first_local_expert;
        let rank_node = self.rank / self.node_size;
        let groups_per_node = self.node_size / self.dp_size;
        let num_nodes = self.world_size / self.node_size;

        // On the sender side, find the start offset of each expert.
        let nets_per_gpu = self.transfer_engine.nets_per_gpu().get() as u32;
        let mut tokens_from_group = vec![0; num_dp_groups];
        let mut src_group_offset = vec![0; num_dp_groups];
        let mut dst_group_offset = vec![0; num_dp_groups];
        let mut tokens_per_expert = vec![0; experts_per_rank];
        let mut num_recv_tokens = 0;
        let mut num_recv_tx = 0;
        let mut source_expert_offset = vec![0; self.num_experts];
        let mut tokens_to_rank: Vec<u32> = vec![0; self.world_size];
        let mut dispatch_src_offset = vec![0; self.world_size];
        {
            let mut rank_offset = 0;
            for dp_group in 0..num_dp_groups {
                let group_node = dp_group / groups_per_node;
                let mut num_tokens = 0usize;
                for i in 0..self.dp_size {
                    let rank = dp_group * self.dp_size + i;
                    dispatch_src_offset[rank] = rank_offset;

                    let first_expert = rank * experts_per_rank;
                    let last_expert =
                        (first_expert + experts_per_rank).min(self.num_experts);
                    for expert in first_expert..last_expert {
                        let n = self.get_num_routed(self.dp_group, expert);
                        source_expert_offset[expert] = rank_offset;
                        tokens_to_rank[rank] += n;
                        rank_offset += n;
                    }
                }

                let mut offset = 0;
                for expert in 0..first_local_expert {
                    offset += self.get_num_routed(dp_group, expert);
                }
                dst_group_offset[dp_group] = offset;

                for expert in first_local_expert..last_local_expert {
                    let n = self.get_num_routed(dp_group, expert);
                    num_tokens += n as usize;
                    tokens_per_expert[expert - first_local_expert] += n;
                }
                tokens_from_group[dp_group] += num_tokens as u32;
                src_group_offset[dp_group] = num_recv_tokens as u32;
                num_recv_tokens += num_tokens;

                if dp_group != self.dp_group && group_node != rank_node {
                    // Private buffer scatter shards by Peers.
                    num_recv_tx += 1;
                    // Overflow token scatter shards by Bytes.
                    if num_tokens > self.max_private_tokens {
                        num_recv_tx += nets_per_gpu;
                    }
                }
            }
        }
        self.tokens_per_expert.copy(&tokens_per_expert);

        // Pad the per-expert counts.
        let mut padded_offset = Vec::with_capacity(num_local_experts);
        let mut base_expert_offset = 0;
        for count in &tokens_per_expert {
            let padded_count =
                (*count as usize).div_ceil(self.expert_padding) * self.expert_padding;
            padded_offset.push(base_expert_offset);
            base_expert_offset += padded_count as u32;
        }

        // Find individual shuffled indices on the local rank.
        let mut source_dispatch_offset = vec![0; num_recv_tokens];
        let mut combine_send_offset = vec![0; num_recv_tokens];
        let mut source_rank = vec![0; num_recv_tokens];
        let mut padded_index = vec![0u32; num_recv_tokens];

        let base_offset = (self.max_private_tokens * num_dp_groups) as u64;

        let mut last = 0;
        let mut src_dispatch_count = vec![0; num_dp_groups];
        let mut src_combine_count = vec![0; num_dp_groups];
        let mut expert_count = vec![0usize; num_local_experts];

        let mut route_group = |peer_group: usize| {
            let mut num_routed = 0;
            for expert in first_local_expert..last_local_expert {
                let private_offset = (self.max_private_tokens * peer_group) as u32;
                let routed = self.get_num_routed(peer_group, expert);
                num_routed += routed as usize;

                let local_expert = expert - first_local_expert;

                let src_offset = src_group_offset[peer_group];
                let dst_offset = dst_group_offset[peer_group];

                let peer_rank = peer_group * self.dp_size + self.dp_rank;
                for _ in 0..routed {
                    if peer_rank == self.rank {
                        let local_offset = source_expert_offset[expert];
                        source_expert_offset[expert] += 1;
                        source_dispatch_offset[last] = local_offset;
                        combine_send_offset[last] = local_offset;
                    } else {
                        let index_on_rank = src_dispatch_count[peer_group];
                        src_dispatch_count[peer_group] += 1;

                        // If the offset is within the private part, select it from there.
                        // Otherwise, pick it off from the remote.
                        if (index_on_rank as usize) < self.max_private_tokens {
                            source_dispatch_offset[last] =
                                private_offset + index_on_rank;
                        } else if peer_rank / self.node_size == rank_node {
                            source_dispatch_offset[last] =
                                (dst_offset + index_on_rank) | (1 << 31);
                        } else {
                            source_dispatch_offset[last] = src_offset
                                + index_on_rank
                                + (base_offset as u32 - self.max_private_tokens as u32);
                        }
                    }

                    if peer_rank != self.rank || self.dp_size > 1 {
                        let combine_index = src_combine_count[peer_group];
                        src_combine_count[peer_group] += 1;
                        if peer_rank / self.node_size == rank_node {
                            combine_send_offset[last] = dst_offset + combine_index;
                        } else {
                            combine_send_offset[last] = src_offset + combine_index;
                        }
                    }
                    source_rank[last] = peer_rank as u32;
                    padded_index[last] = padded_offset[local_expert]
                        + (expert_count[local_expert] as u32);
                    expert_count[local_expert] += 1;
                    last += 1;
                }
            }
            num_routed
        };

        // Order the tokens by the cost of copying them. First, ship out remote
        // tokens to start transferring them via fabric-lib. Next, handle NVLink
        // tokens to other ranks on the same node. Finally, process local tokens.
        let mut num_recv_efa_tokens = 0;
        for i in 1..num_nodes {
            for j in 0..groups_per_node {
                num_recv_efa_tokens +=
                    route_group((rank_node + i) % num_nodes * groups_per_node + j);
            }
        }
        for local_group in 1..groups_per_node {
            route_group(
                rank_node * groups_per_node
                    + (self.dp_group + local_group) % groups_per_node,
            );
        }
        route_group(self.dp_group);

        // Copy the buffers to the device.
        self.padded_index.copy(&padded_index);
        self.source_rank.copy(&source_rank);
        self.source_dispatch_offset.copy(&source_dispatch_offset);
        self.combine_send_offset.copy(&combine_send_offset);
        self.num_recv_tokens
            .copy(&[num_recv_tokens as u32, num_recv_efa_tokens as u32]);
        self.num_recv_tokens_flag.set(true);

        // Prepare the dispatch commands, beyond the private recv buffers.
        let mut dispatch_ranges = Vec::with_capacity(self.world_size - 1);
        for peer_node in 1..(self.world_size / self.node_size) {
            for index in (self.dp_rank..self.node_size).step_by(self.dp_size) {
                let peer_rank = ((rank_node + peer_node) * self.node_size + index)
                    % self.world_size;
                let dst_mr = self.rank_handles[peer_rank].recv_buffer_desc.clone();
                let num_tokens = tokens_to_rank[peer_rank] as usize;
                if num_tokens <= self.max_private_tokens {
                    continue;
                }

                let first_expert = peer_rank * experts_per_rank;
                let last_expert =
                    ((peer_rank + 1) * experts_per_rank).min(self.num_experts);

                let mut dst_offset = 0;
                for src_group in 0..self.dp_group {
                    for expert in first_expert..last_expert {
                        dst_offset += self.get_num_routed(src_group, expert) as u64;
                    }
                }

                let token_dim = self.get_dispatch_token_dim();
                let length = token_dim * (num_tokens - self.max_private_tokens);
                let src_offset = self.max_private_tokens as u64
                    + dispatch_src_offset[peer_rank] as u64;
                dispatch_ranges.push(ScatterTarget {
                    length: length as u64,
                    src_offset: src_offset * token_dim as u64,
                    dst_offset: (base_offset + dst_offset) * token_dim as u64,
                    dst_mr,
                });
            }
        }

        // Prepare the combine commands to remote nodes over EFA.
        let mut combine_ranges = Vec::with_capacity(self.world_size - 1);
        for peer_node in 1..(self.world_size / self.node_size) {
            for index in 0..(self.node_size / self.dp_size) {
                let peer_group =
                    ((rank_node + peer_node) * groups_per_node + index) % num_dp_groups;
                for index in 0..self.dp_size {
                    let token_dim = self.get_combine_token_dim();

                    let peer_rank = peer_group * self.dp_size + index;
                    let length = token_dim * tokens_from_group[peer_group] as usize;

                    let src_offset =
                        src_group_offset[peer_group] as u64 * token_dim as u64;
                    let dst_offset =
                        dst_group_offset[peer_group] as u64 * token_dim as u64;

                    let dst_mr = self.rank_handles[peer_rank].recv_buffer_desc.clone();
                    combine_ranges.push(ScatterTarget {
                        length: length as u64,
                        src_offset,
                        dst_offset,
                        dst_mr,
                    });
                }
            }
        }

        range_end!(process_routing_info_range);

        RoutingInfo {
            num_recv_tx,
            dispatch_ranges: Arc::new(dispatch_ranges),
            combine_ranges: Arc::new(combine_ranges),
        }
    }

    fn barrier(&self, request: TransferRequest, imm_counter: &ImmCounter, num_tx: u32) {
        let barrier = range_start!("barrier");

        self.transfer_engine
            .submit_transfer_atomic(
                request,
                self.tx_counter.clone(),
                self.err_counter.clone(),
            )
            .unwrap();

        // Wait for all payloads to be received.
        imm_counter.wait((self.world_size - 1) as u32);

        // Wait for the sends to complete.
        let old = self.tx_counter.fetch_sub(num_tx as i64, Ordering::Relaxed);
        if old < num_tx as i64 {
            while self.tx_counter.load(Ordering::Relaxed) < 0 {
                std::thread::yield_now();
            }
        }
        self.tx_ready.set(true);

        range_end!(barrier);
    }
}

fn extract_addrs(
    dst_mr: &MemoryRegionDescriptor,
) -> fabric_lib::api::SmallVec<DomainAddress> {
    dst_mr.addr_rkey_list.iter().map(|(addr, _)| addr.clone()).collect()
}
