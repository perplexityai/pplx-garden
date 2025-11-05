use std::{cmp::min, collections::HashMap, ffi::c_void, ptr::NonNull, sync::Arc};

use crate::{
    api::{
        DomainAddress, DomainGroupRouting, GroupTransferRouting,
        MemoryRegionDescriptor, MemoryRegionHandle, PagedTransferRequest,
        PeerGroupHandle, ScatterTransferRequest, SingleTransferRequest, SmallVec,
        TransferCounter, TransferId, TransferRequest,
    },
    error::{FabricLibError, Result},
    mr::MemoryRegion,
    provider::{DomainCompletionEntry, RdmaDomain},
    rdma_op::{
        GroupWriteOp, ImmWriteOp, PagedWriteOp, RecvOp, ScatterGroupWriteOp, SendOp,
        SingleWriteOp, WriteOp,
    },
};

pub struct DomainGroup<D: RdmaDomain, const N: usize> {
    domains: [D; N],
    write_ops: HashMap<TransferId, WriteOpContext>,
    rr_next: usize,
    next_peer_group_handle: u32,
    peer_groups: HashMap<PeerGroupHandle, PeerGroup>,
}

struct WriteOpContext {
    num_used_domains: usize,
    cnt_domain_completion: usize,
    tx_counter: Option<TransferCounter>,
}

struct PeerGroup {
    addrs: Vec<SmallVec<DomainAddress>>,
}

impl<D: RdmaDomain, const N: usize> DomainGroup<D, N> {
    pub fn new(domains: [D; N]) -> Self {
        Self {
            domains,
            write_ops: HashMap::new(),
            rr_next: 0,
            next_peer_group_handle: 0,
            peer_groups: HashMap::new(),
        }
    }

    pub fn aggregate_link_speed(&self) -> u64 {
        self.domains.iter().map(|d| d.link_speed()).sum()
    }

    pub fn register_mr_allow_remote(
        &mut self,
        region: &MemoryRegion,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)> {
        let mut addr_rkey_list = SmallVec::new();
        for domain in self.domains.iter_mut() {
            let rkey = domain.register_mr_allow_remote(region)?;
            addr_rkey_list.push((domain.addr(), rkey));
        }
        let ptr = region.ptr();
        let local = MemoryRegionHandle::new(ptr);
        let remote =
            MemoryRegionDescriptor { ptr: ptr.as_ptr() as u64, addr_rkey_list };
        Ok((local, remote))
    }

    pub fn register_mr_local(
        &mut self,
        region: &MemoryRegion,
    ) -> Result<MemoryRegionHandle> {
        for domain in self.domains.iter_mut() {
            domain.register_mr_local(region)?;
        }
        Ok(MemoryRegionHandle::new(region.ptr()))
    }

    pub fn unregister_mr(&mut self, ptr: NonNull<c_void>) {
        for domain in self.domains.iter_mut() {
            domain.unregister_mr(ptr);
        }
    }

    pub fn add_peer_group(
        &mut self,
        addrs: Vec<SmallVec<DomainAddress>>,
    ) -> Result<PeerGroupHandle> {
        for (handle, group) in self.peer_groups.iter() {
            if group.addrs == addrs {
                return Ok(*handle);
            }
        }
        let handle = PeerGroupHandle(self.next_peer_group_handle);
        self.next_peer_group_handle += 1;
        self.peer_groups.insert(handle, PeerGroup { addrs: addrs.clone() });

        for (i, domain) in self.domains.iter_mut().enumerate() {
            let domain_addrs = addrs.iter().map(|addr| addr[i].clone()).collect();
            domain.add_peer_group(handle, domain_addrs)?;
        }

        Ok(handle)
    }

    pub fn submit_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: TransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        match request {
            TransferRequest::Imm(request) => self.submit_imm_transfer_request(
                transfer_id,
                request.imm_data,
                &[request.dst_mr],
                request.domain,
                tx_counter,
            ),
            TransferRequest::Barrier(request) => self.submit_imm_transfer_request(
                transfer_id,
                request.imm_data,
                &request.dst_mrs,
                request.domain,
                tx_counter,
            ),
            TransferRequest::Single(request) => {
                self.submit_single_transfer_request(transfer_id, request, tx_counter)
            }
            TransferRequest::Paged(request) => {
                self.submit_paged_transfer_request(transfer_id, request, tx_counter)
            }
            TransferRequest::Scatter(request) => {
                self.submit_scatter_transfer_request(transfer_id, request, tx_counter)
            }
        }
    }

    pub fn submit_imm_transfer_request(
        &mut self,
        transfer_id: TransferId,
        imm_data: u32,
        dst_mrs: &[MemoryRegionDescriptor],
        domain: DomainGroupRouting,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Sanity check the input descriptors.
        for dst_mr in dst_mrs {
            if dst_mr.addr_rkey_list.len() != self.domains.len() {
                return Err(FabricLibError::Custom(
                    "Number of target addresses must match the number of domains",
                ));
            }
        }

        // Bookkeeping.
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: dst_mrs.len(),
                cnt_domain_completion: 0,
                tx_counter,
            },
        );

        // Determine the number of imms to send via each domain.
        let num_domains = self.domains.len();
        let (first_domain, imm_per_domain) = match domain {
            DomainGroupRouting::RoundRobinSharded { num_shards } => {
                if num_shards.get() != 1 {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::RoundRobinSharded should have num_shards = 1 for BarrierTransferRequest",
                    ));
                }
                let first_domain = self.rr_next;
                self.rr_next =
                    (self.rr_next + num_domains.min(dst_mrs.len())) % num_domains;
                (first_domain, dst_mrs.len().div_ceil(num_domains))
            }
            DomainGroupRouting::Pinned { domain_idx } => {
                if domain_idx as usize >= self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::Pinned.domain_idx is out of bounds",
                    ));
                }
                (domain_idx as usize, dst_mrs.len())
            }
        };

        // Chunk by domain and submit the writes.
        for (i, dst_mrs) in dst_mrs.chunks(imm_per_domain).enumerate() {
            let domain = &mut self.domains[(first_domain + i) % num_domains];
            for dst_mr in dst_mrs {
                let (dst_addr, dst_rkey) = &dst_mr.addr_rkey_list[i];

                // Construct rdma op
                let rdma_op = WriteOp::Imm(ImmWriteOp {
                    imm_data,
                    dst_ptr: dst_mr.ptr,
                    dst_rkey: *dst_rkey,
                });

                // Submit the transfer request to the domain
                domain.submit_write(transfer_id, dst_addr.clone(), rdma_op);
            }
        }
        Ok(())
    }

    pub fn submit_single_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: SingleTransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Validate
        if request.dst_mr.addr_rkey_list.len() != self.domains.len() {
            return Err(FabricLibError::Custom(
                "Number of target addresses must match the number of domains",
            ));
        }

        // Statically shard the bytes across domains
        let num_shards = match request.domain {
            DomainGroupRouting::RoundRobinSharded { num_shards } => {
                if num_shards.get() as usize > self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::RoundRobinSharded.num_shards is greater than the number of domains",
                    ));
                }
                num_shards.get() as usize
            }
            DomainGroupRouting::Pinned { domain_idx } => {
                if domain_idx as usize >= self.domains.len() {
                    return Err(FabricLibError::Custom(
                        "DomainGroupRouting::Pinned.domain_idx is out of bounds",
                    ));
                }
                1usize
            }
        };
        let ranges = shard_single_transfer(request.length as usize, num_shards);

        // Construct rdma ops
        let mut rdma_ops = SmallVec::with_capacity(ranges.len());
        for (offset, len) in ranges {
            // Sharding
            let i = match request.domain {
                DomainGroupRouting::RoundRobinSharded { .. } => {
                    let ret = self.rr_next;
                    self.rr_next = (self.rr_next + 1) % self.domains.len();
                    ret
                }
                DomainGroupRouting::Pinned { domain_idx } => domain_idx as usize,
            };

            // Address lookup
            let domain = &mut self.domains[i];
            let (dst_addr, dst_rkey) = &request.dst_mr.addr_rkey_list[i];

            // Get the source memory region descriptor
            let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;

            // Build the rdma op iter
            let op = WriteOp::Single(SingleWriteOp {
                src_ptr: request.src_mr.ptr,
                src_desc,
                src_offset: request.src_offset + offset as u64,
                length: len as u64,
                imm_data: request.imm_data,
                dst_ptr: request.dst_mr.ptr,
                dst_rkey: *dst_rkey,
                dst_offset: request.dst_offset + offset as u64,
            });
            rdma_ops.push((op, dst_addr));
        }

        // Bookkeeping
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: rdma_ops.len(),
                cnt_domain_completion: 0,
                tx_counter,
            },
        );

        // Submit the transfer request to each domain
        for (domain, (rdma_op, dst_addr)) in self.domains.iter_mut().zip(rdma_ops) {
            domain.submit_write(transfer_id, dst_addr.clone(), rdma_op);
        }
        Ok(())
    }

    pub fn submit_paged_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: PagedTransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Validate
        if request.dst_mr.addr_rkey_list.len() != self.domains.len() {
            return Err(FabricLibError::Custom(
                "Number of target addresses must match the number of domains",
            ));
        }
        if request.src_page_indices.len() != request.dst_page_indices.len() {
            return Err(FabricLibError::Custom(
                "Length of source and destination page indices must match",
            ));
        }

        // Statically shard the page indices across domains
        let page_range =
            divide_evenly(request.src_page_indices.len(), self.domains.len());

        // Bookkeeping
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: page_range.len(),
                cnt_domain_completion: 0,
                tx_counter,
            },
        );

        // Construct rdma op iter
        for (beg, end) in page_range {
            // Round-robin
            let i = self.rr_next;
            self.rr_next = (self.rr_next + 1) % self.domains.len();

            // Address lookup
            let domain = &mut self.domains[i];
            let (dst_addr, dst_rkey) = &request.dst_mr.addr_rkey_list[i];

            // Get the source memory region descriptor
            let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;

            // Build the rdma op
            let op = WriteOp::Paged(PagedWriteOp {
                src_page_indices: Arc::clone(&request.src_page_indices),
                dst_page_indices: Arc::clone(&request.dst_page_indices),
                page_indices_beg: beg,
                page_indices_end: end,
                length: request.length,
                src_ptr: request.src_mr.ptr,
                src_desc,
                src_stride: request.src_stride,
                src_offset: request.src_offset,
                dst_ptr: request.dst_mr.ptr,
                dst_rkey: *dst_rkey,
                dst_stride: request.dst_stride,
                dst_offset: request.dst_offset,
                imm_data: request.imm_data,
            });
            domain.submit_write(transfer_id, dst_addr.clone(), op);
        }
        Ok(())
    }

    pub fn submit_scatter_transfer_request(
        &mut self,
        transfer_id: TransferId,
        request: ScatterTransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        // Validate
        if request.dsts.is_empty() {
            return Err(FabricLibError::Custom("Empty scatter targets"));
        }
        if let Some(dst_handle) = &request.dst_handle {
            let group = self
                .peer_groups
                .get(dst_handle)
                .ok_or(FabricLibError::Custom("PeerGroupHandle not found"))?;
            if request.dsts.len() != group.addrs.len() {
                return Err(FabricLibError::Custom(
                    "Number of scatter targets must match the number of peer group addresses",
                ));
            }
        }

        // Statically shard the transfer across domains.
        let mut domain_indices = SmallVec::new();
        let mut rdma_ops = SmallVec::new();
        match request.domain {
            GroupTransferRouting::AllDomainsShardPeers => {
                let dst_range = divide_evenly(request.dsts.len(), self.domains.len());
                for ((domain_idx, domain), (beg, end)) in
                    self.domains.iter_mut().enumerate().zip(dst_range.into_iter())
                {
                    let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;
                    let op = GroupWriteOp::Scatter(ScatterGroupWriteOp {
                        domain_idx,
                        src_ptr: request.src_mr.ptr,
                        src_desc,
                        imm_data: request.imm_data,
                        dsts: Arc::clone(&request.dsts),
                        dst_beg: beg,
                        dst_end: end,
                        byte_shards: 1,
                        byte_shard_idx: 0,
                    });
                    rdma_ops.push(op);
                    domain_indices.push(domain_idx);
                }
            }
            GroupTransferRouting::AllDomainsShardBytes => {
                for (domain_idx, domain) in self.domains.iter_mut().enumerate() {
                    let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;
                    let op = GroupWriteOp::Scatter(ScatterGroupWriteOp {
                        domain_idx,
                        src_ptr: request.src_mr.ptr,
                        src_desc,
                        imm_data: request.imm_data,
                        dsts: Arc::clone(&request.dsts),
                        dst_beg: 0,
                        dst_end: request.dsts.len(),
                        byte_shards: N as u32,
                        byte_shard_idx: domain_idx as u32,
                    });
                    rdma_ops.push(op);
                    domain_indices.push(domain_idx);
                }
            }
            GroupTransferRouting::Single { domain_idx } => {
                let domain = &mut self.domains[domain_idx as usize];
                let src_desc = domain.get_mem_desc(request.src_mr.ptr)?;
                let op = GroupWriteOp::Scatter(ScatterGroupWriteOp {
                    domain_idx: domain_idx as usize,
                    src_ptr: request.src_mr.ptr,
                    src_desc,
                    imm_data: request.imm_data,
                    dsts: Arc::clone(&request.dsts),
                    dst_beg: 0,
                    dst_end: request.dsts.len(),
                    byte_shards: 1,
                    byte_shard_idx: 0,
                });
                rdma_ops.push(op);
                domain_indices.push(domain_idx as usize);
            }
        }

        // Bookkeeping
        self.write_ops.insert(
            transfer_id,
            WriteOpContext {
                num_used_domains: rdma_ops.len(),
                cnt_domain_completion: 0,
                tx_counter,
            },
        );

        // Submit the transfer request to each domain
        for (domain_idx, rdma_op) in domain_indices.into_iter().zip(rdma_ops) {
            let domain = &mut self.domains[domain_idx];
            domain.submit_group_write(transfer_id, request.dst_handle, rdma_op);
        }

        Ok(())
    }

    pub fn submit_send(
        &mut self,
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        addr: DomainAddress,
    ) -> Result<()> {
        let domain = &mut self.domains[0];
        let desc = domain.get_mem_desc(mr.ptr)?;
        domain.submit_send(transfer_id, addr, SendOp { ptr, len, desc });
        Ok(())
    }

    pub fn submit_recv(
        &mut self,
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<()> {
        let domain = &mut self.domains[0];
        let desc = domain.get_mem_desc(mr.ptr)?;
        domain.submit_recv(transfer_id, RecvOp { ptr, len, desc });
        Ok(())
    }

    /// Poll RDMA completion queue and send pending RDMA operations.
    pub fn poll_progress(&mut self) {
        for domain in self.domains.iter_mut() {
            domain.poll_progress();
        }
    }

    /// Return Error if any domain fails the transfer.
    /// Return Transfer if all domains have completed the transfer.
    /// Return Recv, Send, ImmData if any domain returns so.
    /// Return None otherwise.
    pub fn get_completion(&mut self) -> Option<DomainCompletionEntry> {
        for domain in self.domains.iter_mut() {
            if let Some(c) = domain.get_completion() {
                match c {
                    DomainCompletionEntry::Error(transfer_id, err) => {
                        if let Some(write_op) = self.write_ops.remove(&transfer_id)
                            && let Some(tx_counter) = write_op.tx_counter
                        {
                            tx_counter.error();
                        }
                        return Some(DomainCompletionEntry::Error(transfer_id, err));
                    }
                    DomainCompletionEntry::Transfer(transfer_id) => {
                        let Some(ctx) = self.write_ops.get_mut(&transfer_id) else {
                            continue; // Transfer not found. Ignore.
                        };
                        ctx.cnt_domain_completion += 1;
                        if ctx.num_used_domains == ctx.cnt_domain_completion {
                            let ctx = self.write_ops.remove(&transfer_id).unwrap();
                            if let Some(tx_counter) = ctx.tx_counter {
                                tx_counter.done();
                                return None;
                            } else {
                                return Some(DomainCompletionEntry::Transfer(
                                    transfer_id,
                                ));
                            }
                        }
                    }
                    DomainCompletionEntry::ImmData(imm_data) => {
                        return Some(DomainCompletionEntry::ImmData(imm_data));
                    }
                    DomainCompletionEntry::Recv { transfer_id, data_len } => {
                        return Some(DomainCompletionEntry::Recv {
                            transfer_id,
                            data_len,
                        });
                    }
                    DomainCompletionEntry::Send(transfer_id) => {
                        return Some(DomainCompletionEntry::Send(transfer_id));
                    }
                    DomainCompletionEntry::ImmCountReached(imm) => {
                        return Some(DomainCompletionEntry::ImmCountReached(imm));
                    }
                }
            }
        }
        None
    }
}

fn divide_evenly(n: usize, k: usize) -> SmallVec<(usize, usize)> {
    let mut result = SmallVec::new();
    let step = n.div_ceil(k);
    let mut remainder = n;
    let mut beg = 0;
    for _ in 0..k {
        let chunk = remainder.min(step);
        if chunk == 0 {
            break;
        }
        result.push((beg, beg + chunk));
        beg += chunk;
        remainder -= chunk;
    }
    debug_assert!(remainder == 0);
    result
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn shard_single_transfer(
    total_size: usize,
    num_shards: usize,
) -> SmallVec<(usize, usize)> {
    const MIN_SIZE: usize = 8192;
    let mut result = SmallVec::new();
    let base = round_up(total_size / num_shards, MIN_SIZE);
    let mut offset = 0;
    for _ in 0..num_shards {
        let len = min(base, total_size - offset);
        if len > 0 {
            result.push((offset, len));
            offset += len;
        } else {
            // For 0-length WRITE, it's possible that the offset is at the end of the MR.
            // To avoid out of bound access, we always use 0 offset for a 0-length WRITE.
            result.push((0, 0));
        }
    }
    result
}
