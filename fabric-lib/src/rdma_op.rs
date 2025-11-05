use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use crate::{
    api::{DomainAddress, MemoryRegionRemoteKey, ScatterTarget},
    mr::MemoryRegionLocalDescriptor,
};

pub struct SingleWriteOp {
    pub src_ptr: NonNull<c_void>,
    pub src_desc: MemoryRegionLocalDescriptor,
    pub src_offset: u64,
    pub length: u64,
    pub imm_data: Option<u32>,
    pub dst_ptr: u64,
    pub dst_rkey: MemoryRegionRemoteKey,
    pub dst_offset: u64,
}

pub struct ImmWriteOp {
    pub imm_data: u32,
    pub dst_ptr: u64,
    pub dst_rkey: MemoryRegionRemoteKey,
}

pub struct PagedWriteOp {
    pub src_page_indices: Arc<Vec<u32>>,
    pub dst_page_indices: Arc<Vec<u32>>,
    pub page_indices_beg: usize,
    pub page_indices_end: usize,
    pub length: u64,
    pub src_ptr: NonNull<c_void>,
    pub src_desc: MemoryRegionLocalDescriptor,
    pub src_stride: u64,
    pub src_offset: u64,
    pub dst_ptr: u64,
    pub dst_rkey: MemoryRegionRemoteKey,
    pub dst_stride: u64,
    pub dst_offset: u64,
    pub imm_data: Option<u32>,
}

pub enum WriteOp {
    Single(SingleWriteOp),
    Imm(ImmWriteOp),
    Paged(PagedWriteOp),
}

pub struct ScatterGroupWriteOp {
    pub domain_idx: usize,
    pub src_ptr: NonNull<c_void>,
    pub src_desc: MemoryRegionLocalDescriptor,
    pub imm_data: Option<u32>,
    pub dsts: Arc<Vec<ScatterTarget>>,
    pub dst_beg: usize,
    pub dst_end: usize,
    pub byte_shards: u32,
    pub byte_shard_idx: u32,
}

pub enum GroupWriteOp {
    Scatter(ScatterGroupWriteOp),
}

impl GroupWriteOp {
    pub fn num_targets(&self) -> usize {
        match self {
            GroupWriteOp::Scatter(op) => op.dsts.len(),
        }
    }

    pub fn peer_addr_iter(&self) -> impl Iterator<Item = &DomainAddress> {
        match self {
            GroupWriteOp::Scatter(op) => {
                op.dsts.iter().map(|dst| &dst.dst_mr.addr_rkey_list[op.domain_idx].0)
            }
        }
    }
}

pub struct SendOp {
    pub ptr: NonNull<c_void>,
    pub len: usize,
    pub desc: MemoryRegionLocalDescriptor,
}

pub struct RecvOp {
    pub ptr: NonNull<c_void>,
    pub len: usize,
    pub desc: MemoryRegionLocalDescriptor,
}
