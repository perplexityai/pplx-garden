//! Types used in public API

use std::{
    ffi::c_void,
    num::NonZeroU8,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    },
};

use bytes::Bytes;
use cuda_lib::gdr::GdrFlag;
use serde::{Deserialize, Serialize};

use crate::{
    error::FabricLibError,
    utils::hex::{fmt_hex, from_hex},
};

pub type SmallVec<T> = ::smallvec::SmallVec<[T; 4]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MemoryRegionHandle {
    pub ptr: NonNull<c_void>,
}

impl MemoryRegionHandle {
    pub fn new(ptr: NonNull<c_void>) -> Self {
        MemoryRegionHandle { ptr }
    }
}

unsafe impl Send for MemoryRegionHandle {}
unsafe impl Sync for MemoryRegionHandle {}

/// A remote key for a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct MemoryRegionRemoteKey(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegionDescriptor {
    pub ptr: u64,
    pub addr_rkey_list: SmallVec<(DomainAddress, MemoryRegionRemoteKey)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(pub u64);

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DomainAddress(pub Bytes);

impl std::fmt::Debug for DomainAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_hex(f, &self.0)
    }
}

impl std::fmt::Display for DomainAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_hex(f, &self.0)
    }
}

impl std::str::FromStr for DomainAddress {
    type Err = FabricLibError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if !s.len().is_multiple_of(2) || s.is_empty() {
            return Err(FabricLibError::Custom("Invalid address length"));
        }
        Ok(Self(from_hex(s).map_err(|_| FabricLibError::Custom("Invalid address"))?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UvmWatcherId(pub(crate) NonNull<c_void>);
unsafe impl Send for UvmWatcherId {}
unsafe impl Sync for UvmWatcherId {}
impl UvmWatcherId {
    pub fn as_non_null(&self) -> NonNull<c_void> {
        self.0
    }

    pub fn as_u64(&self) -> u64 {
        self.0.as_ptr() as u64
    }
}

/// Determines how to shard the transfer across domains.
/// Relevant if NICs per GPU is greater than 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DomainGroupRouting {
    /// Shard the transfer across `num_shards` domains.
    /// Domains are selected in a round-robin manner.
    RoundRobinSharded { num_shards: NonZeroU8 },
    /// Send the transfer via a specific domain.
    Pinned { domain_idx: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupTransferRouting {
    /// Use all domains. Each domain handles a subset of peers.
    AllDomainsShardPeers,
    /// Use all domains. Each domain handles all peers but a subset of bytes.
    AllDomainsShardBytes,
    /// Use a single domain of the given index.
    Single { domain_idx: u8 },
}

#[derive(Clone, Debug)]
pub struct ImmTransferRequest {
    pub imm_data: u32,
    pub dst_mr: MemoryRegionDescriptor,
    pub domain: DomainGroupRouting,
}

#[derive(Clone, Debug)]
pub struct BarrierTransferRequest {
    pub imm_data: u32,
    pub dst_mrs: Vec<MemoryRegionDescriptor>,
    pub domain: DomainGroupRouting,
}

#[derive(Clone, Debug)]
pub struct SingleTransferRequest {
    pub src_mr: MemoryRegionHandle,
    pub src_offset: u64,
    pub length: u64,
    pub imm_data: Option<u32>,
    pub dst_mr: MemoryRegionDescriptor,
    pub dst_offset: u64,
    pub domain: DomainGroupRouting,
}

#[derive(Clone, Debug)]
pub struct PagedTransferRequest {
    pub length: u64,
    pub src_mr: MemoryRegionHandle,
    pub src_page_indices: Arc<Vec<u32>>,
    pub src_stride: u64,
    pub src_offset: u64,
    pub dst_mr: MemoryRegionDescriptor,
    pub dst_page_indices: Arc<Vec<u32>>,
    pub dst_stride: u64,
    pub dst_offset: u64,
    pub imm_data: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerGroupHandle(pub(crate) u32);

#[derive(Clone, Debug)]
pub struct ScatterTarget {
    pub dst_mr: MemoryRegionDescriptor,
    pub length: u64,
    pub src_offset: u64,
    pub dst_offset: u64,
}

#[derive(Clone, Debug)]
pub struct ScatterTransferRequest {
    pub src_mr: MemoryRegionHandle,
    /// You can get lower overhead by providing a PeerGroupHandle.
    /// When PeerGroupHandle is provided, the order of ScatterTarget needs to
    /// match the order of the peer group.
    pub dst_handle: Option<PeerGroupHandle>,
    pub dsts: Arc<Vec<ScatterTarget>>,
    pub imm_data: Option<u32>,
    pub domain: GroupTransferRouting,
}

/// Use static dispatch for performance.
#[derive(Clone)]
pub enum TransferRequest {
    Imm(ImmTransferRequest),
    Single(SingleTransferRequest),
    Paged(PagedTransferRequest),
    Scatter(ScatterTransferRequest),
    Barrier(BarrierTransferRequest),
}

#[derive(Debug)]
pub enum TransferCompletionEntry {
    Recv { transfer_id: TransferId, data_len: usize },
    Send(TransferId),
    Transfer(TransferId),
    ImmData(u32),
    ImmCountReached(u32),
    UvmWatch { id: UvmWatcherId, old: u64, new: u64 },
    Error(TransferId, FabricLibError),
}

/// A free-range immediate counter exposed to users.
#[derive(Clone)]
pub struct ImmCounter {
    counter: Arc<AtomicI64>,
}

impl ImmCounter {
    pub fn new(counter: Arc<AtomicI64>) -> Self {
        Self { counter }
    }

    pub fn wait(&self, target: u32) {
        let old = self.counter.fetch_sub(target as i64, Ordering::Relaxed);
        if old >= target as i64 {
            return;
        }
        while self.counter.load(Ordering::Relaxed) < 0 {
            std::hint::spin_loop();
        }
    }
}

/// An immediate counter that sets a flag via GdrCopy.
#[derive(Clone)]
pub struct GdrCounter {
    counter: Arc<AtomicI64>,
    flag: Arc<GdrFlag>,
}

impl GdrCounter {
    pub fn new(counter: Arc<AtomicI64>, flag: Arc<GdrFlag>) -> Self {
        Self { counter, flag }
    }

    pub fn wait(&self, target: u32) {
        let old = self.counter.fetch_sub(target as i64, Ordering::Relaxed);
        if old >= target as i64 {
            self.flag.set(true);
        }
    }
}

/// Transfer counter exposing a pollable interface to transfer completion.
pub struct TransferCounter {
    counter: Arc<AtomicI64>,
    err_counter: Arc<AtomicI64>,
}

impl TransferCounter {
    pub fn new(counter: Arc<AtomicI64>, err_counter: Arc<AtomicI64>) -> Self {
        Self { counter, err_counter }
    }

    pub(crate) fn error(&self) {
        self.err_counter.fetch_add(1, Ordering::Release);
        self.counter.fetch_add(1, Ordering::Release);
    }

    pub(crate) fn done(&self) {
        self.counter.fetch_add(1, Ordering::Release);
    }
}
