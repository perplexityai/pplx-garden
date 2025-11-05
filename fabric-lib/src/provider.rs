use std::{borrow::Cow, ffi::c_void, ptr::NonNull, sync::Arc};

use crate::{
    api::{DomainAddress, MemoryRegionRemoteKey, PeerGroupHandle, TransferId},
    error::{FabricLibError, Result},
    imm_count::ImmCountMap,
    mr::{MemoryRegion, MemoryRegionLocalDescriptor},
    rdma_op::{GroupWriteOp, RecvOp, SendOp, WriteOp},
};

pub trait RdmaDomainInfo {
    fn name(&self) -> Cow<'_, str>;
    fn link_speed(&self) -> u64;
}

pub trait RdmaDomain {
    type Info: RdmaDomainInfo;

    fn open(info: Self::Info, imm_count_map: Arc<ImmCountMap>) -> Result<Self>
    where
        Self: Sized;

    fn link_speed(&self) -> u64;
    fn addr(&self) -> DomainAddress;

    fn register_mr_local(&mut self, region: &MemoryRegion) -> Result<()>;
    fn register_mr_allow_remote(
        &mut self,
        region: &MemoryRegion,
    ) -> Result<MemoryRegionRemoteKey>;
    fn unregister_mr(&mut self, ptr: NonNull<c_void>);
    fn get_mem_desc(&self, ptr: NonNull<c_void>)
    -> Result<MemoryRegionLocalDescriptor>;

    fn submit_recv(&mut self, transfer_id: TransferId, op: RecvOp);
    fn submit_send(
        &mut self,
        transfer_id: TransferId,
        dest_addr: DomainAddress,
        op: SendOp,
    );
    fn submit_write(
        &mut self,
        transfer_id: TransferId,
        dest_addr: DomainAddress,
        op: WriteOp,
    );

    fn add_peer_group(
        &mut self,
        handle: PeerGroupHandle,
        addrs: Vec<DomainAddress>,
    ) -> Result<()>;
    fn submit_group_write(
        &mut self,
        transfer_id: TransferId,
        handle: Option<PeerGroupHandle>,
        op: GroupWriteOp,
    );

    fn poll_progress(&mut self);
    fn get_completion(&mut self) -> Option<DomainCompletionEntry>;
}

pub enum DomainCompletionEntry {
    Recv { transfer_id: TransferId, data_len: usize },
    Send(TransferId),
    Transfer(TransferId),
    ImmData(u32),
    ImmCountReached(u32),
    Error(TransferId, FabricLibError),
}
