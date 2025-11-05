use mockall::{automock, mock};

use std::{
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
    ptr::NonNull,
    sync::Arc,
};

use cuda_lib::Device;

use crate::{
    api::{DomainAddress, MemoryRegionDescriptor, MemoryRegionHandle, TransferRequest},
    error::{FabricLibError, Result},
};

pub trait RdmaEngine {
    fn main_address(&self) -> DomainAddress;

    fn nets_per_gpu(&self) -> NonZeroU8;

    fn register_memory_local(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<MemoryRegionHandle>;

    fn register_memory_allow_remote(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)>;

    fn unregister_memory(&self, ptr: NonNull<c_void>) -> Result<()>;
}

#[derive(Clone, Copy)]
pub struct SendBuffer {
    pub(crate) ptr: NonNull<c_void>,
    pub(crate) len: usize,
    pub(crate) mr_handle: MemoryRegionHandle,
}

unsafe impl Send for SendBuffer {}
unsafe impl Sync for SendBuffer {}

impl SendBuffer {
    pub fn new(
        ptr: NonNull<c_void>,
        len: usize,
        mr_handle: MemoryRegionHandle,
    ) -> Self {
        Self { ptr, len, mr_handle }
    }
}
pub type CallbackResult = std::result::Result<(), String>;

pub type SendCallback = Box<dyn FnOnce(Result<()>) -> CallbackResult + Send + Sync>;

pub type RecvCallback = Box<dyn Fn(usize) -> CallbackResult + Send + Sync>;

pub type ErrorCallback =
    Box<dyn FnOnce(FabricLibError) -> CallbackResult + Send + Sync>;

pub type BouncingRecvCallback = Arc<Box<dyn Fn(&[u8]) -> CallbackResult + Send + Sync>>;

pub type BouncingErrorCallback =
    Arc<Box<dyn Fn(FabricLibError) -> CallbackResult + Send + Sync>>;

#[automock]
pub trait SendRecvEngine {
    fn submit_send(
        &self,
        addr: DomainAddress,
        buffer: SendBuffer,
        callback: SendCallback,
    ) -> Result<()>;

    fn submit_recv(
        &self,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        on_recv: RecvCallback,
        on_error: ErrorCallback,
    ) -> Result<()>;

    fn submit_bouncing_recvs(
        &self,
        len: usize,
        count: usize,
        on_recv: BouncingRecvCallback,
        on_error: BouncingErrorCallback,
    ) -> Result<()>;
}

#[automock]
pub trait AsyncTransferEngine {
    fn wait_for_imm_count(
        &self,
        imm: u32,
        expected_count: NonZeroU32,
    ) -> impl Future<Output = Result<()>> + Send + Sync;

    fn submit_send_async(
        &self,
        addr: DomainAddress,
        buffer: SendBuffer,
    ) -> impl Future<Output = Result<()>> + Send + Sync;

    fn submit_transfer_async(
        &self,
        request: TransferRequest,
    ) -> impl Future<Output = Result<()>> + Send + Sync;
}

mock! {
    pub TestTransferEngine {}

    impl RdmaEngine for TestTransferEngine {
        fn main_address(&self) -> DomainAddress;

        fn nets_per_gpu(&self) -> NonZeroU8;

        fn register_memory_local(
            &self,
            ptr: NonNull<c_void>,
            len: usize,
            device: Device,
        ) -> Result<MemoryRegionHandle>;

        fn register_memory_allow_remote(
            &self,
            ptr: NonNull<c_void>,
            len: usize,
            device: Device,
        ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)>;

        fn unregister_memory(&self, ptr: NonNull<c_void>) -> Result<()>;
    }

    impl SendRecvEngine for TestTransferEngine {
        fn submit_send(
            &self,
            addr: DomainAddress,
            buffer: SendBuffer,
            callback: SendCallback,
        ) -> Result<()>;

        fn submit_recv(
            &self,
            mr: MemoryRegionHandle,
            ptr: NonNull<c_void>,
            len: usize,
            on_recv: RecvCallback,
            on_error: ErrorCallback,
        ) -> Result<()>;

        fn submit_bouncing_recvs(
            &self,
            len: usize,
            count: usize,
            on_recv: BouncingRecvCallback,
            on_error: BouncingErrorCallback,
        ) -> Result<()>;
    }

    impl AsyncTransferEngine for TestTransferEngine {
        fn wait_for_imm_count(
            &self,
            imm: u32,
            expected_count: NonZeroU32,
        ) -> impl Future<Output = Result<()>> + Send + Sync;

        fn submit_send_async(
            &self,
            addr: DomainAddress,
            buffer: SendBuffer,
        ) -> impl Future<Output = Result<()>> + Send + Sync;

        fn submit_transfer_async(
            &self,
            request: TransferRequest,
        ) -> impl Future<Output = Result<()>> + Send + Sync;
    }
}
