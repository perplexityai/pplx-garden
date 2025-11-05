use std::{
    ffi::c_void,
    mem::MaybeUninit,
    num::NonZeroU8,
    num::NonZeroU32,
    ptr::NonNull,
    sync::Arc,
    sync::atomic::{AtomicI64, AtomicU64, Ordering::SeqCst},
    thread::JoinHandle,
};

use cuda_lib::{Device, gdr::GdrFlag};
use dashmap::DashMap;
use parking_lot::Mutex;
use parking_lot::RwLock;
use tracing::{error, warn};

use crate::{
    BouncingErrorCallback, BouncingRecvCallback, ErrorCallback, FabricLibError,
    RdmaEngine, RecvCallback, SendBuffer, SendCallback, SendRecvEngine,
    api::{
        DomainAddress, GdrCounter, ImmCounter, MemoryRegionDescriptor,
        MemoryRegionHandle, PeerGroupHandle, SmallVec, TransferCompletionEntry,
        TransferCounter, TransferId, TransferRequest, UvmWatcherId,
    },
    error::Result,
    fabric_engine::FabricEngine,
    imm_count::ImmCount,
    worker::Worker,
};
#[cfg(feature = "tokio")]
use {
    crate::AsyncTransferEngine,
    tokio::sync::{mpsc, oneshot},
};

pub type CallbackResult = std::result::Result<(), String>;

pub struct TransferCallback {
    pub on_done: Box<dyn FnOnce() -> CallbackResult + Send + Sync>,
    pub on_error: ErrorCallback,
}

struct RecvContext {
    mr: MemoryRegionHandle,
    ptr: NonNull<c_void>,
    len: usize,
    on_recv: RecvCallback,
    on_error: ErrorCallback,
}
unsafe impl Send for RecvContext {}
unsafe impl Sync for RecvContext {}

pub type ImmCallbackFn = Box<dyn Fn(u32) -> CallbackResult + Send + Sync>;
pub type UvmWatcherCallback =
    Box<dyn Fn(u64, u64) -> std::result::Result<bool, String> + Send + Sync>;

pub type ImmCountCallback =
    Box<dyn Fn() -> std::result::Result<bool, String> + Send + Sync>;

enum ImmCountFn {
    /// The callback will be called once when the expected count is reached.
    #[allow(dead_code)]
    Once(Box<dyn FnOnce() + Send + Sync>),
    /// The callback will be called every time the expected count is reached.
    Repeated(Box<dyn Fn() -> std::result::Result<bool, String> + Send + Sync>),
}
struct Callbacks {
    imm: RwLock<Vec<ImmCallbackFn>>,
    recv_ops: DashMap<TransferId, RecvContext>,
    send_ops: DashMap<TransferId, SendCallback>,
    transfer_ops: DashMap<TransferId, TransferCallback>,
    imm_count: DashMap<u32, ImmCountFn>,
    watchers: DashMap<UvmWatcherId, UvmWatcherCallback>,
}

pub struct TransferEngine {
    next_transfer_id: AtomicU64,
    engine: Arc<FabricEngine>,
    callbacks: Arc<Callbacks>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl TransferEngine {
    pub fn new(workers: Vec<(u8, Worker)>) -> Result<Self> {
        let engine = Arc::new(FabricEngine::new(workers)?);

        let callbacks = Arc::new(Callbacks {
            imm: RwLock::new(Vec::new()),
            recv_ops: DashMap::new(),
            send_ops: DashMap::new(),
            transfer_ops: DashMap::new(),
            imm_count: DashMap::new(),
            watchers: DashMap::new(),
        });

        let thread = {
            let thread_engine = engine.clone();
            let thread_calbacks = callbacks.clone();
            std::thread::Builder::new()
                .name("tx_engine_callback".to_string())
                .spawn(move || callback_worker_thread(thread_engine, thread_calbacks))
                .map_err(|_| {
                    FabricLibError::Custom("failed to launch callback worker thread")
                })
        }?;

        Ok(TransferEngine {
            next_transfer_id: AtomicU64::new(0),
            engine,
            callbacks,
            thread: Mutex::new(Some(thread)),
        })
    }

    pub fn num_domains(&self) -> usize {
        self.engine.num_domains()
    }

    pub fn num_groups(&self) -> usize {
        self.engine.num_groups()
    }

    pub fn aggregated_link_speed(&self) -> u64 {
        self.engine.aggregated_link_speed()
    }

    pub fn add_imm_callback(&self, callback: ImmCallbackFn) {
        self.callbacks.imm.write().push(callback);
    }

    pub fn set_imm_count_expected(
        &self,
        imm: u32,
        expected_count: NonZeroU32,
        callback: ImmCountCallback,
    ) -> Option<ImmCount> {
        self.callbacks.imm_count.insert(imm, ImmCountFn::Repeated(callback));
        self.engine.set_imm_count_expected(imm, expected_count)
    }

    pub fn remove_imm_count(&self, imm: u32) -> Option<ImmCount> {
        self.callbacks.imm_count.remove(&imm);
        self.engine.remove_imm_count(imm)
    }

    pub fn get_imm_counter(&self, imm: u32) -> ImmCounter {
        self.engine.get_imm_counter(imm)
    }

    pub fn get_gdr_counter(&self, imm: u32, flag: Arc<GdrFlag>) -> GdrCounter {
        self.engine.get_gdr_counter(imm, flag)
    }

    pub fn add_peer_group(
        &self,
        addrs: Vec<SmallVec<DomainAddress>>,
        device: Device,
    ) -> Result<PeerGroupHandle> {
        self.engine.add_peer_group(addrs, device)
    }

    pub fn alloc_scalar_watcher(
        &self,
        callback: UvmWatcherCallback,
    ) -> Result<UvmWatcherId> {
        let watcher_id = self.engine.acquire_uvm_watcher()?;
        self.callbacks.watchers.insert(watcher_id, callback);
        Ok(watcher_id)
    }

    pub fn submit_transfer(
        &self,
        request: TransferRequest,
        callback: TransferCallback,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.callbacks.transfer_ops.insert(transfer_id, callback);
        self.engine.submit_transfer(transfer_id, request, None)
    }

    pub fn submit_transfer_atomic(
        &self,
        request: TransferRequest,
        tx_counter: Arc<AtomicI64>,
        err_counter: Arc<AtomicI64>,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.engine.submit_transfer(
            transfer_id,
            request,
            Some(TransferCounter::new(tx_counter, err_counter)),
        )
    }

    pub fn stop(&self) {
        self.engine.stop();
        let mut thread = self.thread.lock();
        if let Some(thread) = thread.take()
            && let Err(error) = thread.join()
        {
            error!(?error, "Failed to join the Transfer Engine callback thread.");
        }
    }

    fn assign_transfer_id(&self) -> TransferId {
        let transfer_id = self.next_transfer_id.fetch_add(1, SeqCst);
        TransferId(transfer_id)
    }
}

impl RdmaEngine for TransferEngine {
    fn main_address(&self) -> DomainAddress {
        self.engine.main_address()
    }

    fn nets_per_gpu(&self) -> NonZeroU8 {
        self.engine.nets_per_gpu()
    }

    fn register_memory_local(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<MemoryRegionHandle> {
        self.engine.register_memory_local(ptr, len, device)
    }

    fn register_memory_allow_remote(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)> {
        self.engine.register_memory_allow_remote(ptr, len, device)
    }

    fn unregister_memory(&self, ptr: NonNull<c_void>) -> Result<()> {
        self.engine.unregister_memory(ptr)
    }
}

impl SendRecvEngine for TransferEngine {
    fn submit_send(
        &self,
        addr: DomainAddress,
        buffer: SendBuffer,
        callback: SendCallback,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.callbacks.send_ops.insert(transfer_id, callback);
        self.engine.submit_send(
            transfer_id,
            addr,
            buffer.mr_handle,
            buffer.ptr,
            buffer.len,
        )?;
        Ok(())
    }

    fn submit_recv(
        &self,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        on_recv: RecvCallback,
        on_error: ErrorCallback,
    ) -> Result<()> {
        let transfer_id = self.assign_transfer_id();
        self.callbacks
            .recv_ops
            .insert(transfer_id, RecvContext { mr, ptr, len, on_recv, on_error });
        self.engine.submit_recv(transfer_id, mr, ptr, len)?;
        Ok(())
    }

    fn submit_bouncing_recvs(
        &self,
        len: usize,
        count: usize,
        on_recv: BouncingRecvCallback,
        on_error: BouncingErrorCallback,
    ) -> Result<()> {
        // Allocate buffers for the recv ops
        let storage: Arc<Vec<MaybeUninit<u8>>> =
            Arc::new(Vec::with_capacity(len * count));
        let buf_base =
            unsafe { NonNull::new_unchecked(storage.as_ref().as_ptr() as *mut c_void) };

        // Register memory regions
        let mr_handle =
            self.engine.register_memory_local(buf_base, len * count, Device::Host)?;

        // Submit RECV ops.
        for i in 0..count {
            let on_recv_ref = on_recv.clone();
            let on_error_ref = on_error.clone();

            let storage_ref: Arc<Vec<MaybeUninit<u8>>> = storage.clone();

            let on_recv_wrapper = Box::new(move |data_len: usize| {
                let ptr = unsafe {
                    NonNull::new_unchecked(
                        storage_ref.as_ref().as_ptr().byte_add(i * len) as *mut c_void,
                    )
                };
                let data = unsafe {
                    std::slice::from_raw_parts(ptr.as_ptr() as *const u8, data_len)
                };

                on_recv_ref(data)?;
                Ok(())
            });

            let on_error_wrapper =
                { Box::new(move |e: FabricLibError| on_error_ref(e)) };

            self.submit_recv(
                mr_handle,
                unsafe {
                    NonNull::new_unchecked(storage.as_ref().as_ptr() as *mut c_void)
                        .byte_add(i * len)
                },
                len,
                on_recv_wrapper,
                on_error_wrapper,
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "tokio")]
impl AsyncTransferEngine for TransferEngine {
    async fn wait_for_imm_count(
        &self,
        imm: u32,
        expected_count: NonZeroU32,
    ) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        let callback = Box::new(move || {
            tx.send(Ok(())).expect("Failed to send through oneshot channel");
        });

        tokio::task::block_in_place(move || {
            self.callbacks.imm_count.insert(imm, ImmCountFn::Once(callback));
            self.engine.set_imm_count_expected(imm, expected_count);
        });

        rx.await.map_err(|e| {
            FabricLibError::CompletionError(format!(
                "Failed to receive result from oneshot channel: {}",
                e
            ))
        })?
    }

    async fn submit_send_async(
        &self,
        addr: DomainAddress,
        buffer: SendBuffer,
    ) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        let callback = Box::new(move |result: Result<()>| {
            if tx.send(result).is_err() {
                Err("Failed to send result through oneshot channel".to_string())
            } else {
                Ok(())
            }
        });

        tokio::task::block_in_place(move || self.submit_send(addr, buffer, callback))?;

        rx.await.map_err(|e| {
            FabricLibError::CompletionError(format!(
                "Failed to receive result from oneshot channel: {}",
                e
            ))
        })?
    }

    async fn submit_transfer_async(&self, request: TransferRequest) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(1);

        let done_tx = tx.clone();
        let error_tx = tx;
        let callback = TransferCallback {
            on_done: Box::new(move || {
                done_tx.blocking_send(Ok(())).map_err(|e| {
                    format!("Failed to send result through oneshot channel: {}", e)
                })
            }),
            on_error: Box::new(move |e: FabricLibError| {
                error_tx.blocking_send(Err(e)).map_err(|e| {
                    format!("Failed to send error through oneshot channel: {}", e)
                })
            }),
        };

        tokio::task::block_in_place(move || self.submit_transfer(request, callback))?;

        rx.recv().await.ok_or_else(|| {
            FabricLibError::CompletionError(
                "Failed to receive result from oneshot channel".to_string(),
            )
        })?
    }
}

fn callback_worker_thread(engine: Arc<FabricEngine>, states: Arc<Callbacks>) {
    while !engine.is_stopped() {
        std::hint::spin_loop();
        let Some(comp) = engine.poll_transfer_completion() else {
            continue;
        };

        if let Err(e) = handle_transfer_completion(&engine, &states, comp) {
            error!(?e, "Transfer Engine callback thread error. Exiting.");
            engine.stop();
            break;
        }
    }
}

fn handle_transfer_completion(
    engine: &FabricEngine,
    states: &Callbacks,
    comp: TransferCompletionEntry,
) -> CallbackResult {
    match comp {
        TransferCompletionEntry::Transfer(transfer_id) => {
            let Some((_, handler)) = states.transfer_ops.remove(&transfer_id) else {
                warn!(?transfer_id, "Transfer callback not found");
                return Ok(());
            };
            (handler.on_done)()
        }
        TransferCompletionEntry::Recv { transfer_id, data_len } => {
            let Some(handler) = states.recv_ops.get(&transfer_id) else {
                warn!(?transfer_id, data_len, "Recv callback not found");
                return Ok(());
            };
            // Run the callback handler.
            (handler.on_recv)(data_len)?;
            // Re-register the operation.
            engine
                .submit_recv(transfer_id, handler.mr, handler.ptr, handler.len)
                .map_err(|e| format!("Failed to re-register recv operation: {}", e))
        }
        TransferCompletionEntry::Send(transfer_id) => {
            let Some((_, handler)) = states.send_ops.remove(&transfer_id) else {
                warn!(?transfer_id, "Send callback not found");
                return Ok(());
            };
            (handler)(Ok(()))
        }
        TransferCompletionEntry::ImmData(imm_data) => {
            for callback in states.imm.read().iter() {
                callback(imm_data)?
            }
            Ok(())
        }
        TransferCompletionEntry::ImmCountReached(imm) => {
            let Some(handler) = states.imm_count.get(&imm) else {
                warn!(imm, "Imm count context not found");
                return Ok(());
            };
            match &*handler {
                ImmCountFn::Once(_) => {
                    // Remove the counter from the engine.
                    drop(handler);
                    let (_, once_handler) = states.imm_count.remove(&imm).unwrap();
                    let ImmCountFn::Once(handler_fn) = once_handler else {
                        unreachable!("Expected ImmCountFn::Once");
                    };
                    handler_fn();
                    Ok(())
                }
                ImmCountFn::Repeated(callback) => {
                    let res = callback();
                    drop(handler);
                    match res {
                        Ok(true) => {
                            // The counter has already been reset as soon as it reached the expected value.
                            // The callback will be called again when reaching the expected value again.
                            Ok(())
                        }
                        Ok(false) | Err(_) => {
                            // Remove ImmCount callback
                            states.imm_count.remove(&imm);

                            // Remove ImmCount from the engine.
                            // Call ImmData callback if there are overflow counts.
                            if let Some(imm_count) = engine.remove_imm_count(imm) {
                                let (count, _expected) = imm_count.consume();
                                if count > 0 {
                                    for _ in 0..count {
                                        for callback in states.imm.read().iter() {
                                            callback(imm)?
                                        }
                                    }
                                }
                            }
                            Ok(())
                        }
                    }
                }
            }
        }
        TransferCompletionEntry::UvmWatch { id, old, new } => {
            let Some(callback) = states.watchers.get(&id) else {
                warn!(?id, "UvmWatcher not found");
                return Ok(());
            };
            match callback(old, new) {
                Ok(true) => Ok(()),
                Ok(false) | Err(_) => {
                    // Stop the watcher if the callback returns false or an error occurs
                    if let Err(e) = engine.release_uvm_watcher(id) {
                        error!("Failed to release UvmWatcher: {}", e);
                    };
                    states.watchers.remove(&id);
                    Ok(())
                }
            }
        }
        TransferCompletionEntry::Error(transfer_id, fabric_lib_error) => {
            let callback_result = {
                if let Some((_, op)) = states.transfer_ops.remove(&transfer_id) {
                    (op.on_error)(fabric_lib_error)
                } else if let Some((_, op)) = states.send_ops.remove(&transfer_id) {
                    op(Err(fabric_lib_error))
                } else if let Some((_, op)) = states.recv_ops.remove(&transfer_id) {
                    (op.on_error)(fabric_lib_error)
                } else {
                    error!(?transfer_id, ?fabric_lib_error, "Unhandled transfer error");
                    return Ok(());
                }
            };
            if let Err(e) = callback_result {
                error!("Failed to call error callback: {}", e);
            };
            Ok(())
        }
    }
}
