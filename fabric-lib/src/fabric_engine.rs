use std::{
    collections::BTreeMap,
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
};

use cuda_lib::{CudaDeviceId, Device, gdr::GdrFlag};
use dashmap::DashMap;

use crate::{
    api::{
        DomainAddress, GdrCounter, ImmCounter, MemoryRegionDescriptor,
        MemoryRegionHandle, PeerGroupHandle, SmallVec, TransferCompletionEntry,
        TransferCounter, TransferId, TransferRequest, UvmWatcherId,
    },
    error::{FabricLibError, Result},
    imm_count::{ImmCount, ImmCountMap},
    mr::MemoryRegion,
    worker::{UvmWatcherCall, Worker, WorkerCall, WorkerCommand, WorkerHandle},
};

pub struct FabricEngine {
    workers: BTreeMap<u8, WorkerContext>,
    main_address: DomainAddress,
    num_groups: usize,
    num_domains: usize,
    aggregated_link_speed: u64,
    nets_per_gpu: NonZeroU8,
    stop_signal: AtomicBool,
    mr_device_map: DashMap<MemoryRegionHandle, Device>,
    imm_count_map: Arc<ImmCountMap>,
}

struct WorkerContext {
    worker: WorkerHandle,
}

impl FabricEngine {
    pub fn new(workers: Vec<(u8, Worker)>) -> Result<Self> {
        let imm_count_map = Arc::new(ImmCountMap::default());

        let spawned_workers = workers
            .into_iter()
            .map(|(device, w)| {
                let worker = w.spawn(imm_count_map.clone())?;
                Ok((device, worker))
            })
            .collect::<Result<Vec<_>>>()?;

        let initialized_workers = spawned_workers
            .into_iter()
            .map(|(device, w)| {
                let worker = w.wait_init()?;
                Ok((device, worker))
            })
            .collect::<Result<Vec<_>>>()?;

        let main_worker = &initialized_workers.first().unwrap().1;
        let main_address = main_worker.address_list[0].clone();
        let nets_per_gpu =
            unsafe { NonZeroU8::new_unchecked(main_worker.address_list.len() as u8) };

        let mut contexts = BTreeMap::new();
        let mut num_groups = 0;
        let mut num_domains = 0;
        let mut aggregated_link_speed = 0;
        for (device, w) in initialized_workers.into_iter() {
            num_groups += 1;
            num_domains += w.address_list.len();
            aggregated_link_speed += w.aggregated_link_speed;
            contexts.insert(device, WorkerContext { worker: w });
        }

        Ok(Self {
            workers: contexts,
            main_address,
            num_groups,
            num_domains,
            aggregated_link_speed,
            nets_per_gpu,
            stop_signal: AtomicBool::new(false),
            mr_device_map: DashMap::new(),
            imm_count_map,
        })
    }

    pub fn main_address(&self) -> DomainAddress {
        self.main_address.clone()
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn num_domains(&self) -> usize {
        self.num_domains
    }

    pub fn aggregated_link_speed(&self) -> u64 {
        self.aggregated_link_speed
    }

    pub fn nets_per_gpu(&self) -> NonZeroU8 {
        self.nets_per_gpu
    }

    pub fn register_memory_local(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<MemoryRegionHandle> {
        let worker = self.get_worker(&device)?;
        let region = MemoryRegion::new(ptr, len, device)?;
        let (tx, rx) = oneshot::channel();
        let cmd = WorkerCall::RegisterMRLocal { region, ret: tx };
        worker
            .worker
            .worker_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        let handle =
            rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))??;
        self.mr_device_map.insert(handle, device);
        Ok(handle)
    }

    pub fn register_memory_allow_remote(
        &self,
        ptr: NonNull<c_void>,
        len: usize,
        device: Device,
    ) -> Result<(MemoryRegionHandle, MemoryRegionDescriptor)> {
        let worker = self.get_worker(&device)?;
        let (tx, rx) = oneshot::channel();
        let region = MemoryRegion::new(ptr, len, device)?;
        let cmd = WorkerCall::RegisterMRAllowRemote { region, ret: tx };
        worker
            .worker
            .worker_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        let (handle, desc) =
            rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))??;
        self.mr_device_map.insert(handle, device);
        Ok((handle, desc))
    }

    pub fn unregister_memory(&self, ptr: NonNull<c_void>) -> Result<()> {
        let worker = self.get_main_worker()?;
        let (tx, rx) = oneshot::channel();
        let cmd = WorkerCall::UnregisterMR { ptr, ret: tx };
        worker
            .worker
            .worker_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))?;
        Ok(())
    }

    pub fn add_peer_group(
        &self,
        addrs: Vec<SmallVec<DomainAddress>>,
        device: Device,
    ) -> Result<PeerGroupHandle> {
        let worker = self.get_worker(&device)?;
        let (tx, rx) = oneshot::channel();
        let cmd = WorkerCall::AddPeerGroup { addrs, ret: tx };
        worker
            .worker
            .worker_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        let handle =
            rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))??;
        Ok(handle)
    }

    pub fn acquire_uvm_watcher(&self) -> Result<UvmWatcherId> {
        let worker = self.get_main_worker()?;
        let (tx, rx) = oneshot::channel();
        let cmd = UvmWatcherCall::AcquireUvmWatcher { ret: tx };
        worker
            .worker
            .uvm_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        let maybe = rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))?;
        maybe.ok_or(FabricLibError::Custom("Failed to acquire UVM watcher"))
    }

    pub fn release_uvm_watcher(&self, watcher: UvmWatcherId) -> Result<()> {
        let worker = self.get_main_worker()?;
        let (tx, rx) = oneshot::channel();
        let cmd = UvmWatcherCall::ReleaseUvmWatcher { watcher, ret: tx };
        worker
            .worker
            .uvm_call_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        rx.recv().map_err(|_| FabricLibError::Custom("Worker is down"))?;
        Ok(())
    }

    pub fn set_imm_count_expected(
        &self,
        imm: u32,
        expected_count: NonZeroU32,
    ) -> Option<ImmCount> {
        self.imm_count_map.set_expected(imm, expected_count)
    }

    pub fn remove_imm_count(&self, imm: u32) -> Option<ImmCount> {
        self.imm_count_map.remove(imm)
    }

    pub fn get_imm_counter(&self, imm: u32) -> ImmCounter {
        self.imm_count_map.get_imm_counter(imm)
    }

    pub fn get_gdr_counter(&self, imm: u32, flag: Arc<GdrFlag>) -> GdrCounter {
        self.imm_count_map.get_gdr_counter(imm, flag)
    }

    pub fn submit_send(
        &self,
        transfer_id: TransferId,
        addr: DomainAddress,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<()> {
        let worker = self.get_main_worker()?;
        let cmd = WorkerCommand::SubmitSend { transfer_id, addr, mr, ptr, len };
        worker.send_command(Box::new(cmd))?;
        Ok(())
    }

    pub fn submit_recv(
        &self,
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    ) -> Result<()> {
        let worker = self.get_main_worker()?;
        worker.send_command(Box::new(WorkerCommand::SubmitRecv {
            transfer_id,
            mr,
            ptr,
            len,
        }))?;
        Ok(())
    }

    pub fn submit_transfer(
        &self,
        transfer_id: TransferId,
        request: TransferRequest,
        tx_counter: Option<TransferCounter>,
    ) -> Result<()> {
        let worker = match &request {
            TransferRequest::Imm(_) => self.get_main_worker()?,
            TransferRequest::Barrier(_) => self.get_main_worker()?,
            TransferRequest::Single(req) => self.get_worker_by_mr(req.src_mr)?,
            TransferRequest::Paged(req) => self.get_worker_by_mr(req.src_mr)?,
            TransferRequest::Scatter(req) => self.get_worker_by_mr(req.src_mr)?,
        };
        worker.send_command(Box::new(WorkerCommand::SubmitTransfer {
            transfer_id,
            request,
            tx_counter,
        }))
    }

    pub fn poll_transfer_completion(&self) -> Option<TransferCompletionEntry> {
        for (_, ctx) in self.workers.iter() {
            if let Ok(completion) = ctx.worker.cq_rx.try_recv() {
                return Some(completion);
            }
        }

        None
    }

    pub fn stop(&self) {
        self.stop_signal.store(true, SeqCst);
        for (_, ctx) in self.workers.iter() {
            ctx.worker.stop();
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stop_signal.load(SeqCst)
    }

    fn get_worker_by_mr(&self, mr: MemoryRegionHandle) -> Result<&WorkerContext> {
        if let Some(device) = self.mr_device_map.get(&mr) {
            self.get_worker(device.value())
        } else {
            Err(FabricLibError::Custom("Invalid memory region"))
        }
    }

    fn get_worker(&self, device: &Device) -> Result<&WorkerContext> {
        match device {
            Device::Host => self.get_main_worker(),
            Device::Cuda(CudaDeviceId(device_id)) => self
                .workers
                .get(device_id)
                .ok_or(FabricLibError::Custom("Worker not found")),
        }
    }

    fn get_main_worker(&self) -> Result<&WorkerContext> {
        Ok(self.workers.first_key_value().unwrap().1)
    }
}

impl Drop for FabricEngine {
    fn drop(&mut self) {
        self.stop();
        while let Some((_, ctx)) = self.workers.pop_first() {
            ctx.worker.stop();
        }
    }
}

impl WorkerContext {
    pub fn send_command(&self, cmd: Box<WorkerCommand>) -> Result<()> {
        self.worker
            .cmd_tx
            .send(cmd)
            .map_err(|_| FabricLibError::Custom("Worker is down"))?;
        Ok(())
    }
}
