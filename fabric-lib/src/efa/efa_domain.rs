use std::{
    collections::{HashMap, VecDeque, hash_map::Entry},
    ffi::{CStr, c_void},
    mem::{MaybeUninit, transmute},
    ptr::{NonNull, null_mut},
    rc::Rc,
    sync::Arc,
};

use bytes::Bytes;
use cuda_lib::{Device, rt::cudaSetDevice};
use libfabric_sys::{
    FI_ADDR_UNSPEC, FI_CQ_FORMAT_DATA, FI_EAGAIN, FI_EAVAIL, FI_ENABLE, FI_HMEM_CUDA,
    FI_MR_DMABUF, FI_MR_ENDPOINT, FI_MR_LOCAL, FI_MR_VIRT_ADDR,
    FI_OPT_CUDA_API_PERMITTED, FI_OPT_ENDPOINT, FI_OPT_SHARED_MEMORY_PERMITTED,
    FI_READ, FI_RECV, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_RMA_EVENT, FI_SEND, FI_WRITE,
    fi_addr_t, fi_av_attr, fi_close, fi_cq_attr, fi_cq_data_entry, fi_cq_err_entry,
    fi_fabric, fi_mr_attr, fi_mr_dmabuf, fi_msg, fid_av, fid_cq, fid_domain, fid_ep,
    fid_fabric, fid_mr, iovec,
};
use tracing::{debug, error, warn};

use crate::{
    api::{DomainAddress, MemoryRegionRemoteKey, PeerGroupHandle, TransferId},
    efa::{
        efa_devinfo::EfaDomainInfo,
        efa_mr::EfaMemDesc,
        efa_rdma_op::{
            PagedWriteOpIter, RmaBuffer, ScatterWriteOpIter, SingleWriteOpIter,
            WriteOpIter, fill_recv_op, fill_send_op,
        },
    },
    error::{FabricLibError, LibfabricError, Result},
    imm_count::{ImmCountMap, ImmCountStatus},
    mr::{Mapping, MemoryRegion, MemoryRegionLocalDescriptor},
    provider::{DomainCompletionEntry, RdmaDomain, RdmaDomainInfo},
    rdma_op::{GroupWriteOp, RecvOp, SendOp, WriteOp},
    utils::{defer::Defer, obj_pool::ObjectPool},
};

pub struct EfaDomain {
    info: EfaDomainInfo,
    fabric: NonNull<fid_fabric>,
    domain: NonNull<fid_domain>,
    ep: NonNull<fid_ep>,
    cq: NonNull<fid_cq>,
    av: NonNull<fid_av>,
    imm_ep: Option<NonNull<fid_ep>>,
    imm_cq: Option<NonNull<fid_cq>>,
    imm_av: Option<NonNull<fid_av>>,
    addr: DomainAddress,

    peer_addr_map: HashMap<DomainAddress, fi_addr_t>,
    imm_peer_addr_map: HashMap<DomainAddress, fi_addr_t>,
    peer_groups: HashMap<PeerGroupHandle, PeerGroup>,
    local_mr_map: HashMap<NonNull<c_void>, NonNull<fid_mr>>,
    use_local_mr_desc: bool,
    use_endpoint_mr: bool,
    use_remote_virt_addr: bool,
    use_cuda_dmabuf: bool,
    use_rma_event_mr: bool,
    use_send_for_imm: bool,
    debug_fi: bool,
    imm_count_map: Arc<ImmCountMap>,
    objpool_write_op: ObjectPool<WriteOpContext>,
    objpool_msg: ObjectPool<RmaBuffer>,

    recv_ops: VecDeque<RecvOpContext>,
    send_ops: VecDeque<SendOpContext>,
    imm_send_ops: VecDeque<ImmSendContext>,
    imm_send_pending: HashMap<TransferId, usize>,
    imm_send_pending_bufs: Vec<(TransferId, Box<[u8; IMM_MSG_BYTES]>)>,
    write_ops: VecDeque<NonNull<WriteOpContext>>,
    imm_recv_slots: Vec<ImmRecvSlot>,
    imm_recvs_posted: bool,
    imm_recv_addr: fi_addr_t,

    completions: VecDeque<DomainCompletionEntry>,
}

struct PeerGroup {
    addrs: Rc<Vec<fi_addr_t>>,
    imm_addrs: Rc<Vec<fi_addr_t>>,
}

struct RecvOpContext {
    transfer_id: TransferId,
    op: RecvOp,
}

struct SendOpContext {
    transfer_id: TransferId,
    dest_addr: fi_addr_t,
    op: SendOp,
}

struct ImmSendContext {
    transfer_id: TransferId,
    dest_addr: fi_addr_t,
    imm: Box<[u8; IMM_MSG_BYTES]>,
}

struct ImmRecvSlot {
    buf: Box<[u8; IMM_MSG_BYTES]>,
}

struct WriteOpContext {
    transfer_id: TransferId,
    rdma_op_iter: WriteOpIter,
    msg_buf: NonNull<RmaBuffer>,
    ep: NonNull<fid_ep>,
    total_ops: usize,
    cnt_posted_ops: usize,
    cnt_finished_ops: usize,
    in_queue: bool,

    /// True when there's a completion error.
    /// No more write ops will be posted.
    /// Once cnt_finished_ops catches up with cnt_posted_ops, the context will be dropped.
    bad: bool,
    imm_after_write: Option<Vec<(fi_addr_t, u32)>>,
}

const EAGAIN: isize = -(FI_EAGAIN as isize);
const IMM_RECV_POOL_SIZE: usize = 4096;
const IMM_MSG_BYTES: usize = 64;

impl EfaDomain {
    fn encode_dual_addr(main_addr: &[u8], imm_addr: &[u8]) -> DomainAddress {
        let mut addr = Vec::with_capacity(2 + main_addr.len() + imm_addr.len());
        addr.extend_from_slice(&(main_addr.len() as u16).to_le_bytes());
        addr.extend_from_slice(main_addr);
        addr.extend_from_slice(imm_addr);
        DomainAddress(Bytes::from(addr))
    }

    fn split_main_addr(addr: &DomainAddress) -> &[u8] {
        if addr.0.len() < 2 {
            return addr.0.as_ref();
        }
        let main_len = u16::from_le_bytes([addr.0[0], addr.0[1]]) as usize;
        if main_len == 0 || 2 + main_len > addr.0.len() {
            return addr.0.as_ref();
        }
        &addr.0[2..2 + main_len]
    }

    fn split_imm_addr(addr: &DomainAddress) -> &[u8] {
        if addr.0.len() < 2 {
            return addr.0.as_ref();
        }
        let main_len = u16::from_le_bytes([addr.0[0], addr.0[1]]) as usize;
        if main_len == 0 || 2 + main_len >= addr.0.len() {
            return addr.0.as_ref();
        }
        &addr.0[2 + main_len..]
    }

    fn open(info: EfaDomainInfo, imm_count_map: Arc<ImmCountMap>) -> Result<Self> {
        unsafe {
            debug!("Domain::open: name: {}", info.name());
            let fi = info.fi();
            let provider_name = CStr::from_ptr((*(*fi.as_ptr()).fabric_attr).prov_name)
                .to_string_lossy();
            let is_efa = provider_name == "efa";
            let mr_mode = (*(*fi.as_ptr()).domain_attr).mr_mode;
            let use_local_mr_desc = mr_mode & FI_MR_LOCAL as i32 != 0;
            let use_endpoint_mr = mr_mode & FI_MR_ENDPOINT as i32 != 0;
            let use_remote_virt_addr = mr_mode & FI_MR_VIRT_ADDR as i32 != 0;
            let use_send_for_imm = provider_name != "efa"
                && std::env::var_os("PPLX_GARDEN_CXI_SEND_IMM_FALLBACK").is_some();
            let debug_fi = std::env::var_os("PPLX_GARDEN_DEBUG_FI").is_some();
            if debug_fi {
                eprintln!(
                    "pplx domain open name={} provider={} mr_mode=0x{:x} local_desc={} endpoint_mr={} virt_addr={} send_imm_fallback={}",
                    info.name(),
                    provider_name,
                    mr_mode,
                    use_local_mr_desc,
                    use_endpoint_mr,
                    use_remote_virt_addr,
                    use_send_for_imm
                );
            }

            // Fabric
            let mut fabric = null_mut();
            let ret = fi_fabric(fi.as_ref().fabric_attr, &raw mut fabric, null_mut());
            let fabric = NonNull::new(fabric)
                .ok_or_else(|| LibfabricError::new(ret, "fi_fabric"))?;
            let mut defer_fabric =
                Defer::new(|| fi_close(&raw mut (*fabric.as_ptr()).fid));

            // Domain
            let mut domain = null_mut();
            let fi_domain = (*(*fabric.as_ptr()).ops).domain.unwrap_unchecked();
            let ret =
                fi_domain(fabric.as_ptr(), fi.as_ptr(), &raw mut domain, null_mut());
            let domain = NonNull::new(domain)
                .ok_or_else(|| LibfabricError::new(ret, "fi_domain"))?;
            let mut defer_domain =
                Defer::new(|| fi_close(&raw mut (*domain.as_ptr()).fid));

            // Completion queue
            let mut cq = null_mut();
            let mut cq_attr =
                fi_cq_attr { format: FI_CQ_FORMAT_DATA, ..Default::default() };
            let fi_cq_open = (*(*domain.as_ptr()).ops).cq_open.unwrap_unchecked();
            let ret =
                fi_cq_open(domain.as_ptr(), &raw mut cq_attr, &raw mut cq, null_mut());
            let cq = NonNull::new(cq)
                .ok_or_else(|| LibfabricError::new(ret, "fi_cq_open"))?;
            let mut defer_cq = Defer::new(|| fi_close(&raw mut (*cq.as_ptr()).fid));

            // Address vector
            let mut av = null_mut();
            let mut av_attr = fi_av_attr::default();
            let fi_av_open = (*(*domain.as_ptr()).ops).av_open.unwrap_unchecked();
            let ret =
                fi_av_open(domain.as_ptr(), &raw mut av_attr, &raw mut av, null_mut());
            let av = NonNull::new(av)
                .ok_or_else(|| LibfabricError::new(ret, "fi_av_open"))?;
            let mut defer_av = Defer::new(|| fi_close(&raw mut (*av.as_ptr()).fid));

            // Endpoint
            let mut ep = null_mut();
            let fi_endpoint = (*(*domain.as_ptr()).ops).endpoint.unwrap_unchecked();
            let ret =
                fi_endpoint(domain.as_ptr(), fi.as_ptr(), &raw mut ep, null_mut());
            let ep = NonNull::new(ep)
                .ok_or_else(|| LibfabricError::new(ret, "fi_endpoint"))?;
            let mut defer_ep = Defer::new(|| fi_close(&raw mut (*ep.as_ptr()).fid));

            // Bind to endpoint
            let ep_fid = &raw mut (*ep.as_ptr()).fid;
            let fi_ep_bind = (*(*ep_fid).ops).bind.unwrap_unchecked();
            let ret = fi_ep_bind(
                ep_fid,
                &raw mut (*cq.as_ptr()).fid,
                (FI_SEND | FI_RECV) as u64,
            );
            if ret != 0 {
                return Err(LibfabricError::new(ret, "fi_ep_bind cq").into());
            }
            let ret = fi_ep_bind(ep_fid, &raw mut (*av.as_ptr()).fid, 0);
            if ret != 0 {
                return Err(LibfabricError::new(ret, "fi_ep_bind av").into());
            }

            if is_efa {
                // Disallow using shm and cuda p2p transfer.
                // All data transfer should go through RDMA.
                let optval = false;
                let fi_setopt = (*(*ep.as_ptr()).ops).setopt.unwrap_unchecked();
                let ret = fi_setopt(
                    ep_fid,
                    FI_OPT_ENDPOINT as i32,
                    FI_OPT_SHARED_MEMORY_PERMITTED as i32,
                    &optval as *const _ as *mut c_void,
                    std::mem::size_of_val(&optval),
                );
                if ret != 0 {
                    return Err(LibfabricError::new(
                        ret,
                        "fi_setopt FI_OPT_SHARED_MEMORY_PERMITTED false",
                    )
                    .into());
                }
                let ret = fi_setopt(
                    ep_fid,
                    FI_OPT_ENDPOINT as i32,
                    FI_OPT_CUDA_API_PERMITTED as i32,
                    &optval as *const _ as *mut c_void,
                    std::mem::size_of_val(&optval),
                );
                if ret != 0 {
                    return Err(LibfabricError::new(
                        ret,
                        "fi_setopt FI_OPT_CUDA_API_PERMITTED false",
                    )
                    .into());
                }
            }

            // Enable endpoint
            let fi_control = (*(*ep_fid).ops).control.unwrap_unchecked();
            let ret = fi_control(ep_fid, FI_ENABLE as i32, null_mut());
            if ret != 0 {
                return Err(LibfabricError::new(ret, "fi_enable").into());
            }

            let (imm_ep, imm_cq, imm_av) = if use_send_for_imm {
                let mut imm_cq = null_mut();
                let mut imm_cq_attr =
                    fi_cq_attr { format: FI_CQ_FORMAT_DATA, ..Default::default() };
                let ret = fi_cq_open(
                    domain.as_ptr(),
                    &raw mut imm_cq_attr,
                    &raw mut imm_cq,
                    null_mut(),
                );
                let imm_cq = NonNull::new(imm_cq)
                    .ok_or_else(|| LibfabricError::new(ret, "fi_cq_open imm"))?;
                let mut defer_imm_cq =
                    Defer::new(|| fi_close(&raw mut (*imm_cq.as_ptr()).fid));

                let mut imm_av = null_mut();
                let mut imm_av_attr = fi_av_attr::default();
                let ret = fi_av_open(
                    domain.as_ptr(),
                    &raw mut imm_av_attr,
                    &raw mut imm_av,
                    null_mut(),
                );
                let imm_av = NonNull::new(imm_av)
                    .ok_or_else(|| LibfabricError::new(ret, "fi_av_open imm"))?;
                let mut defer_imm_av =
                    Defer::new(|| fi_close(&raw mut (*imm_av.as_ptr()).fid));

                let mut imm_ep = null_mut();
                let ret = fi_endpoint(
                    domain.as_ptr(),
                    fi.as_ptr(),
                    &raw mut imm_ep,
                    null_mut(),
                );
                let imm_ep = NonNull::new(imm_ep)
                    .ok_or_else(|| LibfabricError::new(ret, "fi_endpoint imm"))?;
                let mut defer_imm_ep =
                    Defer::new(|| fi_close(&raw mut (*imm_ep.as_ptr()).fid));

                let imm_ep_fid = &raw mut (*imm_ep.as_ptr()).fid;
                let ret = fi_ep_bind(
                    imm_ep_fid,
                    &raw mut (*imm_cq.as_ptr()).fid,
                    (FI_SEND | FI_RECV) as u64,
                );
                if ret != 0 {
                    return Err(LibfabricError::new(ret, "fi_ep_bind imm cq").into());
                }
                let ret = fi_ep_bind(imm_ep_fid, &raw mut (*imm_av.as_ptr()).fid, 0);
                if ret != 0 {
                    return Err(LibfabricError::new(ret, "fi_ep_bind imm av").into());
                }
                let ret = fi_control(imm_ep_fid, FI_ENABLE as i32, null_mut());
                if ret != 0 {
                    return Err(LibfabricError::new(ret, "fi_enable imm").into());
                }

                defer_imm_cq.cancel();
                defer_imm_av.cancel();
                defer_imm_ep.cancel();
                (Some(imm_ep), Some(imm_cq), Some(imm_av))
            } else {
                (None, None, None)
            };

            // Save address
            let mut addrbuf = vec![0u8; 128];
            let mut addrlen = addrbuf.len();
            let fi_getname = (*(*ep.as_ptr()).cm).getname.unwrap_unchecked();
            let ret = fi_getname(
                ep_fid,
                addrbuf.as_mut_ptr() as *mut c_void,
                &raw mut addrlen,
            );
            if ret != 0 {
                return Err(LibfabricError::new(ret, "fi_getname").into());
            }
            let addr = if let Some(imm_ep) = imm_ep {
                let mut imm_addrbuf = vec![0u8; 128];
                let mut imm_addrlen = imm_addrbuf.len();
                let ret = fi_getname(
                    &raw mut (*imm_ep.as_ptr()).fid,
                    imm_addrbuf.as_mut_ptr() as *mut c_void,
                    &raw mut imm_addrlen,
                );
                if ret != 0 {
                    return Err(LibfabricError::new(ret, "fi_getname imm").into());
                }
                Self::encode_dual_addr(&addrbuf[..addrlen], &imm_addrbuf[..imm_addrlen])
            } else {
                DomainAddress(Bytes::copy_from_slice(&addrbuf[..addrlen]))
            };

            // Cancel all defer
            defer_fabric.cancel();
            defer_domain.cancel();
            defer_cq.cancel();
            defer_av.cancel();
            defer_ep.cancel();

            let slf = Self {
                info,
                fabric,
                domain,
                ep,
                cq,
                av,
                imm_ep,
                imm_cq,
                imm_av,
                addr,

                peer_addr_map: HashMap::new(),
                imm_peer_addr_map: HashMap::new(),
                peer_groups: HashMap::new(),
                local_mr_map: HashMap::new(),
                use_local_mr_desc,
                use_endpoint_mr,
                use_remote_virt_addr,
                use_cuda_dmabuf: is_efa,
                use_rma_event_mr: std::env::var_os("PPLX_GARDEN_CXI_RMA_EVENT_MR")
                    .is_some(),
                use_send_for_imm,
                debug_fi,
                objpool_write_op: ObjectPool::with_chunk_size(1024),
                objpool_msg: ObjectPool::with_chunk_size(1024),
                imm_count_map,

                recv_ops: VecDeque::new(),
                send_ops: VecDeque::new(),
                imm_send_ops: VecDeque::new(),
                imm_send_pending: HashMap::new(),
                imm_send_pending_bufs: Vec::new(),
                write_ops: VecDeque::new(),
                imm_recv_slots: if use_send_for_imm {
                    (0..IMM_RECV_POOL_SIZE)
                        .map(|_| ImmRecvSlot { buf: Box::new([0; IMM_MSG_BYTES]) })
                        .collect()
                } else {
                    Vec::new()
                },
                imm_recvs_posted: false,
                imm_recv_addr: FI_ADDR_UNSPEC,

                completions: VecDeque::new(),
            };
            Ok(slf)
        }
    }

    fn get_or_add_remote_addr(
        &mut self,
        peer_addr: &DomainAddress,
    ) -> Result<fi_addr_t> {
        match self.peer_addr_map.entry(peer_addr.clone()) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => unsafe {
                let fi_av_insert = (*(*self.av.as_ptr()).ops).insert.unwrap_unchecked();
                let mut addr_id: fi_addr_t = FI_ADDR_UNSPEC;
                let raw_addr = Self::split_main_addr(peer_addr);
                let ret = fi_av_insert(
                    self.av.as_ptr(),
                    raw_addr.as_ptr() as *const c_void,
                    1,
                    &raw mut addr_id,
                    0,
                    null_mut(),
                );
                if ret == 1 {
                    entry.insert(addr_id);
                    Ok(addr_id)
                } else {
                    Err(LibfabricError::new(ret, "fi_av_insert").into())
                }
            },
        }
    }

    fn get_or_add_imm_remote_addr(
        &mut self,
        peer_addr: &DomainAddress,
    ) -> Result<fi_addr_t> {
        if self.imm_av.is_none() {
            return self.get_or_add_remote_addr(peer_addr);
        }
        let imm_av = self
            .imm_av
            .ok_or(FabricLibError::Custom("Immediate endpoint is not enabled"))?;
        match self.imm_peer_addr_map.entry(peer_addr.clone()) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => unsafe {
                let fi_av_insert = (*(*imm_av.as_ptr()).ops).insert.unwrap_unchecked();
                let mut addr_id: fi_addr_t = FI_ADDR_UNSPEC;
                let raw_addr = Self::split_imm_addr(peer_addr);
                let ret = fi_av_insert(
                    imm_av.as_ptr(),
                    raw_addr.as_ptr() as *const c_void,
                    1,
                    &raw mut addr_id,
                    0,
                    null_mut(),
                );
                if ret == 1 {
                    entry.insert(addr_id);
                    Ok(addr_id)
                } else {
                    Err(LibfabricError::new(ret, "fi_av_insert imm").into())
                }
            },
        }
    }

    fn register_mr(
        &mut self,
        region: &MemoryRegion,
        allow_remote: bool,
    ) -> Result<MemoryRegionRemoteKey> {
        if let Some(mr) = self.local_mr_map.get(&region.ptr()) {
            return Ok(MemoryRegionRemoteKey(unsafe { mr.as_ref() }.key));
        }

        let mut access = (FI_SEND | FI_RECV | FI_WRITE | FI_READ) as u64;
        if allow_remote {
            access |= (FI_REMOTE_WRITE | FI_REMOTE_READ) as u64;
            if self.use_rma_event_mr {
                access |= FI_RMA_EVENT;
            }
        }

        let mut mr = null_mut();
        let mut mr_attr = fi_mr_attr { iov_count: 1, access, ..Default::default() };

        let iov = iovec { iov_base: region.ptr().as_ptr(), iov_len: region.len() };
        let mut dmabuf = fi_mr_dmabuf {
            len: region.len(),
            base_addr: region.ptr().as_ptr(),
            ..Default::default()
        };
        let mut flags = 0;
        match region.mapping() {
            Mapping::Host => {
                mr_attr.__bindgen_anon_1.mr_iov = &iov;
            }
            Mapping::Device { device_id, dmabuf_fd: None } => {
                mr_attr.iface = FI_HMEM_CUDA;
                mr_attr.device.cuda = device_id.0 as i32;
                mr_attr.__bindgen_anon_1.mr_iov = &iov;
            }
            Mapping::Device { device_id, dmabuf_fd: Some(dmabuf_fd) }
                if self.use_cuda_dmabuf =>
            {
                mr_attr.iface = FI_HMEM_CUDA;
                mr_attr.device.cuda = device_id.0 as i32;
                dmabuf.fd = *dmabuf_fd;
                mr_attr.__bindgen_anon_1.dmabuf = &dmabuf;
                flags = FI_MR_DMABUF;
            }
            Mapping::Device { device_id, dmabuf_fd: Some(_) } => {
                mr_attr.iface = FI_HMEM_CUDA;
                mr_attr.device.cuda = device_id.0 as i32;
                mr_attr.__bindgen_anon_1.mr_iov = &iov;
            }
        }
        if let Mapping::Device { device_id, .. } = region.mapping() {
            cudaSetDevice(device_id.0 as i32)?;
        }
        if self.debug_fi {
            let mapping = match region.mapping() {
                Mapping::Host => "host".to_string(),
                Mapping::Device { device_id, dmabuf_fd } => format!(
                    "cuda device={} dmabuf={} use_cuda_dmabuf={}",
                    device_id.0,
                    dmabuf_fd.is_some(),
                    self.use_cuda_dmabuf
                ),
            };
            eprintln!(
                "pplx mr register ptr={:?} len={} allow_remote={} access=0x{:x} flags=0x{:x} iface={} mapping={}",
                region.ptr(),
                region.len(),
                allow_remote,
                access,
                flags,
                mr_attr.iface,
                mapping
            );
        }

        let ret = unsafe {
            let fi_mr_regattr =
                (*(*self.domain.as_ptr()).mr).regattr.unwrap_unchecked();
            let domain_fid = &raw mut (*self.domain.as_ptr()).fid;
            fi_mr_regattr(domain_fid, &mr_attr, flags, &raw mut mr)
        };
        if self.debug_fi {
            eprintln!("pplx mr register ret={} mr={:?}", ret, mr);
        }

        let mr = NonNull::new(mr)
            .ok_or_else(|| LibfabricError::new(ret, "fi_mr_regattr"))?;
        if self.use_endpoint_mr {
            unsafe {
                let mr_fid = &raw mut (*mr.as_ptr()).fid;
                let ep_fid = &raw mut (*self.ep.as_ptr()).fid;
                let fi_bind = (*(*mr_fid).ops).bind.unwrap_unchecked();
                let ret = fi_bind(mr_fid, ep_fid, 0);
                if ret != 0 {
                    fi_close(mr_fid);
                    return Err(LibfabricError::new(ret, "fi_mr_bind ep").into());
                }
                let fi_control = (*(*mr_fid).ops).control.unwrap_unchecked();
                let ret = fi_control(mr_fid, FI_ENABLE as i32, null_mut());
                if ret != 0 {
                    fi_close(mr_fid);
                    return Err(LibfabricError::new(ret, "fi_mr_enable").into());
                }
            }
        }
        self.local_mr_map.insert(region.ptr(), mr);
        Ok(MemoryRegionRemoteKey(unsafe { mr.as_ref() }.key))
    }

    fn progress_ops(&mut self) {
        self.progress_rdma_recv_ops();
        self.progress_imm_send_ops();
        self.progress_rdma_send_ops();
        self.progress_rdma_write_ops();
    }

    fn post_imm_recv_slot(&mut self, idx: usize) -> Result<()> {
        let slot = &mut self.imm_recv_slots[idx];
        let mut iov = iovec {
            iov_base: slot.buf.as_mut_ptr() as *mut c_void,
            iov_len: slot.buf.len(),
        };
        let mut msg = fi_msg {
            msg_iov: &raw mut iov,
            desc: null_mut(),
            iov_count: 1,
            addr: self.imm_recv_addr,
            context: (&mut self.imm_recv_slots[idx]) as *mut ImmRecvSlot as *mut c_void,
            data: 0,
        };
        let ret = unsafe {
            let ep = self.imm_ep.unwrap_or(self.ep);
            let fi_recvmsg = (*(*ep.as_ptr()).msg).recvmsg.unwrap_unchecked();
            fi_recvmsg(ep.as_ptr(), &raw mut msg, 0)
        };
        if ret != 0 {
            return Err(LibfabricError::new(ret as i32, "fi_recvmsg imm").into());
        }
        Ok(())
    }

    fn ensure_imm_recvs_posted(&mut self, _addr: fi_addr_t) -> Result<()> {
        if self.imm_recvs_posted {
            return Ok(());
        }
        self.imm_recv_addr = FI_ADDR_UNSPEC;
        for idx in 0..self.imm_recv_slots.len() {
            let ptr =
                NonNull::new(self.imm_recv_slots[idx].buf.as_mut_ptr() as *mut c_void)
                    .unwrap();
            let region = MemoryRegion::new(ptr, IMM_MSG_BYTES, Device::Host)?;
            self.register_mr(&region, false)?;
            self.post_imm_recv_slot(idx)?;
        }
        self.imm_recvs_posted = true;
        Ok(())
    }

    fn progress_imm_send_ops(&mut self) {
        while let Some(ctx) = self.imm_send_ops.front() {
            let mut iov = iovec {
                iov_base: ctx.imm.as_ptr() as *mut c_void,
                iov_len: ctx.imm.len(),
            };
            let mut msg = fi_msg {
                msg_iov: &raw mut iov,
                desc: null_mut(),
                iov_count: 1,
                addr: ctx.dest_addr,
                context: unsafe {
                    transmute::<TransferId, *mut libc::c_void>(ctx.transfer_id)
                },
                data: 0,
            };
            let ret = unsafe {
                let ep = self.imm_ep.unwrap_or(self.ep);
                let fi_sendmsg = (*(*ep.as_ptr()).msg).sendmsg.unwrap_unchecked();
                fi_sendmsg(ep.as_ptr(), &raw mut msg, 0)
            };
            match ret {
                0 => {
                    let ctx = self.imm_send_ops.pop_front().unwrap();
                    *self.imm_send_pending.entry(ctx.transfer_id).or_insert(0) += 1;
                    self.imm_send_pending_bufs.push((ctx.transfer_id, ctx.imm));
                }
                EAGAIN => break,
                _ => panic!("fi_sendmsg imm returned undocumented error: {}", ret),
            }
        }
    }

    fn progress_rdma_recv_ops(&mut self) {
        let mut iov = MaybeUninit::uninit();
        let mut msg = MaybeUninit::uninit();
        while let Some(ctx) = self.recv_ops.front() {
            fill_recv_op(&ctx.op, &mut iov, &mut msg, unsafe {
                transmute::<TransferId, *mut libc::c_void>(ctx.transfer_id)
            });
            let ret = unsafe {
                let fi_recvmsg = (*(*self.ep.as_ptr()).msg).recvmsg.unwrap_unchecked();
                fi_recvmsg(self.ep.as_ptr(), msg.as_ptr(), 0)
            };
            match ret {
                0 => {
                    self.recv_ops.pop_front();
                }
                EAGAIN => break,
                _ => panic!("fi_recvmsg returned undocumented error: {}", ret),
            }
        }
    }

    fn progress_rdma_send_ops(&mut self) {
        const EAGAIN: isize = -(FI_EAGAIN as isize);
        let mut iov = MaybeUninit::uninit();
        let mut msg = MaybeUninit::uninit();
        while let Some(ctx) = self.send_ops.front() {
            // Populate the libfabric RDMA op
            fill_send_op(&ctx.op, &mut iov, &mut msg, unsafe {
                transmute::<TransferId, *mut libc::c_void>(ctx.transfer_id)
            });

            // Set the destination address
            unsafe { (*msg.as_mut_ptr()).addr = ctx.dest_addr };

            // Submit the RDMA op
            let ret = unsafe {
                let fi_sendmsg = (*(*self.ep.as_ptr()).msg).sendmsg.unwrap_unchecked();
                fi_sendmsg(self.ep.as_ptr(), msg.as_ptr(), 0)
            };
            match ret {
                0 => {
                    self.send_ops.pop_front();
                }
                EAGAIN => break,
                _ => panic!("fi_sendmsg returned undocumented error: {}", ret),
            }
        }
    }

    fn do_submit_write<F: FnOnce(*mut c_void, NonNull<RmaBuffer>) -> WriteOpIter>(
        &mut self,
        transfer_id: TransferId,
        imm_after_write: Option<Vec<(fi_addr_t, u32)>>,
        construct_rdma_op_iter: F,
    ) {
        // Allocate the memory for the context and make it float in the heap.
        // We'll delete the object once the transfer is done.
        // This is because we're creating a self-referential struct.
        let mut context = unsafe { self.objpool_write_op.alloc_uninit() };
        let msg_buf = unsafe { self.objpool_msg.alloc_uninit() };
        let msg_buf = unsafe { (*msg_buf.as_ptr()).assume_init_mut() };
        let msg_buf = unsafe { NonNull::new_unchecked(msg_buf) };

        // Convert the RDMA op to an iterator.
        let rawctx = context.as_ptr() as *mut c_void;
        let rdma_op_iter = construct_rdma_op_iter(rawctx, msg_buf);

        // Initialize the context
        let total_ops = rdma_op_iter.total_ops();
        let context = unsafe {
            context.as_mut().write(WriteOpContext {
                transfer_id,
                rdma_op_iter,
                msg_buf,
                ep: self.ep,
                total_ops,
                cnt_posted_ops: 0,
                cnt_finished_ops: 0,
                in_queue: false,
                bad: false,
                imm_after_write,
            })
        };
        let context_ptr = unsafe { NonNull::new_unchecked(context) };

        // Try to eagerly post the first op if currently there's no pending write ops.
        //
        // NOTE(lequn): max_submit=1 is better than max_submit=32 for the eager posting.
        // I guess this is because on EFA we have mutliple NICs per GPU. So it's better
        // to switch to the next NIC to do its eager posting.
        Self::progress_rdma_write_op_context(context, 1);

        // Add to the pending queue if there are more ops to post.
        if context.cnt_posted_ops != context.total_ops {
            self.write_ops.push_back(context_ptr);
            context.in_queue = true;
        }
    }

    fn do_submit_group_write(
        &mut self,
        transfer_id: TransferId,
        addrs: Rc<Vec<fi_addr_t>>,
        op: GroupWriteOp,
    ) {
        self.do_submit_write(transfer_id, None, |rawctx, msg_buf| match op {
            GroupWriteOp::Scatter(op) => WriteOpIter::Scatter(ScatterWriteOpIter::new(
                op, addrs, msg_buf, rawctx,
            )),
        });
    }

    fn send_imm_after_write(
        &mut self,
        transfer_id: TransferId,
        imm_after_write: Option<Vec<(fi_addr_t, u32)>>,
    ) -> Option<DomainCompletionEntry> {
        let Some(imm_after_write) = imm_after_write else {
            return Some(DomainCompletionEntry::Transfer(transfer_id));
        };
        if imm_after_write.is_empty() {
            return Some(DomainCompletionEntry::Transfer(transfer_id));
        }
        for (dest_addr, imm_data) in imm_after_write {
            self.imm_send_ops.push_back(ImmSendContext {
                transfer_id,
                dest_addr,
                imm: {
                    let mut buf = Box::new([0; IMM_MSG_BYTES]);
                    buf[..4].copy_from_slice(&imm_data.to_ne_bytes());
                    buf
                },
            });
        }
        self.progress_imm_send_ops();
        None
    }

    fn normalize_remote_addr_write_op(&self, mut op: WriteOp) -> WriteOp {
        if self.use_remote_virt_addr {
            return op;
        }
        match &mut op {
            WriteOp::Single(op) => op.dst_ptr = 0,
            WriteOp::Imm(op) => op.dst_ptr = 0,
            WriteOp::Paged(op) => op.dst_ptr = 0,
        }
        op
    }

    fn normalize_remote_addr_group_write_op(
        &self,
        mut op: GroupWriteOp,
    ) -> GroupWriteOp {
        if self.use_remote_virt_addr {
            return op;
        }
        match &mut op {
            GroupWriteOp::Scatter(op) => {
                for dst in Arc::make_mut(&mut op.dsts) {
                    dst.dst_mr.ptr = 0;
                }
            }
        }
        op
    }

    fn progress_rdma_write_op_context(context: &mut WriteOpContext, max_submit: usize) {
        if context.bad {
            return;
        }

        let fi_writemsg =
            unsafe { (*(*context.ep.as_ptr()).rma).writemsg.unwrap_unchecked() };
        let mut cnt_submits = 0;
        loop {
            let (msg, flags) = context.rdma_op_iter.peek();
            if msg.is_null() {
                break;
            }

            let ret = unsafe { fi_writemsg(context.ep.as_ptr(), msg, flags) };
            match ret {
                0 => {
                    context.rdma_op_iter.advance();
                    context.cnt_posted_ops += 1;
                    cnt_submits += 1;
                    if cnt_submits >= max_submit {
                        break;
                    }
                }
                EAGAIN => {
                    // Busy. Break and try again later.
                    break;
                }
                _ => panic!("fi_writemsg returned undocumented error: {}", ret),
            }
        }
    }

    fn maybe_drop_write_op_context(&mut self, mut ptr: NonNull<WriteOpContext>) {
        // There are three ways to finalize a WriteOpContext:
        // 1. All ops completed successfully. Drop from poll_cq when last op is completed.
        // 2. All ops finished posting, but encountered an completion error.
        //    Drop from poll_cq when last posted op is completed.
        // 3. Posted some ops, but encountered an completion error.
        //    Context is still in queue so can't drop from poll_cq.
        //    Next progress_rdma_write_ops removes it from the queue and stops posting.
        //    3a. If all posted ops are completed, drop from progress_rdma_write_ops.
        //    3b. Otherwise, drop from poll_cq when last posted op is completed.
        let context = unsafe { ptr.as_mut() };
        if context.cnt_finished_ops != context.cnt_posted_ops {
            return;
        }
        if context.in_queue {
            return;
        }
        unsafe { self.objpool_msg.free_and_drop(context.msg_buf) };
        unsafe { self.objpool_write_op.free_and_drop(ptr) };
    }

    fn progress_rdma_write_ops(&mut self) {
        while let Some(mut ptr) = self.write_ops.front().cloned() {
            let context = unsafe { ptr.as_mut() };
            assert!(
                context.cnt_finished_ops <= context.cnt_posted_ops,
                "Invariant: context in queue should have more ops to post"
            );

            if context.bad {
                // If there's an error, remove from queue and try to drop.
                context.in_queue = false;
                self.write_ops.pop_front();
                self.maybe_drop_write_op_context(ptr);
                continue;
            }

            // NOTE(lequn): Without limiting max_submit, EFA small packet rate would be lower.
            Self::progress_rdma_write_op_context(context, 32);
            if context.cnt_posted_ops != context.total_ops {
                // More ops to post. Break and try again later.
                break;
            }

            // This transfer is done. Progress the next one.
            self.write_ops.pop_front();
        }
    }

    fn handle_cqe(&mut self, cqe: &fi_cq_data_entry) -> Option<DomainCompletionEntry> {
        if cqe.flags & FI_REMOTE_WRITE as u64 != 0 {
            if self.use_send_for_imm {
                return None;
            }
            let imm = cqe.data as u32;
            return match self.imm_count_map.inc(imm) {
                ImmCountStatus::Vacant => Some(DomainCompletionEntry::ImmData(imm)),
                ImmCountStatus::NotReached => None,
                ImmCountStatus::Reached => {
                    Some(DomainCompletionEntry::ImmCountReached(imm))
                }
            };
        }

        if cqe.flags & FI_WRITE as u64 != 0 {
            let context = unsafe { (cqe.op_context as *mut WriteOpContext).as_mut() }?;
            context.cnt_finished_ops += 1;
            if context.cnt_finished_ops < context.total_ops {
                return None;
            }

            // Transfer is done.
            let transfer_id = context.transfer_id;
            let imm_after_write = context.imm_after_write.take();
            self.maybe_drop_write_op_context(unsafe {
                NonNull::new_unchecked(context)
            });
            return self.send_imm_after_write(transfer_id, imm_after_write);
        }

        if cqe.flags & FI_SEND as u64 != 0 {
            let transfer_id: TransferId = unsafe { transmute(cqe.op_context) };
            if let Some(count) = self.imm_send_pending.get_mut(&transfer_id) {
                *count -= 1;
                if *count == 0 {
                    self.imm_send_pending.remove(&transfer_id);
                    self.imm_send_pending_bufs.retain(|(id, _)| *id != transfer_id);
                    return Some(DomainCompletionEntry::Transfer(transfer_id));
                }
                return None;
            }
            return Some(DomainCompletionEntry::Send(transfer_id));
        }

        if cqe.flags & FI_RECV as u64 != 0 {
            if let Some(idx) = self.imm_recv_slots.iter().position(|slot| {
                std::ptr::eq(slot, cqe.op_context as *const ImmRecvSlot)
            }) {
                let imm = u32::from_ne_bytes(
                    self.imm_recv_slots[idx].buf[..4].try_into().unwrap(),
                );
                if let Err(err) = self.post_imm_recv_slot(idx) {
                    error!(?err, idx, "Failed to repost CXI immediate receive");
                }
                return match self.imm_count_map.inc(imm) {
                    ImmCountStatus::Vacant => Some(DomainCompletionEntry::ImmData(imm)),
                    ImmCountStatus::NotReached => None,
                    ImmCountStatus::Reached => {
                        Some(DomainCompletionEntry::ImmCountReached(imm))
                    }
                };
            }
            let transfer_id: TransferId = unsafe { transmute(cqe.op_context) };
            return Some(DomainCompletionEntry::Recv {
                transfer_id,
                data_len: cqe.len,
            });
        }

        None
    }

    fn poll_cq(&mut self) {
        self.poll_one_cq(self.cq);
        if let Some(imm_cq) = self.imm_cq {
            self.poll_one_cq(imm_cq);
        }
    }

    fn poll_one_cq(&mut self, cq: NonNull<fid_cq>) {
        const READ_COUNT: usize = 16;
        let mut cqes = MaybeUninit::<[fi_cq_data_entry; READ_COUNT]>::uninit();
        loop {
            let ret = unsafe {
                let fi_cq_read = (*(*cq.as_ptr()).ops).read.unwrap_unchecked();
                fi_cq_read(cq.as_ptr(), cqes.as_mut_ptr() as *mut c_void, READ_COUNT)
            };
            if ret > 0 {
                // Process the completions
                let cqes = unsafe { cqes.assume_init() };
                for cqe in cqes.iter().take(ret as usize) {
                    if let Some(c) = self.handle_cqe(cqe) {
                        self.completions.push_back(c);
                    }
                }
            } else if ret == -(FI_EAVAIL as isize) {
                // Check errors
                let mut err_entry = fi_cq_err_entry::default();
                let ret = unsafe {
                    let fi_cq_readerr =
                        (*(*cq.as_ptr()).ops).readerr.unwrap_unchecked();
                    fi_cq_readerr(cq.as_ptr(), &raw mut err_entry, 0)
                };
                if ret > 0 {
                    // RDMA op error.

                    let errmsg = unsafe {
                        let fi_cq_strerror =
                            (*(*cq.as_ptr()).ops).strerror.unwrap_unchecked();
                        CStr::from_ptr(fi_cq_strerror(
                            cq.as_ptr(),
                            err_entry.prov_errno,
                            err_entry.err_data,
                            null_mut(),
                            0,
                        ))
                        .to_string_lossy()
                        .into_owned()
                    };

                    let is_imm_recv = err_entry.flags & FI_RECV as u64 != 0
                        && self.imm_recv_slots.iter().any(|slot| {
                            std::ptr::eq(
                                slot,
                                err_entry.op_context as *const ImmRecvSlot,
                            )
                        });
                    if self.debug_fi {
                        eprintln!(
                            "pplx cq error domain={} flags=0x{:x} err={} prov_errno={} msg={} op_context={:?} is_imm_recv={}",
                            self.info.name(),
                            err_entry.flags,
                            err_entry.err,
                            err_entry.prov_errno,
                            errmsg,
                            err_entry.op_context,
                            is_imm_recv,
                        );
                    }

                    let transfer_id = if err_entry.flags & FI_SEND as u64 != 0 {
                        Some(unsafe {
                            transmute::<*mut c_void, TransferId>(err_entry.op_context)
                        })
                    } else if err_entry.flags & FI_RECV as u64 != 0 && !is_imm_recv {
                        Some(unsafe {
                            transmute::<*mut c_void, TransferId>(err_entry.op_context)
                        })
                    } else if err_entry.flags & FI_WRITE as u64 != 0 {
                        if let Some(context) = unsafe {
                            (err_entry.op_context as *mut WriteOpContext).as_mut()
                        } {
                            context.cnt_finished_ops += 1;
                            let ret = if context.bad {
                                None
                            } else {
                                // Return error to the caller only once.
                                context.bad = true;
                                Some(context.transfer_id)
                            };
                            self.maybe_drop_write_op_context(unsafe {
                                NonNull::new_unchecked(context)
                            });
                            ret
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(transfer_id) = transfer_id {
                        warn!(
                            domain = ?self.info.name(),
                            ?err_entry,
                            msg = errmsg,
                            "Encountered RDMA op error. Send DomainCompletionEntry::Error to the caller."
                        );
                        self.completions.push_back(DomainCompletionEntry::Error(
                            transfer_id,
                            FabricLibError::CompletionError(errmsg),
                        ));
                    } else {
                        error!(
                            domain = ?self.info.name(),
                            ?err_entry,
                            msg = errmsg,
                            "Unhandled RDMA op error."
                        );
                    }

                    return;
                } else {
                    panic!("fi_cq_readerr returned undocumented error: {}", ret);
                }
            } else if ret == -(FI_EAGAIN as isize) {
                // No more completions
                return;
            } else {
                panic!("fi_cq_read returned undocumented error: {}", ret);
            }
        }
    }
}

impl RdmaDomain for EfaDomain {
    type Info = EfaDomainInfo;

    fn open(info: Self::Info, imm_count_map: Arc<ImmCountMap>) -> Result<Self> {
        Self::open(info, imm_count_map)
    }

    fn addr(&self) -> DomainAddress {
        self.addr.clone()
    }

    fn link_speed(&self) -> u64 {
        self.info.link_speed()
    }

    fn register_mr_local(&mut self, region: &MemoryRegion) -> Result<()> {
        self.register_mr(region, false).map(|_| ())
    }

    fn register_mr_allow_remote(
        &mut self,
        region: &MemoryRegion,
    ) -> Result<MemoryRegionRemoteKey> {
        self.register_mr(region, true)
    }

    fn unregister_mr(&mut self, ptr: NonNull<c_void>) {
        if let Some(mut mr) = self.local_mr_map.remove(&ptr) {
            unsafe { fi_close(&raw mut mr.as_mut().fid) };
        }
    }

    fn get_mem_desc(
        &self,
        ptr: NonNull<c_void>,
    ) -> Result<MemoryRegionLocalDescriptor> {
        if !self.use_local_mr_desc {
            return Ok(MemoryRegionLocalDescriptor(0));
        }
        let mr = self
            .local_mr_map
            .get(&ptr)
            .ok_or(FabricLibError::Custom("Local MR not found"))?;
        Ok(EfaMemDesc::from(*mr).into())
    }

    fn submit_recv(&mut self, transfer_id: TransferId, op: RecvOp) {
        self.recv_ops.push_back(RecvOpContext { transfer_id, op });
    }

    fn submit_send(
        &mut self,
        transfer_id: TransferId,
        dest_addr: DomainAddress,
        op: SendOp,
    ) {
        // Resolve the remote address
        let Ok(dest_fi_addr) = self.get_or_add_remote_addr(&dest_addr) else {
            self.completions.push_back(DomainCompletionEntry::Error(
                transfer_id,
                FabricLibError::CompletionError(format!(
                    "Failed to resolve remote address: {}",
                    dest_addr
                )),
            ));
            return;
        };
        if self.use_send_for_imm {
            if let Err(err) = self.get_or_add_imm_remote_addr(&dest_addr) {
                self.completions
                    .push_back(DomainCompletionEntry::Error(transfer_id, err));
                return;
            }
            if let Err(err) = self.ensure_imm_recvs_posted(dest_fi_addr) {
                self.completions
                    .push_back(DomainCompletionEntry::Error(transfer_id, err));
                return;
            }
        }

        // Add to the flying transfer queue
        self.send_ops.push_back(SendOpContext {
            transfer_id,
            dest_addr: dest_fi_addr,
            op,
        });
    }

    fn submit_write(
        &mut self,
        transfer_id: TransferId,
        dest_addr: DomainAddress,
        op: WriteOp,
    ) {
        // Resolve the remote address
        let Ok(dest_fi_addr) = self.get_or_add_remote_addr(&dest_addr) else {
            self.completions.push_back(DomainCompletionEntry::Error(
                transfer_id,
                FabricLibError::CompletionError(format!(
                    "Failed to resolve remote address: {}",
                    dest_addr
                )),
            ));
            return;
        };
        let imm_dest_fi_addr = if self.use_send_for_imm {
            let Ok(imm_dest_fi_addr) = self.get_or_add_imm_remote_addr(&dest_addr)
            else {
                self.completions.push_back(DomainCompletionEntry::Error(
                    transfer_id,
                    FabricLibError::CompletionError(format!(
                        "Failed to resolve remote immediate address: {}",
                        dest_addr
                    )),
                ));
                return;
            };
            if let Err(err) = self.ensure_imm_recvs_posted(dest_fi_addr) {
                self.completions
                    .push_back(DomainCompletionEntry::Error(transfer_id, err));
                return;
            }
            Some(imm_dest_fi_addr)
        } else {
            None
        };

        if self.use_send_for_imm
            && let WriteOp::Imm(op) = op
        {
            self.imm_send_ops.push_back(ImmSendContext {
                transfer_id,
                dest_addr: imm_dest_fi_addr.unwrap(),
                imm: {
                    let mut buf = Box::new([0; IMM_MSG_BYTES]);
                    buf[..4].copy_from_slice(&op.imm_data.to_ne_bytes());
                    buf
                },
            });
            self.progress_imm_send_ops();
            return;
        }

        let mut op = op;
        let mut imm_after_write = None;
        if self.use_send_for_imm {
            match &mut op {
                WriteOp::Single(op) => {
                    if let Some(imm_data) = op.imm_data.take() {
                        imm_after_write =
                            Some(vec![(imm_dest_fi_addr.unwrap(), imm_data)]);
                    }
                }
                WriteOp::Paged(op) => {
                    if let Some(imm_data) = op.imm_data.take() {
                        imm_after_write =
                            Some(vec![(imm_dest_fi_addr.unwrap(), imm_data)]);
                    }
                }
                WriteOp::Imm(_) => {}
            }
        }

        let op = self.normalize_remote_addr_write_op(op);
        self.do_submit_write(
            transfer_id,
            imm_after_write,
            |rawctx, msg_buf| match op {
                WriteOp::Single(op) => WriteOpIter::Single(
                    SingleWriteOpIter::new_single(op, dest_fi_addr, msg_buf, rawctx),
                ),
                WriteOp::Imm(op) => WriteOpIter::Single(SingleWriteOpIter::new_imm(
                    op,
                    dest_fi_addr,
                    msg_buf,
                    rawctx,
                )),
                WriteOp::Paged(op) => WriteOpIter::Paged(PagedWriteOpIter::new(
                    op,
                    dest_fi_addr,
                    msg_buf,
                    rawctx,
                )),
            },
        );
    }

    fn add_peer_group(
        &mut self,
        handle: PeerGroupHandle,
        addrs: Vec<DomainAddress>,
    ) -> Result<()> {
        if self.peer_groups.contains_key(&handle) {
            return Ok(());
        }
        let mut fi_addrs = Vec::with_capacity(addrs.len());
        let mut imm_fi_addrs = Vec::with_capacity(addrs.len());
        for addr in addrs.iter() {
            let fi_addr = self.get_or_add_remote_addr(addr)?;
            fi_addrs.push(fi_addr);
            if self.use_send_for_imm {
                let imm_fi_addr = self.get_or_add_imm_remote_addr(addr)?;
                imm_fi_addrs.push(imm_fi_addr);
            }
        }
        if !self.use_send_for_imm {
            imm_fi_addrs = fi_addrs.clone();
        }
        self.peer_groups.insert(
            handle,
            PeerGroup { addrs: Rc::new(fi_addrs), imm_addrs: Rc::new(imm_fi_addrs) },
        );
        Ok(())
    }

    fn submit_group_write(
        &mut self,
        transfer_id: TransferId,
        handle: Option<PeerGroupHandle>,
        op: GroupWriteOp,
    ) {
        let (addrs, imm_addrs) = if let Some(handle) = handle {
            let Some(peer_group) = self.peer_groups.get_mut(&handle) else {
                self.completions.push_back(DomainCompletionEntry::Error(
                    transfer_id,
                    FabricLibError::Custom("Peer group not found"),
                ));
                return;
            };
            (Rc::clone(&peer_group.addrs), Rc::clone(&peer_group.imm_addrs))
        } else {
            let mut fi_addrs = Vec::with_capacity(op.num_targets());
            let mut imm_fi_addrs = Vec::with_capacity(op.num_targets());
            for addr in op.peer_addr_iter() {
                let Ok(fi_addr) = self.get_or_add_remote_addr(addr) else {
                    self.completions.push_back(DomainCompletionEntry::Error(
                        transfer_id,
                        FabricLibError::CompletionError(format!(
                            "Failed to resolve remote address: {}",
                            addr
                        )),
                    ));
                    return;
                };
                fi_addrs.push(fi_addr);
                if self.use_send_for_imm {
                    let Ok(imm_fi_addr) = self.get_or_add_imm_remote_addr(addr) else {
                        self.completions.push_back(DomainCompletionEntry::Error(
                            transfer_id,
                            FabricLibError::CompletionError(format!(
                                "Failed to resolve remote immediate address: {}",
                                addr
                            )),
                        ));
                        return;
                    };
                    imm_fi_addrs.push(imm_fi_addr);
                }
            }
            if !self.use_send_for_imm {
                imm_fi_addrs = fi_addrs.clone();
            }
            (Rc::new(fi_addrs), Rc::new(imm_fi_addrs))
        };
        if self.use_send_for_imm {
            let has_imm = match &op {
                GroupWriteOp::Scatter(op) => op.imm_data.is_some(),
            };
            if has_imm && let Some(addr) = addrs.first() {
                if let Err(err) = self.ensure_imm_recvs_posted(*addr) {
                    self.completions
                        .push_back(DomainCompletionEntry::Error(transfer_id, err));
                    return;
                }
            }
        }
        let mut imm_after_write = None;
        let mut op = op;
        if self.use_send_for_imm {
            match &mut op {
                GroupWriteOp::Scatter(op) => {
                    if let Some(imm_data) = op.imm_data.take() {
                        imm_after_write = Some(
                            imm_addrs
                                .iter()
                                .copied()
                                .skip(op.dst_beg)
                                .take(op.dst_end - op.dst_beg)
                                .map(|addr| (addr, imm_data))
                                .collect(),
                        );
                    }
                }
            }
        }
        let op = self.normalize_remote_addr_group_write_op(op);
        self.do_submit_write(
            transfer_id,
            imm_after_write,
            |rawctx, msg_buf| match op {
                GroupWriteOp::Scatter(op) => WriteOpIter::Scatter(
                    ScatterWriteOpIter::new(op, addrs, msg_buf, rawctx),
                ),
            },
        );
    }

    fn poll_progress(&mut self) {
        self.progress_ops();
        self.poll_cq();
    }

    fn get_completion(&mut self) -> Option<DomainCompletionEntry> {
        self.completions.pop_front()
    }
}

impl Drop for EfaDomain {
    fn drop(&mut self) {
        debug!("Domain::drop. name: {}", self.info.name());
        unsafe {
            for (_, mut mr) in self.local_mr_map.drain() {
                fi_close(&raw mut mr.as_mut().fid);
            }
            if let Some(mut imm_ep) = self.imm_ep {
                fi_close(&raw mut imm_ep.as_mut().fid);
            }
            if let Some(mut imm_av) = self.imm_av {
                fi_close(&raw mut imm_av.as_mut().fid);
            }
            if let Some(mut imm_cq) = self.imm_cq {
                fi_close(&raw mut imm_cq.as_mut().fid);
            }
            fi_close(&raw mut self.ep.as_mut().fid);
            fi_close(&raw mut self.av.as_mut().fid);
            fi_close(&raw mut self.cq.as_mut().fid);
            fi_close(&raw mut self.domain.as_mut().fid);
            fi_close(&raw mut self.fabric.as_mut().fid);
        }
    }
}
