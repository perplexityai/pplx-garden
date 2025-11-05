use std::{
    collections::{HashMap, VecDeque, hash_map::Entry},
    ffi::{CStr, c_void},
    mem::{MaybeUninit, transmute},
    ptr::{NonNull, null_mut},
    rc::Rc,
    sync::Arc,
};

use bytes::Bytes;
use libfabric_sys::{
    FI_ADDR_UNSPEC, FI_CQ_FORMAT_DATA, FI_EAGAIN, FI_EAVAIL, FI_ENABLE, FI_HMEM_CUDA,
    FI_MR_DMABUF, FI_OPT_CUDA_API_PERMITTED, FI_OPT_ENDPOINT,
    FI_OPT_SHARED_MEMORY_PERMITTED, FI_READ, FI_RECV, FI_REMOTE_READ, FI_REMOTE_WRITE,
    FI_SEND, FI_WRITE, fi_addr_t, fi_av_attr, fi_close, fi_cq_attr, fi_cq_data_entry,
    fi_cq_err_entry, fi_fabric, fi_mr_attr, fi_mr_dmabuf, fid_av, fid_cq, fid_domain,
    fid_ep, fid_fabric, fid_mr, iovec,
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
    addr: DomainAddress,

    peer_addr_map: HashMap<DomainAddress, fi_addr_t>,
    peer_groups: HashMap<PeerGroupHandle, PeerGroup>,
    local_mr_map: HashMap<NonNull<c_void>, NonNull<fid_mr>>,
    imm_count_map: Arc<ImmCountMap>,
    objpool_write_op: ObjectPool<WriteOpContext>,
    objpool_msg: ObjectPool<RmaBuffer>,

    recv_ops: VecDeque<RecvOpContext>,
    send_ops: VecDeque<SendOpContext>,
    write_ops: VecDeque<NonNull<WriteOpContext>>,

    completions: VecDeque<DomainCompletionEntry>,
}

struct PeerGroup {
    addrs: Rc<Vec<fi_addr_t>>,
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
}

const EAGAIN: isize = -(FI_EAGAIN as isize);

impl EfaDomain {
    fn open(info: EfaDomainInfo, imm_count_map: Arc<ImmCountMap>) -> Result<Self> {
        unsafe {
            debug!("Domain::open: name: {}", info.name());
            let fi = info.fi();

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

            // Enable endpoint
            let fi_control = (*(*ep_fid).ops).control.unwrap_unchecked();
            let ret = fi_control(ep_fid, FI_ENABLE as i32, null_mut());
            if ret != 0 {
                return Err(LibfabricError::new(ret, "fi_enable").into());
            }

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
            let addr = DomainAddress(Bytes::copy_from_slice(&addrbuf[..addrlen]));

            // Cancel all defer
            defer_fabric.cancel();
            defer_domain.cancel();
            defer_cq.cancel();
            defer_av.cancel();
            defer_ep.cancel();

            Ok(Self {
                info,
                fabric,
                domain,
                ep,
                cq,
                av,
                addr,

                peer_addr_map: HashMap::new(),
                peer_groups: HashMap::new(),
                local_mr_map: HashMap::new(),
                objpool_write_op: ObjectPool::with_chunk_size(1024),
                objpool_msg: ObjectPool::with_chunk_size(1024),
                imm_count_map,

                recv_ops: VecDeque::new(),
                send_ops: VecDeque::new(),
                write_ops: VecDeque::new(),

                completions: VecDeque::new(),
            })
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
                let ret = fi_av_insert(
                    self.av.as_ptr(),
                    peer_addr.0.as_ptr() as *const c_void,
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
            Mapping::Device { device_id, dmabuf_fd: Some(dmabuf_fd) } => {
                mr_attr.iface = FI_HMEM_CUDA;
                mr_attr.device.cuda = device_id.0 as i32;
                dmabuf.fd = *dmabuf_fd;
                mr_attr.__bindgen_anon_1.dmabuf = &dmabuf;
                flags = FI_MR_DMABUF;
            }
        }

        let ret = unsafe {
            let fi_mr_regattr =
                (*(*self.domain.as_ptr()).mr).regattr.unwrap_unchecked();
            let domain_fid = &raw mut (*self.domain.as_ptr()).fid;
            fi_mr_regattr(domain_fid, &mr_attr, flags, &raw mut mr)
        };

        let mr = NonNull::new(mr)
            .ok_or_else(|| LibfabricError::new(ret, "fi_mr_regattr"))?;
        self.local_mr_map.insert(region.ptr(), mr);
        Ok(MemoryRegionRemoteKey(unsafe { mr.as_ref() }.key))
    }

    fn progress_ops(&mut self) {
        self.progress_rdma_recv_ops();
        self.progress_rdma_send_ops();
        self.progress_rdma_write_ops();
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
        self.do_submit_write(transfer_id, |rawctx, msg_buf| match op {
            GroupWriteOp::Scatter(op) => WriteOpIter::Scatter(ScatterWriteOpIter::new(
                op, addrs, msg_buf, rawctx,
            )),
        });
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
            self.maybe_drop_write_op_context(unsafe {
                NonNull::new_unchecked(context)
            });
            return Some(DomainCompletionEntry::Transfer(transfer_id));
        }

        if cqe.flags & FI_SEND as u64 != 0 {
            let transfer_id: TransferId = unsafe { transmute(cqe.op_context) };
            return Some(DomainCompletionEntry::Send(transfer_id));
        }

        if cqe.flags & FI_RECV as u64 != 0 {
            let transfer_id: TransferId = unsafe { transmute(cqe.op_context) };
            return Some(DomainCompletionEntry::Recv {
                transfer_id,
                data_len: cqe.len,
            });
        }

        None
    }

    fn poll_cq(&mut self) {
        const READ_COUNT: usize = 16;
        let mut cqes = MaybeUninit::<[fi_cq_data_entry; READ_COUNT]>::uninit();
        loop {
            let ret = unsafe {
                let fi_cq_read = (*(*self.cq.as_ptr()).ops).read.unwrap_unchecked();
                fi_cq_read(
                    self.cq.as_ptr(),
                    cqes.as_mut_ptr() as *mut c_void,
                    READ_COUNT,
                )
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
                        (*(*self.cq.as_ptr()).ops).readerr.unwrap_unchecked();
                    fi_cq_readerr(self.cq.as_ptr(), &raw mut err_entry, 0)
                };
                if ret > 0 {
                    // RDMA op error.

                    let errmsg = unsafe {
                        let fi_cq_strerror =
                            (*(*self.cq.as_ptr()).ops).strerror.unwrap_unchecked();
                        CStr::from_ptr(fi_cq_strerror(
                            self.cq.as_ptr(),
                            err_entry.prov_errno,
                            err_entry.err_data,
                            null_mut(),
                            0,
                        ))
                        .to_string_lossy()
                        .into_owned()
                    };

                    #[allow(clippy::if_same_then_else)]
                    let transfer_id = if err_entry.flags & FI_SEND as u64 != 0 {
                        Some(unsafe {
                            transmute::<*mut c_void, TransferId>(err_entry.op_context)
                        })
                    } else if err_entry.flags & FI_RECV as u64 != 0 {
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

        self.do_submit_write(transfer_id, |rawctx, msg_buf| match op {
            WriteOp::Single(op) => WriteOpIter::Single(SingleWriteOpIter::new_single(
                op,
                dest_fi_addr,
                msg_buf,
                rawctx,
            )),
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
        });
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
        for addr in addrs.iter() {
            let fi_addr = self.get_or_add_remote_addr(addr)?;
            fi_addrs.push(fi_addr);
        }
        self.peer_groups.insert(handle, PeerGroup { addrs: Rc::new(fi_addrs) });
        Ok(())
    }

    fn submit_group_write(
        &mut self,
        transfer_id: TransferId,
        handle: Option<PeerGroupHandle>,
        op: GroupWriteOp,
    ) {
        let addrs = if let Some(handle) = handle {
            let Some(peer_group) = self.peer_groups.get_mut(&handle) else {
                self.completions.push_back(DomainCompletionEntry::Error(
                    transfer_id,
                    FabricLibError::Custom("Peer group not found"),
                ));
                return;
            };
            Rc::clone(&peer_group.addrs)
        } else {
            let mut fi_addrs = Vec::with_capacity(op.num_targets());
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
            }
            Rc::new(fi_addrs)
        };
        self.do_submit_group_write(transfer_id, addrs, op);
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
            fi_close(&raw mut self.ep.as_mut().fid);
            fi_close(&raw mut self.av.as_mut().fid);
            fi_close(&raw mut self.cq.as_mut().fid);
            fi_close(&raw mut self.domain.as_mut().fid);
            fi_close(&raw mut self.fabric.as_mut().fid);
        }
    }
}
