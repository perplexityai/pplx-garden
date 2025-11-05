use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::{CStr, c_void},
    mem::{MaybeUninit, transmute},
    ptr::{NonNull, null_mut},
    rc::Rc,
    sync::Arc,
};

use cuda_lib::Device;
use libc::ENOMEM;
use libibverbs_sys::{
    IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_RELAXED_ORDERING, IBV_ACCESS_REMOTE_READ,
    IBV_ACCESS_REMOTE_WRITE, IBV_LINK_LAYER_INFINIBAND, IBV_SEND_SIGNALED,
    IBV_WC_RDMA_WRITE, IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM, IBV_WC_SEND,
    IBV_WC_SUCCESS, IBV_WR_SEND, ibv_ah, ibv_ah_attr, ibv_alloc_parent_domain,
    ibv_alloc_pd, ibv_alloc_td, ibv_close_device, ibv_context, ibv_cq, ibv_create_ah,
    ibv_create_cq, ibv_create_srq, ibv_dealloc_pd, ibv_dealloc_td, ibv_dereg_mr,
    ibv_destroy_ah, ibv_destroy_cq, ibv_destroy_srq, ibv_gid, ibv_global_route, ibv_mr,
    ibv_mtu, ibv_open_device, ibv_parent_domain_init_attr, ibv_pd, ibv_poll_cq,
    ibv_port_attr, ibv_post_recv, ibv_post_send, ibv_post_srq_recv, ibv_qp,
    ibv_query_gid, ibv_query_port_wrap, ibv_recv_wr, ibv_reg_dmabuf_mr, ibv_reg_mr,
    ibv_send_wr, ibv_sge, ibv_srq, ibv_srq_attr, ibv_srq_init_attr, ibv_td,
    ibv_td_init_attr, ibv_wc, ibv_wc_status_str,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

const MAX_OPS: usize = 1024;
const CQ_DEPTH: usize = 4096;
const NUM_IMM_RECVS: usize = 128;
const GRH_BYTES: usize = 40;
const MAX_UD_MSG_BYTES: usize = 4096;
const NUM_UD_RECVS: usize = 128;
const MAX_UD_SENDS: usize = 128;

use crate::{
    api::{DomainAddress, MemoryRegionRemoteKey, PeerGroupHandle, TransferId},
    error::{FabricLibError, Result, VerbsError},
    imm_count::{ImmCountMap, ImmCountStatus},
    mr::{Mapping, MemoryRegion, MemoryRegionLocalDescriptor},
    provider::{DomainCompletionEntry, RdmaDomain, RdmaDomainInfo},
    rdma_op::{GroupWriteOp, RecvOp, SendOp, WriteOp},
    utils::{defer::Defer, memory::MemoryPool, obj_pool::ObjectPool},
    verbs::{
        verbs_address::{Gid, VerbsUDAddress},
        verbs_devinfo::VerbsDeviceInfo,
        verbs_qp::{RCQueuePair, UDQueuePair},
        verbs_rdma_op::{
            PagedWriteOpIter, ScatterWriteOpIter, SingleWriteOpIter, WrChainBuffer,
            WriteOpIter, fill_recv_op, fill_send_op,
        },
    },
};

pub struct VerbsDomain {
    name: String,
    port_num: u8,
    gid_index: u8,

    context: NonNull<ibv_context>,
    lid: u16,
    gid: Gid,
    mtu: ibv_mtu,
    is_infiniband: bool,
    link_speed: u64,

    td: NonNull<ibv_td>,
    mt_pd: NonNull<ibv_pd>,
    pd: NonNull<ibv_pd>,
    cq: NonNull<ibv_cq>,
    msg_srq: NonNull<ibv_srq>,
    rma_srq: NonNull<ibv_srq>,

    // TODO: av_map
    // TODO: peer_qp_map
    ud: UDQueuePair,
    addr: DomainAddress,
    peers: HashMap<DomainAddress, Peer>,
    connecting_peer_groups: HashMap<PeerGroupHandle, ConnectingPeerGroup>,
    peer_groups: HashMap<PeerGroupHandle, PeerGroup>,
    local_mr_map: HashMap<NonNull<c_void>, NonNull<ibv_mr>>,
    imm_count_map: Arc<ImmCountMap>,
    objpool_write_op: ObjectPool<WriteOpContext>,
    objpool_wr: ObjectPool<WrChainBuffer>,
    objpool_pending_group_write_op: ObjectPool<PendingGroupWriteOp>,

    recv_ops: VecDeque<RecvOpContext>,
    send_ops: VecDeque<SendOpContext>,
    write_ops: VecDeque<NonNull<WriteOpContext>>,
    completions: VecDeque<DomainCompletionEntry>,

    ud_mempool: MemoryPool,
    ud_mempool_lkey: MemoryRegionLocalDescriptor,
}

struct Peer {
    // TODO: Graceful disconnect RC
    // TODO: Remove from peers map when disconnected
    // TODO: Remove from collectives map when disconnected
    ud_addr: VerbsUDAddress,
    ah: NonNull<ibv_ah>,
    msg_rc: RCQueuePair,
    rma_rc: RCQueuePair,
    state: PeerState,
}

enum PeerState {
    Connecting {
        pending_submits: Vec<(TransferId, OutboundOp)>,
        pending_group_write_ops: Vec<NonNull<PendingGroupWriteOp>>,
    },
    Established,
}

/// A GroupWriteOp (without PeerGroupHandle) that is waiting for all peers to connect.
struct PendingGroupWriteOp {
    transfer_id: TransferId,
    op: Option<GroupWriteOp>,
    rma_qps: Vec<NonNull<ibv_qp>>,
    pending_peers: HashSet<NonNull<ibv_qp>>,
}

enum OutboundOp {
    Send(SendOp),
    Write(WriteOp),
}

struct PeerGroup {
    rma_qps: Rc<Vec<NonNull<ibv_qp>>>,
}

struct ConnectingPeerGroup {
    rma_qps: Vec<NonNull<ibv_qp>>,
    pending_peers: HashSet<NonNull<ibv_qp>>,
    pending_group_ops: Vec<(TransferId, GroupWriteOp)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerHandshakeInfo {
    ud_addr: VerbsUDAddress,
    lid: u16,
    gid: Gid,
    msg_qp_num: u32,
    msg_psn: u32,
    rma_qp_num: u32,
    rma_psn: u32,
}

struct RecvOpContext {
    transfer_id: TransferId,
    op: RecvOp,
}

struct SendOpContext {
    transfer_id: TransferId,
    msg_qp: NonNull<ibv_qp>,
    op: SendOp,
}

struct WriteOpContext {
    transfer_id: TransferId,
    rdma_op_iter: WriteOpIter,
    wr_chain_buffer: NonNull<WrChainBuffer>,
    total_ops: usize,
    cnt_posted_ops: usize,
    cnt_finished_ops: usize,
    in_queue: bool,

    /// True when there's a completion error.
    /// No more write ops will be posted.
    /// Once cnt_finished_ops catches up with cnt_posted_ops, the context will be dropped.
    bad: bool,
}

impl VerbsDomain {
    fn open(info: VerbsDeviceInfo, imm_count_map: Arc<ImmCountMap>) -> Result<Self> {
        unsafe {
            let device = info.device();
            let name = info.name().into_owned();
            debug!(domain = name, "VerbsDomain::open");

            // Context
            let context = NonNull::new(ibv_open_device(device))
                .ok_or_else(|| VerbsError::with_last_os_error("ibv_open_device"))?;
            let mut defer_context = Defer::new(|| {
                ibv_close_device(context.as_ptr());
            });

            // Query Port
            let mut port_attr = ibv_port_attr::default();
            let errno = ibv_query_port_wrap(
                context.as_ptr(),
                info.port_num,
                &raw mut port_attr,
            );
            if errno != 0 {
                return Err(VerbsError::with_code(errno, "ibv_query_port_wrap").into());
            }
            let lid = port_attr.lid;
            let mtu = port_attr.active_mtu;
            let is_infiniband = port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND as u8;
            let mut gid = ibv_gid::default();
            let errno = ibv_query_gid(
                context.as_ptr(),
                info.port_num,
                info.gid_index as i32,
                &raw mut gid,
            );
            if errno != 0 {
                return Err(VerbsError::with_code(errno, "ibv_query_gid").into());
            }
            let gid = Gid { raw: gid.raw };
            // TODO: p_key partition key

            // Thread domain to remove locks now that we are single-threaded.
            let mut td_init_attr = ibv_td_init_attr { comp_mask: 0 };
            let td =
                NonNull::new(ibv_alloc_td(context.as_ptr(), &raw mut td_init_attr))
                    .ok_or_else(|| VerbsError::with_last_os_error("ibv_alloc_td"))?;
            let mut defer_td = Defer::new(|| {
                ibv_dealloc_td(td.as_ptr());
            });

            // Protection Domain (Multi-threaded)
            let mt_pd = NonNull::new(ibv_alloc_pd(context.as_ptr()))
                .ok_or_else(|| VerbsError::with_last_os_error("ibv_alloc_pd"))?;
            let mut defer_mt_pd = Defer::new(|| {
                ibv_dealloc_pd(mt_pd.as_ptr());
            });

            // Protection Domain (Single-threaded)
            let mut parent_attr = ibv_parent_domain_init_attr {
                pd: mt_pd.as_ptr(),
                td: td.as_ptr(),
                comp_mask: 0,
                ..Default::default()
            };
            let pd = NonNull::new(ibv_alloc_parent_domain(
                context.as_ptr(),
                &raw mut parent_attr,
            ))
            .ok_or_else(|| VerbsError::with_last_os_error("ibv_alloc_parent_domain"))?;
            let mut defer_pd = Defer::new(|| {
                ibv_dealloc_pd(pd.as_ptr());
            });

            // Completion Queue
            let cq = NonNull::new(ibv_create_cq(
                context.as_ptr(),
                CQ_DEPTH as i32,
                null_mut(),
                null_mut(),
                0,
            ))
            .ok_or_else(|| VerbsError::with_last_os_error("ibv_create_cq"))?;
            let mut defer_cq = Defer::new(|| {
                ibv_destroy_cq(cq.as_ptr());
            });

            // Shared Receive Queue
            let mut srq_init_attr = ibv_srq_init_attr {
                srq_context: null_mut(),
                attr: ibv_srq_attr {
                    max_wr: MAX_OPS as u32,
                    max_sge: 1,
                    ..Default::default()
                },
            };
            let msg_srq =
                NonNull::new(ibv_create_srq(pd.as_ptr(), &raw mut srq_init_attr))
                    .ok_or_else(|| VerbsError::with_last_os_error("ibv_create_srq"))?;
            let mut defer_msg_srq = Defer::new(|| {
                ibv_destroy_srq(msg_srq.as_ptr());
            });
            let rma_srq =
                NonNull::new(ibv_create_srq(pd.as_ptr(), &raw mut srq_init_attr))
                    .ok_or_else(|| VerbsError::with_last_os_error("ibv_create_srq"))?;
            let mut defer_rma_srq = Defer::new(|| {
                ibv_destroy_srq(rma_srq.as_ptr());
            });

            // Create UD QP for exchanging RC handshake info.
            let qkey = 0x11111111; // TODO: randomize qkey
            let ud = UDQueuePair::new(cq, pd, gid, lid, qkey, MAX_OPS as u32)?;

            // Activate UD QP
            let pkey_index = 0; // TODO: get pkey_index
            ud.ud_reset_to_init(pkey_index, info.port_num)?;
            ud.ud_init_to_rtr()?;
            ud.ud_rtr_to_rts()?;
            let addr = DomainAddress::from(&ud.addr);

            // Cancel all defer
            defer_context.cancel();
            defer_td.cancel();
            defer_mt_pd.cancel();
            defer_pd.cancel();
            defer_cq.cancel();
            defer_msg_srq.cancel();
            defer_rma_srq.cancel();

            let ud_mempool =
                MemoryPool::new(MAX_UD_MSG_BYTES, NUM_UD_RECVS + MAX_UD_SENDS);
            let mut domain = Self {
                name,
                port_num: info.port_num,
                gid_index: info.gid_index,

                context,
                lid,
                gid,
                mtu,
                is_infiniband,
                link_speed: info.link_speed(),

                td,
                mt_pd,
                pd,
                cq,
                msg_srq,
                rma_srq,

                ud,
                addr,
                peers: HashMap::new(),
                connecting_peer_groups: HashMap::new(),
                peer_groups: HashMap::new(),
                local_mr_map: HashMap::new(),
                imm_count_map,
                objpool_write_op: ObjectPool::with_chunk_size(MAX_OPS),
                objpool_wr: ObjectPool::with_chunk_size(MAX_OPS),
                objpool_pending_group_write_op: ObjectPool::with_chunk_size(MAX_OPS),

                recv_ops: VecDeque::with_capacity(MAX_OPS),
                send_ops: VecDeque::with_capacity(MAX_OPS),
                write_ops: VecDeque::with_capacity(MAX_OPS),
                completions: VecDeque::with_capacity(MAX_OPS),

                ud_mempool,
                ud_mempool_lkey: MemoryRegionLocalDescriptor(0), // Set in post_init
            };

            domain.post_init()?;
            Ok(domain)
        }
    }

    fn post_init(&mut self) -> Result<()> {
        // Register internal MR
        let region = MemoryRegion::new(
            self.ud_mempool.buffer_ptr().cast::<c_void>(),
            self.ud_mempool.buffer_len(),
            Device::Host,
        )?;
        self.register_mr(&region, true)?;
        let lkey = self.get_mem_desc(self.ud_mempool.buffer_ptr().cast::<c_void>())?;
        self.ud_mempool_lkey = lkey;

        // Post RECV for UD QP
        for _ in 0..NUM_UD_RECVS {
            let ptr = unsafe { self.ud_mempool.alloc().unwrap_unchecked() };
            self.post_ud_recv(ptr)?;
        }

        // Post RECV for IMM
        for _ in 0..NUM_IMM_RECVS {
            self.post_imm_recv()?;
        }

        Ok(())
    }

    fn post_ud_recv(&mut self, buf: NonNull<u8>) -> Result<()> {
        let mut sge = ibv_sge {
            addr: buf.as_ptr() as u64,
            length: self.ud_mempool.chunk_size() as u32,
            lkey: self.ud_mempool_lkey.0 as u32,
        };
        let mut wr = ibv_recv_wr {
            wr_id: buf.as_ptr() as u64,
            next: null_mut(),
            sg_list: &raw mut sge,
            num_sge: 1,
        };
        let ec = unsafe { ibv_post_recv(self.ud.qp.as_ptr(), &raw mut wr, null_mut()) };
        if ec != 0 {
            return Err(VerbsError::with_code(ec, "ibv_post_recv: UD RECV").into());
        }
        Ok(())
    }

    fn post_imm_recv(&mut self) -> Result<()> {
        let mut wr =
            ibv_recv_wr { wr_id: 0, next: null_mut(), sg_list: null_mut(), num_sge: 0 };
        let ec = unsafe {
            ibv_post_srq_recv(self.rma_srq.as_ptr(), &raw mut wr, null_mut())
        };
        if ec != 0 {
            return Err(VerbsError::with_code(ec, "ibv_post_srq_recv: IMM RECV").into());
        }
        Ok(())
    }

    fn ud_send<T: Serialize + std::fmt::Debug>(
        &self,
        peer_ud_addr: &VerbsUDAddress,
        peer_ah: NonNull<ibv_ah>,
        buf: NonNull<u8>,
        msg: &T,
    ) -> Result<()> {
        debug!(domain = self.name, ?peer_ud_addr, ?msg, "ud_send");
        let length = postcard::to_slice(msg, unsafe {
            std::slice::from_raw_parts_mut(buf.as_ptr(), self.ud_mempool.chunk_size())
        })
        .map_err(|_| FabricLibError::Custom("Failed to serialize UD message"))?
        .len();

        let mut sge = ibv_sge {
            addr: buf.as_ptr() as u64,
            length: length as u32,
            lkey: self.ud_mempool_lkey.0 as u32,
        };
        let mut wr = ibv_send_wr {
            wr_id: buf.as_ptr() as u64,
            next: null_mut(),
            sg_list: &raw mut sge,
            num_sge: 1,
            opcode: IBV_WR_SEND,
            send_flags: IBV_SEND_SIGNALED,
            ..Default::default()
        };
        wr.wr.ud.ah = peer_ah.as_ptr();
        wr.wr.ud.remote_qpn = peer_ud_addr.qp_num;
        wr.wr.ud.remote_qkey = peer_ud_addr.qkey;

        let ec = unsafe { ibv_post_send(self.ud.qp.as_ptr(), &raw mut wr, null_mut()) };
        if ec != 0 {
            return Err(VerbsError::with_code(ec, "ibv_post_send: UD SEND").into());
        }
        Ok(())
    }

    fn create_peer(
        &self,
        peer_addr: &DomainAddress,
        pending_submits: Vec<(TransferId, OutboundOp)>,
        pending_group_write_ops: Vec<NonNull<PendingGroupWriteOp>>,
    ) -> Result<Peer> {
        let peer_ud_addr = VerbsUDAddress::from_bytes(&peer_addr.0)
            .ok_or(FabricLibError::Custom("Invalid peer address"))?;

        // Create QP
        let msg_psn = 123; // TODO: randomize psn
        let rma_psn = 456; // TODO: randomize psn
        let msg_rc = RCQueuePair::new(
            self.cq,
            self.pd,
            self.msg_srq,
            self.gid,
            self.lid,
            MAX_OPS as u32,
            msg_psn,
        )?;
        let rma_rc = RCQueuePair::new(
            self.cq,
            self.pd,
            self.rma_srq,
            self.gid,
            self.lid,
            MAX_OPS as u32,
            rma_psn,
        )?;

        // Create Address Handle
        let mut ah_attr = ibv_ah_attr {
            grh: ibv_global_route {
                hop_limit: 64,
                dgid: peer_ud_addr.gid.into(),
                sgid_index: self.gid_index,
                ..Default::default()
            },
            dlid: peer_ud_addr.lid,
            is_global: if self.is_infiniband { 0 } else { 1 },
            port_num: self.port_num,
            ..Default::default()
        };
        let ah =
            NonNull::new(unsafe { ibv_create_ah(self.pd.as_ptr(), &raw mut ah_attr) })
                .ok_or_else(|| VerbsError::with_last_os_error("ibv_create_ah"))?;

        Ok(Peer {
            ud_addr: peer_ud_addr.clone(),
            ah,
            msg_rc,
            rma_rc,
            state: PeerState::Connecting { pending_submits, pending_group_write_ops },
        })
    }

    fn connect_peer(&self, peer: &Peer, ud_buf: NonNull<u8>) -> Result<()> {
        debug!(domain = self.name, ?peer.ud_addr, "connect_peer");

        // Send handshake to peer
        let handshake_info = PeerHandshakeInfo {
            ud_addr: self.ud.addr.clone(),
            lid: self.lid,
            gid: self.gid,
            msg_qp_num: peer.msg_rc.addr.qp_num,
            msg_psn: peer.msg_rc.addr.psn,
            rma_qp_num: peer.rma_rc.addr.qp_num,
            rma_psn: peer.rma_rc.addr.psn,
        };
        self.ud_send(&peer.ud_addr, peer.ah, ud_buf, &handshake_info)?;

        Ok(())
    }

    fn handle_peer_handshake(&mut self, info: &PeerHandshakeInfo) -> Result<()> {
        debug!(domain = self.name, ?info, "handle_peer_handshake");
        let peer_addr = DomainAddress::from(&info.ud_addr);

        let peer = if let Some(peer) = self.peers.get_mut(&peer_addr) {
            if let PeerState::Established = peer.state {
                return Ok(());
            }
            peer
        } else {
            // TODO: unify the code with do_submit
            let peer = self.create_peer(&peer_addr, vec![], vec![])?;
            self.peers.insert(peer_addr.clone(), peer);
            let peer = unsafe { self.peers.get(&peer_addr).unwrap_unchecked() };
            let buf = unsafe { self.ud_mempool.alloc() }.ok_or(
                FabricLibError::Custom("Failed to allocate UD message buffer"),
            )?;
            self.connect_peer(peer, buf)?;
            unsafe { self.peers.get_mut(&peer_addr).unwrap_unchecked() }
        };

        // Activate QP

        let pkey_index = 0; // TODO: get pkey_index
        peer.msg_rc.rc_reset_to_init(self.port_num, pkey_index)?;
        peer.rma_rc.rc_reset_to_init(self.port_num, pkey_index)?;

        peer.msg_rc.rc_init_to_rtr(
            self.is_infiniband,
            self.gid_index,
            self.port_num,
            self.mtu,
            info.msg_qp_num,
            info.msg_psn,
            info.lid,
            info.gid,
        )?;
        peer.rma_rc.rc_init_to_rtr(
            self.is_infiniband,
            self.gid_index,
            self.port_num,
            self.mtu,
            info.rma_qp_num,
            info.rma_psn,
            info.lid,
            info.gid,
        )?;

        peer.msg_rc.rc_rtr_to_rts(info.msg_psn)?;
        peer.rma_rc.rc_rtr_to_rts(info.rma_psn)?;
        // TODO: return error for pending submits if fails to connect.
        debug!(domain = self.name, ?peer_addr, "RC handshake completed");

        // Transition to established state
        let (pending_submits, pending_group_write_ops) = match &mut peer.state {
            PeerState::Connecting { pending_submits, pending_group_write_ops } => (
                std::mem::take(pending_submits),
                std::mem::take(pending_group_write_ops),
            ),
            PeerState::Established => {
                unreachable!("Peer should not be in established state")
            }
        };
        peer.state = PeerState::Established;

        // Submit pending submits
        let msg_qp = peer.msg_rc.qp;
        let rma_qp = peer.rma_rc.qp;
        for (transfer_id, op) in pending_submits {
            self.do_submit_outbound_op(transfer_id, op, msg_qp, rma_qp);
        }

        // Check peer group
        let mut connected_peer_groups = Vec::new();
        for (handle, group) in self.connecting_peer_groups.iter_mut() {
            if group.pending_peers.remove(&rma_qp) && group.pending_peers.is_empty() {
                connected_peer_groups.push(*handle);
            }
        }
        for handle in connected_peer_groups {
            let group = unsafe {
                self.connecting_peer_groups.remove(&handle).unwrap_unchecked()
            };
            let rma_qps = Rc::new(group.rma_qps);
            self.peer_groups.insert(handle, PeerGroup { rma_qps: Rc::clone(&rma_qps) });
            for (transfer_id, op) in group.pending_group_ops {
                self.do_submit_group_write(transfer_id, Rc::clone(&rma_qps), op);
            }
        }

        // Check pending group write ops
        for mut op_ptr in pending_group_write_ops {
            let op = unsafe { op_ptr.as_mut() };
            if op.pending_peers.remove(&rma_qp) && op.pending_peers.is_empty() {
                let rma_qps = Rc::new(std::mem::take(&mut op.rma_qps));
                let transfer_id = op.transfer_id;
                if let Some(op) = op.op.take() {
                    // It's always Some.
                    self.do_submit_group_write(transfer_id, rma_qps, op);
                }
                unsafe { self.objpool_pending_group_write_op.free_and_drop(op_ptr) };
            }
        }

        Ok(())
    }

    fn register_mr(
        &mut self,
        region: &MemoryRegion,
        allow_remote: bool,
    ) -> Result<MemoryRegionRemoteKey> {
        // TODO: new type for rkey
        if let Some(mr) = self.local_mr_map.get(&region.ptr()) {
            return Ok(MemoryRegionRemoteKey(unsafe { mr.as_ref() }.rkey as u64));
        }

        // NOTE(lequn): Need to set IBV_ACCESS_RELAXED_ORDERING otherwise it's under 200 Gbps.
        let mut access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
        if allow_remote {
            access |= IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        }

        let (mr, fname) = match region.mapping() {
            Mapping::Host | Mapping::Device { dmabuf_fd: None, .. } => unsafe {
                let mr = ibv_reg_mr(
                    self.pd.as_ptr(),
                    region.ptr().as_ptr(),
                    region.len(),
                    access as i32,
                );
                (mr, "ibv_reg_mr")
            },
            Mapping::Device { dmabuf_fd: Some(dmabuf_fd), .. } => unsafe {
                let mr = ibv_reg_dmabuf_mr(
                    self.pd.as_ptr(),
                    0,
                    region.len(),
                    region.ptr().as_ptr() as u64,
                    *dmabuf_fd,
                    access as i32,
                );
                (mr, "ibv_reg_dmabuf_mr")
            },
        };

        let mr =
            NonNull::new(mr).ok_or_else(|| VerbsError::with_last_os_error(fname))?;
        self.local_mr_map.insert(region.ptr(), mr);
        Ok(MemoryRegionRemoteKey(unsafe { mr.as_ref() }.rkey as u64))
    }

    fn submit_outbound_op(
        &mut self,
        transfer_id: TransferId,
        dest_addr: DomainAddress,
        op: OutboundOp,
    ) {
        if let Some(peer) = self.peers.get_mut(&dest_addr) {
            match &mut peer.state {
                // If connected, add to the op queue directly.
                PeerState::Established => {
                    let msg_qp = peer.msg_rc.qp;
                    let rma_qp = peer.rma_rc.qp;
                    self.do_submit_outbound_op(transfer_id, op, msg_qp, rma_qp)
                }
                // If connecting, queue the submit
                PeerState::Connecting { pending_submits, .. } => {
                    pending_submits.push((transfer_id, op));
                }
            }
        } else {
            // If it's a new peer, initiate the connection and queue the submit
            let res = (|| -> Result<()> {
                let peer =
                    self.create_peer(&dest_addr, vec![(transfer_id, op)], vec![])?;
                self.peers.insert(dest_addr.clone(), peer);
                let peer = unsafe { self.peers.get(&dest_addr).unwrap_unchecked() };
                let buf = unsafe { self.ud_mempool.alloc() }.ok_or(
                    FabricLibError::Custom("Failed to allocate UD message buffer"),
                )?;
                self.connect_peer(peer, buf)?;
                Ok(())
            })();
            if let Err(e) = res {
                self.completions.push_back(DomainCompletionEntry::Error(
                    transfer_id,
                    FabricLibError::Domain(format!(
                        "Failed to create peer: {}. Reason: {}",
                        dest_addr, e
                    )),
                ));
            }
        }
    }

    fn do_submit_outbound_op(
        &mut self,
        transfer_id: TransferId,
        op: OutboundOp,
        msg_qp: NonNull<ibv_qp>,
        rma_qp: NonNull<ibv_qp>,
    ) {
        match op {
            OutboundOp::Send(op) => {
                self.send_ops.push_back(SendOpContext { transfer_id, msg_qp, op });
            }
            OutboundOp::Write(op) => {
                self.do_submit_write(transfer_id, |rawctx, wr_chain_buffer| match op {
                    WriteOp::Single(op) => {
                        WriteOpIter::Single(SingleWriteOpIter::new_single(
                            op,
                            rma_qp,
                            wr_chain_buffer,
                            rawctx,
                        ))
                    }
                    WriteOp::Imm(op) => WriteOpIter::Single(
                        SingleWriteOpIter::new_imm(op, rma_qp, wr_chain_buffer, rawctx),
                    ),
                    WriteOp::Paged(op) => WriteOpIter::Paged(PagedWriteOpIter::new(
                        op,
                        rma_qp,
                        wr_chain_buffer,
                        rawctx,
                    )),
                });
            }
        }
    }

    fn do_submit_write<
        F: FnOnce(*mut c_void, NonNull<WrChainBuffer>) -> WriteOpIter,
    >(
        &mut self,
        transfer_id: TransferId,
        construct_rdma_op_iter: F,
    ) {
        // Allocate the memory for the context and make it float in the heap.
        // We'll delete the object once the transfer is done.
        // This is because we're creating a self-referential struct.
        let mut context = unsafe { self.objpool_write_op.alloc_uninit() };
        let wr_chain_buffer = unsafe { self.objpool_wr.alloc_uninit() };
        let wr_chain_buffer = unsafe { (*wr_chain_buffer.as_ptr()).assume_init_mut() };
        let wr_chain_buffer = unsafe { NonNull::new_unchecked(wr_chain_buffer) };

        // Convert the RDMA op to an iterator.
        let rawctx = context.as_ptr() as *mut c_void;
        let rdma_op_iter = construct_rdma_op_iter(rawctx, wr_chain_buffer);

        // Initialize the context
        let total_ops = rdma_op_iter.total_ops();
        let context = unsafe {
            context.as_mut().write(WriteOpContext {
                transfer_id,
                rdma_op_iter,
                wr_chain_buffer,
                total_ops,
                cnt_posted_ops: 0,
                cnt_finished_ops: 0,
                in_queue: false,
                bad: false,
            })
        };
        let context_ptr = unsafe { NonNull::new_unchecked(context) };

        // Try to eagerly post if currently there's no pending write ops.
        Self::progress_rdma_write_op_context(context);

        // Add to the pending queue if there are more ops to post.
        if context.cnt_posted_ops != context.total_ops {
            self.write_ops.push_back(context_ptr);
            context.in_queue = true;
        }
    }

    fn do_submit_group_write(
        &mut self,
        transfer_id: TransferId,
        rma_qps: Rc<Vec<NonNull<ibv_qp>>>,
        op: GroupWriteOp,
    ) {
        self.do_submit_write(transfer_id, |rawctx, wr_chain_buffer| match op {
            GroupWriteOp::Scatter(op) => WriteOpIter::Scatter(ScatterWriteOpIter::new(
                op,
                rma_qps,
                wr_chain_buffer,
                rawctx,
            )),
        });
    }

    fn progress_ops(&mut self) {
        self.progress_rdma_recv_ops();
        self.progress_rdma_send_ops();
        self.progress_rdma_write_ops();
    }

    fn progress_rdma_recv_ops(&mut self) {
        let mut sge = MaybeUninit::uninit();
        let mut wr = MaybeUninit::uninit();
        let mut bad_wr = MaybeUninit::uninit();
        while let Some(ctx) = self.recv_ops.front() {
            fill_recv_op(&ctx.op, &mut sge, &mut wr, unsafe {
                transmute::<TransferId, *mut c_void>(ctx.transfer_id)
            });
            let ret = unsafe {
                ibv_post_srq_recv(
                    self.msg_srq.as_ptr(),
                    wr.as_mut_ptr(),
                    bad_wr.as_mut_ptr(),
                )
            };
            match ret {
                0 => {
                    self.recv_ops.pop_front();
                }
                ENOMEM => break,
                _ => panic!("ibv_post_srq_recv returned undocumented error: {}", ret),
            }
        }
    }

    fn progress_rdma_send_ops(&mut self) {
        let mut sge = MaybeUninit::uninit();
        let mut wr = MaybeUninit::uninit();
        let mut bad_wr = MaybeUninit::uninit();
        while let Some(ctx) = self.send_ops.front() {
            fill_send_op(&ctx.op, &mut sge, &mut wr, unsafe {
                transmute::<TransferId, *mut libc::c_void>(ctx.transfer_id)
            });
            let ret = unsafe {
                ibv_post_send(ctx.msg_qp.as_ptr(), wr.as_mut_ptr(), bad_wr.as_mut_ptr())
            };
            match ret {
                0 => {
                    self.send_ops.pop_front();
                }
                ENOMEM => break,
                _ => panic!("ibv_post_send returned undocumented error: {}", ret),
            }
        }
    }

    fn progress_rdma_write_op_context(context: &mut WriteOpContext) {
        if context.bad {
            return;
        }
        let mut bad_wr = MaybeUninit::uninit();
        loop {
            let (rma_qp, wr, wr_len) = context.rdma_op_iter.peek();
            if wr_len == 0 {
                break;
            }

            let ret = unsafe { ibv_post_send(rma_qp, wr, bad_wr.as_mut_ptr()) };
            match ret {
                0 => {
                    context.rdma_op_iter.advance(wr_len);
                    context.cnt_posted_ops += wr_len;
                }
                ENOMEM => {
                    // Count the number of ops that are posted.
                    let bad_wr = unsafe { bad_wr.assume_init() };
                    let mut cur = wr;
                    let mut cnt = 0;
                    while cur != bad_wr {
                        cur = unsafe { (*cur).next };
                        cnt += 1;
                    }
                    context.rdma_op_iter.advance(cnt);
                    context.cnt_posted_ops += cnt;

                    // Busy. Break and try again later.
                    break;
                }
                _ => panic!("ibv_post_send returned undocumented error: {}", ret),
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
        unsafe { self.objpool_wr.free_and_drop(context.wr_chain_buffer) };
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

            Self::progress_rdma_write_op_context(context);
            if context.cnt_posted_ops != context.total_ops {
                // More ops to post. Break and try again later.
                break;
            }

            // This transfer is done. Progress the next one.
            self.write_ops.pop_front();
        }
    }

    fn handle_cqe(&mut self, wc: &ibv_wc) -> Option<DomainCompletionEntry> {
        if wc.qp_num == unsafe { *self.ud.qp.as_ptr() }.qp_num {
            if wc.status != IBV_WC_SUCCESS {
                panic!("TODO: handle UD errors. status: {}", wc.status);
            }

            match wc.opcode {
                IBV_WC_RECV => {
                    // Handle RC handshake
                    let ptr = unsafe { NonNull::new_unchecked(wc.wr_id as *mut u8) };
                    let buf = unsafe {
                        std::slice::from_raw_parts_mut(
                            ptr.as_ptr().byte_add(GRH_BYTES),
                            wc.byte_len as usize - GRH_BYTES,
                        )
                    };
                    let msg: PeerHandshakeInfo =
                        postcard::from_bytes(buf).expect("TODO: handle UD error");
                    self.handle_peer_handshake(&msg).expect("TODO: handle UD error");

                    // Post RECV for UD
                    self.post_ud_recv(ptr).expect("TODO: handle UD error");
                }
                IBV_WC_SEND => {
                    debug!(domain = self.name, wr_id = wc.wr_id, "UD SEND completed");
                    // Return the buffer
                    unsafe {
                        let ptr = NonNull::new_unchecked(wc.wr_id as *mut u8);
                        self.ud_mempool.free(ptr);
                    }
                }
                _ => {
                    warn!(
                        domain = self.name,
                        wr_id = wc.wr_id,
                        status = wc.status,
                        opcode = wc.opcode,
                        "Unhandled UD completion event."
                    );
                }
            }

            return None;
        }

        // Check if the completion is an error.
        if wc.status != IBV_WC_SUCCESS {
            let transfer_id: Option<TransferId> = match wc.opcode {
                IBV_WC_RECV | IBV_WC_SEND => {
                    Some(unsafe { transmute::<u64, TransferId>(wc.wr_id) })
                }
                IBV_WC_RDMA_WRITE => {
                    if let Some(context) =
                        unsafe { (wc.wr_id as *mut WriteOpContext).as_mut() }
                    {
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
                }
                _ => None,
            };
            let errmsg = unsafe {
                CStr::from_ptr(ibv_wc_status_str(wc.status))
                    .to_string_lossy()
                    .into_owned()
            };
            return if let Some(transfer_id) = transfer_id {
                warn!(
                    domain = self.name,
                    wr_id = wc.wr_id,
                    status = wc.status,
                    opcode = wc.opcode,
                    "Encountered RDMA op error. Send DomainCompletionEntry::Error to the caller."
                );
                Some(DomainCompletionEntry::Error(
                    transfer_id,
                    FabricLibError::VerbsCompletionError(errmsg),
                ))
            } else {
                error!(
                    domain = self.name,
                    wr_id = wc.wr_id,
                    status = wc.status,
                    msg = errmsg,
                    "Unhandled RDMA op error."
                );
                None
            };
        }

        // Handle successful completion
        match wc.opcode {
            IBV_WC_RECV => {
                let transfer_id: TransferId = unsafe { transmute(wc.wr_id) };
                Some(DomainCompletionEntry::Recv {
                    transfer_id,
                    data_len: wc.byte_len as usize,
                })
            }
            IBV_WC_SEND => {
                let transfer_id: TransferId = unsafe { transmute(wc.wr_id) };
                Some(DomainCompletionEntry::Send(transfer_id))
            }
            IBV_WC_RDMA_WRITE => {
                let context = unsafe { (wc.wr_id as *mut WriteOpContext).as_mut() }?;
                context.cnt_finished_ops += 1;
                if context.cnt_finished_ops < context.total_ops {
                    return None;
                }
                // Transfer is done.
                let transfer_id = context.transfer_id;
                self.maybe_drop_write_op_context(unsafe {
                    NonNull::new_unchecked(context)
                });
                Some(DomainCompletionEntry::Transfer(transfer_id))
            }
            IBV_WC_RECV_RDMA_WITH_IMM => {
                // Submit zero-byte RECV for WRITE_IMM.
                self.post_imm_recv()
                    .expect("TODO: handle error. maybe make ImmRecv a Op");

                // Return different types of completions.
                let imm = unsafe { wc.__bindgen_anon_1.imm_data };
                match self.imm_count_map.inc(imm) {
                    ImmCountStatus::Vacant => Some(DomainCompletionEntry::ImmData(imm)),
                    ImmCountStatus::NotReached => None,
                    ImmCountStatus::Reached => {
                        Some(DomainCompletionEntry::ImmCountReached(imm))
                    }
                }
            }
            _ => {
                warn!(
                    domain = self.name,
                    wr_id = wc.wr_id,
                    status = wc.status,
                    opcode = wc.opcode,
                    "Unhandled RDMA completion event."
                );
                None
            }
        }
    }

    fn poll_cq(&mut self) {
        const READ_COUNT: usize = 16;
        let mut cqes = MaybeUninit::<[ibv_wc; READ_COUNT]>::uninit();
        loop {
            let ret = unsafe {
                ibv_poll_cq(
                    self.cq.as_ptr(),
                    READ_COUNT as i32,
                    cqes.as_mut_ptr() as *mut _,
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
            } else if ret == 0 {
                // No more completions
                return;
            } else {
                panic!("ibv_poll_cq returned undocumented error: {}", ret);
            }
        }
    }
}

impl RdmaDomain for VerbsDomain {
    type Info = VerbsDeviceInfo;

    fn open(info: Self::Info, imm_count_map: Arc<ImmCountMap>) -> Result<Self> {
        Self::open(info, imm_count_map)
    }

    fn addr(&self) -> DomainAddress {
        self.addr.clone()
    }

    fn link_speed(&self) -> u64 {
        self.link_speed
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
        if let Some(mr) = self.local_mr_map.remove(&ptr) {
            unsafe { ibv_dereg_mr(mr.as_ptr()) };
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
        Ok(MemoryRegionLocalDescriptor(unsafe { mr.as_ref() }.lkey as u64))
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
        self.submit_outbound_op(transfer_id, dest_addr, OutboundOp::Send(op));
    }

    fn submit_write(
        &mut self,
        transfer_id: TransferId,
        dest_addr: DomainAddress,
        op: WriteOp,
    ) {
        self.submit_outbound_op(transfer_id, dest_addr, OutboundOp::Write(op));
    }

    fn add_peer_group(
        &mut self,
        handle: PeerGroupHandle,
        addrs: Vec<DomainAddress>,
    ) -> Result<()> {
        if self.connecting_peer_groups.contains_key(&handle) {
            return Ok(());
        }
        if self.peer_groups.contains_key(&handle) {
            return Ok(());
        }
        let mut rma_qps = Vec::with_capacity(addrs.len());
        let mut pending_peers = HashSet::new();
        for addr in addrs.iter() {
            if let Some(peer) = self.peers.get(addr) {
                rma_qps.push(peer.rma_rc.qp);
            } else {
                // Initiate peer connection
                let peer = self.create_peer(addr, vec![], vec![])?;
                self.peers.insert(addr.clone(), peer);
                let peer = unsafe { self.peers.get(addr).unwrap_unchecked() };
                let buf = unsafe { self.ud_mempool.alloc() }.ok_or(
                    FabricLibError::Custom("Failed to allocate UD message buffer"),
                )?;
                self.connect_peer(peer, buf)?;

                let rma_qp = peer.rma_rc.qp;
                rma_qps.push(rma_qp);
                pending_peers.insert(rma_qp);
            }
        }
        if pending_peers.is_empty() {
            self.peer_groups.insert(handle, PeerGroup { rma_qps: Rc::new(rma_qps) });
        } else {
            self.connecting_peer_groups.insert(
                handle,
                ConnectingPeerGroup {
                    rma_qps,
                    pending_peers,
                    pending_group_ops: vec![],
                },
            );
        }
        Ok(())
    }

    fn submit_group_write(
        &mut self,
        transfer_id: TransferId,
        handle: Option<PeerGroupHandle>,
        op: GroupWriteOp,
    ) {
        let rma_qps = if let Some(handle) = handle {
            if let Some(connecting_peer_group) =
                self.connecting_peer_groups.get_mut(&handle)
            {
                connecting_peer_group.pending_group_ops.push((transfer_id, op));
                return;
            }
            let Some(peer_group) = self.peer_groups.get_mut(&handle) else {
                self.completions.push_back(DomainCompletionEntry::Error(
                    transfer_id,
                    FabricLibError::Custom("Peer group not found"),
                ));
                return;
            };
            Rc::clone(&peer_group.rma_qps)
        } else {
            let mut rma_qps = Vec::with_capacity(op.num_targets());
            let mut pending_peers = HashSet::new();
            let mut maybe_ctx_ptr = None;
            for addr in op.peer_addr_iter() {
                if let Some(peer) = self.peers.get(addr) {
                    rma_qps.push(peer.rma_rc.qp);
                } else {
                    // Initiate peer connection
                    let ctx_ptr = if let Some(ctx_ptr) = maybe_ctx_ptr {
                        ctx_ptr
                    } else {
                        let ctx_ptr = unsafe {
                            self.objpool_pending_group_write_op.alloc_uninit().cast()
                        };
                        maybe_ctx_ptr = Some(ctx_ptr);
                        ctx_ptr
                    };
                    let res = (|| -> Result<()> {
                        let peer = self.create_peer(addr, vec![], vec![ctx_ptr])?;
                        self.peers.insert(addr.clone(), peer);
                        let peer = unsafe { self.peers.get(addr).unwrap_unchecked() };
                        let buf = unsafe { self.ud_mempool.alloc() }.ok_or(
                            FabricLibError::Custom(
                                "Failed to allocate UD message buffer",
                            ),
                        )?;
                        self.connect_peer(peer, buf)?;
                        Ok(())
                    })();
                    if let Err(e) = res {
                        self.completions.push_back(DomainCompletionEntry::Error(
                            transfer_id,
                            FabricLibError::Domain(format!(
                                "Failed to create peer: {}. Reason: {}",
                                addr, e
                            )),
                        ));
                        return;
                    }

                    let peer = unsafe { self.peers.get_mut(addr).unwrap_unchecked() };
                    let rma_qp = peer.rma_rc.qp;
                    rma_qps.push(rma_qp);
                    pending_peers.insert(rma_qp);
                }
            }
            if !pending_peers.is_empty() {
                let mut ptr =
                    maybe_ctx_ptr.unwrap().cast::<MaybeUninit<PendingGroupWriteOp>>();
                unsafe {
                    ptr.as_mut().write(PendingGroupWriteOp {
                        transfer_id,
                        op: Some(op),
                        rma_qps,
                        pending_peers,
                    })
                };
                return;
            }
            Rc::new(rma_qps)
        };
        self.do_submit_group_write(transfer_id, rma_qps, op);
    }

    fn poll_progress(&mut self) {
        self.poll_cq();
        self.progress_ops();
    }

    fn get_completion(&mut self) -> Option<DomainCompletionEntry> {
        self.completions.pop_front()
    }
}

impl Drop for VerbsDomain {
    fn drop(&mut self) {
        debug!(name = self.name, "VerbsDomain::drop");
        // TODO: drop only_qp and ud
        unsafe {
            for (_, mr) in self.local_mr_map.drain() {
                ibv_dereg_mr(mr.as_ptr());
            }
            for (_, mut peer) in self.peers.drain() {
                peer.msg_rc.destroy();
                peer.rma_rc.destroy();
            }
            self.ud.destroy();
            ibv_destroy_cq(self.cq.as_ptr());
            ibv_dealloc_pd(self.pd.as_ptr());
            ibv_dealloc_pd(self.mt_pd.as_ptr());
            ibv_dealloc_td(self.td.as_ptr());
            ibv_close_device(self.context.as_ptr());
        }
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        unsafe { ibv_destroy_ah(self.ah.as_ptr()) };
    }
}
