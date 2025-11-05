use std::{
    ffi::c_void,
    mem::MaybeUninit,
    ptr::{NonNull, null_mut},
    rc::Rc,
    sync::Arc,
};

use libibverbs_sys::{
    IBV_SEND_SIGNALED, IBV_WR_RDMA_WRITE, IBV_WR_RDMA_WRITE_WITH_IMM, IBV_WR_SEND,
    ibv_qp, ibv_recv_wr, ibv_send_wr, ibv_sge,
};

use crate::{
    api::ScatterTarget,
    rdma_op::{
        ImmWriteOp, PagedWriteOp, RecvOp, ScatterGroupWriteOp, SendOp, SingleWriteOp,
    },
};

/// The maximum number of WRs in a WR chain.
/// NOTE(lequn): Benchmark shows no performance gain above 4.
pub const WR_CHAIN_LEN: usize = 4;

pub type WrChainBuffer =
    [(MaybeUninit<ibv_send_wr>, MaybeUninit<ibv_sge>); WR_CHAIN_LEN];

pub struct SingleWriteOpIter {
    rma_qp: NonNull<ibv_qp>,
    wr_chain_buffer: NonNull<WrChainBuffer>,
    done: bool,
}

impl SingleWriteOpIter {
    pub fn new_single(
        op: SingleWriteOp,
        rma_qp: NonNull<ibv_qp>,
        mut wr_chain_buffer: NonNull<WrChainBuffer>,
        context: *mut c_void,
    ) -> Self {
        let buf = unsafe { wr_chain_buffer.as_mut() };
        let (wr, sge) = &mut buf[0];
        let sge = sge.write(ibv_sge {
            addr: unsafe { op.src_ptr.as_ptr().byte_add(op.src_offset as usize) }
                as u64,
            length: op.length as u32,
            lkey: op.src_desc.0 as u32,
        });
        let (opcode, imm) = opcode_imm(op.imm_data);
        let wr = wr.write(ibv_send_wr {
            wr_id: context as u64,
            next: null_mut(),
            sg_list: sge,
            num_sge: 1,
            opcode,
            send_flags: IBV_SEND_SIGNALED,
            ..Default::default()
        });
        wr.__bindgen_anon_1.imm_data = imm;
        wr.wr.rdma.remote_addr = op.dst_ptr + op.dst_offset;
        wr.wr.rdma.rkey = op.dst_rkey.0 as u32;
        Self { rma_qp, wr_chain_buffer, done: false }
    }

    pub fn new_imm(
        op: ImmWriteOp,
        rma_qp: NonNull<ibv_qp>,
        mut wr_chain_buffer: NonNull<WrChainBuffer>,
        context: *mut c_void,
    ) -> Self {
        let buf = unsafe { wr_chain_buffer.as_mut() };
        let (wr, _) = &mut buf[0];
        let wr = wr.write(ibv_send_wr {
            wr_id: context as u64,
            next: null_mut(),
            sg_list: null_mut(),
            num_sge: 0,
            opcode: IBV_WR_RDMA_WRITE_WITH_IMM,
            send_flags: IBV_SEND_SIGNALED,
            ..Default::default()
        });
        wr.__bindgen_anon_1.imm_data = op.imm_data;
        wr.wr.rdma.remote_addr = op.dst_ptr;
        wr.wr.rdma.rkey = op.dst_rkey.0 as u32;
        Self { rma_qp, wr_chain_buffer, done: false }
    }

    pub fn peek(&self) -> (*mut ibv_qp, *mut ibv_send_wr, usize) {
        if self.done {
            (self.rma_qp.as_ptr(), null_mut(), 0)
        } else {
            let buf = unsafe { &mut *self.wr_chain_buffer.as_ptr() };
            (self.rma_qp.as_ptr(), buf[0].0.as_mut_ptr(), 1)
        }
    }

    pub fn mark_done(&mut self) {
        self.done = true;
    }
}

pub struct PagedWriteOpIter {
    rma_qp: NonNull<ibv_qp>,
    // Buffer
    wr_chain_buffer: NonNull<WrChainBuffer>,
    i_wr_head: usize,
    i_wr_tail: usize,
    wr_len: usize,
    // Request
    src_page_indices: Arc<Vec<u32>>,
    dst_page_indices: Arc<Vec<u32>>,
    page_indices_beg: usize,
    page_indices_end: usize,
    src_ptr: NonNull<c_void>,
    src_stride: u64,
    dst_ptr: u64,
    dst_stride: u64,
    // Loop variables
    i_page: usize,
}

impl PagedWriteOpIter {
    pub fn new(
        op: PagedWriteOp,
        rma_qp: NonNull<ibv_qp>,
        mut wr_chain_buffer: NonNull<WrChainBuffer>,
        context: *mut c_void,
    ) -> Self {
        assert!(op.page_indices_beg < op.page_indices_end);
        assert!(op.src_page_indices.len() == op.dst_page_indices.len());
        assert!(op.page_indices_end <= op.src_page_indices.len());

        // Prepare WR template
        let (opcode, imm) = opcode_imm(op.imm_data);
        let chain_len =
            std::cmp::min(op.page_indices_end - op.page_indices_beg, WR_CHAIN_LEN);
        let buf = unsafe { wr_chain_buffer.as_mut() };
        for (wr, sge) in buf.iter_mut().take(chain_len) {
            let sge = sge.write(ibv_sge {
                addr: 0,
                length: op.length as u32,
                lkey: op.src_desc.0 as u32,
            });
            let wr = wr.write(ibv_send_wr {
                wr_id: context as u64,
                next: null_mut(),
                sg_list: sge,
                num_sge: 1,
                opcode,
                send_flags: IBV_SEND_SIGNALED,
                ..Default::default()
            });
            wr.__bindgen_anon_1.imm_data = imm;
            wr.wr.rdma.rkey = op.dst_rkey.0 as u32;
        }
        let mut slf = Self {
            rma_qp,
            wr_chain_buffer,
            i_wr_head: 0,
            i_wr_tail: 0,
            wr_len: 0,
            src_page_indices: op.src_page_indices,
            dst_page_indices: op.dst_page_indices,
            page_indices_beg: op.page_indices_beg,
            page_indices_end: op.page_indices_end,
            src_ptr: unsafe { op.src_ptr.byte_add(op.src_offset as usize) },
            src_stride: op.src_stride,
            dst_ptr: op.dst_ptr + op.dst_offset,
            dst_stride: op.dst_stride,
            i_page: op.page_indices_beg,
        };

        // Fill the first batch
        while slf.i_page < slf.page_indices_end && slf.wr_len < WR_CHAIN_LEN {
            slf.fill_wr();
        }

        slf
    }

    pub fn total_ops(&self) -> usize {
        self.page_indices_end - self.page_indices_beg
    }

    pub fn peek(&self) -> (*mut ibv_qp, *mut ibv_send_wr, usize) {
        if self.wr_len == 0 {
            return (null_mut(), null_mut(), 0);
        }
        let buf = unsafe { &mut *self.wr_chain_buffer.as_ptr() };
        (self.rma_qp.as_ptr(), buf[self.i_wr_head].0.as_mut_ptr(), self.wr_len)
    }

    pub fn advance(&mut self, n: usize) {
        self.i_wr_head = (self.i_wr_head + n) % WR_CHAIN_LEN;
        self.wr_len -= n;
        while self.i_page < self.page_indices_end && self.wr_len < WR_CHAIN_LEN {
            self.fill_wr();
        }
    }

    fn fill_wr(&mut self) {
        let buf = unsafe { self.wr_chain_buffer.as_mut() };
        let (prev_wr, _) = &mut buf[(self.i_wr_tail + WR_CHAIN_LEN - 1) % WR_CHAIN_LEN];
        let prev_wr = unsafe { &mut *prev_wr.as_mut_ptr() };
        let (wr, sge) = &mut buf[self.i_wr_tail];
        let wr = unsafe { wr.assume_init_mut() };
        let sge = unsafe { sge.assume_init_mut() };

        // Update WR next pointer
        prev_wr.next = wr;
        wr.next = null_mut();
        self.i_wr_tail = (self.i_wr_tail + 1) % WR_CHAIN_LEN;
        self.wr_len += 1;

        // Update output buffers
        let src_page_idx = self.src_page_indices[self.i_page] as usize;
        let dst_page_idx = self.dst_page_indices[self.i_page] as usize;
        sge.addr = unsafe {
            self.src_ptr.as_ptr().byte_add(self.src_stride as usize * src_page_idx)
        } as u64;
        wr.wr.rdma.remote_addr = self.dst_ptr + self.dst_stride * dst_page_idx as u64;
        self.i_page += 1;
    }
}

pub struct ScatterWriteOpIter {
    qp_list: Rc<Vec<NonNull<ibv_qp>>>,
    // Buffer
    wr_chain_buffer: NonNull<WrChainBuffer>,
    // Request
    domain_idx: usize,
    src_ptr: NonNull<c_void>,
    dsts: Arc<Vec<ScatterTarget>>,
    dst_beg: usize,
    dst_end: usize,
    byte_shards: u32,
    byte_shard_idx: u32,
    // Loop variables
    i_dst: usize,
}

impl ScatterWriteOpIter {
    pub fn new(
        op: ScatterGroupWriteOp,
        qp_list: Rc<Vec<NonNull<ibv_qp>>>,
        mut wr_chain_buffer: NonNull<WrChainBuffer>,
        context: *mut c_void,
    ) -> Self {
        assert!(op.dst_beg < op.dst_end);
        assert!(op.dst_end <= op.dsts.len());
        assert!(qp_list.len() == op.dsts.len());

        // Prepare WR template
        let (opcode, imm) = opcode_imm(op.imm_data);
        let buf = unsafe { wr_chain_buffer.as_mut() };
        let (wr, sge) = &mut buf[0];
        let sge = sge.write(ibv_sge { addr: 0, length: 0, lkey: op.src_desc.0 as u32 });
        let wr = wr.write(ibv_send_wr {
            wr_id: context as u64,
            next: null_mut(),
            sg_list: sge,
            num_sge: 1,
            opcode,
            send_flags: IBV_SEND_SIGNALED,
            ..Default::default()
        });
        wr.__bindgen_anon_1.imm_data = imm;
        let mut slf = Self {
            qp_list,
            wr_chain_buffer,
            domain_idx: op.domain_idx,
            src_ptr: op.src_ptr,
            dsts: op.dsts,
            dst_beg: op.dst_beg,
            dst_end: op.dst_end,
            byte_shards: op.byte_shards,
            byte_shard_idx: op.byte_shard_idx,
            i_dst: op.dst_beg,
        };

        // Fill the first WR
        slf.fill_wr();

        slf
    }

    pub fn total_ops(&self) -> usize {
        self.dst_end - self.dst_beg
    }

    pub fn peek(&self) -> (*mut ibv_qp, *mut ibv_send_wr, usize) {
        if self.i_dst == self.dst_end {
            return (null_mut(), null_mut(), 0);
        }
        let buf = unsafe { &mut *self.wr_chain_buffer.as_ptr() };
        let (wr, _) = &mut buf[0];
        (self.qp_list[self.i_dst].as_ptr(), wr.as_mut_ptr(), 1)
    }

    pub fn advance_one(&mut self) {
        self.i_dst += 1;
        if self.i_dst != self.dst_end {
            self.fill_wr();
        }
    }

    fn fill_wr(&mut self) {
        let buf = unsafe { self.wr_chain_buffer.as_mut() };
        let (wr, sge) = &mut buf[0];
        let wr = unsafe { wr.assume_init_mut() };
        let sge = unsafe { sge.assume_init_mut() };

        // Update output buffers
        let dst = &self.dsts[self.i_dst];
        let len = dst.length as u32 / self.byte_shards;
        let offset = len * self.byte_shard_idx;
        sge.addr = unsafe {
            self.src_ptr.as_ptr().byte_add(dst.src_offset as usize + offset as usize)
        } as u64;
        sge.length = len;
        wr.wr.rdma.remote_addr = dst.dst_mr.ptr + dst.dst_offset + offset as u64;
        wr.wr.rdma.rkey = dst.dst_mr.addr_rkey_list[self.domain_idx].1.0 as u32;
    }
}

fn opcode_imm(imm_data: Option<u32>) -> (u32, u32) {
    if let Some(imm_data) = imm_data {
        (IBV_WR_RDMA_WRITE_WITH_IMM, imm_data)
    } else {
        (IBV_WR_RDMA_WRITE, 0)
    }
}

pub enum WriteOpIter {
    Single(SingleWriteOpIter),
    Paged(PagedWriteOpIter),
    Scatter(ScatterWriteOpIter),
}

impl WriteOpIter {
    pub fn total_ops(&self) -> usize {
        match self {
            WriteOpIter::Single(_) => 1,
            WriteOpIter::Paged(iter) => iter.total_ops(),
            WriteOpIter::Scatter(iter) => iter.total_ops(),
        }
    }

    pub fn peek(&self) -> (*mut ibv_qp, *mut ibv_send_wr, usize) {
        match self {
            WriteOpIter::Single(iter) => iter.peek(),
            WriteOpIter::Paged(iter) => iter.peek(),
            WriteOpIter::Scatter(iter) => iter.peek(),
        }
    }

    pub fn advance(&mut self, n: usize) {
        match self {
            WriteOpIter::Single(iter) => {
                assert!(n == 1);
                iter.mark_done();
            }
            WriteOpIter::Paged(iter) => iter.advance(n),
            WriteOpIter::Scatter(iter) => {
                assert!(n == 1);
                iter.advance_one();
            }
        }
    }
}

pub fn fill_send_op(
    op: &SendOp,
    sge: &mut MaybeUninit<ibv_sge>,
    wr: &mut MaybeUninit<ibv_send_wr>,
    context: *mut c_void,
) {
    unsafe {
        *sge.as_mut_ptr() = ibv_sge {
            addr: op.ptr.as_ptr() as u64,
            length: op.len as u32,
            lkey: op.desc.0 as u32,
        };
        *wr.as_mut_ptr() = ibv_send_wr {
            wr_id: context as u64,
            next: null_mut(),
            sg_list: sge.as_mut_ptr(),
            num_sge: 1,
            opcode: IBV_WR_SEND,
            send_flags: IBV_SEND_SIGNALED,
            ..Default::default()
        };
    }
}

pub fn fill_recv_op(
    op: &RecvOp,
    sge: &mut MaybeUninit<ibv_sge>,
    wr: &mut MaybeUninit<ibv_recv_wr>,
    context: *mut c_void,
) {
    unsafe {
        *sge.as_mut_ptr() = ibv_sge {
            addr: op.ptr.as_ptr() as u64,
            length: op.len as u32,
            lkey: op.desc.0 as u32,
        };
        *wr.as_mut_ptr() = ibv_recv_wr {
            wr_id: context as u64,
            next: null_mut(),
            sg_list: sge.as_mut_ptr(),
            num_sge: 1,
        };
    }
}
