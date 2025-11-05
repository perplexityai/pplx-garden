use std::{
    ffi::c_void,
    mem::MaybeUninit,
    ptr::{NonNull, null, null_mut},
    rc::Rc,
    sync::Arc,
};

use libfabric_sys::{
    FI_ADDR_UNSPEC, FI_REMOTE_CQ_DATA, fi_addr_t, fi_msg, fi_msg_rma, fi_rma_iov, iovec,
};

use crate::{
    api::ScatterTarget,
    efa::EfaMemDesc,
    rdma_op::{
        ImmWriteOp, PagedWriteOp, RecvOp, ScatterGroupWriteOp, SendOp, SingleWriteOp,
    },
};

pub type RmaBuffer =
    (MaybeUninit<iovec>, MaybeUninit<fi_rma_iov>, MaybeUninit<fi_msg_rma>);

#[derive(Debug)]
pub struct SingleWriteOpIter {
    buf: NonNull<RmaBuffer>,
    flags: u64,
    done: bool,
}

impl SingleWriteOpIter {
    pub fn new_single(
        op: SingleWriteOp,
        addr: fi_addr_t,
        mut buf: NonNull<RmaBuffer>,
        context: *mut c_void,
    ) -> Self {
        let (iov, rma_iov, msg) = unsafe { buf.as_mut() };
        let iov = iov.write(iovec {
            iov_base: unsafe { op.src_ptr.as_ptr().byte_add(op.src_offset as usize) },
            iov_len: op.length as usize,
        });
        let rma_iov = rma_iov.write(fi_rma_iov {
            addr: op.dst_ptr + op.dst_offset,
            len: op.length as usize,
            key: op.dst_rkey.0,
        });
        let (imm, flags) = flags_imm(op.imm_data);
        msg.write(fi_msg_rma {
            msg_iov: iov,
            desc: EfaMemDesc::from(op.src_desc).0,
            iov_count: 1,
            addr,
            rma_iov,
            rma_iov_count: 1,
            context,
            data: imm,
        });
        Self { buf, flags, done: false }
    }

    pub fn new_imm(
        op: ImmWriteOp,
        addr: fi_addr_t,
        mut buf: NonNull<RmaBuffer>,
        context: *mut c_void,
    ) -> Self {
        let (_, rma_iov, msg) = unsafe { buf.as_mut() };
        // Even though RDMA spec says no need to specify rma_iov,
        // EFA still requires a valid rma_iov.
        let rma_iov =
            rma_iov.write(fi_rma_iov { addr: op.dst_ptr, len: 0, key: op.dst_rkey.0 });
        msg.write(fi_msg_rma {
            msg_iov: null(),
            desc: null_mut(),
            iov_count: 0,
            addr,
            rma_iov,
            rma_iov_count: 1,
            context,
            data: op.imm_data as u64,
        });
        Self { buf, flags: FI_REMOTE_CQ_DATA as u64, done: false }
    }

    pub fn peek(&self) -> (*mut fi_msg_rma, u64) {
        if self.done {
            (null_mut(), 0)
        } else {
            let (_, _, msg) = unsafe { &mut *self.buf.as_ptr() };
            (msg.as_mut_ptr(), self.flags)
        }
    }

    pub fn mark_done(&mut self) {
        assert!(!self.done);
        self.done = true;
    }
}

pub struct PagedWriteOpIter {
    // Request
    src_page_indices: Arc<Vec<u32>>,
    dst_page_indices: Arc<Vec<u32>>,
    page_indices_beg: usize,
    page_indices_end: usize,
    src_ptr: NonNull<c_void>,
    src_stride: u64,
    dst_ptr: u64,
    dst_stride: u64,
    // Output buffers
    buf: NonNull<RmaBuffer>,
    flags: u64,
    // Loop variables
    i_page: usize,
}

impl PagedWriteOpIter {
    pub fn new(
        op: PagedWriteOp,
        addr: fi_addr_t,
        mut buf: NonNull<RmaBuffer>,
        context: *mut c_void,
    ) -> Self {
        assert!(op.page_indices_beg < op.page_indices_end);
        assert!(op.src_page_indices.len() == op.dst_page_indices.len());
        assert!(op.page_indices_end <= op.src_page_indices.len());

        let (iov, rma_iov, msg) = unsafe { buf.as_mut() };
        let iov =
            iov.write(iovec { iov_base: null_mut(), iov_len: op.length as usize });
        let rma_iov = rma_iov.write(fi_rma_iov {
            addr: 0,
            len: op.length as usize,
            key: op.dst_rkey.0,
        });

        let (imm, flags) = flags_imm(op.imm_data);
        msg.write(fi_msg_rma {
            msg_iov: iov,
            desc: EfaMemDesc::from(op.src_desc).0,
            iov_count: 1,
            addr,
            rma_iov,
            rma_iov_count: 1,
            context,
            data: imm,
        });

        let mut slf = Self {
            src_page_indices: op.src_page_indices,
            dst_page_indices: op.dst_page_indices,
            page_indices_beg: op.page_indices_beg,
            page_indices_end: op.page_indices_end,
            src_ptr: unsafe { op.src_ptr.byte_add(op.src_offset as usize) },
            src_stride: op.src_stride,
            dst_ptr: op.dst_ptr + op.dst_offset,
            dst_stride: op.dst_stride,
            buf,
            flags,
            i_page: op.page_indices_beg,
        };
        slf.fill_msg();
        slf
    }

    pub fn total_ops(&self) -> usize {
        self.page_indices_end - self.page_indices_beg
    }

    pub fn peek(&self) -> (*mut fi_msg_rma, u64) {
        if self.i_page == self.page_indices_end {
            (null_mut(), 0)
        } else {
            let (_, _, msg) = unsafe { &mut *self.buf.as_ptr() };
            (msg.as_mut_ptr(), self.flags)
        }
    }

    pub fn advance(&mut self) {
        assert!(self.i_page < self.page_indices_end);
        self.i_page += 1;
        if self.i_page < self.page_indices_end {
            self.fill_msg();
        }
    }

    fn fill_msg(&mut self) {
        let src_page_idx = self.src_page_indices[self.i_page] as usize;
        let dst_page_idx = self.dst_page_indices[self.i_page] as usize;

        let (iov, rma_iov, _) = unsafe { self.buf.as_mut() };
        let iov = unsafe { iov.assume_init_mut() };
        let rma_iov = unsafe { rma_iov.assume_init_mut() };
        iov.iov_base = unsafe {
            self.src_ptr.as_ptr().byte_add(self.src_stride as usize * src_page_idx)
        };
        rma_iov.addr = self.dst_ptr + self.dst_stride * dst_page_idx as u64;
    }
}

pub struct ScatterWriteOpIter {
    addrs: Rc<Vec<fi_addr_t>>,
    // Buffer
    buf: NonNull<RmaBuffer>,
    flags: u64,
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
        addrs: Rc<Vec<fi_addr_t>>,
        mut buf: NonNull<RmaBuffer>,
        context: *mut c_void,
    ) -> Self {
        assert!(op.dst_beg < op.dst_end);
        assert!(op.dst_end <= op.dsts.len());
        assert!(addrs.len() == op.dsts.len());

        let (imm, flags) = flags_imm(op.imm_data);
        let (iov, rma_iov, msg) = unsafe { buf.as_mut() };
        msg.write(fi_msg_rma {
            msg_iov: iov.as_mut_ptr(),
            desc: EfaMemDesc::from(op.src_desc).0,
            iov_count: 1,
            addr: FI_ADDR_UNSPEC,
            rma_iov: rma_iov.as_mut_ptr(),
            rma_iov_count: 1,
            context,
            data: imm,
        });

        let mut slf = Self {
            addrs,
            buf,
            flags,
            domain_idx: op.domain_idx,
            src_ptr: op.src_ptr,
            dsts: op.dsts,
            dst_beg: op.dst_beg,
            dst_end: op.dst_end,
            byte_shards: op.byte_shards,
            byte_shard_idx: op.byte_shard_idx,
            i_dst: op.dst_beg,
        };
        slf.fill_msg();
        slf
    }

    pub fn total_ops(&self) -> usize {
        self.dst_end - self.dst_beg
    }

    pub fn peek(&self) -> (*mut fi_msg_rma, u64) {
        if self.i_dst == self.dst_end {
            (null_mut(), 0)
        } else {
            let (_, _, msg) = unsafe { &mut *self.buf.as_ptr() };
            (msg.as_mut_ptr(), self.flags)
        }
    }

    pub fn advance(&mut self) {
        assert!(self.i_dst < self.dst_end);
        self.i_dst += 1;
        if self.i_dst < self.dst_end {
            self.fill_msg();
        }
    }

    fn fill_msg(&mut self) {
        let (iov, rma_iov, msg) = unsafe { self.buf.as_mut() };
        let msg = unsafe { msg.assume_init_mut() };

        let dst = &self.dsts[self.i_dst];
        let len = dst.length as u32 / self.byte_shards;
        let offset = len * self.byte_shard_idx;
        msg.addr = self.addrs[self.i_dst];
        iov.write(iovec {
            iov_base: unsafe {
                self.src_ptr
                    .as_ptr()
                    .byte_add(dst.src_offset as usize + offset as usize)
            },
            iov_len: len as usize,
        });
        rma_iov.write(fi_rma_iov {
            addr: dst.dst_mr.ptr + dst.dst_offset + offset as u64,
            len: len as usize,
            key: dst.dst_mr.addr_rkey_list[self.domain_idx].1.0,
        });
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

    pub fn peek(&self) -> (*mut fi_msg_rma, u64) {
        match self {
            WriteOpIter::Single(iter) => iter.peek(),
            WriteOpIter::Paged(iter) => iter.peek(),
            WriteOpIter::Scatter(iter) => iter.peek(),
        }
    }

    pub fn advance(&mut self) {
        match self {
            WriteOpIter::Single(iter) => iter.mark_done(),
            WriteOpIter::Paged(iter) => iter.advance(),
            WriteOpIter::Scatter(iter) => iter.advance(),
        }
    }
}

fn flags_imm(imm_data: Option<u32>) -> (u64, u64) {
    if let Some(imm_data) = imm_data {
        (imm_data as u64, FI_REMOTE_CQ_DATA as u64)
    } else {
        (0, 0)
    }
}

pub fn fill_send_op(
    op: &SendOp,
    iov: &mut MaybeUninit<iovec>,
    msg: &mut MaybeUninit<fi_msg>,
    context: *mut c_void,
) {
    unsafe {
        *iov.as_mut_ptr() = iovec { iov_base: op.ptr.as_ptr(), iov_len: op.len };
        *msg.as_mut_ptr() = fi_msg {
            msg_iov: iov.as_mut_ptr(),
            desc: EfaMemDesc::from(op.desc).0,
            iov_count: 1,
            addr: FI_ADDR_UNSPEC,
            context,
            data: 0,
        };
    }
}

pub fn fill_recv_op(
    op: &RecvOp,
    iov: &mut MaybeUninit<iovec>,
    msg: &mut MaybeUninit<fi_msg>,
    context: *mut c_void,
) {
    unsafe {
        *iov.as_mut_ptr() = iovec { iov_base: op.ptr.as_ptr(), iov_len: op.len };
        *msg.as_mut_ptr() = fi_msg {
            msg_iov: iov.as_mut_ptr(),
            desc: EfaMemDesc::from(op.desc).0,
            iov_count: 1,
            addr: FI_ADDR_UNSPEC,
            context,
            data: 0,
        };
    }
}
