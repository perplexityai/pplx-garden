use std::{ffi::c_void, ptr::NonNull};

use libfabric_sys::fid_mr;

use crate::mr::MemoryRegionLocalDescriptor;

#[derive(Debug, Clone, Copy)]
pub struct EfaMemDesc(pub *mut *mut c_void);

impl From<NonNull<fid_mr>> for EfaMemDesc {
    fn from(mr: NonNull<fid_mr>) -> Self {
        EfaMemDesc(unsafe { &raw mut (*mr.as_ptr()).mem_desc })
    }
}

impl From<MemoryRegionLocalDescriptor> for EfaMemDesc {
    fn from(desc: MemoryRegionLocalDescriptor) -> Self {
        EfaMemDesc(desc.0 as *mut *mut c_void)
    }
}

impl From<EfaMemDesc> for MemoryRegionLocalDescriptor {
    fn from(desc: EfaMemDesc) -> Self {
        MemoryRegionLocalDescriptor(desc.0 as u64)
    }
}
