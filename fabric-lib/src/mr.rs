use std::{ffi::c_void, ptr::NonNull};

use cuda_lib::driver::cu_get_dma_buf_fd;
use cuda_lib::rt::{cudaMemoryTypeDevice, cudaPointerGetAttributes};
use cuda_lib::{CudaDeviceId, Device};

use crate::error::{FabricLibError, Result};

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Mapping {
    Host,
    Device { device_id: CudaDeviceId, dmabuf_fd: Option<i32> },
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MemoryRegion {
    ptr: NonNull<c_void>,
    len: usize,
    mapping: Mapping,
}

impl MemoryRegion {
    pub fn new(ptr: NonNull<c_void>, len: usize, device: Device) -> Result<Self> {
        let mapping = match device {
            Device::Host => Mapping::Host,
            Device::Cuda(device_id) => {
                let attrs = cudaPointerGetAttributes(ptr)?;
                if attrs.type_ != cudaMemoryTypeDevice {
                    return Err(FabricLibError::Custom("not a device pointer"));
                }
                let dmabuf_fd = cu_get_dma_buf_fd(ptr, len).ok();
                Mapping::Device { device_id, dmabuf_fd }
            }
        };
        Ok(MemoryRegion { ptr, len, mapping })
    }

    pub fn ptr(&self) -> NonNull<c_void> {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn mapping(&self) -> &Mapping {
        &self.mapping
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        match self.mapping {
            Mapping::Host => {}
            Mapping::Device { dmabuf_fd: None, .. } => {}
            Mapping::Device { dmabuf_fd: Some(dmabuf_fd), .. } => unsafe {
                libc::close(dmabuf_fd);
            },
        }
    }
}

/// A local descriptor for a memory region.
/// For verbs, this is the MR LKEY.
/// For libfabric, this is the MR descriptor.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct MemoryRegionLocalDescriptor(pub u64);
