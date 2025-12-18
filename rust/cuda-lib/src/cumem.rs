#![allow(clippy::macro_metavars_in_unsafe)]

use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use bincode::{
    Decode, Encode,
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
};

use crate::{
    CudaDeviceId, Device, cuda_check, cuda_unwrap,
    error::{CudaError, CudaResult},
};

/// The list of supported IPC handles.
#[derive(Clone, Copy, Debug)]
pub enum CUMemHandleKind {
    Local,
    FileDescriptor,
    Fabric,
}

pub trait CUAllocHandle {
    /// Map the allocation into a new mapping.
    fn map(&self, device: Device) -> CudaResult<CUMemMapping>;
    /// Re-map the allocation into an existing mapping.
    fn map_to(&self, mapping: CUMemMapping) -> CudaResult<()>;
}

/// An owning handle to a CUmem allocation.
pub struct CUMemAlloc {
    handle: cuda_sys::CUmemGenericAllocationHandle,
    alloc_size: usize,
    device_id: CudaDeviceId,
    handle_kind: CUMemHandleKind,
}

impl Drop for CUMemAlloc {
    fn drop(&mut self) {
        cuda_unwrap!(cuda_sys::cuMemRelease(self.handle));
    }
}

impl CUMemAlloc {
    fn map(alloc: CUMemMappingRef, device: Device) -> CudaResult<CUMemMapping> {
        let device_id = match device {
            Device::Cuda(id) => id,
            Device::Host => alloc.device_id(),
        };

        // Reserve a virtual address range for the memory.
        let mut ptr: cuda_sys::CUdeviceptr = 0;
        cuda_check!(cuda_sys::cuMemAddressReserve(&mut ptr, alloc.size(), 0, 0, 0,))?;

        // Map the memory to the virtual address range.
        cuda_check!(cuda_sys::cuMemMap(ptr, alloc.size(), 0, alloc.handle(), 0,))?;

        // Allow read/write access to the memory.
        let mut access_desc =
            unsafe { std::mem::zeroed::<cuda_sys::CUmemAccessDesc>() };
        access_desc.location.type_ = cuda_sys::CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_id.0 as i32;
        access_desc.flags = cuda_sys::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuda_check!(cuda_sys::cuMemSetAccess(ptr, alloc.size(), &access_desc, 1,))?;

        Ok(CUMemMapping {
            handle: Arc::new(CUMemMappingHolder {
                alloc_size: alloc.size(),
                device_id,
                mapped: AtomicBool::new(true),
                ptr,
                _ref: alloc,
            }),
        })
    }

    fn map_to(alloc: CUMemMappingRef, mapping: CUMemMapping) -> CudaResult<()> {
        // Ensure the mapping size matches the allocation size.
        if mapping.size() != alloc.size() {
            return Err(CudaError::CustomError(format!(
                "Mapping size {} does not match allocation size {}",
                mapping.size(),
                alloc.size()
            )));
        }

        // Map the memory to the existing virtual address range.
        cuda_check!(cuda_sys::cuMemMap(
            mapping.handle.ptr,
            alloc.size(),
            0,
            alloc.handle(),
            0,
        ))?;

        // Allow read/write access to the memory.
        let mut access_desc =
            unsafe { std::mem::zeroed::<cuda_sys::CUmemAccessDesc>() };
        access_desc.location.type_ = cuda_sys::CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = mapping.handle.device_id.0 as i32;
        access_desc.flags = cuda_sys::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuda_check!(cuda_sys::cuMemSetAccess(
            mapping.handle.ptr,
            alloc.size(),
            &access_desc,
            1,
        ))?;

        // Mark the mapping as mapped.
        mapping.handle.mapped.store(true, Ordering::Relaxed);

        Ok(())
    }
}

/// A reference which keeps a target mapping alive.
enum CUMemMappingRef {
    Alloc(Arc<CUMemAlloc>),
    Multicast(Arc<CUMulticast>),
}

impl CUMemMappingRef {
    fn size(&self) -> usize {
        match self {
            CUMemMappingRef::Alloc(alloc) => alloc.alloc_size,
            CUMemMappingRef::Multicast(mcast) => mcast.alloc_size,
        }
    }

    fn handle(&self) -> cuda_sys::CUmemGenericAllocationHandle {
        match self {
            CUMemMappingRef::Alloc(alloc) => alloc.handle,
            CUMemMappingRef::Multicast(mcast) => mcast.handle,
        }
    }

    fn device_id(&self) -> CudaDeviceId {
        match self {
            CUMemMappingRef::Alloc(alloc) => alloc.device_id,
            CUMemMappingRef::Multicast(_) => {
                // For multicast, we default to device 0.
                CudaDeviceId(0)
            }
        }
    }
}

/// A handle to a physical memory region allocated via CUmem.
#[derive(Clone)]
pub struct CUMemAllocHandle {
    handle: Arc<CUMemAlloc>,
}

impl CUMemAllocHandle {
    /// Create a new CUmemAllocHandle of at least `size` bytes on `device`.
    pub fn new(
        size: usize,
        device_id: CudaDeviceId,
        handle_kind: CUMemHandleKind,
    ) -> CudaResult<Self> {
        let mut props = unsafe { std::mem::zeroed::<cuda_sys::CUmemAllocationProp>() };
        props.type_ = cuda_sys::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = cuda_sys::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = device_id.0 as i32;
        props.requestedHandleTypes = match handle_kind {
            CUMemHandleKind::Local => cuda_sys::CU_MEM_HANDLE_TYPE_NONE,
            CUMemHandleKind::FileDescriptor => {
                cuda_sys::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            }
            CUMemHandleKind::Fabric => cuda_sys::CU_MEM_HANDLE_TYPE_FABRIC,
        };

        props.allocFlags.gpuDirectRDMACapable = 1;

        let mut granularity: usize = 0;
        cuda_check!(cuda_sys::cuMemGetAllocationGranularity(
            &mut granularity,
            &props,
            cuda_sys::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        ))?;

        let alloc_size = size.div_ceil(granularity) * granularity;

        let mut handle =
            unsafe { std::mem::zeroed::<cuda_sys::CUmemGenericAllocationHandle>() };
        cuda_check!(cuda_sys::cuMemCreate(&mut handle, alloc_size, &props, 0,))?;

        Ok(Self {
            handle: Arc::new(CUMemAlloc { handle, alloc_size, device_id, handle_kind }),
        })
    }

    /// Exports the mapping as a file descriptor that can be shared with other processes.
    pub fn export(&self) -> CudaResult<CUMemExportHandle> {
        Ok(CUMemExportHandle(
            CUMemExport {
                handle: GenericExportHandle::new(
                    self.handle.handle_kind,
                    self.handle.handle,
                )?,
                alloc_size: self.handle.alloc_size,
                device_id: self.handle.device_id,
            },
            Some(self.handle.clone()),
        ))
    }

    /// Return the allocation size.
    pub fn size(&self) -> usize {
        self.handle.alloc_size
    }
}

impl CUAllocHandle for CUMemAllocHandle {
    fn map(&self, device: Device) -> CudaResult<CUMemMapping> {
        CUMemAlloc::map(CUMemMappingRef::Alloc(self.handle.clone()), device)
    }

    fn map_to(&self, mapping: CUMemMapping) -> CudaResult<()> {
        CUMemAlloc::map_to(CUMemMappingRef::Alloc(self.handle.clone()), mapping)
    }
}

/// An owning handle to a CUmem memory mapping.
struct CUMemMappingHolder {
    alloc_size: usize,
    device_id: CudaDeviceId,
    mapped: AtomicBool,
    ptr: cuda_sys::CUdeviceptr,
    _ref: CUMemMappingRef,
}

impl Drop for CUMemMappingHolder {
    fn drop(&mut self) {
        if self.mapped.load(Ordering::Relaxed) {
            cuda_unwrap!(cuda_sys::cuMemUnmap(self.ptr, self.alloc_size));
        }
        cuda_unwrap!(cuda_sys::cuMemAddressFree(self.ptr, self.alloc_size));
    }
}

#[derive(Clone)]
pub struct CUMemMapping {
    handle: Arc<CUMemMappingHolder>,
}

impl CUMemMapping {
    /// Return the size of the mapped region in bytes.
    pub fn size(&self) -> usize {
        self.handle.alloc_size
    }

    /// Return a device pointer to the mapped region.
    pub fn data_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked(self.handle.ptr as *mut c_void) }
    }

    /// Return the device index.
    pub fn device_id(&self) -> CudaDeviceId {
        self.handle.device_id
    }

    /// Unmap the memory mapping from the backing physical memory.
    pub fn unmap(&self) -> CudaResult<()> {
        if self.handle.mapped.swap(false, Ordering::Relaxed) {
            cuda_check!(cuda_sys::cuMemUnmap(self.handle.ptr, self.handle.alloc_size))?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct CUMemImportHandle {
    handle: Arc<CUMemAlloc>,
}

impl CUAllocHandle for CUMemImportHandle {
    fn map(&self, device: Device) -> CudaResult<CUMemMapping> {
        CUMemAlloc::map(CUMemMappingRef::Alloc(self.handle.clone()), device)
    }

    fn map_to(&self, mapping: CUMemMapping) -> CudaResult<()> {
        CUMemAlloc::map_to(CUMemMappingRef::Alloc(self.handle.clone()), mapping)
    }
}

/// The serialized payload for an exportable handle.
#[derive(Encode, Decode)]
pub enum GenericSerializedHandle {
    /// A handle exported via pidfd_getfd.
    FD { fd: i32, pid: u32 },
    /// A fabric handle exported via IMEX.
    Fabric(Vec<u8>),
}

/// The de-serialized, in-process representation of an exportable handle.
pub enum GenericExportHandle {
    ExportedFD { fd: i32, pid: u32 },
    ImportedFD { local_fd: i32 },
    Fabric { handle: cuda_sys::CUmemFabricHandle },
}

impl Drop for GenericExportHandle {
    fn drop(&mut self) {
        match self {
            GenericExportHandle::ExportedFD { fd, .. } => {
                unsafe { libc::close(*fd) };
            }
            GenericExportHandle::ImportedFD { local_fd } => {
                unsafe { libc::close(*local_fd) };
            }
            GenericExportHandle::Fabric { .. } => {}
        }
    }
}

impl GenericExportHandle {
    fn new(
        handle_kind: CUMemHandleKind,
        handle: cuda_sys::CUmemGenericAllocationHandle,
    ) -> CudaResult<Self> {
        match handle_kind {
            CUMemHandleKind::Local => Err(CudaError::CustomError(
                "Cannot export mapping with no handle".to_string(),
            )),
            CUMemHandleKind::FileDescriptor => {
                let pid = std::process::id();

                let mut fd: i32 = -1;
                cuda_check!(cuda_sys::cuMemExportToShareableHandle(
                    &mut fd as *mut _ as *mut std::ffi::c_void,
                    handle,
                    cuda_sys::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                ))?;

                Ok(GenericExportHandle::ExportedFD { fd, pid })
            }
            CUMemHandleKind::Fabric => {
                let mut fabric_handle =
                    unsafe { std::mem::zeroed::<cuda_sys::CUmemFabricHandle>() };
                cuda_check!(cuda_sys::cuMemExportToShareableHandle(
                    &mut fabric_handle as *mut _ as *mut std::ffi::c_void,
                    handle,
                    cuda_sys::CU_MEM_HANDLE_TYPE_FABRIC,
                    0,
                ))?;
                Ok(GenericExportHandle::Fabric { handle: fabric_handle })
            }
        }
    }

    fn bind(&self) -> CudaResult<cuda_sys::CUmemGenericAllocationHandle> {
        match self {
            GenericExportHandle::ExportedFD { .. } => {
                Err(CudaError::CustomError("Cannot bind exported FD".to_string()))
            }
            GenericExportHandle::ImportedFD { local_fd } => {
                let mut handle = unsafe {
                    std::mem::zeroed::<cuda_sys::CUmemGenericAllocationHandle>()
                };
                cuda_check!(cuda_sys::cuMemImportFromShareableHandle(
                    &mut handle,
                    *local_fd as *mut std::ffi::c_void,
                    cuda_sys::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                ))?;

                Ok(handle)
            }
            GenericExportHandle::Fabric { handle } => {
                let mut imported_handle = unsafe {
                    std::mem::zeroed::<cuda_sys::CUmemGenericAllocationHandle>()
                };
                cuda_check!(cuda_sys::cuMemImportFromShareableHandle(
                    &mut imported_handle,
                    handle as *const _ as *mut std::ffi::c_void,
                    cuda_sys::CU_MEM_HANDLE_TYPE_FABRIC,
                ))?;

                Ok(imported_handle)
            }
        }
    }
}

impl Encode for GenericExportHandle {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        match self {
            GenericExportHandle::ImportedFD { .. } => Err(EncodeError::OtherString(
                "Cannot encode imported handle".to_string(),
            )),
            GenericExportHandle::ExportedFD { fd, pid } => Encode::encode(
                &GenericSerializedHandle::FD { fd: *fd, pid: *pid },
                encoder,
            ),
            GenericExportHandle::Fabric { handle } => {
                let raw_handle = unsafe {
                    std::slice::from_raw_parts(
                        handle as *const _ as *const u8,
                        std::mem::size_of::<cuda_sys::CUmemFabricHandle>(),
                    )
                };
                Encode::encode(
                    &GenericSerializedHandle::Fabric(raw_handle.to_vec()),
                    encoder,
                )
            }
        }
    }
}

impl<Context> Decode<Context> for GenericExportHandle {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let payload = GenericSerializedHandle::decode(decoder)?;
        match payload {
            GenericSerializedHandle::FD { fd, pid } => {
                // Open the pidfd for the source process.
                let pidfd =
                    unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
                if pidfd < 0 {
                    return Err(DecodeError::OtherString(format!(
                        "failed to open import handle: {}",
                        CudaError::Errno(pidfd as i32)
                    )));
                }

                // Get the file descriptor from the source process.
                let local_fd =
                    unsafe { libc::syscall(libc::SYS_pidfd_getfd, pidfd, fd, 0) }
                        as i32;
                if local_fd < 0 {
                    unsafe { libc::close(pidfd) };
                    return Err(DecodeError::OtherString(format!(
                        "failed to open import handle: {}",
                        CudaError::Errno(local_fd as i32)
                    )));
                }

                // Close the pidfd as it's no longer needed.
                let ret = unsafe { libc::close(pidfd) };
                if ret != 0 {
                    return Err(DecodeError::OtherString(format!(
                        "failed to open import handle: {}",
                        CudaError::Errno(ret)
                    )));
                }

                Ok(GenericExportHandle::ImportedFD { local_fd })
            }
            GenericSerializedHandle::Fabric(data) => {
                if data.len() != std::mem::size_of::<cuda_sys::CUmemFabricHandle>() {
                    return Err(DecodeError::OtherString(
                        "invalid fabric handle size".to_string(),
                    ));
                }
                let mut handle =
                    unsafe { std::mem::zeroed::<cuda_sys::CUmemFabricHandle>() };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        &mut handle as *mut _ as *mut u8,
                        data.len(),
                    );
                }
                Ok(GenericExportHandle::Fabric { handle })
            }
        }
    }
}

bincode::impl_borrow_decode!(GenericExportHandle);

/// A wrapper over importable and exportable handles.
#[derive(Encode, Decode)]
pub struct CUMemExport {
    handle: GenericExportHandle,
    alloc_size: usize,
    device_id: CudaDeviceId,
}

pub struct CUMemExportHandle(pub CUMemExport, pub Option<Arc<CUMemAlloc>>);

impl CUMemExportHandle {
    pub fn bind(&self) -> CudaResult<CUMemImportHandle> {
        assert!(self.1.is_none(), "Handle is already bound");

        let handle = self.0.handle.bind()?;
        Ok(CUMemImportHandle {
            handle: Arc::new(CUMemAlloc {
                handle,
                alloc_size: self.0.alloc_size,
                device_id: self.0.device_id,
                handle_kind: CUMemHandleKind::Local,
            }),
        })
    }
}

pub struct CUMulticast {
    handle: cuda_sys::CUmemGenericAllocationHandle,
    alloc_size: usize,
    handle_kind: CUMemHandleKind,
}

impl Drop for CUMulticast {
    fn drop(&mut self) {
        cuda_unwrap!(cuda_sys::cuMemRelease(self.handle));
    }
}

pub struct CUMulticastHandle {
    handle: Arc<CUMulticast>,
    buffers: Vec<Arc<CUMemAlloc>>,
}

impl CUMulticastHandle {
    pub fn new(
        num_devices: u32,
        size: usize,
        handle_kind: CUMemHandleKind,
    ) -> CudaResult<Self> {
        let mut props =
            unsafe { std::mem::zeroed::<cuda_sys::CUmulticastObjectProp>() };
        props.numDevices = num_devices;
        props.size = size;
        props.handleTypes = match handle_kind {
            CUMemHandleKind::Local => cuda_sys::CU_MEM_HANDLE_TYPE_NONE,
            CUMemHandleKind::FileDescriptor => {
                cuda_sys::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            }
            CUMemHandleKind::Fabric => cuda_sys::CU_MEM_HANDLE_TYPE_FABRIC,
        }
        .into();

        let mut granularity: usize = 0;
        cuda_check!(cuda_sys::cuMulticastGetGranularity(
            &mut granularity,
            &props,
            cuda_sys::CU_MULTICAST_GRANULARITY_MINIMUM,
        ))?;

        let alloc_size = size.div_ceil(granularity) * granularity;
        props.size = alloc_size;

        let mut handle =
            unsafe { std::mem::zeroed::<cuda_sys::CUmemGenericAllocationHandle>() };
        cuda_check!(cuda_sys::cuMulticastCreate(&mut handle, &props))?;

        Ok(Self {
            handle: Arc::new(CUMulticast { handle, alloc_size, handle_kind }),
            buffers: Vec::new(),
        })
    }

    /// Exports the mapping as a file descriptor that can be shared with other processes.
    pub fn export(&self) -> CudaResult<CUMulticastExportHandle> {
        Ok(CUMulticastExportHandle(
            CUMulticastExport {
                handle: GenericExportHandle::new(
                    self.handle.handle_kind,
                    self.handle.handle,
                )?,
                alloc_size: self.handle.alloc_size,
            },
            Some(self.handle.clone()),
        ))
    }

    /// Add a device to the multicast group.
    pub fn add_device(&self, device_id: CudaDeviceId) -> CudaResult<()> {
        cuda_check!(cuda_sys::cuMulticastAddDevice(
            self.handle.handle,
            device_id.0 as i32,
        ))?;
        Ok(())
    }

    /// Bind a memory allocation to the multicast handle.
    pub fn bind_mem(&mut self, alloc_handle: &CUMemAllocHandle) -> CudaResult<()> {
        self.buffers.push(alloc_handle.handle.clone());
        cuda_check!(cuda_sys::cuMulticastBindMem(
            self.handle.handle,
            0,
            alloc_handle.handle.handle,
            0,
            alloc_handle.size(),
            0,
        ))?;
        Ok(())
    }
}

impl CUAllocHandle for CUMulticastHandle {
    fn map(&self, device: Device) -> CudaResult<CUMemMapping> {
        CUMemAlloc::map(CUMemMappingRef::Multicast(self.handle.clone()), device)
    }

    fn map_to(&self, mapping: CUMemMapping) -> CudaResult<()> {
        CUMemAlloc::map_to(CUMemMappingRef::Multicast(self.handle.clone()), mapping)
    }
}

/// A wrapper over importable and exportable handles.
#[derive(Encode, Decode)]
pub struct CUMulticastExport {
    handle: GenericExportHandle,
    alloc_size: usize,
}

pub struct CUMulticastExportHandle(pub CUMulticastExport, pub Option<Arc<CUMulticast>>);

impl CUMulticastExportHandle {
    pub fn bind(&self) -> CudaResult<CUMulticastHandle> {
        assert!(self.1.is_none(), "Handle is already bound");

        Ok(CUMulticastHandle {
            handle: Arc::new(CUMulticast {
                handle: self.0.handle.bind()?,
                alloc_size: self.0.alloc_size,
                handle_kind: CUMemHandleKind::Local,
            }),
            buffers: Vec::new(),
        })
    }
}
