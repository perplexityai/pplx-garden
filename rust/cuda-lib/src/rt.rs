use std::{
    ffi::{CStr, c_void},
    ptr::NonNull,
};

pub type CudaResult<T> = std::result::Result<T, CudartError>;

#[derive(Clone, Debug)]
pub struct CudartError {
    pub code: u32,
    pub context: &'static str,
}

impl CudartError {
    pub fn new(code: u32, context: &'static str) -> Self {
        Self { code, context }
    }
}

impl std::fmt::Display for CudartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudartError: code {} ({:?}), context: {}",
            self.code,
            unsafe { CStr::from_ptr(cudart_sys::cudaGetErrorString(self.code)) },
            self.context
        )
    }
}

impl std::error::Error for CudartError {}

pub use cudart_sys::{cudaMemoryTypeDevice, cudaPointerAttributes};
pub fn cudaPointerGetAttributes(
    ptr: NonNull<c_void>,
) -> CudaResult<cudaPointerAttributes> {
    let mut attrs = cudaPointerAttributes::default();
    let ret =
        unsafe { cudart_sys::cudaPointerGetAttributes(&raw mut attrs, ptr.as_ptr()) };
    match ret {
        0 => Ok(attrs),
        _ => Err(CudartError::new(ret, "cudaPointerGetAttributes")),
    }
}

pub use cudart_sys::cudaDeviceProp;
pub fn cudaGetDeviceProperties(device: i32) -> CudaResult<cudaDeviceProp> {
    let mut prop = cudaDeviceProp::default();
    let ret = unsafe { cudart_sys::cudaGetDeviceProperties(&raw mut prop, device) };
    match ret {
        0 => Ok(prop),
        _ => Err(CudartError::new(ret, "cudaGetDeviceProperties")),
    }
}

pub fn cudaGetDeviceCount() -> CudaResult<i32> {
    let mut count = 0;
    let ret = unsafe { cudart_sys::cudaGetDeviceCount(&raw mut count) };
    match ret {
        0 => Ok(count),
        _ => Err(CudartError::new(ret, "cudaGetDeviceCount")),
    }
}

pub fn cudaSetDevice(device: i32) -> CudaResult<()> {
    let ret = unsafe { cudart_sys::cudaSetDevice(device) };
    match ret {
        0 => Ok(()),
        _ => Err(CudartError::new(ret, "cudaSetDevice")),
    }
}

pub fn cudaHostAlloc(size: usize, flags: u32) -> CudaResult<NonNull<c_void>> {
    let mut ptr = std::ptr::null_mut();
    let ret = unsafe { cudart_sys::cudaHostAlloc(&raw mut ptr, size, flags) };
    match ret {
        0 => Ok(NonNull::new(ptr).unwrap()),
        _ => Err(CudartError::new(ret, "cudaHostAlloc")),
    }
}

pub fn cudaFreeHost(ptr: NonNull<c_void>) -> CudaResult<()> {
    let ret = unsafe { cudart_sys::cudaFreeHost(ptr.as_ptr()) };
    match ret {
        0 => Ok(()),
        _ => Err(CudartError::new(ret, "cudaFreeHost")),
    }
}

pub fn cudaGetNumSMs(device: u8) -> CudaResult<usize> {
    let mut numSMs = 0;
    let ret = unsafe {
        cuda_sys::cuDeviceGetAttribute(
            &mut numSMs,
            cudart_sys::cudaDevAttrMultiProcessorCount,
            device as i32,
        )
    };
    match ret {
        0 => Ok(numSMs as usize),
        _ => Err(CudartError::new(ret, "cudaGetNumSMs")),
    }
}
