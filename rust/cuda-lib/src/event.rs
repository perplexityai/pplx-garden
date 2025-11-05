use crate::rt::{CudaResult, CudartError};

pub struct CudaEvent {
    pub event: cudart_sys::cudaEvent_t,
}

impl CudaEvent {
    pub fn new() -> CudaResult<Self> {
        let mut event = std::ptr::null_mut();
        let ret = unsafe { cudart_sys::cudaEventCreate(&mut event) };
        if ret != 0 {
            return Err(CudartError::new(ret, "cudaEventCreate"));
        }
        Ok(CudaEvent { event })
    }

    pub fn record(&self) -> CudaResult<()> {
        let ret =
            unsafe { cudart_sys::cudaEventRecord(self.event, std::ptr::null_mut()) };
        if ret != 0 {
            return Err(CudartError::new(ret, "cudaEventRecord"));
        }
        Ok(())
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        let ret = unsafe { cudart_sys::cudaEventSynchronize(self.event) };
        if ret != 0 {
            return Err(CudartError::new(ret, "cudaEventSynchronize"));
        }
        Ok(())
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let ret = unsafe { cudart_sys::cudaEventDestroy(self.event) };
        if ret != 0 {
            panic!("cudaEventDestroy failed: {}", ret);
        }
    }
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}
