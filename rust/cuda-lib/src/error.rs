use crate::{driver::CudaDriverError, rt::CudartError};

pub type CudaResult<T> = ::std::result::Result<T, CudaError>;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("{0}")]
    CudaDriver(#[from] CudaDriverError),
    #[error("{0}")]
    Cudart(#[from] CudartError),
    #[error("{0}")]
    CudaError(cuda_sys::CUresult),
    #[error("{0}")]
    GdrCopyError(&'static str),
    #[error("{0}")]
    CustomError(String),
    #[error("{0}")]
    Errno(i32),
}

#[macro_export]
macro_rules! cuda_check {
    ($x:expr) => {{
        let code = unsafe { $x } as u32;
        if code != $crate::cuda_sys::CUDA_SUCCESS {
            Err($crate::CudaError::Cudart($crate::rt::CudartError {
                code,
                context: "cuda_check call failed",
            }))
        } else {
            Ok(())
        }
    }};
}

#[macro_export]
macro_rules! cuda_unwrap {
    ($x:expr) => {{
        let ret = unsafe { $x } as u32;
        if ret != $crate::cuda_sys::CUDA_SUCCESS {
            panic!("cuda_unwrap call failed: {}", ret);
        }
    }};
}
