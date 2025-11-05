#![allow(non_snake_case)]

pub use cuda_sys;
pub use cudart_sys;
pub use gdrapi_sys;
pub mod driver;
pub mod event;
pub mod gdr;
pub mod rt;

pub mod cumem;
mod error;
mod mem;
pub use error::{CudaError, CudaResult};
pub use mem::{CudaDeviceMemory, CudaHostMemory};
mod device;
pub use device::{CudaDeviceId, Device};

#[cfg(test)]
mod test_driver;

#[cfg(test)]
mod test_gdr;
