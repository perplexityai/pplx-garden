use bincode::{Decode, Encode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct CudaDeviceId(pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Host,
    Cuda(CudaDeviceId),
}
