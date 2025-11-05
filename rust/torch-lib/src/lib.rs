use std::any::Any;
use std::{
    ffi::{c_char, c_void},
    ptr::NonNull,
};

use cuda_lib::{CudaDeviceId, Device};
use cxx::UniquePtr;
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, PyAny, PyErr, PyResult, Python,
    exceptions::PyValueError,
};

#[cxx::bridge(namespace = "torch_lib")]
mod ffi {
    extern "Rust" {
        type FromBlobContext;
    }

    enum DeviceType {
        Cpu,
        Cuda,
    }

    #[allow(dead_code)]
    struct Device {
        device_type: DeviceType,
        device_index: u8,
    }

    #[allow(non_camel_case_types)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ScalarType {
        BOOL,
        I8,
        U8,
        I16,
        U16,
        I32,
        U32,
        I64,
        U64,
        F8_E4M3,
        F8_E5M2,
        F16,
        BF16,
        F32,
        F64,
    }

    unsafe extern "C++" {
        include!("torch-lib/src/torch_lib.h");
        unsafe fn from_blob(
            data_ptr: *mut c_char,
            shape: &[i64],
            dtype: ScalarType,
            device: Device,
            context: Box<FromBlobContext>,
        ) -> *mut c_char;

        unsafe fn torch_to_scalar_type(obj: *mut c_char) -> Result<ScalarType>;

        unsafe fn scalar_to_torch_type(scalar_type: ScalarType) -> Result<*mut c_char>;

        unsafe fn current_stream() -> u64;

        type TorchProfilerGuard;
        unsafe fn profile_range(name: String) -> UniquePtr<TorchProfilerGuard>;
    }
}

impl From<Device> for ffi::Device {
    fn from(device: Device) -> Self {
        match device {
            Device::Host => {
                ffi::Device { device_type: ffi::DeviceType::Cpu, device_index: 0 }
            }
            Device::Cuda(CudaDeviceId(device_id)) => ffi::Device {
                device_type: ffi::DeviceType::Cuda,
                device_index: device_id,
            },
        }
    }
}

#[allow(dead_code)]
struct FromBlobContext(Box<dyn Any + Send + Sync>);

pub use ffi::ScalarType;

impl ScalarType {
    pub fn element_size(self) -> usize {
        match self {
            ScalarType::BOOL => 1,
            ScalarType::U8 => 1,
            ScalarType::I8 => 1,
            ScalarType::I16 => 2,
            ScalarType::U16 => 2,
            ScalarType::I32 => 4,
            ScalarType::U32 => 4,
            ScalarType::I64 => 8,
            ScalarType::U64 => 8,
            ScalarType::F8_E4M3 => 1,
            ScalarType::F8_E5M2 => 1,
            ScalarType::F16 => 2,
            ScalarType::BF16 => 2,
            ScalarType::F32 => 4,
            ScalarType::F64 => 8,
            _ => panic!("Unsupported scalar type"),
        }
    }
}

/// Attempts to convert a PyTorch DType object to a scalar dtype.
impl<'py> FromPyObject<'_, 'py> for ScalarType {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        unsafe { ffi::torch_to_scalar_type(obj.as_ptr() as *mut c_char) }.map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to convert PyTorch dtype to ScalarType: {:?}",
                e
            ))
        })
    }
}

/// Wraps a scalar dtype into a PyTorch DType object.
impl<'py> IntoPyObject<'py> for ScalarType {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ptr = unsafe { ffi::scalar_to_torch_type(self) }.map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to convert ScalarType to PyTorch dtype: {:?}",
                e
            ))
        })?;
        let py_ptr = ptr as *mut pyo3::ffi::PyObject;
        Ok(unsafe { Bound::from_borrowed_ptr(py, py_ptr) })
    }
}

pub fn from_blob(
    data_ptr: NonNull<c_void>,
    shape: &[i64],
    dtype: ScalarType,
    device: Device,
    context: Box<dyn Any + Send + Sync>,
) -> *mut pyo3::ffi::PyObject {
    unsafe {
        ffi::from_blob(
            data_ptr.as_ptr() as *mut c_char,
            shape,
            dtype,
            device.into(),
            Box::new(FromBlobContext(context)),
        ) as *mut pyo3::ffi::PyObject
    }
}

pub fn current_stream() -> u64 {
    unsafe { ffi::current_stream() }
}

#[allow(dead_code)]
pub struct TorchProfilerGuard(UniquePtr<ffi::TorchProfilerGuard>);

unsafe impl Send for TorchProfilerGuard {}
unsafe impl Sync for TorchProfilerGuard {}

pub fn torch_profile_range(name: String) -> TorchProfilerGuard {
    TorchProfilerGuard(unsafe { ffi::profile_range(name) })
}

#[cfg(test)]
mod test_torch;
