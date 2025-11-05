use pyo3::{
    Bound, Py, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAny, PyAnyMethods, PyModule, PyModuleMethods},
};

use cuda_lib::{
    Device,
    cumem::{
        CUAllocHandle, CUMemAllocHandle, CUMemExportHandle, CUMemHandleKind,
        CUMemImportHandle, CUMemMapping, CUMulticastExportHandle, CUMulticastHandle,
    },
};
use torch_lib::ScalarType;

use crate::py_device::PyDevice;

#[pyclass(name = "CUMemHandleKind", module = "pplx_garden._rust")]
#[derive(Clone)]
pub enum PyCUMemHandleKind {
    Local,
    FileDescriptor,
    Fabric,
}

impl From<PyCUMemHandleKind> for CUMemHandleKind {
    fn from(py_handle_kind: PyCUMemHandleKind) -> Self {
        match py_handle_kind {
            PyCUMemHandleKind::Local => CUMemHandleKind::Local,
            PyCUMemHandleKind::FileDescriptor => CUMemHandleKind::FileDescriptor,
            PyCUMemHandleKind::Fabric => CUMemHandleKind::Fabric,
        }
    }
}

#[pyclass(name = "CUMemMapping", module = "pplx_garden._rust")]
#[derive(Clone)]
pub struct PyCUMemMapping(pub CUMemMapping);

#[pymethods]
impl PyCUMemMapping {
    fn to_tensor<'py>(
        &self,
        py: Python<'py>,
        shape: Vec<i64>,
        dtype: ScalarType,
    ) -> PyResult<Py<PyAny>> {
        let handle = &self.0;

        let data_ptr = handle.data_ptr();
        let size = handle.size();
        let device = Device::Cuda(handle.device_id());

        // Size check, as the tensor must fit into the allocation.
        let mut numel = dtype.element_size();
        for &dim in &shape {
            numel *= dim as usize;
        }
        if numel > size {
            return Err(PyRuntimeError::new_err(format!(
                "Requested shape {:?} with {} elements exceeds allocated size of {} bytes",
                shape, numel, size
            )));
        }

        // Build the python object.
        let context = Box::new(handle.clone());
        let object = torch_lib::from_blob(data_ptr, &shape, dtype, device, context);
        Ok(unsafe { Py::from_owned_ptr(py, object) })
    }

    fn data_ptr(&self) -> u64 {
        self.0.data_ptr().as_ptr() as u64
    }

    #[getter]
    fn get_size(&self) -> usize {
        self.0.size()
    }

    fn unmap(&self) -> PyResult<()> {
        self.0.unmap().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to unmap CUMemMapping: {}", e))
        })
    }
}

#[pyclass(name = "CUMemAllocHandle", module = "pplx_garden._rust")]
struct PyCUMemAllocHandle(CUMemAllocHandle);

#[pymethods]
impl PyCUMemAllocHandle {
    #[new]
    #[pyo3(signature = (size, device, handle_kind=PyCUMemHandleKind::Local))]
    fn new<'py>(
        size: usize,
        device: &Bound<'py, PyAny>,
        handle_kind: PyCUMemHandleKind,
    ) -> PyResult<Self> {
        let PyDevice(device) = device.extract()?;
        let device_id = match device {
            Device::Cuda(id) => id,
            Device::Host => return Err(PyRuntimeError::new_err("Device is not cuda")),
        };
        let handle = CUMemAllocHandle::new(size, device_id, handle_kind.into())
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to create CUMemAllocHandle: {}",
                    e
                ))
            })?;
        Ok(Self(handle))
    }

    #[pyo3(signature = (device=None))]
    fn map<'py>(
        &self,
        py: Python<'py>,
        device: Option<Py<PyAny>>,
    ) -> PyResult<PyCUMemMapping> {
        let device = device
            .map(|dev| dev.extract::<PyDevice>(py))
            .transpose()?
            .map(|PyDevice(d)| d)
            .unwrap_or(Device::Host);

        let mapping = self.0.map(device).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to map CUMemAllocHandle: {}", e))
        })?;
        Ok(PyCUMemMapping(mapping))
    }

    fn map_to(&self, mapping: &PyCUMemMapping) -> PyResult<()> {
        self.0.map_to(mapping.0.clone()).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to map_to CUMemImportHandle: {}",
                e
            ))
        })
    }

    fn export(&self) -> PyResult<PyCUMemExportHandle> {
        let handle = self.0.export().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to export CUMemAllocHandle: {}", e))
        })?;
        Ok(PyCUMemExportHandle(handle))
    }
}

#[pyclass(name = "CUMemImportHandle", module = "pplx_garden._rust")]
struct PyCUMemImportHandle(CUMemImportHandle);

#[pymethods]
impl PyCUMemImportHandle {
    #[pyo3(signature = (device=None))]
    fn map<'py>(
        &self,
        py: Python<'py>,
        device: Option<Py<PyAny>>,
    ) -> PyResult<PyCUMemMapping> {
        let device = device
            .map(|dev| dev.extract::<PyDevice>(py))
            .transpose()?
            .map(|PyDevice(d)| d)
            .unwrap_or(Device::Host);

        let mapping = self.0.map(device).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to map CUMemAllocHandle: {}", e))
        })?;
        Ok(PyCUMemMapping(mapping))
    }

    fn map_to(&self, mapping: &PyCUMemMapping) -> PyResult<()> {
        self.0.map_to(mapping.0.clone()).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to map_to CUMemImportHandle: {}",
                e
            ))
        })
    }
}

#[pyclass(name = "CUMemExportHandle", module = "pplx_garden._rust")]
struct PyCUMemExportHandle(CUMemExportHandle);

#[pymethods]
impl PyCUMemExportHandle {
    #[new]
    fn new(payload: Vec<u8>) -> PyResult<Self> {
        let (handle, _) =
            bincode::decode_from_slice(&payload, bincode::config::standard()).map_err(
                |e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to deserialize CUMemExportHandle: {}",
                        e
                    ))
                },
            )?;
        Ok(Self(CUMemExportHandle(handle, None)))
    }

    fn __getnewargs__(&self) -> PyResult<(Vec<u8>,)> {
        let payload = bincode::encode_to_vec(&self.0.0, bincode::config::standard())
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to serialize CUMemExportHandle: {}",
                    e
                ))
            })?;
        Ok((payload,))
    }

    fn bind(&self) -> PyResult<PyCUMemImportHandle> {
        let handle = self.0.bind().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to bind CUMemExportHandle: {}", e))
        })?;
        Ok(PyCUMemImportHandle(handle))
    }
}

#[pyclass(name = "CUMulticastHandle", module = "pplx_garden._rust")]
struct PyCUMulticastHandle(CUMulticastHandle);

#[pymethods]
impl PyCUMulticastHandle {
    #[new]
    #[pyo3(signature = (num_devices, size, handle_kind=PyCUMemHandleKind::Local))]
    fn new(
        num_devices: u32,
        size: usize,
        handle_kind: PyCUMemHandleKind,
    ) -> PyResult<Self> {
        let handle = CUMulticastHandle::new(num_devices, size, handle_kind.into())
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to create CUMulticastHandle: {}",
                    e
                ))
            })?;
        Ok(Self(handle))
    }

    fn add_device<'py>(&self, device: Bound<'py, PyAny>) -> PyResult<()> {
        let PyDevice(Device::Cuda(device_id)) = device.extract::<PyDevice>()? else {
            return Err(PyRuntimeError::new_err("Expected CUDA device"));
        };
        self.0.add_device(device_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to add_device: {}", e))
        })
    }

    fn bind_mem(&mut self, alloc_handle: &PyCUMemAllocHandle) -> PyResult<()> {
        self.0
            .bind_mem(&alloc_handle.0)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind_mem: {}", e)))
    }

    #[pyo3(signature = (device=None))]
    fn map<'py>(
        &self,
        py: Python<'py>,
        device: Option<Py<PyAny>>,
    ) -> PyResult<PyCUMemMapping> {
        let device = device
            .map(|dev| dev.extract::<PyDevice>(py))
            .transpose()?
            .map(|PyDevice(d)| d)
            .unwrap_or(Device::Host);

        let mapping = self.0.map(device).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to map CUMulticastHandle: {}", e))
        })?;
        Ok(PyCUMemMapping(mapping))
    }

    fn export(&self) -> PyResult<PyCUMulticastExportHandle> {
        let handle = self.0.export().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to export CUMulticastHandle: {}",
                e
            ))
        })?;
        Ok(PyCUMulticastExportHandle(handle))
    }
}

#[pyclass(name = "CUMulticastExportHandle", module = "pplx_garden._rust")]
struct PyCUMulticastExportHandle(CUMulticastExportHandle);

#[pymethods]
impl PyCUMulticastExportHandle {
    #[new]
    fn new(payload: Vec<u8>) -> PyResult<Self> {
        let (handle, _) =
            bincode::decode_from_slice(&payload, bincode::config::standard()).map_err(
                |e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to deserialize CUMulticastExportHandle: {}",
                        e
                    ))
                },
            )?;
        Ok(Self(CUMulticastExportHandle(handle, None)))
    }

    fn __getnewargs__(&self) -> PyResult<(Vec<u8>,)> {
        let payload = bincode::encode_to_vec(&self.0.0, bincode::config::standard())
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to serialize CUMulticastExportHandle: {}",
                    e
                ))
            })?;
        Ok((payload,))
    }

    fn bind(&self) -> PyResult<PyCUMulticastHandle> {
        let handle = self.0.bind().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to bind CUMulticastExportHandle: {}",
                e
            ))
        })?;
        Ok(PyCUMulticastHandle(handle))
    }
}

pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCUMemHandleKind>()?;
    m.add_class::<PyCUMemMapping>()?;
    m.add_class::<PyCUMemAllocHandle>()?;
    m.add_class::<PyCUMemImportHandle>()?;
    m.add_class::<PyCUMemExportHandle>()?;
    m.add_class::<PyCUMulticastHandle>()?;
    m.add_class::<PyCUMulticastExportHandle>()?;
    Ok(())
}
