#![allow(clippy::too_many_arguments)]
use std::{
    collections::HashSet,
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
    ptr::NonNull,
    sync::Arc,
};

use bytes::Bytes;
use fabric_lib::{
    BouncingErrorCallback, BouncingRecvCallback, DomainInfo, FabricLibError,
    HostBufferAllocator, ImmCountCallback, RdmaDomainInfo, RdmaEngine, SendBuffer,
    SendRecvEngine, TopologyGroup, TransferCallback, TransferEngine,
    TransferEngineBuilder, UvmWatcherCallback,
    api::{
        DomainAddress, DomainGroupRouting, ImmTransferRequest, MemoryRegionDescriptor,
        MemoryRegionHandle, PagedTransferRequest, SingleTransferRequest,
        TransferRequest,
    },
    detect_topology,
};
use parking_lot::RwLock;
use pyo3::{
    Bound, Py, PyAny, PyRefMut, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods, PyModule, PyModuleMethods, PyType},
};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::py_device::PyDevice;

#[pyclass(name = "TopologyGroup", module = "pplx_garden._rust")]
pub struct PyTopologyGroup(pub TopologyGroup);

#[pymethods]
impl PyTopologyGroup {
    #[getter]
    fn get_cuda_device(&self) -> u8 {
        self.0.cuda_device
    }

    #[getter]
    fn get_numa(&self) -> u8 {
        self.0.numa
    }

    #[getter]
    fn get_domains<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyDomainInfo>>> {
        self.0.domains.iter().map(|d| Py::new(py, PyDomainInfo(d.clone()))).collect()
    }

    #[getter]
    fn get_cpus(&self) -> Vec<u16> {
        self.0.cpus.clone()
    }
}

#[pyclass(name = "DomainInfo", module = "pplx_garden._rust")]
#[derive(Clone)]
pub struct PyDomainInfo(pub DomainInfo);

#[pymethods]
impl PyDomainInfo {
    #[getter]
    fn get_name(&self) -> String {
        self.0.name().to_string()
    }

    #[getter]
    fn get_link_speed(&self) -> u64 {
        self.0.link_speed()
    }
}

#[pyclass(name = "PageIndices", module = "pplx_garden._rust")]
#[derive(Clone)]
pub(crate) struct PyPageIndices(pub(crate) Arc<Vec<u32>>);

#[pymethods]
impl PyPageIndices {
    #[new]
    fn new(indices: Vec<u32>) -> Self {
        PyPageIndices(Arc::new(indices))
    }
}

#[pyclass(frozen, module = "pplx_garden._rust")]
struct UvmWatcher {
    #[pyo3(get)]
    ptr: u64,
}

#[pyclass(name = "DomainAddress", frozen, eq, hash, module = "pplx_garden._rust")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PyDomainAddress(pub DomainAddress);

#[pymethods]
impl PyDomainAddress {
    #[new]
    fn new(bytes: Vec<u8>) -> Self {
        PyDomainAddress(DomainAddress(Bytes::from(bytes)))
    }

    fn __getnewargs__(&self) -> (Vec<u8>,) {
        (self.0.0.to_vec(),)
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, bytes: &[u8]) -> Self {
        PyDomainAddress(DomainAddress(Bytes::copy_from_slice(bytes)))
    }

    fn as_bytes(&self) -> PyResult<Vec<u8>> {
        Ok(self.0.0.to_vec())
    }

    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        Ok(PyDomainAddress(
            <DomainAddress as std::str::FromStr>::from_str(s)
                .map_err(|_| PyValueError::new_err("Invalid Address"))?,
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("DomainAddress.from_str('{}')", self.0))
    }
}

#[pyclass(name = "MemoryRegionDescriptor", module = "pplx_garden._rust")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyMemoryRegionDescriptor(pub MemoryRegionDescriptor);

#[pymethods]
impl PyMemoryRegionDescriptor {
    #[new]
    fn new(bytes: &[u8]) -> PyResult<Self> {
        Ok(PyMemoryRegionDescriptor(postcard::from_bytes(bytes).map_err(|e| {
            PyValueError::new_err(format!("Failed to deserialize: {}", e))
        })?))
    }

    fn __getnewargs__(&self) -> PyResult<(Vec<u8>,)> {
        self.as_bytes().map(|bytes| (bytes,))
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, bytes: &[u8]) -> PyResult<Self> {
        PyMemoryRegionDescriptor::new(bytes)
    }

    fn as_bytes(&self) -> PyResult<Vec<u8>> {
        postcard::to_allocvec(&self)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize: {}", e)))
    }

    fn debug_str(&self) -> String {
        let l: Vec<_> = self
            .0
            .addr_rkey_list
            .iter()
            .map(|(addr, rkey)| {
                format!(
                    "{{addr: {}, ptr: 0x{:012x}, rkey: 0x{:08x}}}",
                    addr, self.0.ptr, rkey.0
                )
            })
            .collect();
        format!("[{}]", l.join(", "))
    }
}

#[pyclass(name = "MemoryRegionHandle", module = "pplx_garden._rust")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyMemoryRegionHandle(pub MemoryRegionHandle);

#[pymethods]
impl PyMemoryRegionHandle {
    fn debug_str(&self) -> String {
        format!("{{ptr: {:p}}}", self.0.ptr.as_ptr())
    }
}

#[pyclass(name = "TransferEngineBuilder", module = "pplx_garden._rust")]
pub(crate) struct PyTransferEngineBuilder {
    builder: TransferEngineBuilder,
}

#[pymethods]
impl PyTransferEngineBuilder {
    fn add_gpu_domains<'py>(
        mut slf: PyRefMut<'py, Self>,
        cuda_device: u8,
        domains: Vec<PyDomainInfo>,
        pin_worker_cpu: u16,
        pin_uvm_cpu: u16,
    ) -> PyRefMut<'py, Self> {
        slf.builder.add_gpu_domains(
            cuda_device,
            domains.into_iter().map(|domain| domain.0).collect(),
            pin_worker_cpu,
            pin_uvm_cpu,
        );
        slf
    }

    fn build(&self) -> PyResult<PyTransferEngine> {
        let engine = Arc::new(self.builder.build()?);
        Ok(PyTransferEngine::wrap(engine))
    }
}

fn make_transfer_callback(on_done: Py<PyAny>, on_error: Py<PyAny>) -> TransferCallback {
    let on_done = Box::new(move || {
        Python::attach(|py| {
            on_done
                .call0(py)
                .map_err(|e| format!("Failed to call on_done callback: {}", e))?;
            Ok(())
        })
    });

    let on_error = Box::new(move |err: FabricLibError| {
        Python::attach(|py| {
            on_error
                .call1(py, (err.to_string(),))
                .map_err(|e| format!("Failed to call on_error callback: {}", e))?;
            Ok(())
        })
    });

    TransferCallback { on_done, on_error }
}

#[pyclass(name = "TransferEngine", module = "pplx_garden._rust")]
pub(crate) struct PyTransferEngine {
    engine: Arc<TransferEngine>,
    allocator: HostBufferAllocator<TransferEngine>,
    imm_callback: Arc<RwLock<Option<Py<PyAny>>>>,
}

impl PyTransferEngine {
    pub fn wrap(engine: Arc<TransferEngine>) -> Self {
        Self {
            allocator: HostBufferAllocator::new(engine.clone()),
            engine,
            imm_callback: Arc::new(RwLock::new(None)),
        }
    }

    pub fn get_fabric_engine(&self) -> Arc<TransferEngine> {
        self.engine.clone()
    }
}

#[pymethods]
impl PyTransferEngine {
    #[new]
    fn new(nets_per_gpu: usize, cuda_devices: Vec<u8>) -> PyResult<Self> {
        let system_topo = detect_topology().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to detect topology: {}", e))
        })?;

        let mut builder = TransferEngineBuilder::default();

        let mut registered_devices = HashSet::new();
        for group in system_topo {
            if !cuda_devices.contains(&group.cuda_device) {
                continue;
            }

            if group.domains.len() < nets_per_gpu {
                return Err(PyRuntimeError::new_err(format!(
                    "Not enough NICs for cuda:{}: Expected {}, got {}",
                    group.cuda_device,
                    nets_per_gpu,
                    group.domains.len()
                )));
            }

            let domains =
                group.domains.iter().take(nets_per_gpu).cloned().collect::<Vec<_>>();
            if group.cpus.len() < 2 {
                return Err(PyRuntimeError::new_err(format!(
                    "Not enough CPUs for cuda:{}: Expected at least 2, got {}",
                    group.cuda_device,
                    group.cpus.len()
                )));
            }
            let worker_cpu = group.cpus[0];
            let uvm_cpu = group.cpus[1];
            builder.add_gpu_domains(
                group.cuda_device,
                domains.clone(),
                worker_cpu,
                uvm_cpu,
            );

            let domain_names: Vec<_> =
                domains.iter().map(|domain| domain.name().clone()).collect();
            tracing::info!(
                "TransferEngine using cuda:{}, Worker CPU #{}, UVM CPU #{}, domains {:?}",
                group.cuda_device,
                worker_cpu,
                uvm_cpu,
                domain_names
            );

            registered_devices.insert(group.cuda_device);
        }

        let missing_devices: Vec<_> = cuda_devices
            .iter()
            .filter(|device| !registered_devices.contains(device))
            .cloned()
            .collect();

        if !missing_devices.is_empty() {
            return Err(PyRuntimeError::new_err(format!(
                "Not all CUDA devices were registered: {:?}",
                missing_devices
            )));
        }

        let engine = Arc::new(builder.build().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to build TransferEngine: {}", e))
        })?);

        tracing::info!(
            "TransferEngine addr={}, nets_per_gpu={}",
            engine.main_address(),
            nets_per_gpu
        );

        Ok(PyTransferEngine::wrap(engine))
    }

    #[staticmethod]
    fn detect_topology() -> PyResult<Vec<PyTopologyGroup>> {
        detect_topology()
            .map(|system_topo| {
                system_topo.iter().map(|topo| PyTopologyGroup(topo.clone())).collect()
            })
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to detect topology: {}", e))
            })
    }

    #[staticmethod]
    fn builder() -> PyTransferEngineBuilder {
        PyTransferEngineBuilder { builder: TransferEngineBuilder::default() }
    }

    #[getter]
    fn get_main_address(&self) -> PyDomainAddress {
        PyDomainAddress(self.engine.main_address())
    }

    #[getter]
    fn get_num_domains(&self) -> usize {
        self.engine.num_domains()
    }

    #[getter]
    fn get_aggregated_link_speed(&self) -> u64 {
        self.engine.aggregated_link_speed()
    }

    #[getter]
    fn get_nets_per_gpu(&self) -> NonZeroU8 {
        self.engine.nets_per_gpu()
    }

    fn register_tensor<'py>(
        &self,
        tensor: &Bound<'py, PyAny>,
    ) -> PyResult<(PyMemoryRegionHandle, PyMemoryRegionDescriptor)> {
        // Determine the device index.
        let PyDevice(device) = tensor.getattr("device")?.extract()?;

        // Get the data pointer.
        let data_ptr: u64 = tensor.call_method0("data_ptr")?.extract()?;
        let ptr = NonNull::new(data_ptr as *mut c_void)
            .ok_or_else(|| PyValueError::new_err("Invalid data pointer"))?;
        let contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
        if !contiguous {
            return Err(PyValueError::new_err("Tensor is not contiguous"));
        }

        // Verify the allocation size.
        let numel = tensor.call_method0("numel")?.extract::<usize>()?;
        let elsz = tensor.call_method0("element_size")?.extract::<usize>()?;
        let len = numel * elsz;
        if len % 4096 != 0 {
            return Err(PyValueError::new_err("Tensor size is not page-aligned"));
        }

        let (handle, desc) = self
            .engine
            .register_memory_allow_remote(ptr, len, device)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to register memory: {}", e))
            })?;
        Ok((PyMemoryRegionHandle(handle), PyMemoryRegionDescriptor(desc)))
    }

    fn register_memory<'py>(
        &self,
        ptr: u64,
        len: usize,
        device: &Bound<'py, PyAny>,
    ) -> PyResult<(PyMemoryRegionHandle, PyMemoryRegionDescriptor)> {
        let PyDevice(device) = device.extract()?;
        // Verify the allocation if page-aligned.
        if !len.is_multiple_of(4096) {
            return Err(PyValueError::new_err("Tensor size is not page-aligned"));
        }

        let alloc = NonNull::new(ptr as *mut c_void)
            .ok_or_else(|| PyValueError::new_err("Invalid data pointer"))?;

        let (handle, desc) = self
            .engine
            .register_memory_allow_remote(alloc, len, device)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to register memory: {}", e))
            })?;
        Ok((PyMemoryRegionHandle(handle), PyMemoryRegionDescriptor(desc)))
    }

    fn unregister_memory(&self, ptr: u64) -> PyResult<()> {
        let ptr = NonNull::new(ptr as *mut c_void)
            .ok_or_else(|| PyValueError::new_err("Invalid data pointer"))?;
        self.engine.unregister_memory(ptr).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to unregister memory: {}", e))
        })
    }

    fn set_imm_callback(&self, callback: Py<PyAny>) -> PyResult<()> {
        let imm_callback_ref = self.imm_callback.clone();
        let mut imm_callback = self.imm_callback.write();
        match *imm_callback {
            Some(_) => {
                imm_callback.replace(callback);
                Ok(())
            }
            None => {
                imm_callback.replace(callback);
                self.engine.add_imm_callback(Box::new(move |imm_data: u32| {
                    Python::attach(|py| {
                        let imm_callback = imm_callback_ref.read();
                        let Some(callback) = imm_callback.as_ref() else {
                            warn!(imm_data, "Imm callback not set. Ignoring imm data");
                            return Ok(());
                        };
                        callback.call1(py, (imm_data,)).map_err(|e| {
                            format!("Failed to call on_error callback: {}", e)
                        })?;
                        Ok(())
                    })
                }));
                Ok(())
            }
        }
    }

    fn set_imm_count_expected(
        &self,
        imm: u32,
        expected_count: u32,
        on_reached: Py<PyAny>,
    ) -> PyResult<Option<(u32, Option<NonZeroU32>)>> {
        let expected_count = NonZeroU32::new(expected_count)
            .ok_or_else(|| PyValueError::new_err("Expected count cannot be zero"))?;

        // Create a single-shot callback from Python.
        let callback: ImmCountCallback = Box::new(move || {
            Python::attach(|py| {
                on_reached.call0(py).map_err(|e| {
                    format!("Failed to call on_reached callback: {}", e)
                })?;
                Ok(false)
            })
        });

        Ok(self
            .engine
            .set_imm_count_expected(imm, expected_count, callback)
            .map(|imm_count| imm_count.consume()))
    }

    fn remove_imm_count(&self, imm: u32) -> Option<(u32, Option<NonZeroU32>)> {
        self.engine.remove_imm_count(imm).map(|imm_count| imm_count.consume())
    }

    fn alloc_scalar_watcher(&self, on_changed: Py<PyAny>) -> PyResult<UvmWatcher> {
        let callback: UvmWatcherCallback = Box::new(move |old: u64, new: u64| {
            Python::attach(|py| on_changed.call1(py, (old, new))?.extract::<bool>(py))
                .map_err(|e| format!("Failed to call on_changed callback: {}", e))
        });

        let watcher_id = self.engine.alloc_scalar_watcher(callback).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to allocate scalar watcher: {}", e))
        })?;

        Ok(UvmWatcher { ptr: watcher_id.as_u64() })
    }

    fn submit_bouncing_recvs(
        &self,
        count: usize,
        len: usize,
        on_recv: Py<PyAny>,
        on_error: Py<PyAny>,
    ) -> PyResult<()> {
        let on_recv_wrapper: BouncingRecvCallback =
            Arc::new(Box::new(move |data: &[u8]| -> Result<(), String> {
                Python::attach(|py| {
                    on_recv
                        .call1(py, (data,))
                        .map_err(|e| format!("failed to call on_recv: {}", e))?;
                    Ok(())
                })
            }));

        let on_error_wrapper: BouncingErrorCallback =
            Arc::new(Box::new(move |err: FabricLibError| -> Result<(), String> {
                Python::attach(|py| {
                    on_error
                        .call1(py, (err.to_string(),))
                        .map_err(|e| format!("failed to call on_error: {}", e))?;
                    Ok(())
                })
            }));

        self.engine
            .submit_bouncing_recvs(len, count, on_recv_wrapper, on_error_wrapper)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to submit RECV: {}", e))
            })
    }

    fn submit_send(
        &self,
        addr: PyDomainAddress,
        data: &[u8],
        on_done: Py<PyAny>,
        on_error: Py<PyAny>,
    ) -> PyResult<()> {
        // NOTE(lequn): Currently we do a copy from python bytes to registered
        // memory region. If we come up with a better way to design Python APIs,
        // this can be zero-copy.

        // Allocate a buffer
        let mut buf = self.allocator.allocate(data.len()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to allocate host buffer: {}", e))
        })?;

        // Copy data to the buffer
        buf.as_mut_slice()[..data.len()].copy_from_slice(data);

        // Create a raw pointer into the storage.
        let send_buffer =
            SendBuffer::new(buf.as_nonnull(), data.len(), buf.mr_handle());

        // Prepare the callbacks
        let callback = Box::new(move |result: Result<(), FabricLibError>| {
            // Release the buffer.
            drop(buf);

            // Call the callbacks
            let ret = Python::attach(|py| match result {
                Ok(_) => on_done
                    .call0(py)
                    .map_err(|e| format!("Failed to call on_done callback: {}", e)),
                Err(err) => on_error
                    .call1(py, (err.to_string(),))
                    .map_err(|e| format!("Failed to call on_error callback: {}", e)),
            });
            ret.map(|_| ()) // Ignore the return value of the callback
        });

        // Submit SEND op
        self.engine.submit_send(addr.0, send_buffer, callback).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to submit SEND: {}", e))
        })
    }

    fn submit_imm<'py>(
        &self,
        py: Python<'py>,
        imm_data: u32,
        dst_mr: PyMemoryRegionDescriptor,
        on_done: Py<PyAny>,
        on_error: Py<PyAny>,
    ) -> PyResult<()> {
        let write_op = ImmTransferRequest {
            imm_data,
            dst_mr: dst_mr.0,
            domain: DomainGroupRouting::Pinned { domain_idx: 0 },
        };

        let callback = make_transfer_callback(on_done, on_error);

        Python::detach(py, || {
            self.engine
                .submit_transfer(TransferRequest::Imm(write_op), callback)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to submit imm: {}", e))
                })
        })
    }

    #[pyo3(signature = (src_mr, offset, length, imm_data, dst_mr, dst_offset, on_done, on_error, num_shards=None))]
    fn submit_write<'py>(
        &self,
        py: Python<'py>,
        src_mr: PyMemoryRegionHandle,
        offset: usize,
        length: usize,
        imm_data: Option<u32>,
        dst_mr: PyMemoryRegionDescriptor,
        dst_offset: usize,
        on_done: Py<PyAny>,
        on_error: Py<PyAny>,
        num_shards: Option<NonZeroU8>,
    ) -> PyResult<()> {
        if let Some(num_shards) = num_shards
            && num_shards.get() as usize > self.engine.num_domains()
        {
            return Err(PyValueError::new_err(format!(
                "num_shards is too big. num_shards={}, num_domains={}",
                num_shards.get(),
                self.engine.num_domains()
            )));
        }

        let write_op = SingleTransferRequest {
            src_mr: src_mr.0,
            src_offset: offset as u64,
            length: length as u64,
            imm_data,
            dst_mr: dst_mr.0,
            dst_offset: dst_offset as u64,
            domain: DomainGroupRouting::RoundRobinSharded {
                num_shards: num_shards.unwrap_or(self.engine.nets_per_gpu()),
            },
        };

        let callback = make_transfer_callback(on_done, on_error);

        Python::detach(py, || {
            self.engine
                .submit_transfer(TransferRequest::Single(write_op), callback)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to submit paged writes: {}",
                        e
                    ))
                })
        })
    }

    fn submit_paged_writes<'py>(
        &self,
        py: Python<'py>,
        length: usize,
        src_mr: PyMemoryRegionHandle,
        src_page_indices: &Bound<'py, PyPageIndices>,
        src_stride: usize,
        src_offset: usize,
        dst_mr: PyMemoryRegionDescriptor,
        dst_page_indices: &Bound<'py, PyPageIndices>,
        dst_stride: usize,
        dst_offset: usize,
        imm_data: Option<u32>,
        on_done: Py<PyAny>,
        on_error: Py<PyAny>,
    ) -> PyResult<()> {
        let write_op = PagedTransferRequest {
            length: length as u64,
            src_mr: src_mr.0,
            src_page_indices: Arc::clone(&src_page_indices.borrow().0),
            src_stride: src_stride as u64,
            src_offset: src_offset as u64,
            dst_mr: dst_mr.0,
            dst_page_indices: Arc::clone(&dst_page_indices.borrow().0),
            dst_stride: dst_stride as u64,
            dst_offset: dst_offset as u64,
            imm_data,
        };

        let callback = make_transfer_callback(on_done, on_error);

        Python::detach(py, || {
            self.engine
                .submit_transfer(TransferRequest::Paged(write_op), callback)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to submit paged writes: {}",
                        e
                    ))
                })
        })
    }

    fn stop(&self) {
        self.engine.stop();
    }
}

impl Drop for PyTransferEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransferEngine>()?;
    m.add_class::<PyTransferEngineBuilder>()?;
    m.add_class::<PyPageIndices>()?;
    m.add_class::<PyDomainAddress>()?;
    m.add_class::<PyMemoryRegionHandle>()?;
    m.add_class::<PyMemoryRegionDescriptor>()?;
    m.add_class::<PyDomainInfo>()?;
    m.add_class::<PyTopologyGroup>()?;
    Ok(())
}
