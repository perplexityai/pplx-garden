use std::ffi::c_void;
use std::ptr::NonNull;

use cuda_lib::Device;
use pyo3::{Py, PyAny, Python};

use crate::{ScalarType, from_blob};

#[test]
fn test_from_blob() {
    let data = vec![1, 2];

    Python::initialize();
    Python::attach(|py| {
        py.import("torch").expect("Failed to import torch");

        let tensor = from_blob(
            NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
            &[1, 2],
            ScalarType::I32,
            Device::Host,
            Box::new(data),
        );

        let tensor: Py<PyAny> = unsafe { Py::from_owned_ptr(py, tensor) };
        let shape = tensor.getattr(py, "shape")?.extract::<Vec<i64>>(py)?;
        let dtype = tensor.getattr(py, "dtype")?.bind(py).to_string();
        let device = tensor.getattr(py, "device")?.bind(py).to_string();

        assert_eq!(shape, vec![1, 2]);
        assert_eq!(dtype, "torch.int32");
        assert_eq!(device, "cpu");
        Ok::<(), pyo3::PyErr>(())
    })
    .unwrap();
}
