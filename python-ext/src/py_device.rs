use cuda_lib::{CudaDeviceId, Device};
use pyo3::{
    Borrowed, FromPyObject, PyAny, PyErr, exceptions::PyValueError, types::PyAnyMethods,
};

pub(crate) struct PyDevice(pub Device);

impl<'py> FromPyObject<'_, 'py> for PyDevice {
    type Error = PyErr;

    fn extract(device: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        match device.getattr("type")?.extract::<&str>()? {
            "cpu" => Ok(PyDevice(Device::Host)),
            "cuda" => {
                let index =
                    device.getattr("index")?.extract::<Option<u8>>()?.unwrap_or(0);
                Ok(PyDevice(Device::Cuda(CudaDeviceId(index))))
            }
            device_type => Err(PyValueError::new_err(format!(
                "Unknown device type: {device_type}"
            ))),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_py_device() {
        Python::initialize();
        Python::attach(|py| {
            let torch = py.import("torch").unwrap();
            let cuda_device = torch.call_method1("device", ("cuda",)).unwrap();
            let PyDevice(device) = cuda_device.extract().unwrap();
            assert_eq!(device, Device::Cuda(CudaDeviceId(0)));

            let cpu_device = torch.call_method1("device", ("cpu",)).unwrap();
            let PyDevice(device) = cpu_device.extract().unwrap();
            assert_eq!(device, Device::Host);

            let cuda_device_2 = torch.call_method1("device", ("cuda:2",)).unwrap();
            let PyDevice(device) = cuda_device_2.extract().unwrap();
            assert_eq!(device, Device::Cuda(CudaDeviceId(2)));
        });
    }
}
