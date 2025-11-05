mod py_cumem;
mod py_device;
mod py_fabric_lib;
mod py_p2p_all_to_all;

use pyo3::{Bound, PyResult, pymodule, types::PyModule};

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = logging_lib::init(&logging_lib::LoggingOpts {
        log_color: logging_lib::LogColor::Auto,
        log_format: logging_lib::LogFormat::Text,
        log_directives: None,
    });

    py_cumem::init(m)?;
    py_p2p_all_to_all::init(m)?;
    py_fabric_lib::init(m)?;

    Ok(())
}
