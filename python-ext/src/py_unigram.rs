use std::path::PathBuf;

use pplx_unigram::{EncodeState, Engine, Error as UnigramError};
use pyo3::{
    Bound, PyResult,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyModule, PyModuleMethods},
};

fn map_err(err: UnigramError) -> pyo3::PyErr {
    match err {
        UnigramError::UnsupportedConfig(msg) | UnigramError::InvalidConfig(msg) => {
            PyValueError::new_err(msg)
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[pyclass(name = "UnigramEncodeState", module = "pplx_garden._rust")]
pub struct PyEncodeState {
    inner: EncodeState,
}

#[pymethods]
impl PyEncodeState {
    #[new]
    fn new() -> Self {
        Self { inner: EncodeState::new() }
    }

    #[getter]
    fn get_tokens(&self) -> Vec<u32> {
        self.inner.tokens.clone()
    }
}

#[pyclass(name = "UnigramEngine", module = "pplx_garden._rust")]
pub struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    #[staticmethod]
    fn from_hf_json(path: PathBuf) -> PyResult<Self> {
        Engine::from_hf_json_path(&path)
            .map(|inner| Self { inner })
            .map_err(map_err)
    }

    #[staticmethod]
    fn from_hf_json_bytes(bytes: &[u8]) -> PyResult<Self> {
        Engine::from_hf_json_bytes(bytes)
            .map(|inner| Self { inner })
            .map_err(map_err)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let mut state = EncodeState::new();
        self.inner.encode(text, &mut state).map_err(map_err)?;
        Ok(state.tokens)
    }

    fn encode_into(&self, text: &str, state: &mut PyEncodeState) -> PyResult<Vec<u32>> {
        self.inner.encode(text, &mut state.inner).map_err(map_err)?;
        Ok(state.inner.tokens.clone())
    }
}

pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEncodeState>()?;
    Ok(())
}
