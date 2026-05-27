//! Error and result types.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid tokenizer config: {0}")]
    InvalidConfig(String),
    #[error("unsupported tokenizer config: {0}")]
    UnsupportedConfig(String),
    #[error("encode failed: {0}")]
    EncodeFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
