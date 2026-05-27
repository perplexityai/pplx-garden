//! Metaspace pre-tokenization.

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct Metaspace {
    replacement_bytes: [u8; 4],
    replacement_len: usize,
    add_prefix_space: bool,
}

impl Metaspace {
    pub fn new(replacement: char, add_prefix_space: bool) -> Self {
        let mut bytes = [0u8; 4];
        replacement.encode_utf8(&mut bytes);
        Self {
            replacement_bytes: bytes,
            replacement_len: replacement.len_utf8(),
            add_prefix_space,
        }
    }

    /// Validates that pre-tokenizer and decoder configs agree, then builds.
    pub fn from_pre_and_decoder(
        pre: (char, bool),
        decoder: (char, bool),
    ) -> Result<Self> {
        if pre.0 != decoder.0 {
            return Err(Error::UnsupportedConfig(
                "metaspace replacement mismatch".into(),
            ));
        }
        if pre.1 != decoder.1 {
            return Err(Error::UnsupportedConfig(
                "metaspace add_prefix_space mismatch".into(),
            ));
        }
        Ok(Self::new(pre.0, pre.1))
    }

    pub fn encode_into(&self, text: &str, out: &mut Vec<u8>) {
        out.clear();
        if text.is_empty() {
            return;
        }
        let repl = &self.replacement_bytes[..self.replacement_len];
        if self.add_prefix_space && !text.starts_with(' ') {
            out.extend_from_slice(repl);
        }
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            if ch == ' ' {
                out.extend_from_slice(repl);
            } else {
                let encoded = ch.encode_utf8(&mut buf);
                out.extend_from_slice(encoded.as_bytes());
            }
        }
    }
}
