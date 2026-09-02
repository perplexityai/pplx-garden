//! Metaspace pre-tokenization, applied to one whitespace-delimited word.

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

    /// Encodes one word emitted by the preceding `WhitespaceSplit` step.
    ///
    /// The word holds no whitespace, so nothing needs substituting: the
    /// replacement character is purely the prefix that marks a word start.
    pub fn encode_word_into(&self, word: &str, out: &mut Vec<u8>) {
        out.clear();
        if self.add_prefix_space {
            out.extend_from_slice(&self.replacement_bytes[..self.replacement_len]);
        }
        out.extend_from_slice(word.as_bytes());
    }
}
