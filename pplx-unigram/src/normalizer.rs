//! SentencePiece precompiled normalizer with fused multi-space collapse.

use crate::{Error, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use spm_precompiled::Precompiled;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone)]
pub struct NormalizerPipeline {
    precompiled: Precompiled,
}

impl NormalizerPipeline {
    /// Builds from a base64-encoded SentencePiece precompiled charsmap.
    pub fn from_charsmap_b64(b64: &str) -> Result<Self> {
        let charsmap = STANDARD
            .decode(b64)
            .map_err(|err| Error::InvalidConfig(err.to_string()))?;
        let precompiled = Precompiled::from(&charsmap)
            .map_err(|err| Error::InvalidConfig(err.to_string()))?;
        Ok(Self { precompiled })
    }

    /// Walks `input` grapheme-by-grapheme (SentencePiece charsmaps are
    /// grapheme-keyed) and collapses runs of spaces in the same pass.
    pub fn normalize_into(&self, input: &str, out: &mut String) {
        out.clear();
        out.reserve(input.len());

        let mut prev_space = false;
        for grapheme in input.graphemes(true) {
            // Fast path: the precompiled charsmap keys are short (one or two
            // UTF-8 codepoints), so try a whole-grapheme lookup first when the
            // grapheme is short enough to plausibly match. Fall back to a
            // per-char loop on longer graphemes.
            if grapheme.len() < 6 {
                if let Some(normalized) = self.precompiled.transform(grapheme) {
                    push_collapsed_str(normalized, out, &mut prev_space);
                    continue;
                }
            }
            for (index, ch) in grapheme.char_indices() {
                let part = &grapheme[index..index + ch.len_utf8()];
                if let Some(normalized) = self.precompiled.transform(part) {
                    push_collapsed_str(normalized, out, &mut prev_space);
                } else {
                    push_collapsed_char(ch, out, &mut prev_space);
                }
            }
        }
    }
}

#[inline]
fn push_collapsed_str(text: &str, out: &mut String, prev_space: &mut bool) {
    for ch in text.chars() {
        push_collapsed_char(ch, out, prev_space);
    }
}

#[inline]
fn push_collapsed_char(ch: char, out: &mut String, prev_space: &mut bool) {
    if ch == ' ' {
        if !*prev_space {
            out.push(' ');
        }
        *prev_space = true;
    } else {
        out.push(ch);
        *prev_space = false;
    }
}
