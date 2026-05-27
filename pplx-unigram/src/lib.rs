mod config;
mod engine;
mod error;
mod metaspace;
mod normalizer;
mod special_tokens;
mod state;
mod trie;

pub use engine::Engine;
pub use error::{Error, Result};
pub use state::EncodeState;

pub type TokenId = u32;

/// A span in the normalized text — either regular text to feed to Viterbi
/// or a matched special token that bypasses it.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    pub special_id: Option<TokenId>,
}

impl Segment {
    pub fn text(start: usize, end: usize) -> Self {
        Self { start, end, special_id: None }
    }
    pub fn special(start: usize, end: usize, special_id: TokenId) -> Self {
        Self { start, end, special_id: Some(special_id) }
    }
}

/// Penalty added to `min_score` to derive the UNK score. Mirrors HuggingFace.
const UNK_PENALTY: f64 = 10.0;
