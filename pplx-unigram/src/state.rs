//! Reusable encode scratch state.

use crate::TokenId;

/// Reusable scratch buffers for `Engine::encode`. Owning these in the caller
/// lets a hot loop encode many strings without per-call heap traffic — every
/// vector is cleared (not freed) between calls and grows monotonically with
/// the longest input the caller has seen.
///
/// Lifecycle of one `encode` call:
/// 1. `tokens` is reset and ultimately holds the result the caller reads.
/// 2. `normalized` gets the normalizer's UTF-8 output for the whole input.
/// 3. `segments` slices `normalized` around any special-token matches.
/// 4. For each non-special segment, `prep` gets the metaspace-encoded bytes.
/// 5. The Viterbi forward pass fills `best_score` / `best_start` / `best_id`
///    over `prep`, then the backward pass uses `backtrack` to reverse the
///    chosen path before appending it to `tokens`.
#[derive(Debug, Default)]
pub struct EncodeState {
    /// Output token IDs — the only field the caller reads after `encode`.
    pub tokens: Vec<TokenId>,
    /// Normalized text for the full input (one per call).
    pub normalized: String,
    /// Metaspace-preprocessed bytes for the current text segment.
    pub prep: Vec<u8>,
    /// Spans of `normalized` separated by special-token matches.
    pub segments: Vec<crate::Segment>,
    /// Viterbi DP: best score ending at each byte position of `prep`.
    pub best_score: Vec<f64>,
    /// Viterbi DP: start byte of the best path ending at each position.
    pub best_start: Vec<u32>,
    /// Viterbi DP: token chosen for the best path ending at each position.
    pub best_id: Vec<TokenId>,
    /// Scratch used by the backward pass to reverse the chosen path before
    /// pushing it onto `tokens`.
    pub backtrack: Vec<TokenId>,
}

impl EncodeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.tokens.clear();
        self.normalized.clear();
        self.prep.clear();
        self.segments.clear();
        self.backtrack.clear();
        // best_score / best_start / best_id are not cleared.
        // Viterbi forward pass refills the prefix it uses before reading from it.
    }
}
