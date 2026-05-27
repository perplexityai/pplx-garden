//! Unigram encoder.

use std::path::Path;

use crate::config;
use crate::metaspace::Metaspace;
use crate::normalizer::NormalizerPipeline;
use crate::special_tokens::{SpecialToken, SpecialTokenMatcher};
use crate::trie::DartsPackedTrie;
use crate::{EncodeState, Error, Result, TokenId, UNK_PENALTY};

pub struct Engine {
    components: Components,
    trie: DartsPackedTrie,
}

struct Components {
    unk_id: TokenId,
    unk_score: f64,
    normalizer: NormalizerPipeline,
    metaspace: Metaspace,
    special_matcher: SpecialTokenMatcher,
    vocab_size: usize,
}

impl Engine {
    /// Load a HuggingFace `tokenizer.json` and build the trie.
    pub fn from_hf_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_hf_json_bytes(&bytes)
    }

    /// Load from raw `tokenizer.json` bytes.
    pub fn from_hf_json_bytes(bytes: &[u8]) -> Result<Self> {
        let (components, vocab_bytes, vocab_scores) = load_components(bytes)?;
        let trie = DartsPackedTrie::from_vocab(&vocab_bytes, &vocab_scores);
        Ok(Self { components, trie })
    }

    /// Returns the full vocabulary size (base vocab + added tokens).
    pub fn vocab_size(&self) -> usize {
        self.components.vocab_size
    }

    /// Encode `text` into `state.tokens`.
    pub fn encode(&self, text: &str, state: &mut EncodeState) -> Result<()> {
        state.reset();
        let mut normalized = std::mem::take(&mut state.normalized);
        let mut segments = std::mem::take(&mut state.segments);
        self.components.normalizer.normalize_into(text, &mut normalized);
        self.components.special_matcher.split(&normalized, &mut segments, true);

        for &segment in &segments {
            if let Some(special_id) = segment.special_id {
                state.tokens.push(special_id);
                continue;
            }
            if segment.start == segment.end {
                continue;
            }
            self.encode_text_segment(&normalized[segment.start..segment.end], state)?;
        }

        // Both `normalize_into` and `split` clear their output at the top, so
        // we only restore the capacity-bearing buffers here.
        state.normalized = normalized;
        state.segments = segments;
        Ok(())
    }

    fn encode_text_segment(&self, text: &str, state: &mut EncodeState) -> Result<()> {
        let mut prep = std::mem::take(&mut state.prep);
        self.components.metaspace.encode_into(text, &mut prep);
        let prep_len = prep.len();

        ensure_dp_capacity(state, prep_len + 1);

        if prep_len > 0 {
            self.viterbi(&prep, state)?;
        }

        prep.clear();
        state.prep = prep;
        Ok(())
    }

    #[inline]
    fn viterbi(&self, prep: &[u8], state: &mut EncodeState) -> Result<()> {
        let n = prep.len();
        state.best_score[..n + 1].fill(f64::NEG_INFINITY);
        state.best_score[0] = 0.0;

        for pos in 0..n {
            if prep[pos] & 0xC0 == 0x80 {
                continue;
            }
            let s0 = state.best_score[pos];
            if s0 == f64::NEG_INFINITY {
                continue;
            }

            let mut h = &self.trie.node_hot[0];
            let mut has_single_char_match = false;
            for end in pos..n {
                let byte = prep[end];
                if !h.bitmap_test(byte) {
                    break;
                }
                // Double-array transition: `base[parent] + byte`. The wrapping
                // arithmetic is purely a typing convenience — `base` is signed
                // during placement (-1 marks free) but always lands on a
                // valid u32 slot for any reachable child here.
                let next_slot = h.base.wrapping_add(byte as i32) as u32;
                h = &self.trie.node_hot[next_slot as usize];

                if h.token_id >= 0 {
                    let ep = end + 1;
                    let total = s0 + h.score;
                    if total > state.best_score[ep] {
                        state.best_score[ep] = total;
                        state.best_start[ep] = pos as u32;
                        state.best_id[ep] = h.token_id as TokenId;
                    }
                    if !has_single_char_match && ep == next_char_boundary(prep, pos) {
                        has_single_char_match = true;
                    }
                }
            }

            if !has_single_char_match {
                let nxt = next_char_boundary(prep, pos);
                let score = s0 + self.components.unk_score;
                if score > state.best_score[nxt] {
                    state.best_score[nxt] = score;
                    state.best_start[nxt] = pos as u32;
                    state.best_id[nxt] = self.components.unk_id;
                }
            }
        }

        state.backtrack.clear();
        let mut end = n;
        while end > 0 {
            if state.best_score[end] == f64::NEG_INFINITY {
                return Err(Error::EncodeFailed("backtrack failed".into()));
            }
            let start = state.best_start[end] as usize;
            state.backtrack.push(state.best_id[end]);
            end = start;
        }

        let original_len = state.tokens.len();
        let unk_id = self.components.unk_id;
        for &token_id in state.backtrack.iter().rev() {
            // Fuse consecutive UNKs within a single text segment.
            let should_fuse_unk = token_id == unk_id
                && state.tokens.len() > original_len
                && state.tokens.last().copied() == Some(unk_id);
            if !should_fuse_unk {
                state.tokens.push(token_id);
            }
        }

        Ok(())
    }
}

/// Advance past UTF-8 continuation bytes — returns the index of the next
/// codepoint boundary at or after `from + 1`.
#[inline]
fn next_char_boundary(prep: &[u8], from: usize) -> usize {
    let mut nxt = from + 1;
    while nxt < prep.len() && prep[nxt] & 0xC0 == 0x80 {
        nxt += 1;
    }
    nxt
}

fn ensure_dp_capacity(state: &mut EncodeState, needed: usize) {
    if state.best_score.len() < needed {
        state.best_score.resize(needed, f64::NEG_INFINITY);
        state.best_start.resize(needed, 0);
        state.best_id.resize(needed, 0);
    }
}

/// Parse `tokenizer.json`, build the preprocessing pipelines and the
/// base-vocab arrays the trie needs.
fn load_components(bytes: &[u8]) -> Result<(Components, Vec<Vec<u8>>, Vec<f64>)> {
    let cfg: config::TokenizerConfig = serde_json::from_slice(bytes)
        .map_err(|e| Error::InvalidConfig(e.to_string()))?;
    let config::TokenizerConfig {
        added_tokens,
        normalizer,
        pre_tokenizer,
        decoder,
        model,
    } = cfg;

    let charsmap = config::extract_charsmap_b64(&normalizer).ok_or_else(|| {
        Error::UnsupportedConfig(
            "normalizer must include a Precompiled charsmap step".into(),
        )
    })?;
    let normalizer = NormalizerPipeline::from_charsmap_b64(charsmap)?;

    let pre = config::extract_metaspace_pre(&pre_tokenizer).ok_or_else(|| {
        Error::UnsupportedConfig("pre_tokenizer must include a Metaspace step".into())
    })?;
    let decoder_pair = config::extract_metaspace_decoder(&decoder);
    let metaspace = Metaspace::from_pre_and_decoder(pre, decoder_pair)?;

    let config::ModelConfig { model_type, unk_id, vocab, byte_fallback } = model;
    if let Some(ty) = model_type.as_deref() {
        if ty != "Unigram" {
            return Err(Error::UnsupportedConfig(format!(
                "model type {ty:?} is not supported (only Unigram)"
            )));
        }
    }
    if byte_fallback {
        return Err(Error::UnsupportedConfig(
            "byte_fallback unigram models are not supported".into(),
        ));
    }
    if vocab.is_empty() {
        return Err(Error::InvalidConfig("unigram vocab cannot be empty".into()));
    }
    if (unk_id as usize) >= vocab.len() {
        return Err(Error::InvalidConfig("unk_id out of range".into()));
    }
    if added_tokens.iter().any(|t| !t.special) {
        return Err(Error::UnsupportedConfig(
            "non-special added tokens are not supported".into(),
        ));
    }

    let mut vocab_bytes: Vec<Vec<u8>> = Vec::with_capacity(vocab.len());
    let mut scores: Vec<f64> = Vec::with_capacity(vocab.len());
    let mut min_score = f64::INFINITY;
    for (token, score) in &vocab {
        let s = *score;
        if s < min_score {
            min_score = s;
        }
        vocab_bytes.push(token.as_bytes().to_vec());
        scores.push(s);
    }
    let unk_score = min_score - UNK_PENALTY;

    let base_vocab_size = vocab_bytes.len();
    let total_vocab_size =
        added_tokens.iter().fold(base_vocab_size, |acc, t| acc.max(t.id as usize + 1));

    let special_matcher = SpecialTokenMatcher::from_specs(
        added_tokens
            .iter()
            .map(|t| SpecialToken {
                id: t.id,
                content: t.content.clone(),
                single_word: t.single_word,
                lstrip: t.lstrip,
                rstrip: t.rstrip,
            })
            .collect(),
    );

    let components = Components {
        unk_id,
        unk_score,
        normalizer,
        metaspace,
        special_matcher,
        vocab_size: total_vocab_size,
    };
    Ok((components, vocab_bytes, scores))
}
