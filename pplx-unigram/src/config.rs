//! HuggingFace `tokenizer.json` parser, minimum Unigram schema.

use serde::Deserialize;

use crate::TokenId;

#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub added_tokens: Vec<AddedTokenConfig>,
    pub normalizer: NormalizerConfig,
    #[serde(rename = "pre_tokenizer")]
    pub pre_tokenizer: PreTokenizerConfig,
    pub decoder: DecoderConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AddedTokenConfig {
    pub id: TokenId,
    pub content: String,
    #[serde(default)]
    pub single_word: bool,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default)]
    pub rstrip: bool,
    #[serde(default)]
    pub special: bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum NormalizerConfig {
    Sequence { normalizers: Vec<NormalizerStepConfig> },
    Precompiled { precompiled_charsmap: String },
    Replace,
    NFKC,
    NFC,
    NFD,
    NFKD,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum NormalizerStepConfig {
    Precompiled { precompiled_charsmap: String },
    Replace,
    NFKC,
    NFC,
    NFD,
    NFKD,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum PreTokenizerConfig {
    Sequence { pretokenizers: Vec<PreTokenizerStepConfig> },
    Metaspace { replacement: char, add_prefix_space: bool },
    WhitespaceSplit,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum PreTokenizerStepConfig {
    Metaspace { replacement: char, add_prefix_space: bool },
    WhitespaceSplit,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum DecoderConfig {
    Metaspace { replacement: char, add_prefix_space: bool },
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    #[serde(rename = "type", default)]
    pub model_type: Option<String>,
    pub unk_id: TokenId,
    pub vocab: Vec<(String, f64)>,
    #[serde(default)]
    pub byte_fallback: bool,
}

/// Returns `(replacement, add_prefix_space)` from a top-level Metaspace or
/// one nested in a Sequence.
pub fn extract_metaspace_pre(cfg: &PreTokenizerConfig) -> Option<(char, bool)> {
    match cfg {
        PreTokenizerConfig::Metaspace { replacement, add_prefix_space } => {
            Some((*replacement, *add_prefix_space))
        }
        PreTokenizerConfig::Sequence { pretokenizers } => {
            pretokenizers.iter().find_map(|p| match p {
                PreTokenizerStepConfig::Metaspace { replacement, add_prefix_space } => {
                    Some((*replacement, *add_prefix_space))
                }
                _ => None,
            })
        }
        PreTokenizerConfig::WhitespaceSplit => None,
    }
}

pub fn extract_metaspace_decoder(cfg: &DecoderConfig) -> (char, bool) {
    let DecoderConfig::Metaspace { replacement, add_prefix_space } = cfg;
    (*replacement, *add_prefix_space)
}

/// Returns the base64 charsmap from a top-level Precompiled normalizer or one
/// nested in a Sequence.
pub fn extract_charsmap_b64(cfg: &NormalizerConfig) -> Option<&str> {
    match cfg {
        NormalizerConfig::Precompiled { precompiled_charsmap } => {
            Some(precompiled_charsmap.as_str())
        }
        NormalizerConfig::Sequence { normalizers } => {
            normalizers.iter().find_map(|n| match n {
                NormalizerStepConfig::Precompiled { precompiled_charsmap } => {
                    Some(precompiled_charsmap.as_str())
                }
                _ => None,
            })
        }
        _ => None,
    }
}
