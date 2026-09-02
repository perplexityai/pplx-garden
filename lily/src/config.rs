//! HF `config.json` parsing for Qwen3.6-35B-A3B. The file is a multimodal
//! wrapper; lily reads only `text_config` and ignores the vision tower.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use serde::Deserialize;

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerType {
    #[serde(rename = "linear_attention")]
    LinearAttention,
    #[serde(rename = "full_attention")]
    FullAttention,
}

#[derive(Deserialize, Debug)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

/// The `quantization` block an mlx-converted checkpoint carries at the top
/// level of `config.json`. Only the affine mode (`w = scales*q + biases`,
/// codes packed low-nibble-first into u32 along the input dim) is supported.
#[derive(Deserialize, Clone, Copy, Debug)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
}

/// `eos_token_id` appears as a scalar in some checkpoints and a list in
/// others (Qwen3.6 chat models stop on two ids).
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum EosTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenIds {
    pub fn as_vec(&self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.clone(),
        }
    }
}

fn default_true() -> bool {
    true
}

/// The `text_config` section; field names match the checkpoint JSON.
#[derive(Deserialize, Debug)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// Checkpoint-declared context window; zero means unspecified.
    #[serde(default)]
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub eos_token_id: EosTokenIds,
    // Sparse MoE.
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    /// 0 means no shared expert.
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    // Full attention.
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attn_output_gate: bool,
    pub rope_parameters: RopeParameters,
    // Gated DeltaNet.
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    /// Set from the required top-level MLX `quantization` block.
    #[serde(skip)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Deserialize)]
struct QuantizationBlock {
    group_size: usize,
    bits: usize,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Deserialize)]
struct WrapperConfig {
    model_type: String,
    text_config: TextConfig,
    quantization: Option<QuantizationBlock>,
    /// Chat checkpoints put the operative stop list at the top level while
    /// `text_config` keeps a scalar; the top level wins when present.
    eos_token_id: Option<EosTokenIds>,
}

impl TextConfig {
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let wrapper: WrapperConfig =
            serde_json::from_slice(&bytes).context("parsing config.json")?;
        let mut config = wrapper.text_config;
        ensure!(
            config.layer_types.len() == config.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            config.layer_types.len(),
            config.num_hidden_layers
        );
        ensure!(
            wrapper.model_type == "qwen3_5_moe",
            "unsupported model_type {:?}; lily only supports Qwen3.6-35B-A3B",
            wrapper.model_type
        );
        if let Some(eos) = wrapper.eos_token_id {
            config.eos_token_id = eos;
        }
        let q = wrapper
            .quantization
            .context("Qwen3.6-35B-A3B must be an MLX affine 4-bit checkpoint")?;
        ensure!(
            q.mode.as_deref().unwrap_or("affine") == "affine"
                && q.bits == 4
                && q.group_size == 64,
            "unsupported quantization: mode={:?}, bits={}, group_size={}; \
             expected MLX affine 4-bit group_size=64",
            q.mode,
            q.bits,
            q.group_size
        );
        config.quantization =
            Some(QuantizationConfig { group_size: q.group_size, bits: q.bits });
        config.validate_qwen36_35b_a3b()?;
        Ok(config)
    }

    fn validate_qwen36_35b_a3b(&self) -> Result<()> {
        let layer_pattern = self.layer_types.iter().enumerate().all(|(i, ty)| {
            *ty == if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        });
        ensure!(
            self.hidden_size == 2048
                && self.num_hidden_layers == 40
                && self.vocab_size == 248_320
                && self.num_experts == 256
                && self.num_experts_per_tok == 8
                && self.moe_intermediate_size == 512
                && self.shared_expert_intermediate_size == 512
                && self.num_attention_heads == 16
                && self.num_key_value_heads == 2
                && self.head_dim == 256
                && self.linear_num_key_heads == 16
                && self.linear_num_value_heads == 32
                && self.linear_key_head_dim == 128
                && self.linear_value_head_dim == 128
                && self.linear_conv_kernel_dim == 4
                && self.attn_output_gate
                && !self.tie_word_embeddings
                && layer_pattern,
            "checkpoint is not the supported Qwen3.6-35B-A3B shape"
        );
        Ok(())
    }

    /// Rotary dims of each attention head (the rest pass through unrotated).
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as usize
    }

    /// GDN qkv channel count: the conv1d operates over `[q | k | v]`.
    pub fn gdn_conv_channels(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }
}
