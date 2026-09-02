//! Checkpoint loading: maps Qwen3.5 tensor names onto device tensors, reading
//! each byte range straight into a shared-storage Metal buffer. Only the
//! MLX-converted Qwen3.6-35B-A3B layout is supported: every linear projection
//! and the embedding use affine packed `{weight, scales, biases}` triples.
//! Every tensor in the
//! file must be consumed or explicitly skip-listed — a name-scheme drift
//! fails loudly at load instead of silently dropping weights.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use objc2_metal::MTLBuffer;

use crate::config::{LayerType, QuantizationConfig, TextConfig};
use crate::metal::MetalContext;
use crate::safetensors::{Checkpoint, SafetensorsDType};
use crate::tensor::{DType, Tensor};

const MLX_PREFIX: &str = "language_model.model.";
/// The text-only engine deliberately ignores the vision tower.
const MLX_SKIP_PREFIXES: &[&str] = &["vision_tower."];

/// One linear layer's weights in MLX affine form: `codes` packs `32 / bits`
/// elements per u32 along the input dim (low element first), dequantized per
/// `group_size` input elements as `w = scale * q + bias`. `bits` is 4 for
/// everything except MoE routers and shared-expert gates, which mlx keeps at
/// 8 bits.
pub struct QuantWeights {
    /// U32 `[out, in * bits / 32]`.
    pub codes: Tensor,
    /// BF16 `[out, in/group_size]`.
    pub scales: Tensor,
    /// BF16 `[out, in/group_size]`.
    pub biases: Tensor,
    pub group_size: usize,
    pub bits: usize,
}

impl QuantWeights {
    pub fn out_features(&self) -> usize {
        self.codes.shape()[0]
    }

    pub fn in_features(&self) -> usize {
        self.codes.shape()[1] * (32 / self.bits)
    }

    /// A row-range view (`[rows, in]` starting at `start_row`) sharing the
    /// underlying buffers — how fused projections hand out their segments.
    pub fn view_rows(&self, start_row: usize, rows: usize) -> Result<Self> {
        let words = self.codes.shape()[1];
        let groups = self.scales.shape()[1];
        Ok(Self {
            codes: self.codes.view(start_row * words, &[rows, words])?,
            scales: self.scales.view(start_row * groups, &[rows, groups])?,
            biases: self.biases.view(start_row * groups, &[rows, groups])?,
            group_size: self.group_size,
            bits: self.bits,
        })
    }

    fn expect_features(&self, out: usize, inp: usize, name: &str) -> Result<()> {
        ensure!(
            self.out_features() == out && self.in_features() == inp,
            "{name} is [{}, {}], expected [{out}, {inp}]",
            self.out_features(),
            self.in_features(),
        );
        Ok(())
    }
}

pub type LinearWeights = QuantWeights;

fn expected_projection_bits(bases: &[&str]) -> usize {
    let q8 = bases.len() == 1
        && (bases[0].ends_with(".mlp.gate")
            || bases[0].ends_with(".mlp.shared_expert_gate"));
    if q8 { 8 } else { 4 }
}

pub struct MlpWeights {
    /// `[2*inter, h]` — gate rows then up rows; `gate_proj`/`up_proj` are
    /// row-range views of it, so prefill's separate GEMMs and decode's single
    /// fused matvec share the same bytes.
    pub gate_up_proj: LinearWeights,
    pub gate_proj: LinearWeights,
    pub up_proj: LinearWeights,
    pub down_proj: LinearWeights,
}

/// Sparse-MoE FFN: a router over stacked expert projections plus the
/// always-on shared expert with a sigmoid gate. Expert stacks are flattened
/// `[E * rows, cols]` so expert `e` is the row range `e*rows .. (e+1)*rows`
/// (`view_rows` for prefill's per-expert GEMMs, base-offset arithmetic for
/// the decode gather kernels).
pub struct MoeWeights {
    /// Router `[num_experts, h]` (8-bit in mlx-quantized checkpoints).
    pub gate: LinearWeights,
    /// `[E * inter, h]`.
    pub expert_gate: LinearWeights,
    /// `[E * inter, h]`.
    pub expert_up: LinearWeights,
    /// `[E * h, inter]`.
    pub expert_down: LinearWeights,
    pub shared: MlpWeights,
    /// `[1, h]`, sigmoid-gating the shared expert's output.
    pub shared_gate: LinearWeights,
}

pub struct GdnWeights {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    /// `[conv_c + dim_v + 2*heads, h]` — qkv | z | a | b rows fused; the four
    /// named projections below are row-range views.
    pub in_proj: LinearWeights,
    pub in_proj_qkv: LinearWeights,
    pub in_proj_z: LinearWeights,
    pub in_proj_a: LinearWeights,
    pub in_proj_b: LinearWeights,
    /// Tap-major `[kernel_dim, conv_channels]` (transposed from the
    /// checkpoint layout at load).
    pub conv_w: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub norm_w: Tensor,
    pub out_proj: LinearWeights,
    pub ffn: Box<MoeWeights>,
}

pub struct AttnWeights {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    /// `[nq*2*hd + 2*nkv*hd, h]` — q(|gate) | k | v rows fused; the three
    /// named projections below are row-range views.
    pub qkv_proj: LinearWeights,
    pub q_proj: LinearWeights,
    pub k_proj: LinearWeights,
    pub v_proj: LinearWeights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub o_proj: LinearWeights,
    pub ffn: Box<MoeWeights>,
}

pub enum LayerWeights {
    Gdn(Box<GdnWeights>),
    Full(Box<AttnWeights>),
}

pub struct ModelWeights {
    /// `[vocab, h]` token table.
    pub embed_tokens: LinearWeights,
    /// Untied LM head.
    pub lm_head: LinearWeights,
    pub final_norm: Tensor,
    pub layers: Vec<LayerWeights>,
}

fn to_dtype(dtype: &SafetensorsDType) -> Result<DType> {
    match dtype {
        SafetensorsDType::BF16 => Ok(DType::BF16),
        SafetensorsDType::F32 => Ok(DType::F32),
        SafetensorsDType::U32 => Ok(DType::U32),
        SafetensorsDType::Other(s) => anyhow::bail!("unsupported checkpoint dtype {s}"),
    }
}

/// Checkpoint access that records every consumed tensor name so `finish` can
/// verify nothing in the file was silently ignored.
struct Loader<'a> {
    ctx: &'a MetalContext,
    ckpt: Checkpoint,
    quant: QuantizationConfig,
    skip_prefixes: &'static [&'static str],
    consumed: RefCell<HashSet<String>>,
}

impl Loader<'_> {
    /// Reads one tensor straight into a fresh shared-storage Metal buffer.
    fn tensor(&self, name: &str) -> Result<Tensor> {
        self.consumed.borrow_mut().insert(name.to_string());
        let meta = self
            .ckpt
            .meta(name)
            .with_context(|| format!("tensor {name} not in checkpoint"))?;
        let dtype = to_dtype(&meta.dtype)?;
        let shape = meta.shape.clone();
        let buf = self.ctx.new_buffer(meta.byte_len())?;
        self.ckpt.read_with(name, |meta| {
            // SAFETY: freshly allocated shared buffer of exactly byte_len bytes.
            Ok(unsafe {
                core::slice::from_raw_parts_mut(
                    buf.contents().as_ptr().cast::<u8>(),
                    meta.byte_len(),
                )
            })
        })?;
        Tensor::from_buffer(buf, &shape, dtype)
    }

    /// Like [`Self::tensor`] but upcasts BF16 to F32 host-side — for the few
    /// small parameters whose kernels want f32 while mlx checkpoints store
    /// bf16 (the GDN GatedNorm weight).
    fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        let t = self.tensor(name)?;
        match t.dtype() {
            DType::F32 => Ok(t),
            DType::BF16 => Tensor::from_f32(self.ctx, &t.to_f32()?, t.shape()),
            DType::U32 => anyhow::bail!("{name} is U32, expected float"),
        }
    }

    /// Loads a zero-centered RMSNorm weight, undoing mlx's baked-in `+1.0`
    /// where present. `bf16(1 + w) - 1.0` has at most bf16-around-1.0
    /// precision, so the subtraction round-trips exactly.
    fn zero_centered_norm(&self, name: &str) -> Result<Tensor> {
        let t = self.tensor(name)?;
        ensure!(t.dtype() == DType::BF16, "{name} norm weight must be BF16");
        let shifted: Vec<f32> = t.to_f32()?.iter().map(|v| v - 1.0).collect();
        Tensor::from_f32_as_bf16(self.ctx, &shifted, t.shape())
    }

    /// Loads several `[n_i, k]` row-major tensors of one dtype into a single
    /// `[sum(n_i), k]` buffer (each read lands at its row offset), so decode
    /// can run one fused matvec while per-segment views keep the original
    /// tensors addressable.
    fn concat_rows(&self, names: &[String], k: usize) -> Result<Tensor> {
        let mut rows = 0usize;
        let mut dtype: Option<DType> = None;
        for name in names {
            self.consumed.borrow_mut().insert(name.clone());
            let meta = self
                .ckpt
                .meta(name)
                .with_context(|| format!("tensor {name} not in checkpoint"))?;
            // Stacked expert tensors are [E, I, k]; flatten leading dims.
            ensure!(
                meta.shape.len() >= 2 && *meta.shape.last().expect("shape") == k,
                "{name} shape {:?} not [.., {k}]",
                meta.shape
            );
            let d = to_dtype(&meta.dtype)?;
            ensure!(
                dtype.is_none() || dtype == Some(d),
                "{name} dtype mismatch in fuse"
            );
            dtype = Some(d);
            rows += meta.shape[..meta.shape.len() - 1].iter().product::<usize>();
        }
        let dtype = dtype.ok_or_else(|| anyhow::anyhow!("empty fuse list"))?;
        let buf = self.ctx.new_buffer(rows * k * dtype.size())?;
        let mut offset = 0usize;
        for name in names {
            self.ckpt.read_with(name, |meta| {
                let start = offset;
                // SAFETY: freshly allocated shared buffer; [start,
                // start+byte_len) is this tensor's disjoint row range.
                Ok(unsafe {
                    core::slice::from_raw_parts_mut(
                        buf.contents().as_ptr().cast::<u8>().add(start),
                        meta.byte_len(),
                    )
                })
            })?;
            let meta =
                self.ckpt.meta(name).with_context(|| format!("{name} missing"))?;
            offset += meta.byte_len();
        }
        Tensor::from_buffer(buf, &[rows, k], dtype)
    }

    /// Loads one quantized projection, or several fused along the output dim.
    fn linear(&self, bases: &[&str], k: usize) -> Result<LinearWeights> {
        let weights: Vec<String> =
            bases.iter().map(|b| format!("{b}.weight")).collect();
        ensure!(
            self.ckpt.meta(&format!("{}.scales", bases[0])).is_some(),
            "{} is not an MLX quantized projection",
            bases[0]
        );
        let q = self.quant;
        ensure!(
            k.is_multiple_of(q.group_size) && k.is_multiple_of(8),
            "in_features {k} not divisible by group size {} / packing",
            q.group_size
        );
        // Per-tensor bit width is inferred from the packed width so router /
        // shared-expert-gate tensors that mlx keeps at 8 bits load correctly
        // regardless of the config's global `bits`.
        let meta = self
            .ckpt
            .meta(&weights[0])
            .with_context(|| format!("tensor {} not in checkpoint", weights[0]))?;
        let cols = *meta.shape.last().expect("shape");
        ensure!(
            k.is_multiple_of(cols) && matches!(32 * cols / k, 4 | 8),
            "cannot infer bit width for {} ({cols} packed cols, in={k})",
            weights[0]
        );
        let bits = 32 * cols / k;
        let expected_bits = expected_projection_bits(bases);
        ensure!(
            bits == expected_bits,
            "{} uses {bits}-bit storage; expected {expected_bits}-bit for this Qwen3.6-35B-A3B projection",
            weights[0]
        );
        let scales: Vec<String> = bases.iter().map(|b| format!("{b}.scales")).collect();
        let biases: Vec<String> = bases.iter().map(|b| format!("{b}.biases")).collect();
        let codes = self.concat_rows(&weights, cols)?;
        let scales = self.concat_rows(&scales, k / q.group_size)?;
        let biases = self.concat_rows(&biases, k / q.group_size)?;
        ensure!(codes.dtype() == DType::U32, "quantized codes must be U32");
        ensure!(
            scales.dtype() == DType::BF16 && biases.dtype() == DType::BF16,
            "quantized scales/biases must be BF16"
        );
        Ok(QuantWeights { codes, scales, biases, group_size: q.group_size, bits })
    }

    /// Loads the conv1d weight, accepting the HF `[C, 1, KD]` or mlx
    /// `[C, KD, 1]` layout and transposing to the tap-major `[KD, C]` the conv
    /// kernel consumes.
    fn conv_weight(&self, name: &str) -> Result<Tensor> {
        self.consumed.borrow_mut().insert(name.to_string());
        let bytes = self.ckpt.read(name)?;
        let meta =
            self.ckpt.meta(name).with_context(|| format!("tensor {name} missing"))?;
        ensure!(meta.dtype == SafetensorsDType::BF16, "conv weight must be BF16");
        ensure!(
            meta.shape.len() == 3 && (meta.shape[1] == 1 || meta.shape[2] == 1),
            "conv weight shape {:?} != [C, 1, KD] or [C, KD, 1]",
            meta.shape
        );
        let (c, kd) = (meta.shape[0], meta.shape[1].max(meta.shape[2]));
        let src: &[bf16] = bytemuck::cast_slice(&bytes);
        let mut transposed = vec![bf16::from_f32(0.0); kd * c];
        for ch in 0..c {
            for t in 0..kd {
                transposed[t * c + ch] = src[ch * kd + t];
            }
        }
        Tensor::from_bytes(
            self.ctx,
            bytemuck::cast_slice(&transposed),
            &[kd, c],
            DType::BF16,
        )
    }

    /// Fails if the checkpoint holds tensors lily neither consumed nor
    /// skip-listed — the guard against silent name-scheme drift.
    fn finish(&self) -> Result<()> {
        let consumed = self.consumed.borrow();
        let mut unconsumed: Vec<&str> = self
            .ckpt
            .names()
            .filter(|n| {
                !consumed.contains(*n)
                    && !self.skip_prefixes.iter().any(|p| n.starts_with(p))
            })
            .collect();
        unconsumed.sort_unstable();
        ensure!(
            unconsumed.is_empty(),
            "{} checkpoint tensors were not consumed (name-scheme drift?): {:?}{}",
            unconsumed.len(),
            &unconsumed[..unconsumed.len().min(8)],
            if unconsumed.len() > 8 { " ..." } else { "" }
        );
        Ok(())
    }
}

fn expect_shape(t: &Tensor, shape: &[usize], name: &str) -> Result<()> {
    ensure!(t.shape() == shape, "{name} shape {:?} != expected {shape:?}", t.shape());
    Ok(())
}

/// Loads a dense SwiGLU MLP whose tensors live at `{prefix}gate_proj` etc.
/// (`prefix` includes the trailing dot), fusing gate|up along output rows.
fn load_mlp(
    loader: &Loader<'_>,
    prefix: &str,
    h: usize,
    i: usize,
) -> Result<MlpWeights> {
    let gate_up_proj = loader
        .linear(&[&format!("{prefix}gate_proj"), &format!("{prefix}up_proj")], h)?;
    gate_up_proj.expect_features(2 * i, h, "gate_up_proj")?;
    let gate_proj = gate_up_proj.view_rows(0, i)?;
    let up_proj = gate_up_proj.view_rows(i, i)?;
    let down_proj = loader.linear(&[&format!("{prefix}down_proj")], i)?;
    down_proj.expect_features(h, i, "down_proj")?;
    Ok(MlpWeights { gate_up_proj, gate_proj, up_proj, down_proj })
}

/// Loads the fixed sparse-MoE block: router, stacked experts and shared expert.
fn load_ffn(
    loader: &Loader<'_>,
    p: &str,
    config: &TextConfig,
) -> Result<Box<MoeWeights>> {
    let h = config.hidden_size;
    let (e, i) = (config.num_experts, config.moe_intermediate_size);
    let gate = loader.linear(&[&format!("{p}mlp.gate")], h)?;
    gate.expect_features(e, h, "router gate")?;
    let expert_gate = loader.linear(&[&format!("{p}mlp.switch_mlp.gate_proj")], h)?;
    expert_gate.expect_features(e * i, h, "expert gate_proj")?;
    let expert_up = loader.linear(&[&format!("{p}mlp.switch_mlp.up_proj")], h)?;
    expert_up.expect_features(e * i, h, "expert up_proj")?;
    let expert_down = loader.linear(&[&format!("{p}mlp.switch_mlp.down_proj")], i)?;
    expert_down.expect_features(e * h, i, "expert down_proj")?;
    let shared = load_mlp(
        loader,
        &format!("{p}mlp.shared_expert."),
        h,
        config.shared_expert_intermediate_size,
    )?;
    let shared_gate = loader.linear(&[&format!("{p}mlp.shared_expert_gate")], h)?;
    shared_gate.expect_features(1, h, "shared_expert_gate")?;
    Ok(Box::new(MoeWeights {
        gate,
        expert_gate,
        expert_up,
        expert_down,
        shared,
        shared_gate,
    }))
}

/// Loads one Gated-DeltaNet (linear-attention) block whose tensors live at
/// `{p}` (a `layers.{i}.` prefix including the trailing dot).
fn load_gdn(loader: &Loader<'_>, p: &str, config: &TextConfig) -> Result<GdnWeights> {
    let h = config.hidden_size;
    let input_norm =
        loader.zero_centered_norm(&format!("{p}input_layernorm.weight"))?;
    let post_norm =
        loader.zero_centered_norm(&format!("{p}post_attention_layernorm.weight"))?;
    let ffn = load_ffn(loader, p, config)?;

    let la = format!("{p}linear_attn.");
    let heads = config.linear_num_value_heads;
    let dim_v = heads * config.linear_value_head_dim;
    let conv_c = config.gdn_conv_channels();

    let in_proj = loader.linear(
        &[
            &format!("{la}in_proj_qkv"),
            &format!("{la}in_proj_z"),
            &format!("{la}in_proj_a"),
            &format!("{la}in_proj_b"),
        ],
        h,
    )?;
    in_proj.expect_features(conv_c + dim_v + 2 * heads, h, "in_proj")?;
    let in_proj_qkv = in_proj.view_rows(0, conv_c)?;
    let in_proj_z = in_proj.view_rows(conv_c, dim_v)?;
    let in_proj_a = in_proj.view_rows(conv_c + dim_v, heads)?;
    let in_proj_b = in_proj.view_rows(conv_c + dim_v + heads, heads)?;
    let conv_w = loader.conv_weight(&format!("{la}conv1d.weight"))?;
    // Some mlx conversions store A_log in bf16 (the 35B lmstudio one); the GDN
    // kernels read f32.
    let a_log = loader.tensor_f32(&format!("{la}A_log"))?;
    let dt_bias = loader.tensor(&format!("{la}dt_bias"))?;
    let norm_w = loader.tensor_f32(&format!("{la}norm.weight"))?;
    let out_proj = loader.linear(&[&format!("{la}out_proj")], dim_v)?;

    expect_shape(&conv_w, &[config.linear_conv_kernel_dim, conv_c], "conv_w")?;
    expect_shape(&a_log, &[heads], "A_log")?;
    expect_shape(&dt_bias, &[heads], "dt_bias")?;
    expect_shape(&norm_w, &[config.linear_value_head_dim], "gdn norm")?;
    out_proj.expect_features(h, dim_v, "out_proj")?;

    Ok(GdnWeights {
        input_norm,
        post_norm,
        in_proj,
        in_proj_qkv,
        in_proj_z,
        in_proj_a,
        in_proj_b,
        conv_w,
        a_log,
        dt_bias,
        norm_w,
        out_proj,
        ffn,
    })
}

/// Loads one full-attention + FFN block whose tensors live at `{p}`.
fn load_attn(loader: &Loader<'_>, p: &str, config: &TextConfig) -> Result<AttnWeights> {
    let h = config.hidden_size;
    let input_norm =
        loader.zero_centered_norm(&format!("{p}input_layernorm.weight"))?;
    let post_norm =
        loader.zero_centered_norm(&format!("{p}post_attention_layernorm.weight"))?;
    let ffn = load_ffn(loader, p, config)?;

    let sa = format!("{p}self_attn.");
    let (hd, nq, nkv) =
        (config.head_dim, config.num_attention_heads, config.num_key_value_heads);
    let q_rows = if config.attn_output_gate { 2 * nq * hd } else { nq * hd };

    let qkv_proj = loader.linear(
        &[&format!("{sa}q_proj"), &format!("{sa}k_proj"), &format!("{sa}v_proj")],
        h,
    )?;
    qkv_proj.expect_features(q_rows + 2 * nkv * hd, h, "qkv_proj")?;
    let q_proj = qkv_proj.view_rows(0, q_rows)?;
    let k_proj = qkv_proj.view_rows(q_rows, nkv * hd)?;
    let v_proj = qkv_proj.view_rows(q_rows + nkv * hd, nkv * hd)?;
    let q_norm = loader.zero_centered_norm(&format!("{sa}q_norm.weight"))?;
    let k_norm = loader.zero_centered_norm(&format!("{sa}k_norm.weight"))?;
    let o_proj = loader.linear(&[&format!("{sa}o_proj")], nq * hd)?;

    expect_shape(&q_norm, &[hd], "q_norm")?;
    expect_shape(&k_norm, &[hd], "k_norm")?;
    o_proj.expect_features(h, nq * hd, "o_proj")?;

    Ok(AttnWeights {
        input_norm,
        post_norm,
        qkv_proj,
        q_proj,
        k_proj,
        v_proj,
        q_norm,
        k_norm,
        o_proj,
        ffn,
    })
}

pub fn load(
    ctx: &MetalContext,
    dir: impl AsRef<Path>,
    config: &TextConfig,
) -> Result<ModelWeights> {
    let ckpt = Checkpoint::open(&dir)?;
    ensure!(
        ckpt.meta(&format!("{MLX_PREFIX}embed_tokens.weight")).is_some(),
        "unsupported checkpoint layout; expected MLX Qwen3.6-35B-A3B"
    );
    let prefix = MLX_PREFIX;
    let loader = Loader {
        ctx,
        ckpt,
        quant: config.quantization.expect("validated config"),
        skip_prefixes: MLX_SKIP_PREFIXES,
        consumed: RefCell::new(HashSet::new()),
    };
    let h = config.hidden_size;

    let embed_tokens = loader.linear(&[&format!("{prefix}embed_tokens")], h)?;
    embed_tokens.expect_features(config.vocab_size, h, "embed_tokens")?;
    let lm_head = loader.linear(&["language_model.lm_head"], h)?;
    lm_head.expect_features(config.vocab_size, h, "lm_head")?;
    let final_norm = loader.zero_centered_norm(&format!("{prefix}norm.weight"))?;
    expect_shape(&final_norm, &[h], "final norm")?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for (idx, layer_type) in config.layer_types.iter().enumerate() {
        let p = format!("{prefix}layers.{idx}.");
        let layer = match layer_type {
            LayerType::LinearAttention => {
                LayerWeights::Gdn(Box::new(load_gdn(&loader, &p, config)?))
            }
            LayerType::FullAttention => {
                LayerWeights::Full(Box::new(load_attn(&loader, &p, config)?))
            }
        };
        layers.push(layer);
    }

    loader.finish()?;
    Ok(ModelWeights { embed_tokens, lm_head, final_norm, layers })
}

#[cfg(test)]
#[path = "../tests/unit/weights.rs"]
mod tests;
