//! Gated DeltaNet recurrence, causal depthwise conv1d, and gated RMSNorm.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// Compiled GDN key/value head dimension.
pub const GDN_HEAD_DIM: usize = 128;

/// Prefill conv1d tile width; must match `CONV_TILE` in gdn.metal.
pub const CONV1D_PREFILL_TILE: usize = 64;

const SOURCE: &str = include_str!("metal/gdn.metal");

/// Recurrent state is F32 because rounding compounds across tokens.
pub const GDN_STATE_DTYPE: DType = DType::F32;

#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gated_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    eps: f32,
) -> Result<()> {
    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let vpk = num_heads / num_k_heads;
    let k_part = num_k_heads * dim;
    let part = num_heads * dim;
    ensure!(qkv.numel() == 2 * k_part + part, "bad fused qkv size");
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    ensure!(state.numel() == num_heads * dim * dim, "bad state size");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    ensure!(z.numel() == part && z.dtype() == DType::BF16, "z must be BF16 [H, dim]");
    ensure!(
        norm_w.numel() == dim && norm_w.dtype() == DType::F32,
        "norm_w must be F32 [dim]"
    );
    ensure!(out.numel() == part && out.dtype() == DType::BF16, "bad output size");
    let elem = qkv.dtype().size();
    let (qkv_buf, qkv_off) = qkv.binding();
    let pipeline = ctx.pipeline("gdn_step_gated", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            (qkv_buf, qkv_off),
            (qkv_buf, qkv_off + k_part * elem),
            (qkv_buf, qkv_off + 2 * k_part * elem),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            state.binding(),
            z.binding(),
            norm_w.binding(),
            out.binding(),
        ],
        &[&scale.to_ne_bytes(), &u32_bytes(vpk), &eps.to_ne_bytes()],
        Grid::Threadgroups { groups: (num_heads, 1, 1), threadgroup: (dim, 1, 1) },
    )
}

/// Staging for normalized q/k rows and precomputed gates.
pub struct GdnRegscanStaging<'a> {
    /// BF16 `[M, 2*HK*128]` normalized q then k rows.
    pub qk_norm: &'a Tensor,
    /// F32 `[M, H]` decay gates.
    pub decay: &'a Tensor,
    /// F32 `[M, H]` sigmoid(b) gates.
    pub beta: &'a Tensor,
}

/// L2-normalizes q/k into BF16 `[M, 2*HK*128]`; q is pre-scaled.
#[allow(clippy::too_many_arguments)]
fn gdn_qk_l2norm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    qk_norm: &Tensor,
    scale: f32,
    num_k_heads: usize,
    num_heads: usize,
) -> Result<()> {
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a positive multiple of k-heads {num_k_heads}"
    );
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    let row_size = (2 * num_k_heads + num_heads) * dim;
    ensure!(qkv.numel().is_multiple_of(row_size), "qkv not [M, (2*HK+H)*{dim}]");
    let m = qkv.numel() / row_size;
    ensure!(m > 0, "qkv must contain at least one row");
    ensure!(
        qk_norm.numel() == m * 2 * num_k_heads * dim && qk_norm.dtype() == DType::BF16,
        "qk_norm must be BF16 [M, 2*HK*{dim}]"
    );

    let pipeline = ctx.pipeline("gdn_qk_l2norm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[qkv.binding(), qk_norm.binding()],
        &[&scale.to_ne_bytes(), &u32_bytes(num_k_heads), &u32_bytes(num_heads)],
        Grid::Threads {
            grid: (32, num_k_heads, m),
            threadgroup: (32, 4.min(num_k_heads), 1),
        },
    )
}

/// Computes F32 decay and beta gates from BF16 `[M, H]` inputs.
#[allow(clippy::too_many_arguments)]
fn gdn_gates(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let num_heads = a_log.numel();
    ensure!(num_heads > 0, "a_log must contain at least one head");
    ensure!(a.dtype() == DType::BF16, "a must be BF16");
    ensure!(b.dtype() == DType::BF16, "b must be BF16");
    ensure!(a_log.dtype() == DType::F32, "a_log must be F32");
    ensure!(dt_bias.dtype() == DType::BF16, "dt_bias must be BF16");
    ensure!(a.numel().is_multiple_of(num_heads), "a not [M, H]");
    let n = a.numel();
    ensure!(n > 0, "a must contain at least one row");
    ensure!(b.numel() == n, "b not [M, H]");
    ensure!(dt_bias.numel() == num_heads, "dt_bias not [H]");
    ensure!(
        decay.numel() == n && decay.dtype() == DType::F32,
        "decay must be F32 [M, H]"
    );
    ensure!(beta.numel() == n && beta.dtype() == DType::F32, "beta must be F32 [M, H]");

    let pipeline = ctx.pipeline("gdn_gates", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            decay.binding(),
            beta.binding(),
        ],
        &[&u32_bytes(num_heads)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256.min(n), 1, 1) },
    )
}

/// Scans `[q | k | v]` rows serially with F32 state held in registers.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    let num_heads = a_log.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let vpk = num_heads / num_k_heads;
    ensure!(a.numel().is_multiple_of(num_heads), "a not [M, H]");
    let m = a.numel() / num_heads;
    ensure!(b.numel() == m * num_heads, "b not [M, H]");
    ensure!(
        qkv.numel() == m * (2 * num_k_heads + num_heads) * dim,
        "qkv not [M, (2*HK+H)*{dim}]"
    );
    ensure!(out.numel() == m * num_heads * dim, "out not [M, H, {dim}]");
    ensure!(state.numel() == num_heads * dim * dim, "state must be [H, {dim}, {dim}]");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    ensure!(a_log.dtype() == DType::F32, "a_log must be F32");
    ensure!(
        staging.qk_norm.numel() == m * 2 * num_k_heads * dim
            && staging.qk_norm.dtype() == DType::BF16,
        "qk_norm must be BF16 [M, 2*HK*{dim}]"
    );
    ensure!(
        staging.decay.numel() == m * num_heads && staging.decay.dtype() == DType::F32,
        "decay must be F32 [M, H]"
    );
    ensure!(
        staging.beta.numel() == m * num_heads && staging.beta.dtype() == DType::F32,
        "beta must be F32 [M, H]"
    );

    gdn_qk_l2norm(ctx, pass, qkv, staging.qk_norm, scale, num_k_heads, num_heads)?;
    gdn_gates(ctx, pass, a, b, a_log, dt_bias, staging.decay, staging.beta)?;
    let scan = ctx.pipeline("gdn_prefill_regscan", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scan,
        &[
            qkv.binding(),
            staging.qk_norm.binding(),
            staging.decay.binding(),
            staging.beta.binding(),
            state.binding(),
            out.binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(num_heads), &u32_bytes(vpk)],
        Grid::Threadgroups { groups: (num_heads, dim / 4, 1), threadgroup: (32, 4, 1) },
    )
}

/// Causal conv1d + SiLU over `[M, C]`; separate window buffers avoid tile races.
pub fn conv1d_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window_in: &Tensor,
    window_out: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
) -> Result<()> {
    ensure!(x.shape().len() == 2, "x must be [M, C]");
    let (m, c) = (x.shape()[0], x.shape()[1]);
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!((2..=9).contains(&kd), "conv kernel dim {kd} outside register window");
    for (name, t) in [("window_in", window_in), ("window_out", window_out)] {
        ensure!(t.numel() == c * (kd - 1), "{name} numel {} != C*(KD-1)", t.numel());
    }
    let (in_buf, in_off) = window_in.binding();
    let (out_buf, out_off) = window_out.binding();
    ensure!(
        !(std::ptr::eq(in_buf, out_buf) && in_off == out_off),
        "window buffers must be distinct (double-buffered)"
    );
    ensure!(out.numel() == m * c, "out numel {} != M*C", out.numel());
    let pipeline = ctx.pipeline("conv1d_prefill_bf16", SOURCE, MslVersion::V3_1)?;
    let tiles = m.div_ceil(CONV1D_PREFILL_TILE).max(1);
    pass.dispatch_at(
        &pipeline,
        &[
            window_in.binding(),
            window_out.binding(),
            x.binding(),
            w.binding(),
            out.binding(),
        ],
        &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(m)],
        Grid::Threads { grid: (c, tiles, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// One causal conv1d + SiLU decode step. `window` (`[C, KD-1]`, bf16) is
/// shifted in place; `w` is tap-major `[KD, C]`.
pub fn conv1d_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let c = x.numel();
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!(kd >= 2, "conv kernel dim {kd} < 2");
    ensure!(
        window.numel() == c * (kd - 1),
        "window numel {} != C*(KD-1)",
        window.numel()
    );
    ensure!(out.numel() == c, "out numel {} != C {c}", out.numel());
    let pipeline = ctx.pipeline("conv1d_step_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[window.binding(), x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(c), &u32_bytes(kd)],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// Gated RMS norm over rows of size `d` with F32 weights.
pub fn gated_rmsnorm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    gate: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
) -> Result<()> {
    let d = w.numel();
    ensure!(w.dtype() == DType::F32, "gated_rmsnorm weight must be F32");
    ensure!(
        x.numel().is_multiple_of(d),
        "x numel {} not a multiple of D {d}",
        x.numel()
    );
    ensure!(gate.numel() == x.numel() && out.numel() == x.numel(), "size mismatch");
    let rows = x.numel() / d;
    let pipeline = ctx.pipeline("gated_rmsnorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), gate.binding(), w.binding(), out.binding()],
        &[&u32_bytes(d), &eps.to_ne_bytes()],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/gdn.rs"]
mod tests;
