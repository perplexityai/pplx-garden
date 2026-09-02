//! Normalization kernels: weighted RMSNorm (row-wise) with fp32 reduction.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// Threads per row-reduction threadgroup.
const TG: usize = 256;
const RESIDUAL_H: usize = 2048;

const SOURCE: &str = include_str!("metal/norm.metal");

/// Fuses BF16-rounded `x += b` with `out = rmsnorm(x) * (w_bias + w)`.
/// The normalized result may differ from separate kernels by one BF16 ULP.
#[allow(clippy::too_many_arguments)]
pub fn add_rmsnorm_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    b: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    let h =
        *x.shape().last().ok_or_else(|| anyhow::anyhow!("rmsnorm on 0-d tensor"))?;
    let m = x.numel() / h;
    ensure!(b.numel() == x.numel(), "addend numel mismatch");
    ensure!(h == RESIDUAL_H, "residual RMSNorm requires H={RESIDUAL_H}, got {h}");
    ensure!(w.numel() == h, "weight numel {} != H {h}", w.numel());
    ensure!(out.numel() == x.numel(), "output numel mismatch");
    for t in [x, b, w, out] {
        ensure!(t.dtype() == DType::BF16, "rmsnorm expects BF16, got {:?}", t.dtype());
    }
    let pipeline = ctx.pipeline("add_rmsnorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), b.binding(), w.binding(), out.binding()],
        &[&eps.to_ne_bytes(), &w_bias.to_ne_bytes()],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

/// RMSNorm over the last dimension with gain `w_bias + w`.
pub fn rmsnorm_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    let h =
        *x.shape().last().ok_or_else(|| anyhow::anyhow!("rmsnorm on 0-d tensor"))?;
    let m = x.numel() / h;
    ensure!(w.numel() == h, "weight numel {} != H {h}", w.numel());
    ensure!(out.numel() == x.numel(), "output numel mismatch");
    for t in [x, w, out] {
        ensure!(t.dtype() == DType::BF16, "rmsnorm expects BF16, got {:?}", t.dtype());
    }
    let pipeline = ctx.pipeline("rmsnorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(h), &eps.to_ne_bytes(), &w_bias.to_ne_bytes()],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/norm.rs"]
mod tests;
