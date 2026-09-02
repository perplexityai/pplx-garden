//! Element-wise, embedding, and argmax kernels.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

const SOURCE: &str = include_str!("metal/elementwise.metal");

fn binary_op(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    fn_name: &'static str,
    dtype: DType,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op_source(ctx, pass, fn_name, dtype, a, b, out, SOURCE)
}

#[allow(clippy::too_many_arguments)]
fn binary_op_source(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    fn_name: &'static str,
    dtype: DType,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    source: &'static str,
) -> Result<()> {
    let n = a.numel();
    ensure!(b.numel() == n && out.numel() == n, "size mismatch in {fn_name}");
    for t in [a, b, out] {
        ensure!(t.dtype() == dtype, "{fn_name} expects {dtype:?}, got {:?}", t.dtype());
    }
    let pipeline = ctx.pipeline(fn_name, source, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[a.binding(), b.binding(), out.binding()],
        &[],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256.min(n), 1, 1) },
    )
}

pub fn add_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op(ctx, pass, "add_bf16", DType::BF16, a, b, out)
}

pub fn silu_mul_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &Tensor,
    up: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op(ctx, pass, "silu_mul_bf16", DType::BF16, gate, up, out)
}

pub fn sigmoid_mul_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &Tensor,
    x: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op(ctx, pass, "sigmoid_mul_bf16", DType::BF16, gate, x, out)
}

/// Splits fused `[m, n_total]` rows into two to four contiguous outputs.
pub fn split_cols_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    src: &Tensor,
    dsts: &[&Tensor],
) -> Result<()> {
    ensure!((2..=4).contains(&dsts.len()), "split_cols takes 2..=4 destinations");
    ensure!(src.dtype() == DType::BF16, "split_cols src must be BF16");
    let m = src.shape()[0];
    let n_total = src.numel() / m;
    let mut widths = [0u32; 4];
    let mut sum = 0usize;
    for (j, d) in dsts.iter().enumerate() {
        ensure!(d.dtype() == DType::BF16, "split_cols dsts must be BF16");
        ensure!(
            d.shape()[0] == m && d.numel().is_multiple_of(m),
            "split_cols dst {j} shape {:?} not [{m}, w]",
            d.shape()
        );
        let w = d.numel() / m;
        widths[j] = u32::try_from(w)?;
        sum += w;
    }
    ensure!(sum == n_total, "split_cols widths sum {sum} != src columns {n_total}");
    let mut widths_bytes = [0u8; 16];
    for (chunk, w) in widths_bytes.chunks_exact_mut(4).zip(widths) {
        chunk.copy_from_slice(&w.to_ne_bytes());
    }
    // Unused slots have zero width and are never written.
    let dst = |j: usize| dsts.get(j).copied().unwrap_or(dsts[0]).binding();
    let pipeline = ctx.pipeline("split_cols_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[src.binding(), dst(0), dst(1), dst(2), dst(3)],
        &[&widths_bytes, &crate::kernels::u32_bytes(n_total)],
        Grid::Threads { grid: (n_total, m, 1), threadgroup: (256.min(n_total), 1, 1) },
    )
}

/// Batched embedding lookup: `out[i, :] = table[ids[i], :]`.
pub fn gather_rows_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &Tensor,
    ids: &Tensor,
    out: &Tensor,
) -> Result<()> {
    ensure!(ids.dtype() == DType::U32, "ids must be U32");
    let m = ids.numel();
    ensure!(m > 0 && out.numel().is_multiple_of(m), "out rows mismatch");
    let h = out.numel() / m;
    ensure!(table.numel().is_multiple_of(h), "table not a multiple of row size");
    let pipeline = ctx.pipeline("gather_rows_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.binding(), ids.binding(), out.binding()],
        &[&crate::kernels::u32_bytes(h)],
        Grid::Threads { grid: (m * h, 1, 1), threadgroup: (256.min(m * h), 1, 1) },
    )
}

/// Copies row `row` of the `[rows, h]` bf16 `table` into `out` (`[h]`).
pub fn gather_row_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &Tensor,
    out: &Tensor,
    row: usize,
) -> Result<()> {
    let h = out.numel();
    ensure!(table.numel().is_multiple_of(h), "table not a multiple of row size");
    ensure!(row < table.numel() / h, "row {row} out of range");
    let pipeline = ctx.pipeline("gather_row_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.binding(), out.binding()],
        &[&crate::kernels::u32_bytes(row), &crate::kernels::u32_bytes(h)],
        Grid::Threads { grid: (h, 1, 1), threadgroup: (256.min(h), 1, 1) },
    )
}

/// Threadgroups and partial pairs used by argmax.
pub const ARGMAX_GROUPS: usize = 64;

/// Reduces `x` into `(value, index)` partials.
fn argmax_f32_partial(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    partials: &Tensor,
) -> Result<()> {
    let n = x.numel();
    ensure!(n > 0, "empty argmax input");
    ensure!(x.dtype() == DType::F32, "argmax input must be F32");
    ensure!(
        partials.numel() * partials.dtype().size() >= ARGMAX_GROUPS * 8,
        "partials scratch too small"
    );
    let chunk = n.div_ceil(ARGMAX_GROUPS);
    let partial = ctx.pipeline("argmax_f32_partial", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &partial,
        &[x.binding(), partials.binding()],
        &[&u32_bytes(n), &u32_bytes(chunk)],
        Grid::Threadgroups { groups: (ARGMAX_GROUPS, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Reduces argmax partials and writes the first-max index.
fn argmax_f32_final(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    partials: &Tensor,
    out: &Tensor,
) -> Result<()> {
    ensure!(out.dtype() == DType::U32 && out.numel() >= 1, "out must be U32[1]");
    ensure!(
        partials.numel() * partials.dtype().size() >= ARGMAX_GROUPS * 8,
        "partials scratch too small"
    );
    let pipeline = ctx.pipeline("argmax_f32_final", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[partials.binding(), out.binding()],
        &[&u32_bytes(ARGMAX_GROUPS)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// First-max argmax into `out`; `partials` needs `ARGMAX_GROUPS * 8` bytes.
pub fn argmax_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    partials: &Tensor,
    out: &Tensor,
) -> Result<()> {
    argmax_f32_partial(ctx, pass, x, partials)?;
    // The final stage consumes the partial stage.
    pass.level_barrier(&[partials])?;
    argmax_f32_final(ctx, pass, partials, out)
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/elementwise.rs"]
mod tests;
