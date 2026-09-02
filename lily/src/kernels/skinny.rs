//! Small-M Q4 GEMMs with staged-A and register-A variants.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/skinny.metal");

/// Weight rows per threadgroup; must match `SKINNY_SG` in skinny.metal.
const ROWS_PER_TG: usize = 4;

/// Largest row count compiled for register-A kernels.
const REG_MAX_M: usize = 8;

/// Minimum output width for register-A; variants use different reduction orders.
const WIDE_N_MIN: usize = 65536;

/// Rows, one simdgroup each, per register-A threadgroup.
const WIDE_ROWS_PER_TG: usize = 2;

/// Selects register-A when shape and packing constraints hold.
fn reg_routes(m: usize, n: usize, block_walk_ok: bool) -> bool {
    block_walk_ok && m <= REG_MAX_M && n >= WIDE_N_MIN
}

/// Requires a fused stack and every slice to use the same reduction variant.
pub fn stack_route_uniform(
    m: usize,
    n_total: usize,
    slice_ns: &[usize],
    block_walk_ok: bool,
) -> bool {
    let stack_reg = reg_routes(m, n_total, block_walk_ok);
    slice_ns.iter().all(|&n| reg_routes(m, n, block_walk_ok) == stack_reg)
}

/// Largest row count routed to the staged small-M family.
pub const DENSE_SMALLM_THRESHOLD: usize = 16;

/// Whether a nonempty dense chunk uses the staged small-M kernel.
pub fn dense_smallm_routes(m: usize) -> bool {
    m > 0 && m <= DENSE_SMALLM_THRESHOLD
}

fn staged_grid(n: usize) -> Grid {
    Grid::Threadgroups {
        groups: (n.div_ceil(ROWS_PER_TG), 1, 1),
        threadgroup: (32 * ROWS_PER_TG, 1, 1),
    }
}

/// Kernel names for the per-m register-A instantiations (index m - 1).
const Q4_REG_FNS: [&str; REG_MAX_M] = [
    "gemm_skinny_q4_bf16_reg_m1",
    "gemm_skinny_q4_bf16_reg_m2",
    "gemm_skinny_q4_bf16_reg_m3",
    "gemm_skinny_q4_bf16_reg_m4",
    "gemm_skinny_q4_bf16_reg_m5",
    "gemm_skinny_q4_bf16_reg_m6",
    "gemm_skinny_q4_bf16_reg_m7",
    "gemm_skinny_q4_bf16_reg_m8",
];

fn reg_grid(n: usize, rows_per_tg: usize) -> Grid {
    Grid::Threadgroups {
        groups: (n.div_ceil(rows_per_tg), 1, 1),
        threadgroup: (32 * rows_per_tg, 1, 1),
    }
}

fn validate_q4(
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<(usize, usize, usize)> {
    let (n, k) = (w.out_features(), w.in_features());
    let m = a.shape()[0];
    ensure!(m > 0, "skinny GEMM needs at least one row");
    ensure!(
        m <= DENSE_SMALLM_THRESHOLD,
        "skinny GEMM is instantiated only through m={DENSE_SMALLM_THRESHOLD} (got {m})"
    );
    ensure!(w.bits == 4, "skinny quantized GEMM is 4-bit only");
    // The Metal body requires complete 64-element quantization groups.
    ensure!(
        w.group_size == 64 && k.is_multiple_of(64),
        "skinny GEMM requires group_size=64 and K % 64 == 0 (k={k}, gs={})",
        w.group_size
    );
    ensure!(a.shape() == [m, k], "A shape {:?} != [{m}, {k}]", a.shape());
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    ensure!(w.codes.dtype() == DType::U32, "codes must be U32");
    ensure!(
        w.scales.shape() == [n, k / w.group_size]
            && w.biases.shape() == [n, k / w.group_size],
        "scales/biases shape mismatch for [{n}, {k}] gs={}",
        w.group_size
    );
    ensure!(
        a.dtype() == DType::BF16 && c.dtype() == DType::BF16,
        "skinny q4 GEMM activations must be BF16"
    );
    Ok((m, k, n))
}

/// Whether the register-A walk can consume complete quant groups.
pub fn q4_block_walk_ok(k: usize, group_size: usize) -> bool {
    group_size == 64 && k.is_multiple_of(64)
}

fn dispatch_q4_staged(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    let fn_name = if m <= REG_MAX_M {
        "gemm_skinny_q4_bf16_m8"
    } else {
        "gemm_skinny_q4_bf16_m16"
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size), &u32_bytes(m)],
        staged_grid(n),
    )
}

fn dispatch_q4_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    let pipeline = ctx.pipeline(Q4_REG_FNS[m - 1], SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        reg_grid(n, WIDE_ROWS_PER_TG),
    )
}

/// Small-M affine-Q4 GEMM with BF16-rounded dequantization. Register-A requires
/// `m <= 8`, `group_size == 64`, `K % 64 == 0`, and a wide output.
pub fn gemm_skinny_q4_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<()> {
    let dims = validate_q4(a, w, c)?;
    let (m, k, n) = dims;
    if reg_routes(m, n, q4_block_walk_ok(k, w.group_size)) {
        return dispatch_q4_reg(ctx, pass, a, w, c, dims);
    }
    dispatch_q4_staged(ctx, pass, a, w, c, dims)
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/skinny.rs"]
mod tests;
