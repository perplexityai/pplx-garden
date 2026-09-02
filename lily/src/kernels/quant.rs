//! Affine Q4/Q8 kernels with BF16 activations and F32 accumulation.
//! Q4 packs eight low-nibble-first codes per u32; `w = scale*q + bias`.

use anyhow::{Result, ensure};

use crate::kernels::gemm::gemm_bf16_nt;
use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/quant.metal");

/// Rows each thread of the Q4 gather GEMV accumulates.
const Q4_GEMV_ROWS: usize = 2;

/// Row-tile height and execution-simdgroup count of a grouped Q4 GEMM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeTile {
    T32Sg4,
    T64Sg4,
}

impl MoeTile {
    /// Row-tile height T — the block map's granularity.
    pub fn rows(self) -> usize {
        match self {
            Self::T32Sg4 => 32,
            Self::T64Sg4 => 64,
        }
    }

    /// Tensor-op simdgroups; dispatch uses `32 * simdgroups` threads.
    pub fn simdgroups(self) -> usize {
        match self {
            Self::T32Sg4 | Self::T64Sg4 => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::T32Sg4 => "32x4",
            Self::T64Sg4 => "64x4",
        }
    }

    fn nax_kernel(self) -> &'static str {
        match self {
            Self::T32Sg4 => "gemm_q4_nt_nax_grouped_t32x4",
            Self::T64Sg4 => "gemm_q4_nt_nax_grouped_t64x4",
        }
    }
}

/// Row count at which grouped Q4 GEMM switches to the 64x4 tile.
pub const MOE_TILE_MROUTE_DEFAULT: usize = 3072;

/// Largest prefill chunk routed to grouped GEMV instead of tiled GEMM.
const MOE_SMALLM_THRESHOLD: usize = 8;

// The small-M route is limited by per-expert register capacity.
const _: () = assert!(MOE_SMALLM_THRESHOLD > 0);
const _: () = assert!(MOE_SMALLM_THRESHOLD <= crate::kernels::moe::MOE_SMALLM_MAX_M);

/// Grouped GEMM or direct small-M GEMV execution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeRoute {
    Grouped(MoeTile),
    SmallmGemv,
}

impl MoeRoute {
    pub fn label(self) -> &'static str {
        match self {
            Self::Grouped(tile) => tile.label(),
            Self::SmallmGemv => "gemv",
        }
    }
}

/// Fixed per-chunk MoE routing policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeTilePolicy;

impl MoeTilePolicy {
    /// Selects small-M through 8 rows, 64x4 from 3072, otherwise 32x4.
    pub fn route_for(&self, chunk_m: usize) -> MoeRoute {
        if chunk_m > 0 && chunk_m <= MOE_SMALLM_THRESHOLD {
            MoeRoute::SmallmGemv
        } else {
            MoeRoute::Grouped(self.tile_for(chunk_m))
        }
    }

    fn tile_for(&self, chunk_m: usize) -> MoeTile {
        if chunk_m >= MOE_TILE_MROUTE_DEFAULT {
            MoeTile::T64Sg4
        } else {
            MoeTile::T32Sg4
        }
    }

    /// Upper bound for grouped block-map rows.
    pub fn alloc_tile(&self) -> MoeTile {
        MoeTile::T32Sg4
    }
}

/// Returns the fixed chunk-routing policy.
pub fn moe_tile_mroute_table() -> MoeTilePolicy {
    MoeTilePolicy
}

fn check_quant(w: &QuantWeights) -> Result<(usize, usize)> {
    let (n, k) = (w.out_features(), w.in_features());
    ensure!(matches!(w.bits, 4 | 8), "unsupported bit width {}", w.bits);
    // A uint4 block must not cross a quantization group.
    ensure!(
        k.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(128 / w.bits),
        "in_features {k} / group size {} not block-packable at {} bits",
        w.group_size,
        w.bits
    );
    ensure!(w.codes.dtype() == DType::U32, "codes must be U32");
    // uint4 code loads require 16-byte alignment.
    ensure!(
        w.codes.binding().1.is_multiple_of(16),
        "codes byte offset {} not 16-aligned; the kernels read codes as uint4",
        w.codes.binding().1
    );
    ensure!(
        w.scales.shape() == [n, k / w.group_size]
            && w.biases.shape() == [n, k / w.group_size],
        "scales/biases shape mismatch for [{n}, {k}] gs={}",
        w.group_size
    );
    Ok((n, k))
}

pub fn gemv_quant(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(x.numel() == k, "x numel {} != K {k}", x.numel());
    ensure!(y.numel() == n, "y numel {} != N {n}", y.numel());
    let requested_rows =
        if w.bits == 4 && y.dtype() == DType::BF16 { Q4_GEMV_ROWS } else { 1 };
    let packed_rows = if n.is_multiple_of(requested_rows) { requested_rows } else { 1 };
    let fn_name = match (w.bits, y.dtype(), packed_rows) {
        (4, DType::BF16, 1) => "gemv_q4_bf16",
        (4, DType::BF16, 2) => "gemv_q4_bf16_2row",
        (4, DType::F32, 1) => "gemv_q4_bf16_f32out",
        (8, DType::BF16, 1) => "gemv_q8_bf16",
        (8, DType::F32, 1) => "gemv_q8_bf16_f32out",
        _ => anyhow::bail!("gemv output must be BF16 or F32"),
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            y.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threadgroups { groups: (n / packed_rows, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Writes `out[N, K] = dequant(W)` in bf16.
fn dequant_to_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    out: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        out.shape() == [n, k] && out.dtype() == DType::BF16,
        "dequant out must be BF16 [{n}, {k}], got {:?} {:?}",
        out.shape(),
        out.dtype()
    );
    let fn_name = if w.bits == 4 { "dequant_q4_bf16" } else { "dequant_q8_bf16" };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(), out.binding()],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threads { grid: (k / (32 / w.bits), n, 1), threadgroup: (32, 1, 1) },
    )
}

/// Dequantizes `W` into bf16 scratch, then computes
/// `C[M, N] = A[M, K] . W[N, K]^T`.
pub fn gemm_quant_bf16_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    scratch: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        scratch.numel() >= n * k && scratch.dtype() == DType::BF16,
        "dequant scratch holds {} bf16 elements, needs {}",
        scratch.numel(),
        n * k
    );
    let staged = scratch.view(0, &[n, k])?;
    dequant_to_bf16(ctx, pass, w, &staged)?;
    gemm_bf16_nt(ctx, pass, a, &staged, c)
}

/// Block-mapped grouped GEMM with Q4 dequantization fused into B-tile staging.
/// `tile` must match the configuration used to build the block map.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_grouped_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    blocks: &Tensor,
    n_blocks: usize,
    tile: MoeTile,
) -> Result<()> {
    let (_, k) = check_quant(w)?;
    ensure!(
        w.bits == 4 && w.group_size == 64 && k.is_multiple_of(64),
        "grouped NAX GEMM needs 4-bit group-64 weights and K % 64 == 0"
    );
    let (s, ak) = (a.shape()[0], a.shape()[1]);
    ensure!(ak == k, "A K {ak} != weight K {k}");
    let n = c.shape()[1];
    ensure!(c.shape() == [s, n], "C shape {:?} != [{s}, {n}]", c.shape());
    ensure!(n.is_multiple_of(64), "grouped GEMM needs N % 64 == 0 (got {n})");
    ensure!(blocks.dtype() == DType::U32, "block map must be U32");
    ensure!(blocks.numel() >= n_blocks * 4 && n_blocks > 0, "bad block map");
    let pipeline = ctx.pipeline(tile.nax_kernel(), SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
            blocks.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        Grid::Threadgroups {
            groups: (n_blocks, 1, 1),
            threadgroup: (32 * tile.simdgroups(), 1, 1),
        },
    )
}

/// `out[M, K]` = dequantized embedding rows for `ids` (batched prefill gather).
pub fn gather_rows_q4(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    ids: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let (_, k) = check_quant(w)?;
    ensure!(w.bits == 4, "embedding gathers are 4-bit only");
    let m = ids.numel();
    ensure!(ids.dtype() == DType::U32, "ids must be U32");
    ensure!(
        out.numel() == m * k && out.dtype() == DType::BF16,
        "gather out must be BF16 [{m}, {k}]"
    );
    let pipeline = ctx.pipeline("gather_rows_q4_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            ids.binding(),
            out.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threads { grid: (k / 8, m, 1), threadgroup: (32, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/quant.rs"]
mod tests;
