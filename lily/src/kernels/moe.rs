//! GPU-resident sparse-MoE routing, expert projection, and combine kernels.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/moe.metal");

/// Router limits compiled into moe.metal.
const MAX_E: usize = 1024;
const MAX_K: usize = 16;

/// Softmax top-k routing from F32 logits; ties choose the lowest expert id.
pub fn moe_router_topk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    renorm: bool,
) -> Result<()> {
    let e = logits.numel();
    let k = indices.numel();
    ensure!(
        e <= MAX_E && logits.dtype() == DType::F32,
        "logits must be F32 [<= {MAX_E}]"
    );
    ensure!(
        k <= MAX_K && k <= e && indices.dtype() == DType::U32,
        "indices must be U32 [<= {MAX_K}]"
    );
    ensure!(
        scores.numel() == k && scores.dtype() == DType::F32,
        "scores must be F32 [k]"
    );
    let tg = e.clamp(32, 256).next_multiple_of(32);
    let pipeline = ctx.pipeline("moe_router_topk", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), indices.binding(), scores.binding()],
        &[&u32_bytes(e), &u32_bytes(k), &u32_bytes(renorm as usize)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (tg, 1, 1) },
    )
}

/// Fused Q4 gate/up gather GEMV with SwiGLU output.
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_gemv_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QuantWeights,
    up: &QuantWeights,
    n_per_expert: usize,
    x: &Tensor,
    indices: &Tensor,
    y: &Tensor,
) -> Result<()> {
    ensure!(gate.bits == 4 && up.bits == 4, "expert gate/up fusion is Q4 only");
    ensure!(
        (gate.out_features(), gate.in_features(), gate.group_size)
            == (up.out_features(), up.in_features(), up.group_size),
        "expert gate/up shapes or group sizes differ"
    );
    let (rows, k_in) = (gate.out_features(), gate.in_features());
    ensure!(rows.is_multiple_of(n_per_expert), "bad stacked expert rows");
    ensure!(
        k_in.is_multiple_of(gate.group_size) && gate.group_size.is_multiple_of(32),
        "in {k_in} / group size {} not block-packable",
        gate.group_size
    );
    ensure!(indices.dtype() == DType::U32, "indices must be U32");
    let slots = indices.numel();
    ensure!(x.numel() == k_in, "gate/up fusion expects one shared input vector");
    ensure!(
        y.numel() == slots * n_per_expert && y.dtype() == DType::BF16,
        "y must be BF16 [{slots}, {n_per_expert}]"
    );
    let pipeline =
        ctx.pipeline("moe_gather_gemv_q4_gate_up", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            x.binding(),
            indices.binding(),
            y.binding(),
        ],
        &[&u32_bytes(k_in), &u32_bytes(gate.group_size), &u32_bytes(n_per_expert)],
        Grid::Threadgroups {
            groups: (n_per_expert, slots, 1),
            threadgroup: (32, 1, 1),
        },
    )
}

/// Largest chunk row count accepted by the small-M gather GEMV.
pub const MOE_SMALLM_MAX_M: usize = 8;

/// Pair-list capacity compiled into moe.metal.
const MOE_SMALLM_MAX_S: usize = 256;

/// Routed-row capacity per union expert.
const MOE_SMALLM_MAX_ROWS: usize = 16;

/// Words in the expert-major map: count, expert/row metadata, then pair ids.
pub const MOE_UNION_MAP_WORDS: usize = 1 + MOE_SMALLM_MAX_S * (2 + MOE_SMALLM_MAX_ROWS);

/// Builds an expert-major map from U32 `[m, top_k]` indices.
pub fn moe_union_experts(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    indices: &Tensor,
    umap: &Tensor,
) -> Result<()> {
    let s = indices.numel();
    ensure!(
        indices.dtype() == DType::U32 && s > 0 && s <= MOE_SMALLM_MAX_S,
        "indices must be U32 [1..={MOE_SMALLM_MAX_S}]"
    );
    ensure!(
        umap.dtype() == DType::U32 && umap.numel() >= MOE_UNION_MAP_WORDS,
        "umap must be U32 [>= {MOE_UNION_MAP_WORDS}]"
    );
    let pipeline = ctx.pipeline("moe_union_experts", SOURCE, MslVersion::V3_1)?;
    // One thread per routed pair.
    pass.dispatch_at(
        &pipeline,
        &[indices.binding(), umap.binding()],
        &[&u32_bytes(s)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (MOE_SMALLM_MAX_S, 1, 1) },
    )
}

/// Expert-major Q4 GEMV over a union map.
#[allow(clippy::too_many_arguments)]
pub fn moe_gemv_smallm_em(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    n_per_expert: usize,
    x: &Tensor,
    umap: &Tensor,
    y: &Tensor,
    s: usize,
    top_k: usize,
    x_per_pair: bool,
) -> Result<()> {
    let (rows, k_in) = (w.out_features(), w.in_features());
    ensure!(w.bits == 4, "small-m gather GEMV is 4-bit only");
    ensure!(
        rows.is_multiple_of(n_per_expert),
        "stacked rows {rows} not a multiple of per-expert rows {n_per_expert}"
    );
    // Each 32-element block must stay within one scale group.
    ensure!(
        k_in.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(32),
        "in {k_in} / group size {} not block-packable",
        w.group_size
    );
    ensure!(top_k > 0 && top_k <= MAX_K, "top_k {top_k} out of 1..={MAX_K}");
    ensure!(
        umap.dtype() == DType::U32 && umap.numel() >= MOE_UNION_MAP_WORDS,
        "umap must be U32 [>= {MOE_UNION_MAP_WORDS}]"
    );
    ensure!(s.is_multiple_of(top_k) && s > 0, "pair count {s} not m * {top_k}");
    ensure!(
        s <= MOE_SMALLM_MAX_S,
        "pair count {s} exceeds the staged pair-list capacity"
    );
    let m = s / top_k;
    ensure!(
        m <= MOE_SMALLM_MAX_M,
        "small-m gather GEMV holds at most {MOE_SMALLM_MAX_M} rows per expert \
         in registers (got m={m})"
    );
    let x_expect = if x_per_pair { s * k_in } else { m * k_in };
    ensure!(
        x.numel() == x_expect && x.dtype() == DType::BF16,
        "x numel {} != {x_expect}",
        x.numel()
    );
    ensure!(
        y.numel() == s * n_per_expert && y.dtype() == DType::BF16,
        "y must be BF16 [{s}, {n_per_expert}]"
    );
    let rows_per_tg = if k_in <= 512 { 16 } else { 8 };
    ensure!(
        n_per_expert.is_multiple_of(rows_per_tg),
        "production small-m N per expert {n_per_expert} must be a multiple of \
         {rows_per_tg}"
    );
    let name = if k_in <= 512 {
        "moe_gemv_smallm_q4_em_r8_w"
    } else {
        "moe_gemv_smallm_q4_em_r8"
    };
    let grid = Grid::Threadgroups {
        groups: (n_per_expert / rows_per_tg, s, 1),
        threadgroup: (64, 1, 1),
    };
    let (kb, gsb, nb, sb, tkb, xpb) = (
        u32_bytes(k_in),
        u32_bytes(w.group_size),
        u32_bytes(n_per_expert),
        u32_bytes(s),
        u32_bytes(top_k),
        u32_bytes(x_per_pair as usize),
    );
    let params: [&[u8]; 6] = [&kb, &gsb, &nb, &sb, &tkb, &xpb];
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            umap.binding(),
            y.binding(),
        ],
        &params,
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn moe_gather_gemv_down_combine(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    h: usize,
    x: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    shared: Option<(&Tensor, &Tensor)>,
    out: &Tensor,
) -> Result<()> {
    let k_in = w.in_features();
    ensure!(w.bits == 4, "fused down/combine requires Q4 weights");
    ensure!(h.is_multiple_of(2), "fused down/combine requires even H");
    ensure!(indices.dtype() == DType::U32, "indices must be U32");
    let slots = indices.numel();
    ensure!(
        w.out_features().is_multiple_of(h),
        "stacked expert rows {} not divisible by H {h}",
        w.out_features()
    );
    ensure!(scores.numel() == slots && scores.dtype() == DType::F32, "bad scores");
    ensure!(x.numel() == slots * k_in, "x must be BF16 [{slots}, {k_in}]");
    ensure!(x.dtype() == DType::BF16, "x must be BF16");
    ensure!(out.numel() == h && out.dtype() == DType::BF16, "out must be BF16 [{h}]");
    ensure!(
        k_in.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(32),
        "in {k_in} / group size {} not block-packable",
        w.group_size
    );
    if let Some((shared_out, gate)) = shared {
        ensure!(
            shared_out.numel() == h && shared_out.dtype() == DType::BF16,
            "shared_out must be BF16 [{h}]"
        );
        ensure!(
            gate.numel() == 1 && gate.dtype() == DType::F32,
            "gate must be F32 [1]"
        );
    }
    let (shared_out, gate) = match shared {
        Some((shared_out, gate)) => (shared_out, gate),
        None => (x, scores),
    };
    let pipeline =
        ctx.pipeline("moe_gather_gemv_q4_down_combine", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            indices.binding(),
            scores.binding(),
            shared_out.binding(),
            gate.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(k_in),
            &u32_bytes(w.group_size),
            &u32_bytes(h),
            &u32_bytes(slots),
            &u32_bytes(shared.is_some() as usize),
        ],
        Grid::Threadgroups { groups: (h / 2, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Routes each BF16 `[m, E]` row into indices and F32 scores.
pub fn moe_router_topk_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    renorm: bool,
) -> Result<()> {
    ensure!(logits.shape().len() == 2, "logits must be [m, E]");
    let (m, e) = (logits.shape()[0], logits.shape()[1]);
    ensure!(e <= MAX_E && logits.dtype() == DType::BF16, "logits must be BF16 [m, E]");
    ensure!(
        indices.dtype() == DType::U32 && indices.numel().is_multiple_of(m),
        "indices must be U32 [m, k]"
    );
    let k = indices.numel() / m;
    ensure!(k <= MAX_K && k <= e, "k {k} out of range");
    ensure!(
        scores.numel() == m * k && scores.dtype() == DType::F32,
        "scores must be F32 [m, k]"
    );
    let tg = e.clamp(32, 256).next_multiple_of(32);
    let pipeline = ctx.pipeline("moe_router_topk_rows", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), indices.binding(), scores.binding()],
        &[&u32_bytes(e), &u32_bytes(k), &u32_bytes(renorm as usize)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (tg, 1, 1) },
    )
}

/// Zero-fills a U32 tensor (histogram/cursor reset).
pub fn fill_zero_u32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dst: &Tensor,
) -> Result<()> {
    ensure!(dst.dtype() == DType::U32, "fill_zero_u32 needs U32");
    let n = dst.numel();
    let pipeline = ctx.pipeline("fill_zero_u32", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[dst.binding()],
        &[&u32_bytes(n)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (64, 1, 1) },
    )
}

/// Device buffers for the counting-sort pipeline (`moe_sort_slots`).
pub struct MoeSortBuffers<'a> {
    pub indices: &'a Tensor,
    pub counts: &'a Tensor,
    pub cursors: &'a Tensor,
    pub offsets: &'a Tensor,
    pub tile_offsets: &'a Tensor,
    pub ids_sorted: &'a Tensor,
    pub slot_of: &'a Tensor,
}

/// Sorts routes by expert; scratch must be zeroed and `tile_m` must match the
/// consuming grouped GEMM.
pub fn moe_sort_slots(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    b: &MoeSortBuffers<'_>,
    num_experts: usize,
    top_k: usize,
    tile_m: usize,
) -> Result<()> {
    let MoeSortBuffers {
        indices,
        counts,
        cursors,
        offsets,
        tile_offsets,
        ids_sorted,
        slot_of,
    } = *b;
    let s = indices.numel();
    ensure!(tile_m > 0, "row-tile height must be positive");
    ensure!(indices.dtype() == DType::U32, "indices must be U32");
    ensure!(
        counts.numel() == num_experts && cursors.numel() == num_experts,
        "counts/cursors must be [E]"
    );
    ensure!(
        offsets.numel() == num_experts + 1 && tile_offsets.numel() == num_experts + 1,
        "offsets must be [E+1]"
    );
    ensure!(
        ids_sorted.numel() == s && slot_of.numel() == s,
        "sorted views must be [S]"
    );
    let hist = ctx.pipeline("moe_histogram", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &hist,
        &[indices.binding(), counts.binding()],
        &[&u32_bytes(s)],
        Grid::Threads { grid: (s, 1, 1), threadgroup: (64, 1, 1) },
    )?;
    let scan = ctx.pipeline("moe_scan_offsets", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scan,
        &[counts.binding(), offsets.binding(), tile_offsets.binding()],
        &[&u32_bytes(num_experts), &u32_bytes(tile_m)],
        Grid::Threads { grid: (1, 1, 1), threadgroup: (1, 1, 1) },
    )?;
    let scatter = ctx.pipeline("moe_scatter_slots", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scatter,
        &[
            indices.binding(),
            offsets.binding(),
            cursors.binding(),
            ids_sorted.binding(),
            slot_of.binding(),
        ],
        &[&u32_bytes(top_k), &u32_bytes(s)],
        Grid::Threads { grid: (s, 1, 1), threadgroup: (64, 1, 1) },
    )
}

/// Builds a tile-major grouped-GEMM map; `tile_m` must match the offset scan.
/// Unused capacity receives sentinel blocks.
#[allow(clippy::too_many_arguments)]
pub fn moe_build_blocks(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    offsets: &Tensor,
    tile_offsets: &Tensor,
    blocks: &Tensor,
    num_experts: usize,
    n_per: usize,
    tile_m: usize,
) -> Result<()> {
    ensure!(n_per.is_multiple_of(64), "N per expert must be a multiple of 64");
    ensure!(tile_m > 0, "row-tile height must be positive");
    let n_tiles = n_per / 64;
    ensure!(
        blocks.dtype() == DType::U32 && blocks.numel().is_multiple_of(n_tiles * 4),
        "block map must be U32 sized as tiles x {n_tiles} x uint4"
    );
    let tile_capacity = blocks.numel() / (n_tiles * 4);
    let pipeline = ctx.pipeline("moe_build_blocks", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[offsets.binding(), tile_offsets.binding(), blocks.binding()],
        &[
            &u32_bytes(num_experts),
            &u32_bytes(n_per),
            &u32_bytes(tile_m),
            &u32_bytes(0),
        ],
        Grid::Threads { grid: (tile_capacity, n_tiles, 1), threadgroup: (32, 1, 1) },
    )
}

/// Writes `out[row] = Σ_k scores[row,k] * ed[slots[row,k]]`.
pub fn moe_combine_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    ed: &Tensor,
    slots: &Tensor,
    scores: &Tensor,
    out: &Tensor,
    top_k: usize,
) -> Result<()> {
    ensure!(out.shape().len() == 2, "out must be [m, h]");
    let (m, h) = (out.shape()[0], out.shape()[1]);
    ensure!(h.is_multiple_of(4), "combine needs h % 4 == 0 (got {h})");
    ensure!(
        ed.dtype() == DType::BF16 && ed.numel().is_multiple_of(h),
        "ed must be [S, h]"
    );
    ensure!(
        slots.dtype() == DType::U32 && slots.numel() == m * top_k,
        "slots must be U32 [m, top_k]"
    );
    ensure!(
        scores.dtype() == DType::F32 && scores.numel() == m * top_k,
        "scores must be F32 [m, top_k]"
    );
    let pipeline = ctx.pipeline("moe_combine_rows", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[ed.binding(), slots.binding(), scores.binding(), out.binding()],
        &[&u32_bytes(h), &u32_bytes(top_k)],
        Grid::Threads { grid: (h / 4, m, 1), threadgroup: (32, 1, 1) },
    )
}

/// `dst[r, :] += sigmoid(gate[r]) * src[r, :]` (prefill shared-expert add).
pub fn moe_row_gate_add(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    src: &Tensor,
    gate: &Tensor,
    dst: &Tensor,
) -> Result<()> {
    ensure!(dst.shape().len() == 2, "dst must be [m, h]");
    let (m, h) = (dst.shape()[0], dst.shape()[1]);
    ensure!(src.numel() == m * h && src.dtype() == DType::BF16, "src must be [m, h]");
    ensure!(
        gate.numel() == m && gate.dtype() == DType::BF16,
        "gate logits must be BF16 [m]"
    );
    let pipeline = ctx.pipeline("moe_row_gate_add", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[src.binding(), gate.binding(), dst.binding()],
        &[&u32_bytes(h)],
        Grid::Threads { grid: (h, m, 1), threadgroup: (32, 1, 1) },
    )
}
#[cfg(test)]
#[path = "../../tests/unit/kernels/moe.rs"]
mod tests;
