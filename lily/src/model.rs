//! The Qwen3.5 model graph: single-token decode steps plus a batched prefill
//! path that processes `PREFILL_CHUNK` prompt tokens per dispatch round (GEMM
//! projections, in-kernel token loops for the conv/GDN recurrences, causal
//! SDPA), reading the weights once per chunk instead of once per token.

use anyhow::{Result, ensure};
use std::path::Path;

use crate::config::TextConfig;
use crate::kernels::attention::{
    MAX_SEQ, k_norm_rope_scatter_decode, q_norm_rope_split_decode, rope_neox,
    scatter_kv, sdpa_decode, sdpa_prefill, sdpa_split_scratch_splits, split_q_gate,
};
use crate::kernels::elementwise::{
    ARGMAX_GROUPS, add_bf16, argmax_f32, gather_row_bf16, gather_rows_bf16,
    sigmoid_mul_bf16, silu_mul_bf16, split_cols_bf16,
};
use crate::kernels::gdn::{
    GDN_HEAD_DIM, GDN_STATE_DTYPE, GdnRegscanStaging, conv1d_prefill, conv1d_step,
    gated_rmsnorm, gdn_prefill, gdn_step_gated_fused,
};
use crate::kernels::norm::{add_rmsnorm_bf16, rmsnorm_bf16};
use crate::kernels::{moe, quant, skinny};
use crate::metal::{ComputePass, MetalContext, PendingPass};
use crate::tensor::{DType, Tensor};
use crate::weights::{
    self, AttnWeights, GdnWeights, LayerWeights, LinearWeights, ModelWeights,
    MoeWeights,
};

/// Qwen3.5's standard RMSNorms are zero-centered: gain = 1 + weight. (The GDN
/// GatedNorm is the exception and uses a plain gain.)
const NORM_WEIGHT_BIAS: f32 = 1.0;

/// Prompt tokens processed by one prefill command buffer.
const PREFILL_CHUNK: usize = 4096;

pub struct Qwen3_5Model {
    pub config: TextConfig,
    weights: ModelWeights,
    attn_scale: f32,
    gdn_scale: f32,
}

/// Projects an already-stacked weight group as one skinny GEMM when eligible;
/// otherwise dispatches the corresponding per-slice projections. Keeping both
/// arms here prevents the fused and fallback projection lists from drifting.
fn project_stack_or_slices<const N: usize>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    stack_w: &LinearWeights,
    stack_c: &Tensor,
    projections: [(&LinearWeights, &Tensor); N],
    scratch: &Tensor,
) -> Result<()> {
    let m = a.shape()[0];
    let n_total = stack_w.out_features();
    let widths: [usize; N] = std::array::from_fn(|i| projections[i].1.numel() / m);
    let walk_ok = skinny::q4_block_walk_ok(stack_w.in_features(), stack_w.group_size);
    let fused = widths.iter().sum::<usize>() == n_total
        && stack_w.bits == 4
        && skinny::dense_smallm_routes(m)
        && skinny::stack_route_uniform(m, n_total, &widths, walk_ok);
    if fused {
        // Fusing is bit-identical only when the stack and every slice choose
        // the same skinny variant; staged and register-A reduce in different
        // f32 orders.
        let c = stack_c.view(0, &[m, n_total])?;
        skinny::gemm_skinny_q4_nt(ctx, pass, a, stack_w, &c)?;
        let outs: [&Tensor; N] = std::array::from_fn(|i| projections[i].1);
        split_cols_bf16(ctx, pass, &c, &outs)?;
        return Ok(());
    }

    for (w, out) in projections {
        project_mat(ctx, pass, a, w, out, scratch)?;
    }
    Ok(())
}

/// Dispatches a GEMM on the projection's storage precision; quantized weights
/// stage a bf16 dequant into `scratch` first (see `quant::gemm_q4_bf16_nt`),
/// unless the site routes to the skinny small-m kernel.
fn project_mat(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &LinearWeights,
    c: &Tensor,
    scratch: &Tensor,
) -> Result<()> {
    if w.bits == 4 && skinny::dense_smallm_routes(a.shape()[0]) {
        skinny::gemm_skinny_q4_nt(ctx, pass, a, w, c)?;
        return Ok(());
    }
    quant::gemm_quant_bf16_nt(ctx, pass, a, w, c, scratch)
}

enum LayerState {
    Gdn {
        /// Recurrent state, fp32 `[H, 128, 128]`.
        state: Tensor,
        /// Double-buffered conv window, bf16 `[C, KD-1]` each: a prefill
        /// chunk's conv kernel reads one buffer and writes the other (an
        /// in-place update would race across its threadgroups), selected by
        /// `DecodeState::conv_slot`.
        conv_windows: [Tensor; 2],
    },
    Full {
        /// `[KVH, max_seq, D]` bf16.
        k_cache: Tensor,
        v_cache: Tensor,
    },
}

pub struct DecodeState {
    pub pos: usize,
    max_seq: usize,
    layers: Vec<LayerState>,
    /// Which `conv_windows` buffer holds the current cross-chunk conv state
    /// (all GDN layers advance in lockstep): prefill chunks read it, write
    /// the other, then flip; decode steps update it in place.
    conv_slot: usize,
}

impl DecodeState {
    /// Returns the state to position zero with zeroed GDN recurrent state
    /// and conv windows, so a session slot can recycle its buffers instead
    /// of reallocating. KV needs no clearing: rows past `pos` are never
    /// read and get overwritten. The GPU must be idle on this state's buffers.
    pub fn reset(&mut self) {
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                state.zero_fill();
                conv_windows[0].zero_fill();
                conv_windows[1].zero_fill();
            }
        }
        self.pos = 0;
        self.conv_slot = 0;
    }
}

/// Per-step intermediates, allocated once. The projection outputs that share
/// one fused matvec (`gdn_in`, `attn_qkv`, `mlp_gu`) are single buffers whose
/// named segments below are views.
pub struct Scratch {
    x: Tensor,
    normed: Tensor,
    branch_out: Tensor,
    gdn_in: Tensor,
    attn_qkv: Tensor,
    mlp_gu: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_gated: Tensor,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    logits: Tensor,
    sdpa_partials: Tensor,
    sdpa_stats: Tensor,
    argmax_partials: Tensor,
    /// Greedy tokens written by the in-graph argmax (`U32[2]`): two ping-pong
    /// slots so the pipelined loop can host-read step N's token while the
    /// in-flight step N+1 writes the other slot.
    pub next_token: Tensor,
    /// Prefill staging for the largest quantized projection.
    dequant: Tensor,
    /// Sparse-MoE decode intermediates.
    moe: MoeScratch,
    /// Session-lived batched prefill intermediates, grown on demand and capped
    /// at `PREFILL_CHUNK` rows.
    prefill: Option<PrefillScratch>,
}

/// Decode-side MoE intermediates: the router distribution and the top-k
/// expert projections, all device-resident (no readback per step).
struct MoeScratch {
    router_logits: Tensor,
    indices: Tensor,
    scores: Tensor,
    act: Tensor,
    shared_out: Tensor,
    shared_gate: Tensor,
}

/// Batched prefill intermediates, owned by [`Scratch`] at a capacity that only
/// grows. Each chunk uses exact row-prefix views of these buffers.
struct PrefillScratch {
    m: usize,
    ids: Tensor,
    /// Router logits `[m, E]` + shared-expert staging.
    moe: PrefillMoeScratch,
    x: Tensor,
    normed: Tensor,
    branch_out: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_out: Tensor,
    gdn_gated: Tensor,
    /// Register-scan staging: normalized q/k rows and precomputed per-token
    /// gates.
    gdn_stage: GdnStageScratch,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    /// The fused-stack GEMM output (`project_stack_or_slices` views
    /// `[m, n_total]` prefixes of it per level), sized for the widest
    /// stacked weight group at the largest row count the small-m route serves.
    stack: Tensor,
}

/// The two `[m, ·]` elementwise outputs the regscan prefill path stages per
/// layer·chunk (plus β, which shares the gates pass).
struct GdnStageScratch {
    qk_norm: Tensor,
    decay: Tensor,
    beta: Tensor,
}

/// GPU-resident MoE prefill intermediates (capacity-sized, viewed per chunk
/// like the rest of [`PrefillScratch`]): the router distribution, the
/// counting-sort state, the sentinel-padded block maps, and the expert-slot
/// activations (`S = m·top_k` rows).
struct PrefillMoeScratch {
    router_logits: Tensor,
    shared_out: Tensor,
    shared_gate: Tensor,
    indices: Tensor,
    scores: Tensor,
    counts: Tensor,
    cursors: Tensor,
    offsets: Tensor,
    tile_offsets: Tensor,
    ids_sorted: Tensor,
    slot_of: Tensor,
    /// Identity slot map `0..S` (`[m, top_k]`, host-filled once, never
    /// written by the GPU): the small-m GEMV route keeps its expert outputs
    /// in natural (row, k) pair order, so `moe_combine_rows` consumes this
    /// instead of the counting sort's `slot_of`.
    slots_iota: Tensor,
    /// the expert-major union map (`moe_union_experts` output,
    /// capacity-sized) the GEMV route's kernels read — built once per
    /// layer·chunk and shared by the three expert projections.
    umap: Tensor,
    blocks_gu: Tensor,
    blocks_dn: Tensor,
    gx: Tensor,
    eg: Tensor,
    eu: Tensor,
    ea: Tensor,
    ed: Tensor,
    /// Row-tile capacity bound `ceil(S/T) + min(E, S)` at `tile.rows()` —
    /// covers any expert split without a readback; block-map slots past the
    /// real count carry the sentinel the grouped GEMM kernels skip.
    tile_capacity: usize,
    /// The grouped-GEMM row-tile config `tile_capacity` was computed at: the
    /// policy's alloc tile for the backing scratch (the sizing bound), the
    /// policy's per-chunk answer for a chunk view. The GPU router's tile
    /// scan and the kernel selection in `prefill_moe` all read this one
    /// value (the lockstep contract; `prefill_moe` re-checks the pair).
    tile: quant::MoeTile,
    /// The chunk's grouped-MoE execution route: block-mapped
    /// grouped GEMM at `tile`, or the small-m GEMV fast path. Set together
    /// with `tile`/`tile_capacity` in [`Self::chunk`]; `prefill_moe`
    /// re-checks it against the fixed shipped policy (the desync guard).
    route: quant::MoeRoute,
}

/// A row-prefix view of `t` (same buffer, leading dimension shrunk to `m`).
fn prefix_rows(t: &Tensor, m: usize) -> Result<Tensor> {
    let mut shape = t.shape().to_vec();
    shape[0] = m;
    t.view(0, &shape)
}

impl PrefillScratch {
    fn new(ctx: &MetalContext, cfg: &TextConfig, capacity: usize) -> Result<Self> {
        ensure!(capacity > 0, "prefill scratch capacity must be nonzero");
        let m = capacity;
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let bf = DType::BF16;
        let moe_policy = quant::moe_tile_mroute_table();
        Ok(Self {
            m,
            ids: Tensor::zeros(ctx, &[m], DType::U32)?,
            x: Tensor::zeros(ctx, &[m, h], bf)?,
            normed: Tensor::zeros(ctx, &[m, h], bf)?,
            branch_out: Tensor::zeros(ctx, &[m, h], bf)?,
            mlp_gate: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_up: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_act: Tensor::zeros(ctx, &[m, inter], bf)?,
            qkv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            qkv_conv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            z: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            a: Tensor::zeros(ctx, &[m, heads], bf)?,
            b: Tensor::zeros(ctx, &[m, heads], bf)?,
            gdn_out: Tensor::zeros(ctx, &[m, heads, GDN_HEAD_DIM], bf)?,
            gdn_gated: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            // Register-scan staging scales with the current row capacity.
            gdn_stage: {
                let hk = cfg.linear_num_key_heads;
                GdnStageScratch {
                    qk_norm: Tensor::zeros(ctx, &[m, 2 * hk * GDN_HEAD_DIM], bf)?,
                    decay: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
                    beta: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
                }
            },
            qg: Tensor::zeros(ctx, &[m, nq * 2 * hd], bf)?,
            q: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            k_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            v_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            attn_o: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[m, nq * hd], bf)?,
            stack: {
                // Rows cap at the largest chunk eligible for fused projection.
                let rows = m.min(skinny::DENSE_SMALLM_THRESHOLD);
                let width = (conv_c + dim_v + 2 * heads)
                    .max((nq * 2 + 2 * nkv) * hd)
                    .max(2 * inter);
                Tensor::zeros(ctx, &[rows, width], bf)?
            },
            moe: {
                let (e, k, mi) = (
                    cfg.num_experts,
                    cfg.num_experts_per_tok,
                    cfg.moe_intermediate_size,
                );
                let s_total = m * k;
                // Sized at the policy's alloc tile: cap is monotone
                // nonincreasing in tile rows, so the smallest tile the
                // policy can emit bounds every m-routed chunk's map.
                let alloc_tile = moe_policy.alloc_tile();
                // Tail chunks have S < E: bounding the partial-tile term by
                // min(E, S) keeps the sentinel share sane on short tails.
                let cap = s_total.div_ceil(alloc_tile.rows()) + e.min(s_total);
                let u32t = DType::U32;
                PrefillMoeScratch {
                    router_logits: Tensor::zeros(ctx, &[m, e], bf)?,
                    shared_out: Tensor::zeros(ctx, &[m, h], bf)?,
                    shared_gate: Tensor::zeros(ctx, &[m], bf)?,
                    indices: Tensor::zeros(ctx, &[m, k], u32t)?,
                    scores: Tensor::zeros(ctx, &[m, k], DType::F32)?,
                    counts: Tensor::zeros(ctx, &[e], u32t)?,
                    cursors: Tensor::zeros(ctx, &[e], u32t)?,
                    offsets: Tensor::zeros(ctx, &[e + 1], u32t)?,
                    tile_offsets: Tensor::zeros(ctx, &[e + 1], u32t)?,
                    ids_sorted: Tensor::zeros(ctx, &[s_total], u32t)?,
                    slot_of: Tensor::zeros(ctx, &[m, k], u32t)?,
                    slots_iota: {
                        let iota: Vec<u32> = (0..s_total as u32).collect();
                        Tensor::from_bytes(
                            ctx,
                            bytemuck::cast_slice(&iota),
                            &[m, k],
                            u32t,
                        )?
                    },
                    umap: Tensor::zeros(ctx, &[moe::MOE_UNION_MAP_WORDS], u32t)?,
                    blocks_gu: Tensor::zeros(ctx, &[cap * (mi / 64) * 4], u32t)?,
                    blocks_dn: Tensor::zeros(ctx, &[cap * (h / 64) * 4], u32t)?,
                    gx: Tensor::zeros(ctx, &[s_total, h], bf)?,
                    eg: Tensor::zeros(ctx, &[s_total, mi], bf)?,
                    eu: Tensor::zeros(ctx, &[s_total, mi], bf)?,
                    ea: Tensor::zeros(ctx, &[s_total, mi], bf)?,
                    ed: Tensor::zeros(ctx, &[s_total, h], bf)?,
                    tile_capacity: cap,
                    tile: alloc_tile,
                    route: quant::MoeRoute::Grouped(alloc_tile),
                }
            },
        })
    }

    /// Views of the capacity scratch for one chunk of `tokens`, uploading the
    /// token ids (the GPU is idle here: chunks are commit_wait-synchronized).
    /// Reuse across chunks and prefill calls needs no re-zeroing: within a
    /// chunk's pass every buffer is written over the full view extent before
    /// it is read. Counts and cursors are filled explicitly, block maps are
    /// fully rewritten, and union-map reads are bounded by fresh counts.
    fn chunk(&self, tokens: &[u32]) -> Result<Self> {
        let m = tokens.len();
        ensure!(m <= self.m, "chunk of {m} tokens exceeds scratch capacity {}", self.m);
        let ids = prefix_rows(&self.ids, m)?;
        let chunk = Self {
            m,
            ids,
            x: prefix_rows(&self.x, m)?,
            normed: prefix_rows(&self.normed, m)?,
            branch_out: prefix_rows(&self.branch_out, m)?,
            mlp_gate: prefix_rows(&self.mlp_gate, m)?,
            mlp_up: prefix_rows(&self.mlp_up, m)?,
            mlp_act: prefix_rows(&self.mlp_act, m)?,
            qkv: prefix_rows(&self.qkv, m)?,
            qkv_conv: prefix_rows(&self.qkv_conv, m)?,
            z: prefix_rows(&self.z, m)?,
            a: prefix_rows(&self.a, m)?,
            b: prefix_rows(&self.b, m)?,
            gdn_out: prefix_rows(&self.gdn_out, m)?,
            gdn_gated: prefix_rows(&self.gdn_gated, m)?,
            gdn_stage: GdnStageScratch {
                qk_norm: prefix_rows(&self.gdn_stage.qk_norm, m)?,
                decay: prefix_rows(&self.gdn_stage.decay, m)?,
                beta: prefix_rows(&self.gdn_stage.beta, m)?,
            },
            qg: prefix_rows(&self.qg, m)?,
            q: prefix_rows(&self.q, m)?,
            gate: prefix_rows(&self.gate, m)?,
            k_new: prefix_rows(&self.k_new, m)?,
            v_new: prefix_rows(&self.v_new, m)?,
            attn_o: prefix_rows(&self.attn_o, m)?,
            attn_gated: prefix_rows(&self.attn_gated, m)?,
            // Kept at full extent (its rows cap at the routing threshold,
            // not the chunk): `project_stack_or_slices` views the exact [m, n_total]
            // prefix per level and never fuses past the threshold.
            stack: self.stack.view(0, self.stack.shape())?,
            moe: self.moe.chunk(m)?,
        };
        chunk.ids.write_bytes(bytemuck::cast_slice(tokens))?;
        Ok(chunk)
    }
}

impl PrefillMoeScratch {
    /// Per-chunk views (`S = m·top_k` slot rows, the chunk's own row-tile
    /// capacity bound). The block-map views are sized to the chunk's bound so
    /// `moe_build_blocks` (grid = view capacity) rewrites exactly the entries
    /// the grouped GEMMs read — stale tiles a larger previous chunk wrote
    /// past this extent are unreachable.
    fn chunk(&self, m: usize) -> Result<Self> {
        let e = self.counts.numel();
        let k = self.indices.shape()[1];
        let s_total = m * k;
        // Set route, tile, and capacity together. The GEMV route retains
        // bounded unused block-map views.
        let policy = quant::moe_tile_mroute_table();
        let route = policy.route_for(m);
        let tile = match route {
            quant::MoeRoute::Grouped(tile) => tile,
            quant::MoeRoute::SmallmGemv => policy.alloc_tile(),
        };
        let cap = s_total.div_ceil(tile.rows()) + e.min(s_total);
        let slot_rows = |t: &Tensor| prefix_rows(t, s_total);
        let block_view = |t: &Tensor| {
            let words_per_tile = t.numel() / self.tile_capacity;
            t.view(0, &[cap * words_per_tile])
        };
        Ok(Self {
            router_logits: prefix_rows(&self.router_logits, m)?,
            shared_out: prefix_rows(&self.shared_out, m)?,
            shared_gate: prefix_rows(&self.shared_gate, m)?,
            indices: prefix_rows(&self.indices, m)?,
            scores: prefix_rows(&self.scores, m)?,
            counts: self.counts.view(0, self.counts.shape())?,
            cursors: self.cursors.view(0, self.cursors.shape())?,
            offsets: self.offsets.view(0, self.offsets.shape())?,
            tile_offsets: self.tile_offsets.view(0, self.tile_offsets.shape())?,
            ids_sorted: self.ids_sorted.view(0, &[s_total])?,
            slot_of: prefix_rows(&self.slot_of, m)?,
            slots_iota: prefix_rows(&self.slots_iota, m)?,
            umap: self.umap.view(0, self.umap.shape())?,
            blocks_gu: block_view(&self.blocks_gu)?,
            blocks_dn: block_view(&self.blocks_dn)?,
            gx: slot_rows(&self.gx)?,
            eg: slot_rows(&self.eg)?,
            eu: slot_rows(&self.eu)?,
            ea: slot_rows(&self.ea)?,
            ed: slot_rows(&self.ed)?,
            tile_capacity: cap,
            tile,
            route,
        })
    }
}

impl Qwen3_5Model {
    pub fn load(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Self> {
        let config = TextConfig::from_model_dir(&dir)?;
        ensure!(
            config.linear_key_head_dim == GDN_HEAD_DIM
                && config.linear_value_head_dim == GDN_HEAD_DIM,
            "GDN head dim {} unsupported (kernel is compiled for {GDN_HEAD_DIM})",
            config.linear_key_head_dim,
        );
        let weights = weights::load(ctx, &dir, &config)?;
        let attn_scale = 1.0 / (config.head_dim as f32).sqrt();
        let gdn_scale = 1.0 / (config.linear_key_head_dim as f32).sqrt();
        Ok(Self { config, weights, attn_scale, gdn_scale })
    }

    pub fn new_state(&self, ctx: &MetalContext, max_seq: usize) -> Result<DecodeState> {
        ensure!(max_seq <= MAX_SEQ, "max_seq {max_seq} exceeds kernel limit {MAX_SEQ}");
        let c = self.config.gdn_conv_channels();
        let kd = self.config.linear_conv_kernel_dim;
        let heads = self.config.linear_num_value_heads;
        let layers = self
            .weights
            .layers
            .iter()
            .map(|layer| match layer {
                LayerWeights::Gdn(_) => Ok(LayerState::Gdn {
                    state: Tensor::zeros(
                        ctx,
                        &[heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                        GDN_STATE_DTYPE,
                    )?,
                    conv_windows: [
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                    ],
                }),
                LayerWeights::Full(_) => Ok(LayerState::Full {
                    k_cache: Tensor::zeros(
                        ctx,
                        &[
                            self.config.num_key_value_heads,
                            max_seq,
                            self.config.head_dim,
                        ],
                        DType::BF16,
                    )?,
                    v_cache: Tensor::zeros(
                        ctx,
                        &[
                            self.config.num_key_value_heads,
                            max_seq,
                            self.config.head_dim,
                        ],
                        DType::BF16,
                    )?,
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DecodeState { pos: 0, max_seq, layers, conv_slot: 0 })
    }

    /// Scratch whose split-decode buffers are sized for `split_capacity_tokens`
    /// of context rather than for the engine's ceiling.
    ///
    /// This avoids sizing the two `sdpa_*` buffers for the model's full
    /// declared context when the server uses a smaller limit.
    pub fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        split_capacity_tokens: usize,
    ) -> Result<Scratch> {
        // Size for 256-token split routes and the 128-block fixed route.
        let splits = sdpa_split_scratch_splits(split_capacity_tokens);
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let bf = DType::BF16;
        let gdn_in = Tensor::zeros(ctx, &[conv_c + dim_v + 2 * heads], bf)?;
        let attn_qkv = Tensor::zeros(ctx, &[(nq * 2 + 2 * nkv) * hd], bf)?;
        let mlp_gu = Tensor::zeros(ctx, &[2 * inter], bf)?;
        // Largest single dense projection that the batched fallback may stage.
        // Expert stacks use native grouped Q4 kernels and never dequant in full.
        let dequant_numel = [
            conv_c * h,
            dim_v * h,
            heads * h,
            h * dim_v,
            nq * 2 * hd * h,
            nkv * hd * h,
            h * nq * hd,
            cfg.num_experts * h,
            cfg.moe_intermediate_size * h,
            h * cfg.moe_intermediate_size,
            inter * h,
            h * inter,
            h,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Ok(Scratch {
            x: Tensor::zeros(ctx, &[h], bf)?,
            normed: Tensor::zeros(ctx, &[h], bf)?,
            branch_out: Tensor::zeros(ctx, &[h], bf)?,
            mlp_gate: mlp_gu.view(0, &[inter])?,
            mlp_up: mlp_gu.view(inter, &[inter])?,
            mlp_act: Tensor::zeros(ctx, &[inter], bf)?,
            qkv: gdn_in.view(0, &[conv_c])?,
            qkv_conv: Tensor::zeros(ctx, &[conv_c], bf)?,
            z: gdn_in.view(conv_c, &[dim_v])?,
            a: gdn_in.view(conv_c + dim_v, &[heads])?,
            b: gdn_in.view(conv_c + dim_v + heads, &[heads])?,
            gdn_gated: Tensor::zeros(ctx, &[dim_v], bf)?,
            qg: attn_qkv.view(0, &[nq, 2 * hd])?,
            q: Tensor::zeros(ctx, &[nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[nq, hd], bf)?,
            k_new: attn_qkv.view(nq * 2 * hd, &[nkv, hd])?,
            v_new: attn_qkv.view((nq * 2 + nkv) * hd, &[nkv, hd])?,
            attn_o: Tensor::zeros(ctx, &[nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[nq * hd], bf)?,
            gdn_in,
            attn_qkv,
            mlp_gu,
            logits: Tensor::zeros(ctx, &[cfg.vocab_size], DType::F32)?,
            sdpa_partials: Tensor::zeros(ctx, &[nq, splits, hd], DType::F32)?,
            sdpa_stats: Tensor::zeros(ctx, &[nq, splits, 2], DType::F32)?,
            argmax_partials: Tensor::zeros(ctx, &[2 * ARGMAX_GROUPS], DType::U32)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
            dequant: Tensor::zeros(ctx, &[dequant_numel], bf)?,
            moe: {
                let (e, k, i) = (
                    cfg.num_experts,
                    cfg.num_experts_per_tok,
                    cfg.moe_intermediate_size,
                );
                MoeScratch {
                    router_logits: Tensor::zeros(ctx, &[e], DType::F32)?,
                    indices: Tensor::zeros(ctx, &[k], DType::U32)?,
                    scores: Tensor::zeros(ctx, &[k], DType::F32)?,
                    act: Tensor::zeros(ctx, &[k, i], bf)?,
                    shared_out: Tensor::zeros(ctx, &[h], bf)?,
                    shared_gate: Tensor::zeros(ctx, &[1], DType::F32)?,
                }
            },
            prefill: None,
        })
    }

    /// Submits one decode step whose input token is read on-GPU from
    /// `next_token[slot_in]` (written by the previous step's argmax) and whose
    /// argmax lands in `next_token[slot_out]`, submitted without waiting.
    /// The depth-2 pipelined decode primitive: the host never needs the token
    /// value to encode the next step, so a second pass can be in flight while
    /// this one executes. `pos` advances at encode time.
    pub fn submit_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
    ) -> Result<PendingPass<'a>> {
        let pass = ctx.begin_concurrent()?;
        self.encode_decode_step(ctx, &pass, state, s, slot_in, slot_out)?;
        state.pos += 1;
        pass.commit()
    }

    /// Runs the prompt in batches of `PREFILL_CHUNK` tokens: one command
    /// buffer per chunk, GEMM-shaped projections over all its rows, and
    /// in-kernel token loops for the sequential recurrences. Logits are
    /// produced for the final prompt token only.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prompt");
        // Power-of-two growth bounds reallocations and total zero-fill work.
        let needed = tokens.len().min(PREFILL_CHUNK);
        let have = s.prefill.as_ref().map_or(0, |p| p.m);
        if have < needed {
            let target = needed.next_power_of_two().min(PREFILL_CHUNK);
            // Drop the old buffers first so peak is max(old, new), not the sum.
            s.prefill = None;
            s.prefill = Some(PrefillScratch::new(ctx, &self.config, target)?);
        }
        let s = &*s;
        let capacity = s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill scratch missing after growth"))?;
        let mut remaining = tokens.len();
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            let ps = capacity.chunk(chunk)?;
            remaining -= chunk.len();
            self.encode_prefill_chunk(ctx, state, s, &ps, remaining == 0)?;
            state.pos += chunk.len();
            // The chunk's conv kernels wrote the other window buffer.
            state.conv_slot = 1 - state.conv_slot;
        }
        Ok(())
    }

    /// Runs one batched prefill chunk (`m = ps.m` tokens) starting at
    /// `state.pos` as a single command buffer (the MoE FFN is GPU-resident,
    /// so no mid-chunk readback splits the pass). When `want_logits`, the
    /// final row is normed and projected into `s.logits`.
    fn encode_prefill_chunk(
        &self,
        ctx: &MetalContext,
        state: &DecodeState,
        s: &Scratch,
        ps: &PrefillScratch,
        want_logits: bool,
    ) -> Result<()> {
        let pass = ctx.begin()?;
        let m = ps.m;
        ensure!(state.pos + m <= state.max_seq, "sequence full ({})", state.max_seq);
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let pos = state.pos;
        // `state` is shadowed by the GDN recurrent tensor in the match arms.
        let conv_slot = state.conv_slot;

        quant::gather_rows_q4(ctx, &pass, &self.weights.embed_tokens, &ps.ids, &ps.x)?;

        for (layer, lstate) in self.weights.layers.iter().zip(state.layers.iter()) {
            let (input_norm, post_norm, ffn) = match layer {
                LayerWeights::Gdn(w) => (&w.input_norm, &w.post_norm, &w.ffn),
                LayerWeights::Full(w) => (&w.input_norm, &w.post_norm, &w.ffn),
            };

            rmsnorm_bf16(
                ctx,
                &pass,
                &ps.x,
                input_norm,
                &ps.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            match (layer, lstate) {
                (LayerWeights::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                    project_stack_or_slices(
                        ctx,
                        &pass,
                        &ps.normed,
                        &w.in_proj,
                        &ps.stack,
                        [
                            (&w.in_proj_qkv, &ps.qkv),
                            (&w.in_proj_z, &ps.z),
                            (&w.in_proj_a, &ps.a),
                            (&w.in_proj_b, &ps.b),
                        ],
                        &s.dequant,
                    )?;
                    conv1d_prefill(
                        ctx,
                        &pass,
                        &conv_windows[conv_slot],
                        &conv_windows[1 - conv_slot],
                        &ps.qkv,
                        &w.conv_w,
                        &ps.qkv_conv,
                    )?;
                    let staging = GdnRegscanStaging {
                        qk_norm: &ps.gdn_stage.qk_norm,
                        decay: &ps.gdn_stage.decay,
                        beta: &ps.gdn_stage.beta,
                    };
                    gdn_prefill(
                        ctx,
                        &pass,
                        &ps.qkv_conv,
                        &ps.a,
                        &ps.b,
                        &w.a_log,
                        &w.dt_bias,
                        &staging,
                        state,
                        &ps.gdn_out,
                        self.gdn_scale,
                        cfg.linear_num_key_heads,
                    )?;
                    gated_rmsnorm(
                        ctx,
                        &pass,
                        &ps.gdn_out,
                        &ps.z,
                        &w.norm_w,
                        &ps.gdn_gated,
                        eps,
                    )?;
                    project_mat(
                        ctx,
                        &pass,
                        &ps.gdn_gated,
                        &w.out_proj,
                        &ps.branch_out,
                        &s.dequant,
                    )?;
                }
                (LayerWeights::Full(w), LayerState::Full { k_cache, v_cache }) => {
                    let rot = cfg.rotary_dim();
                    let theta = cfg.rope_parameters.rope_theta;
                    project_stack_or_slices(
                        ctx,
                        &pass,
                        &ps.normed,
                        &w.qkv_proj,
                        &ps.stack,
                        [
                            (&w.q_proj, &ps.qg),
                            (&w.k_proj, &ps.k_new),
                            (&w.v_proj, &ps.v_new),
                        ],
                        &s.dequant,
                    )?;
                    split_q_gate(ctx, &pass, &ps.qg, &ps.q, &ps.gate)?;
                    rmsnorm_bf16(
                        ctx,
                        &pass,
                        &ps.q,
                        &w.q_norm,
                        &ps.q,
                        eps,
                        NORM_WEIGHT_BIAS,
                    )?;
                    rmsnorm_bf16(
                        ctx,
                        &pass,
                        &ps.k_new,
                        &w.k_norm,
                        &ps.k_new,
                        eps,
                        NORM_WEIGHT_BIAS,
                    )?;
                    rope_neox(
                        ctx,
                        &pass,
                        &ps.q,
                        cfg.num_attention_heads,
                        rot,
                        pos,
                        theta,
                    )?;
                    rope_neox(
                        ctx,
                        &pass,
                        &ps.k_new,
                        cfg.num_key_value_heads,
                        rot,
                        pos,
                        theta,
                    )?;
                    scatter_kv(ctx, &pass, k_cache, &ps.k_new, pos)?;
                    scatter_kv(ctx, &pass, v_cache, &ps.v_new, pos)?;
                    sdpa_prefill(
                        ctx,
                        &pass,
                        &ps.q,
                        k_cache,
                        v_cache,
                        &ps.attn_o,
                        pos,
                        self.attn_scale,
                    )?;
                    sigmoid_mul_bf16(ctx, &pass, &ps.gate, &ps.attn_o, &ps.attn_gated)?;
                    project_mat(
                        ctx,
                        &pass,
                        &ps.attn_gated,
                        &w.o_proj,
                        &ps.branch_out,
                        &s.dequant,
                    )?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            add_bf16(ctx, &pass, &ps.x, &ps.branch_out, &ps.x)?;

            rmsnorm_bf16(
                ctx,
                &pass,
                &ps.x,
                post_norm,
                &ps.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            self.prefill_moe(ctx, &pass, ffn, s, ps)?;
            add_bf16(ctx, &pass, &ps.x, &ps.branch_out, &ps.x)?;
        }

        if want_logits {
            // Only the last prompt token feeds decoding: pull its row into the
            // single-token scratch and reuse the decode logits path.
            gather_row_bf16(ctx, &pass, &ps.x, &s.x, m - 1)?;
            rmsnorm_bf16(
                ctx,
                &pass,
                &s.x,
                &self.weights.final_norm,
                &s.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            quant::gemv_quant(ctx, &pass, &self.weights.lm_head, &s.normed, &s.logits)?;
            // Slot 0 by convention: the pipelined loop's first decode step
            // consumes the prefill token from this slot.
            let out = s.next_token.view(0, &[1])?;
            argmax_f32(ctx, &pass, &s.logits, &s.argmax_partials, &out)?;
        }
        pass.commit_wait()
    }

    /// The prefill MoE FFN, GPU-resident end to end: router GEMM → per-row
    /// softmax/top-k → counting sort (histogram, serial scans, atomic
    /// scatter) → GPU-built block map (sentinel-padded to a readback-free
    /// capacity bound) → grouped expert GEMMs → per-row combine. No host
    /// round-trip and no per-layer allocations, so the whole chunk stays in
    /// one command buffer.
    fn prefill_moe(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        moe_w: &MoeWeights,
        s: &Scratch,
        ps: &PrefillScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let ms = &ps.moe;
        let (e, top_k, inter) =
            (cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        let h = cfg.hidden_size;
        let policy = quant::moe_tile_mroute_table();
        // desync guard, extended to the route: the chunk view
        // and this function must agree on the policy's answer — a mismatch
        // would consume block maps scanned at another tile (dropping rows
        // SILENTLY) or run the GEMV path over unsorted state the grouped
        // kernels needed. Re-derive the route and, on the grouped arm, the
        // (tile, cap) pair the chunk view set together.
        let s_slots = ps.m * top_k;
        ensure!(
            ms.route == policy.route_for(ps.m),
            "prefill MoE route desync: chunk m={} carries route={} but the \
             policy routes {}",
            ps.m,
            ms.route.label(),
            policy.route_for(ps.m).label(),
        );
        if let quant::MoeRoute::Grouped(tile) = ms.route {
            ensure!(
                ms.tile == tile
                    && ms.tile_capacity
                        == s_slots.div_ceil(ms.tile.rows()) + e.min(s_slots),
                "prefill MoE tile desync: chunk m={} carries tile={} cap={} but \
                 the policy routes {}",
                ps.m,
                ms.tile.label(),
                ms.tile_capacity,
                tile.label(),
            );
        }
        project_mat(ctx, pass, &ps.normed, &moe_w.gate, &ms.router_logits, &s.dequant)?;
        moe::moe_router_topk_rows(
            ctx,
            pass,
            &ms.router_logits,
            &ms.indices,
            &ms.scores,
            cfg.norm_topk_prob,
        )?;
        // The GEMV route consumes the router output directly in natural
        // (row, k) pair order: no counting sort, no gather, no block maps.
        if matches!(ms.route, quant::MoeRoute::Grouped(_)) {
            moe::fill_zero_u32(ctx, pass, &ms.counts)?;
            moe::fill_zero_u32(ctx, pass, &ms.cursors)?;
            moe::moe_sort_slots(
                ctx,
                pass,
                &moe::MoeSortBuffers {
                    indices: &ms.indices,
                    counts: &ms.counts,
                    cursors: &ms.cursors,
                    offsets: &ms.offsets,
                    tile_offsets: &ms.tile_offsets,
                    ids_sorted: &ms.ids_sorted,
                    slot_of: &ms.slot_of,
                },
                e,
                top_k,
                ms.tile.rows(),
            )?;
            gather_rows_bf16(ctx, pass, &ps.normed, &ms.ids_sorted, &ms.gx)?;
            moe::moe_build_blocks(
                ctx,
                pass,
                &ms.offsets,
                &ms.tile_offsets,
                &ms.blocks_gu,
                e,
                inter,
                ms.tile.rows(),
            )?;
            moe::moe_build_blocks(
                ctx,
                pass,
                &ms.offsets,
                &ms.tile_offsets,
                &ms.blocks_dn,
                e,
                h,
                ms.tile.rows(),
            )?;
        }

        match ms.route {
            quant::MoeRoute::Grouped(_) => {
                let nb_gu = ms.tile_capacity * (inter / 64);
                let nb_dn = ms.tile_capacity * (h / 64);
                quant::gemm_q4_grouped_nt(
                    ctx,
                    pass,
                    &ms.gx,
                    &moe_w.expert_gate,
                    &ms.eg,
                    &ms.blocks_gu,
                    nb_gu,
                    ms.tile,
                )?;
                quant::gemm_q4_grouped_nt(
                    ctx,
                    pass,
                    &ms.gx,
                    &moe_w.expert_up,
                    &ms.eu,
                    &ms.blocks_gu,
                    nb_gu,
                    ms.tile,
                )?;
                silu_mul_bf16(ctx, pass, &ms.eg, &ms.eu, &ms.ea)?;
                quant::gemm_q4_grouped_nt(
                    ctx,
                    pass,
                    &ms.ea,
                    &moe_w.expert_down,
                    &ms.ed,
                    &ms.blocks_dn,
                    nb_dn,
                    ms.tile,
                )?;
                moe::moe_combine_rows(
                    ctx,
                    pass,
                    &ms.ed,
                    &ms.slot_of,
                    &ms.scores,
                    &ps.branch_out,
                    top_k,
                )?;
            }
            quant::MoeRoute::SmallmGemv => {
                ensure!(
                    moe_w.expert_gate.bits == 4
                        && moe_w.expert_up.bits == 4
                        && moe_w.expert_down.bits == 4,
                    "small-m route requires q4 experts"
                );
                let run_gemv = |w: &LinearWeights,
                                n_per: usize,
                                x: &Tensor,
                                y: &Tensor,
                                x_per_pair: bool|
                 -> Result<()> {
                    moe::moe_gemv_smallm_em(
                        ctx, pass, w, n_per, x, &ms.umap, y, s_slots, top_k, x_per_pair,
                    )
                };
                // Build one union map shared by the three expert projections.
                moe::moe_union_experts(ctx, pass, &ms.indices, &ms.umap)?;
                run_gemv(&moe_w.expert_gate, inter, &ps.normed, &ms.eg, false)?;
                run_gemv(&moe_w.expert_up, inter, &ps.normed, &ms.eu, false)?;
                silu_mul_bf16(ctx, pass, &ms.eg, &ms.eu, &ms.ea)?;
                run_gemv(&moe_w.expert_down, h, &ms.ea, &ms.ed, true)?;
                // GEMV outputs are in natural (row, k) pair order: the
                // combine reads the identity slot map.
                moe::moe_combine_rows(
                    ctx,
                    pass,
                    &ms.ed,
                    &ms.slots_iota,
                    &ms.scores,
                    &ps.branch_out,
                    top_k,
                )?;
            }
        }

        let shared = &moe_w.shared;
        project_stack_or_slices(
            ctx,
            pass,
            &ps.normed,
            &shared.gate_up_proj,
            &ps.stack,
            [(&shared.gate_proj, &ps.mlp_gate), (&shared.up_proj, &ps.mlp_up)],
            &s.dequant,
        )?;
        silu_mul_bf16(ctx, pass, &ps.mlp_gate, &ps.mlp_up, &ps.mlp_act)?;
        project_mat(
            ctx,
            pass,
            &ps.mlp_act,
            &shared.down_proj,
            &ms.shared_out,
            &s.dequant,
        )?;
        project_mat(
            ctx,
            pass,
            &ps.normed,
            &moe_w.shared_gate,
            &ms.shared_gate,
            &s.dequant,
        )?;
        moe::moe_row_gate_add(
            ctx,
            pass,
            &ms.shared_out,
            &ms.shared_gate,
            &ps.branch_out,
        )?;
        Ok(())
    }

    /// Encodes one concurrent-dispatch decode graph at `state.pos`. The input
    /// token comes from `next_token[slot_in]`; greedy argmax writes
    /// `next_token[slot_out]` for the next submitted pass.
    #[allow(clippy::too_many_arguments)]
    fn encode_decode_step(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
    ) -> Result<()> {
        ensure!(state.pos < state.max_seq, "sequence full ({})", state.max_seq);
        let eps = self.config.rms_norm_eps;

        let ids = s.next_token.view(slot_in, &[1])?;
        quant::gather_rows_q4(ctx, pass, &self.weights.embed_tokens, &ids, &s.x)?;
        pass.level_barrier(&[&s.x])?;

        // Every rmsnorm after layer 0's input norm is preceded by a residual
        // add in the graph, so the pairs run as one fused launch: the layer's
        // post-norm fuses with the branch add, and the trailing FFN add fuses
        // with the NEXT layer's input norm (or the final norm when logits are
        // output) — halving the norm-boundary launches per step.
        //
        // The concurrent encoder has no implicit dispatch ordering:
        // `level_barrier` marks every true inter-level data edge. Each branch
        // barriers between its own levels but leaves its final edge to the
        // caller, allowing independent dispatches inside a level to overlap.
        let layers = &self.weights.layers;
        // `state` is shadowed by the GDN recurrent tensor in the match arms.
        let conv_slot = state.conv_slot;
        for (idx, (layer, lstate)) in layers.iter().zip(state.layers.iter()).enumerate()
        {
            let (input_norm, post_norm, ffn) = match layer {
                LayerWeights::Gdn(w) => (&w.input_norm, &w.post_norm, &w.ffn),
                LayerWeights::Full(w) => (&w.input_norm, &w.post_norm, &w.ffn),
            };

            if idx == 0 {
                rmsnorm_bf16(
                    ctx,
                    pass,
                    &s.x,
                    input_norm,
                    &s.normed,
                    eps,
                    NORM_WEIGHT_BIAS,
                )?;
                pass.level_barrier(&[&s.normed])?;
            }
            match (layer, lstate) {
                (LayerWeights::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                    self.gdn_branch(ctx, pass, w, s, state, &conv_windows[conv_slot])?;
                }
                (LayerWeights::Full(w), LayerState::Full { k_cache, v_cache }) => {
                    self.attn_branch(ctx, pass, w, s, k_cache, v_cache, state.pos)?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            pass.level_barrier(&[&s.branch_out])?;
            add_rmsnorm_bf16(
                ctx,
                pass,
                &s.x,
                &s.branch_out,
                post_norm,
                &s.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            // The fused kernel writes both: `x` is the running residual and
            // `normed` is what the FFN reads.
            pass.level_barrier(&[&s.x, &s.normed])?;

            self.moe_branch(ctx, pass, ffn, s)?;
            pass.level_barrier(&[&s.branch_out])?;
            let next_norm = if idx + 1 < layers.len() {
                match &layers[idx + 1] {
                    LayerWeights::Gdn(w) => &w.input_norm,
                    LayerWeights::Full(w) => &w.input_norm,
                }
            } else {
                &self.weights.final_norm
            };
            add_rmsnorm_bf16(
                ctx,
                pass,
                &s.x,
                &s.branch_out,
                next_norm,
                &s.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            pass.level_barrier(&[&s.x, &s.normed])?;
        }

        // The last layer's fused tail already normed `x` with final_norm.
        quant::gemv_quant(ctx, pass, &self.weights.lm_head, &s.normed, &s.logits)?;
        pass.level_barrier(&[&s.logits])?;
        let out = s.next_token.view(slot_out, &[1])?;
        argmax_f32(ctx, pass, &s.logits, &s.argmax_partials, &out)?;
        pass.level_barrier(&[&s.next_token])?;
        Ok(())
    }

    fn gdn_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        gdn_state: &Tensor,
        conv_window: &Tensor,
    ) -> Result<()> {
        // One fused matvec covers qkv | z | a | b (the scratch segments are
        // views of s.gdn_in). Five dispatches in five dependency levels — the
        // GDN branch is a pure chain with no removable edge.
        quant::gemv_quant(ctx, pass, &w.in_proj, &s.normed, &s.gdn_in)?;
        pass.level_barrier(&[&s.gdn_in])?;
        conv1d_step(ctx, pass, conv_window, &s.qkv, &w.conv_w, &s.qkv_conv)?;
        pass.level_barrier(&[&s.qkv_conv, conv_window])?;
        {
            gdn_step_gated_fused(
                ctx,
                pass,
                &s.qkv_conv,
                &s.a,
                &s.b,
                &w.a_log,
                &w.dt_bias,
                gdn_state,
                &s.z,
                &w.norm_w,
                &s.gdn_gated,
                self.gdn_scale,
                self.config.linear_num_key_heads,
                self.config.rms_norm_eps,
            )?;
        }
        pass.level_barrier(&[&s.gdn_gated, gdn_state])?;
        quant::gemv_quant(ctx, pass, &w.out_proj, &s.gdn_gated, &s.branch_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        k_cache: &Tensor,
        v_cache: &Tensor,
        pos: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_parameters.rope_theta;

        // One fused matvec covers q(|gate) | k | v.
        quant::gemv_quant(ctx, pass, &w.qkv_proj, &s.normed, &s.attn_qkv)?;
        pass.level_barrier(&[&s.attn_qkv])?;
        {
            q_norm_rope_split_decode(
                ctx, pass, &s.qg, &w.q_norm, &s.q, &s.gate, rot, pos, theta, eps,
            )?;
            k_norm_rope_scatter_decode(
                ctx, pass, &s.k_new, &w.k_norm, k_cache, rot, pos, theta, eps,
            )?;
            scatter_kv(ctx, pass, v_cache, &s.v_new, pos)?;
            pass.level_barrier(&[&s.q, &s.gate, k_cache, v_cache])?;
        }
        sdpa_decode(
            ctx,
            pass,
            &s.q,
            k_cache,
            v_cache,
            &s.attn_o,
            pos + 1,
            self.attn_scale,
            Some((&s.sdpa_partials, &s.sdpa_stats)),
        )?;
        pass.level_barrier(&[&s.attn_o])?;
        sigmoid_mul_bf16(ctx, pass, &s.gate, &s.attn_o, &s.attn_gated)?;
        pass.level_barrier(&[&s.attn_gated])?;
        quant::gemv_quant(ctx, pass, &w.o_proj, &s.attn_gated, &s.branch_out)
    }

    /// The decode MoE FFN, fully on-GPU: router GEMV + softmax/top-k kernel,
    /// gather-GEMVs over the stacked experts through the device-resident
    /// indices, the shared expert through the dense-MLP scratch, and the
    /// score-weighted combine into `s.branch_out`.
    fn moe_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        moe_w: &MoeWeights,
        s: &Scratch,
    ) -> Result<()> {
        let ms = &s.moe;
        let inter = self.config.moe_intermediate_size;
        // The shared expert reads only `normed` and writes only its own scratch,
        // so its whole chain is independent of the router and of the expert
        // gathers. Encoding it interleaved with them (rather than after) is what
        // puts the two on the same dependency levels; under a serial encoder the
        // order is immaterial, since every data edge still holds.
        let shared = &moe_w.shared;

        // Level 0 — everything whose only input is `normed`.
        quant::gemv_quant(ctx, pass, &moe_w.gate, &s.normed, &ms.router_logits)?;
        quant::gemv_quant(ctx, pass, &shared.gate_up_proj, &s.normed, &s.mlp_gu)?;
        quant::gemv_quant(ctx, pass, &moe_w.shared_gate, &s.normed, &ms.shared_gate)?;
        pass.level_barrier(&[&ms.router_logits, &s.mlp_gu, &ms.shared_gate])?;

        // Level 1 — the routing decision, and the shared expert's activation.
        moe::moe_router_topk(
            ctx,
            pass,
            &ms.router_logits,
            &ms.indices,
            &ms.scores,
            self.config.norm_topk_prob,
        )?;
        silu_mul_bf16(ctx, pass, &s.mlp_gate, &s.mlp_up, &s.mlp_act)?;
        pass.level_barrier(&[&ms.indices, &ms.scores, &s.mlp_act])?;

        moe::moe_gather_gemv_gate_up(
            ctx,
            pass,
            &moe_w.expert_gate,
            &moe_w.expert_up,
            inter,
            &s.normed,
            &ms.indices,
            &ms.act,
        )?;
        quant::gemv_quant(ctx, pass, &shared.down_proj, &s.mlp_act, &ms.shared_out)?;
        pass.level_barrier(&[&ms.act, &ms.shared_out])?;

        moe::moe_gather_gemv_down_combine(
            ctx,
            pass,
            &moe_w.expert_down,
            self.config.hidden_size,
            &ms.act,
            &ms.indices,
            &ms.scores,
            Some((&ms.shared_out, &ms.shared_gate)),
            &s.branch_out,
        )
    }
}
