const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/moe.metal"),
    "\n",
    include_str!("../../metal/moe_test.metal")
);

/// Includes extended-row variants from test-only MSL.
const MOE_SMALLM_TEST_MAX_M: usize = 16;

enum TestSmallmIndex<'a> {
    Pairs(&'a Tensor),
    Union(&'a Tensor, usize),
}

/// Test-only pair-major small-M grouped GEMV.
#[allow(clippy::too_many_arguments)]
fn moe_gemv_smallm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    n_per_expert: usize,
    x: &Tensor,
    indices: &Tensor,
    y: &Tensor,
    top_k: usize,
    x_per_pair: bool,
) -> Result<()> {
    moe_gemv_smallm_test_inner(
        ctx,
        pass,
        w,
        n_per_expert,
        x,
        TestSmallmIndex::Pairs(indices),
        y,
        top_k,
        x_per_pair,
    )
}

#[allow(clippy::too_many_arguments)]
fn moe_gemv_smallm_em_test(
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
    moe_gemv_smallm_test_inner(
        ctx,
        pass,
        w,
        n_per_expert,
        x,
        TestSmallmIndex::Union(umap, s),
        y,
        top_k,
        x_per_pair,
    )
}

#[allow(clippy::too_many_arguments)]
fn moe_gemv_smallm_test_inner(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    n_per_expert: usize,
    x: &Tensor,
    index: TestSmallmIndex<'_>,
    y: &Tensor,
    top_k: usize,
    x_per_pair: bool,
) -> Result<()> {
    let (rows, k_in) = (w.out_features(), w.in_features());
    ensure!(w.bits == 4, "small-m gather GEMV is 4-bit only");
    ensure!(
        rows.is_multiple_of(n_per_expert),
        "stacked rows {rows} not a multiple of per-expert rows {n_per_expert}"
    );
    ensure!(
        k_in.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(32),
        "in {k_in} / group size {} not block-packable",
        w.group_size
    );
    ensure!(top_k > 0 && top_k <= MAX_K, "top_k {top_k} out of 1..={MAX_K}");
    let (expert_major, routing, s) = match index {
        TestSmallmIndex::Pairs(indices) => {
            ensure!(indices.dtype() == DType::U32, "indices must be U32 [m, top_k]");
            (false, indices, indices.numel())
        }
        TestSmallmIndex::Union(umap, s) => {
            ensure!(
                umap.dtype() == DType::U32 && umap.numel() >= MOE_UNION_MAP_WORDS,
                "umap must be U32 [>= {MOE_UNION_MAP_WORDS}]"
            );
            (true, umap, s)
        }
    };
    ensure!(s.is_multiple_of(top_k) && s > 0, "pair count {s} not m * {top_k}");
    ensure!(
        s <= MOE_SMALLM_MAX_S,
        "pair count {s} exceeds the staged pair-list capacity"
    );
    let m = s / top_k;
    ensure!(
        m <= MOE_SMALLM_TEST_MAX_M,
        "test small-m GEMV holds at most {MOE_SMALLM_TEST_MAX_M} rows per expert"
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

    let r8 = m <= MOE_SMALLM_MAX_M;
    let rows_per_sg = if r8 { 4 } else { 2 } * if k_in <= 512 { 2 } else { 1 };
    let (name, rows_per_tg) = if n_per_expert.is_multiple_of(2 * rows_per_sg) {
        let name = match (expert_major, r8, k_in <= 512) {
            (false, true, false) => "moe_gemv_smallm_q4_r8",
            (false, false, false) => "moe_gemv_smallm_q4_r16",
            (false, true, true) => "moe_gemv_smallm_q4_r8_w",
            (false, false, true) => "moe_gemv_smallm_q4_r16_w",
            (true, true, false) => "moe_gemv_smallm_q4_em_r8",
            (true, false, false) => "moe_gemv_smallm_q4_em_r16",
            (true, true, true) => "moe_gemv_smallm_q4_em_r8_w",
            (true, false, true) => "moe_gemv_smallm_q4_em_r16_w",
        };
        (name, 2 * rows_per_sg)
    } else {
        ensure!(n_per_expert.is_multiple_of(2), "N per expert must be even");
        let name = match (expert_major, r8) {
            (false, true) => "moe_gemv_smallm_q4_r8_n2",
            (false, false) => "moe_gemv_smallm_q4_r16_n2",
            (true, true) => "moe_gemv_smallm_q4_em_r8_n2",
            (true, false) => "moe_gemv_smallm_q4_em_r16_n2",
        };
        (name, 2)
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
    let pipeline = ctx.pipeline(name, TEST_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            routing.binding(),
            y.binding(),
        ],
        &params,
        Grid::Threadgroups {
            groups: (n_per_expert / rows_per_tg, s, 1),
            threadgroup: (64, 1, 1),
        },
    )
}

use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

fn random_vec(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(lo..hi)).collect()
}

/// CPU reference: softmax over all logits, top-k by probability (ties to
/// the lowest id), optional renorm.
fn ref_router(logits: &[f32], k: usize, renorm: bool) -> (Vec<u32>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut probs: Vec<(u32, f32)> =
        exps.iter().enumerate().map(|(i, &e)| (i as u32, e / sum)).collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    probs.truncate(k);
    let denom: f32 = if renorm { probs.iter().map(|&(_, p)| p).sum() } else { 1.0 };
    (
        probs.iter().map(|&(i, _)| i).collect(),
        probs.iter().map(|&(_, p)| p / denom).collect(),
    )
}

#[test]
fn router_topk_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(30);
    for (e, k, renorm) in [
        (8, 2, true),
        (256, 8, true),
        (256, 8, false),
        (33, 4, true),
        // More -inf logits than remaining k slots: the mask sentinel must
        // not collide (a selected winner reappearing as a duplicate).
        (6, 4, true),
    ] {
        let mut logits = random_vec(&mut rng, e, -4.0, 4.0);
        if e == 6 {
            for x in logits[2..].iter_mut() {
                *x = f32::NEG_INFINITY;
            }
        }
        let t_logits = Tensor::from_f32(&ctx, &logits, &[e]).expect("logits");
        let t_idx = Tensor::zeros(&ctx, &[k], DType::U32).expect("idx");
        let t_scores = Tensor::zeros(&ctx, &[k], DType::F32).expect("scores");

        let pass = ctx.begin().expect("pass");
        moe_router_topk(&ctx, &pass, &t_logits, &t_idx, &t_scores, renorm)
            .expect("router");
        pass.commit_wait().expect("commit");

        let (exp_idx, exp_scores) = ref_router(&logits, k, renorm);
        assert_eq!(t_idx.to_u32().expect("idx"), exp_idx, "E={e} k={k}");
        cpu_ref::assert_close(
            &t_scores.to_f32().expect("scores"),
            &exp_scores,
            1e-5,
            1e-5,
        );
    }
}

/// Per-row combine of the grouped prefill path: rows draw from arbitrary
/// slots (shared and repeated across rows) with distinct scores.
#[test]
fn combine_rows_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(41);
    let (s, m, h, k) = (11, 4, 64, 3);
    let ed = random_vec(&mut rng, s * h, -1.0, 1.0);
    let slots: Vec<u32> = (0..m * k).map(|_| rng.gen_range(0..s as u32)).collect();
    let scores = random_vec(&mut rng, m * k, 0.0, 1.0);

    let t_ed = Tensor::from_f32_as_bf16(&ctx, &ed, &[s, h]).expect("ed");
    let t_slots =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&slots), &[m, k], DType::U32)
            .expect("slots");
    let t_scores = Tensor::from_f32(&ctx, &scores, &[m, k]).expect("scores");
    let t_out = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    moe_combine_rows(&ctx, &pass, &t_ed, &t_slots, &t_scores, &t_out, k)
        .expect("combine rows");
    pass.commit_wait().expect("commit");

    let red = cpu_ref::round_bf16(&ed);
    let mut expected = vec![0.0f32; m * h];
    for row in 0..m {
        for kk in 0..k {
            let slot = slots[row * k + kk] as usize;
            let sc = scores[row * k + kk];
            for c in 0..h {
                expected[row * h + c] += sc * red[slot * h + c];
            }
        }
    }
    cpu_ref::assert_close(&t_out.to_f32().expect("read"), &expected, 1e-2, 1e-2);
}

/// Builds a random 4-bit stacked expert weight (`[e * n, k]`, gs 64 when
/// `gs` is 0) plus its exact dequantized f32 image.
fn random_quant_stack(
    ctx: &MetalContext,
    rng: &mut StdRng,
    rows: usize,
    k_in: usize,
    gs: usize,
) -> (QuantWeights, Vec<f32>) {
    let words = k_in / 8;
    let groups = k_in / gs;
    let codes: Vec<u32> = (0..rows * words).map(|_| rng.r#gen()).collect();
    let scales: Vec<f32> = (0..rows * groups)
        .map(|_| bf16::from_f32(rng.gen_range(0.01f32..0.4)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..rows * groups)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0f32..0.0)).to_f32())
        .collect();
    let dequant = cpu_ref::dequant_q4(&codes, &scales, &biases, rows, k_in, gs);
    let to_bf16 =
        |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
    let w = QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[rows, words],
            DType::U32,
        )
        .expect("codes"),
        scales: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(&scales)),
            &[rows, groups],
            DType::BF16,
        )
        .expect("scales"),
        biases: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(&biases)),
            &[rows, groups],
            DType::BF16,
        )
        .expect("biases"),
        group_size: gs,
        bits: 4,
    };
    (w, dequant)
}

/// Per-row routings for the small-m GEMV edges. Every routing keeps a
/// row's top-k experts distinct (the router contract the kernel's
/// register bound relies on).
fn smallm_routing(
    rng: &mut StdRng,
    m: usize,
    top_k: usize,
    e: usize,
    mode: &str,
) -> Vec<u32> {
    assert!(top_k <= e);
    let mut indices = Vec::with_capacity(m * top_k);
    for _row in 0..m {
        match mode {
            // Union = top_k: every row routes to the same expert set, so
            // each union expert carries the full m rows (the register
            // accumulator edge) and all later pairs are duplicates.
            "all_dup" => indices.extend((0..top_k as u32).rev()),
            // Distinct random experts per row (partial shuffle).
            "random" => {
                let mut pool: Vec<u32> = (0..e as u32).collect();
                for k in 0..top_k {
                    let pick = rng.gen_range(k..e);
                    pool.swap(k, pick);
                }
                indices.extend(&pool[..top_k]);
            }
            other => panic!("unknown routing mode {other}"),
        }
    }
    indices
}

/// Checks small-M grouped GEMV across expert, row, K, group-size, and routing
/// boundaries using both per-token and per-pair activations.
#[test]
fn gemv_smallm_q4_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(61);
    // (E, n_per, K, m, top_k, x_per_pair, gs, routing)
    type Case = (usize, usize, usize, usize, usize, bool, usize, &'static str);
    let cases: &[Case] = &[
        // One-row walk (K > 512), r8 bucket, random skew.
        (8, 8, 576, 5, 2, false, 64, "random"),
        // All rows on one expert set: union = top_k, m rows per expert.
        (4, 8, 576, 8, 2, false, 64, "all_dup"),
        // Two-row walk (K <= 512), single expert, top_k = 1.
        (2, 64, 64, 5, 1, false, 64, "all_dup"),
        // r16 bucket at the register edge: m = 16 rows on every expert.
        (4, 16, 128, 16, 4, false, 64, "all_dup"),
        // r16 bucket, random routing, E > union.
        (16, 16, 128, 9, 8, false, 64, "random"),
        // r16 bucket on the K > 512 one-x-chunk-per-row-pair walk.
        (8, 8, 576, 12, 4, false, 64, "random"),
        // Down-projection shape: per-pair activations.
        (8, 16, 64, 4, 3, true, 64, "random"),
        // E == S with a wide union; N below the tile widths.
        (40, 4, 64, 5, 8, false, 64, "random"),
        // Odd N-per (one-row walk despite K <= 512: N % 4 != 0), gs 32.
        (3, 6, 96, 9, 2, false, 32, "random"),
        // Compact gate/up and down boundary shapes.
        (8, 64, 256, 6, 2, false, 64, "random"),
        (8, 256, 64, 6, 2, true, 64, "random"),
        // Smallest K the block walk supports.
        (4, 8, 32, 2, 2, false, 32, "random"),
    ];
    for &(e, n, k_in, m, top_k, x_per_pair, gs, mode) in cases {
        let s = m * top_k;
        let (w, dequant) = random_quant_stack(&ctx, &mut rng, e * n, k_in, gs);
        let indices = smallm_routing(&mut rng, m, top_k, e, mode);
        let x_rows = if x_per_pair { s } else { m };
        let x = random_vec(&mut rng, x_rows * k_in, -1.0, 1.0);

        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[x_rows, k_in]).expect("x");
        let tidx = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&indices),
            &[m, top_k],
            DType::U32,
        )
        .expect("idx");
        let ty = Tensor::zeros(&ctx, &[s, n], DType::BF16).expect("y");
        let ty_em = Tensor::zeros(&ctx, &[s, n], DType::BF16).expect("y em");
        let production_eligible =
            m <= MOE_SMALLM_MAX_M && n.is_multiple_of(if k_in <= 512 { 16 } else { 8 });
        let ty_prod = production_eligible
            .then(|| Tensor::zeros(&ctx, &[s, n], DType::BF16).expect("y prod"));
        let umap =
            Tensor::zeros(&ctx, &[MOE_UNION_MAP_WORDS], DType::U32).expect("umap");

        let pass = ctx.begin().expect("pass");
        moe_gemv_smallm(&ctx, &pass, &w, n, &tx, &tidx, &ty, top_k, x_per_pair)
            .expect("smallm gemv");
        // The expert-major twin over the same routing must be
        // bit-identical: same per-row walk, grouping read from the map.
        moe_union_experts(&ctx, &pass, &tidx, &umap).expect("union map");
        moe_gemv_smallm_em_test(
            &ctx, &pass, &w, n, &tx, &umap, &ty_em, s, top_k, x_per_pair,
        )
        .expect("smallm gemv em");
        if let Some(ty_prod) = &ty_prod {
            moe_gemv_smallm_em(
                &ctx, &pass, &w, n, &tx, &umap, ty_prod, s, top_k, x_per_pair,
            )
            .expect("production smallm gemv em");
        }
        pass.commit_wait().expect("commit");

        let rx = cpu_ref::round_bf16(&x);
        let got = ty.to_f32().expect("read");
        let got_em = ty_em.to_f32().expect("read em");
        for (i, (a, b)) in got.iter().zip(&got_em).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "expert-major diverged from pair-major at {i} ({mode}, \
                     K={k_in}, n={n})"
            );
        }
        if let Some(ty_prod) = &ty_prod {
            let got_prod = ty_prod.to_f32().expect("read prod");
            for (i, (a, b)) in got_em.iter().zip(&got_prod).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "production expert-major diverged at {i} ({mode}, K={k_in}, n={n})"
                );
            }
        }
        for (pair, &ei) in indices.iter().enumerate() {
            let x_row = if x_per_pair { pair } else { pair / top_k };
            let w_rows =
                &dequant[(ei as usize) * n * k_in..(ei as usize + 1) * n * k_in];
            let expected = cpu_ref::gemm_nt(
                &rx[x_row * k_in..(x_row + 1) * k_in],
                w_rows,
                1,
                k_in,
                n,
            );
            cpu_ref::assert_close(
                &got[pair * n..(pair + 1) * n],
                &expected,
                2e-2,
                2e-2,
            );
        }
    }
}

/// The register bound is a hard host-side error, not a truncation: m
/// above [`MOE_SMALLM_MAX_M`] must be rejected before any dispatch.
#[test]
fn gemv_smallm_rejects_m_above_register_bound() {
    let ctx = MetalContext::new().expect("metal context");
    let (e, n, k_in, top_k) = (4usize, 8usize, 64usize, 2usize);
    let m = MOE_SMALLM_MAX_M + 1;
    let w = QuantWeights {
        codes: Tensor::zeros(&ctx, &[e * n, k_in / 8], DType::U32).expect("codes"),
        scales: Tensor::zeros(&ctx, &[e * n, k_in / 64], DType::BF16).expect("scales"),
        biases: Tensor::zeros(&ctx, &[e * n, k_in / 64], DType::BF16).expect("biases"),
        group_size: 64,
        bits: 4,
    };
    let tx = Tensor::zeros(&ctx, &[m, k_in], DType::BF16).expect("x");
    let tidx = Tensor::zeros(&ctx, &[m, top_k], DType::U32).expect("idx");
    let ty = Tensor::zeros(&ctx, &[m * top_k, n], DType::BF16).expect("y");
    let umap = Tensor::zeros(&ctx, &[MOE_UNION_MAP_WORDS], DType::U32).expect("umap");
    let pass = ctx.begin().expect("pass");
    let err = moe_gemv_smallm_em(
        &ctx,
        &pass,
        &w,
        n,
        &tx,
        &umap,
        &ty,
        tidx.numel(),
        top_k,
        false,
    )
    .expect_err("m above the production register bound must be rejected");
    assert!(err.to_string().contains("registers"), "unexpected error: {err:#}");
}

/// Checks the GPU routing pipeline against a host reference across skew, ties,
/// empty experts, E > S, and every grouped-GEMM tile height.
#[test]
fn router_pipeline_matches_host() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(52);
    for (tile_m, (m, e, k, mode)) in [16usize, 32, 64]
        .into_iter()
        .flat_map(|t| {
            [
                (12usize, 8usize, 2usize, "random"),
                (7, 256, 8, "random"),
                (33, 8, 2, "all_to_one"),
                (5, 64, 4, "ties"),
                (3, 16, 8, "e_gt_s"),
            ]
            .into_iter()
            .map(move |case| (t, case))
        })
        .collect::<Vec<_>>()
    {
        let s_total = m * k;
        let mut logits = random_vec(&mut rng, m * e, -4.0, 4.0);
        match mode {
            "all_to_one" => {
                for row in 0..m {
                    logits[row * e + 3] = 20.0;
                }
            }
            "ties" => {
                // Whole rows of one constant: every expert ties, so
                // selection must be ids 0..k in order on both sides.
                for row in 0..m {
                    for x in &mut logits[row * e..(row + 1) * e] {
                        *x = 0.25;
                    }
                }
            }
            _ => {}
        }
        let t_logits =
            Tensor::from_f32_as_bf16(&ctx, &logits, &[m, e]).expect("logits");
        let t_idx = Tensor::zeros(&ctx, &[m, k], DType::U32).expect("idx");
        let t_scores = Tensor::zeros(&ctx, &[m, k], DType::F32).expect("scores");
        let t_counts = Tensor::zeros(&ctx, &[e], DType::U32).expect("counts");
        let t_cursors = Tensor::zeros(&ctx, &[e], DType::U32).expect("cursors");
        let t_offsets = Tensor::zeros(&ctx, &[e + 1], DType::U32).expect("offsets");
        let t_toffsets =
            Tensor::zeros(&ctx, &[e + 1], DType::U32).expect("tile offsets");
        let t_sorted = Tensor::zeros(&ctx, &[s_total], DType::U32).expect("sorted");
        let t_slot_of = Tensor::zeros(&ctx, &[m, k], DType::U32).expect("slot_of");
        let cap = s_total.div_ceil(tile_m) + e.min(s_total);
        let n_per = 128usize;
        let t_blocks =
            Tensor::zeros(&ctx, &[cap * (n_per / 64) * 4], DType::U32).expect("bl");

        let pass = ctx.begin().expect("pass");
        moe_router_topk_rows(&ctx, &pass, &t_logits, &t_idx, &t_scores, true)
            .expect("topk rows");
        fill_zero_u32(&ctx, &pass, &t_counts).expect("zero counts");
        fill_zero_u32(&ctx, &pass, &t_cursors).expect("zero cursors");
        moe_sort_slots(
            &ctx,
            &pass,
            &MoeSortBuffers {
                indices: &t_idx,
                counts: &t_counts,
                cursors: &t_cursors,
                offsets: &t_offsets,
                tile_offsets: &t_toffsets,
                ids_sorted: &t_sorted,
                slot_of: &t_slot_of,
            },
            e,
            k,
            tile_m,
        )
        .expect("sort");
        moe_build_blocks(
            &ctx,
            &pass,
            &t_offsets,
            &t_toffsets,
            &t_blocks,
            e,
            n_per,
            tile_m,
        )
        .expect("blocks");
        pass.commit_wait().expect("commit");

        let indices = t_idx.to_u32().expect("idx read");
        let scores = t_scores.to_f32().expect("scores read");
        let rounded = cpu_ref::round_bf16(&logits);
        for row in 0..m {
            let (ref_idx, ref_scores) =
                ref_router(&rounded[row * e..(row + 1) * e], k, true);
            assert_eq!(
                indices[row * k..(row + 1) * k],
                ref_idx[..],
                "row {row} indices diverge ({mode})"
            );
            for j in 0..k {
                assert!(
                    (scores[row * k + j] - ref_scores[j]).abs() < 1e-5,
                    "row {row} score {j} ({mode})"
                );
            }
        }

        // Counting sort invariants: offsets are the exclusive prefix of
        // the per-expert selection counts; slot_of is a bijection into
        // [0, S) landing inside the owning expert's range.
        let offsets = t_offsets.to_u32().expect("offsets read");
        let toffsets = t_toffsets.to_u32().expect("toffsets read");
        let sorted = t_sorted.to_u32().expect("sorted read");
        let slot_of = t_slot_of.to_u32().expect("slot_of read");
        let mut ref_counts = vec![0u32; e];
        for &ei in &indices {
            ref_counts[ei as usize] += 1;
        }
        let mut acc = 0u32;
        let mut tacc = 0u32;
        for ei in 0..e {
            assert_eq!(offsets[ei], acc, "offset {ei} ({mode}, T={tile_m})");
            assert_eq!(toffsets[ei], tacc, "tile offset {ei} ({mode}, T={tile_m})");
            acc += ref_counts[ei];
            tacc += ref_counts[ei].div_ceil(tile_m as u32);
        }
        assert_eq!(offsets[e], acc);
        assert_eq!(toffsets[e], tacc);
        let mut seen = vec![false; s_total];
        for row in 0..m {
            for j in 0..k {
                let slot = slot_of[row * k + j] as usize;
                assert!(!seen[slot], "slot {slot} reused ({mode})");
                seen[slot] = true;
                let ei = indices[row * k + j] as usize;
                assert!(
                    (offsets[ei]..offsets[ei + 1]).contains(&(slot as u32)),
                    "slot {slot} outside expert {ei} range ({mode})"
                );
                assert_eq!(sorted[slot], row as u32, "ids_sorted[{slot}] ({mode})");
            }
        }

        // Block map: each expert's tiles cover its rows exactly at height
        // `tile_m`, with m_end at the expert boundary; slots past the
        // real tile count hold the sentinel.
        let blocks = t_blocks.to_u32().expect("blocks read");
        let n_tiles = n_per / 64;
        let total_tiles = toffsets[e] as usize;
        for t_i in 0..cap {
            for n_i in 0..n_tiles {
                let b = &blocks[(t_i * n_tiles + n_i) * 4..][..4];
                if t_i >= total_tiles {
                    assert_eq!(
                        b,
                        [0, 0, 0, 0],
                        "sentinel expected ({mode}, T={tile_m})"
                    );
                    continue;
                }
                let ei = (0..e)
                    .find(|&x| {
                        toffsets[x] <= t_i as u32 && (t_i as u32) < toffsets[x + 1]
                    })
                    .expect("tile owner");
                let tile_in_e = t_i as u32 - toffsets[ei];
                assert_eq!(
                    b,
                    [
                        offsets[ei] + tile_in_e * tile_m as u32,
                        (ei * n_per) as u32 + (n_i as u32) * 64,
                        (n_i as u32) * 64,
                        offsets[ei + 1],
                    ],
                    "block ({t_i},{n_i}) ({mode}, T={tile_m})"
                );
            }
        }
    }
}
