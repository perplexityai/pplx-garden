use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use super::*;
use crate::cpu_ref;

const GROUP_SIZE: usize = 64;

const FIXTURE: &[u8] = include_bytes!("../../goldens/quant_fixture.json");

#[derive(Deserialize)]
struct FixtureCase {
    name: String,
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: usize,
    codes: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    expected: Vec<f32>,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<FixtureCase>,
}

#[test]
fn dequant_matches_mlx_bit_exactly() {
    let fixture: Fixture = serde_json::from_slice(FIXTURE).expect("fixture");
    assert!(!fixture.cases.is_empty());
    for case in &fixture.cases {
        let got = cpu_ref::dequant_affine(
            &case.codes,
            &case.scales,
            &case.biases,
            case.rows,
            case.cols,
            case.group_size,
            case.bits,
        );
        assert_eq!(got.len(), case.expected.len(), "{}", case.name);
        for (i, (got, expected)) in got.iter().zip(&case.expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{}: bit mismatch at {i}: got {got} expected {expected}",
                case.name
            );
        }
    }
}

/// Builds a random quantized weight whose dequantization is exactly
/// representable, plus its dequantized f32 image.
fn random_quant(
    ctx: &MetalContext,
    rng: &mut StdRng,
    n: usize,
    k: usize,
) -> (QuantWeights, Vec<f32>) {
    let words = k / 8;
    let groups = k / GROUP_SIZE;
    let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(0.01f32..0.5)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(-2.0f32..0.0)).to_f32())
        .collect();
    let dequant = cpu_ref::dequant_q4(&codes, &scales, &biases, n, k, GROUP_SIZE);

    let to_bf16 =
        |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
    let w = QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, words],
            DType::U32,
        )
        .expect("codes"),
        scales: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(&scales)),
            &[n, groups],
            DType::BF16,
        )
        .expect("scales"),
        biases: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(&biases)),
            &[n, groups],
            DType::BF16,
        )
        .expect("biases"),
        group_size: GROUP_SIZE,
        bits: 4,
    };
    (w, dequant)
}

/// Fused-q4 grouped GEMM vs the dequant-then-matmul reference over a
/// multi-expert stack; expert row counts cover empty/sub-tile/exact/
/// straddle/multi-tile, so m_end clipping is exercised on every edge.
#[test]
fn gemm_q4_grouped_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(21);
    for (k, n_per) in [(64, 64), (2048, 64), (512, 128)] {
        let tile = MoeTile::T32Sg4;
        let counts = [0usize, 3, 32, 33, 72];
        let e = counts.len();
        let s: usize = counts.iter().sum();
        let a: Vec<f32> = (0..s * k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let (w, w_deq) = random_quant(&ctx, &mut rng, e * n_per, k);

        let mut blocks: Vec<u32> = Vec::new();
        let mut base = 0u32;
        for (expert, &t) in counts.iter().enumerate() {
            for m0 in (0..t as u32).step_by(tile.rows()) {
                for n0 in (0..n_per as u32).step_by(64) {
                    blocks.extend_from_slice(&[
                        base + m0,
                        (expert * n_per) as u32 + n0,
                        n0,
                        base + t as u32,
                    ]);
                }
            }
            base += t as u32;
        }

        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[s, k]).expect("a");
        let tc = Tensor::zeros(&ctx, &[s, n_per], DType::BF16).expect("c");
        let tblocks = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&blocks),
            &[blocks.len()],
            DType::U32,
        )
        .expect("blocks");

        let pass = ctx.begin().expect("pass");
        gemm_q4_grouped_nt(&ctx, &pass, &ta, &w, &tc, &tblocks, blocks.len() / 4, tile)
            .expect("fused grouped gemm");
        pass.commit_wait().expect("commit");

        let ar = cpu_ref::round_bf16(&a);
        let br = cpu_ref::round_bf16(&w_deq);
        let mut expected = vec![0f32; s * n_per];
        let mut base = 0usize;
        for (expert, &t) in counts.iter().enumerate() {
            if t == 0 {
                continue;
            }
            let sub = cpu_ref::gemm_nt(
                &ar[base * k..(base + t) * k],
                &br[expert * n_per * k..(expert + 1) * n_per * k],
                t,
                k,
                n_per,
            );
            expected[base * n_per..(base + t) * n_per].copy_from_slice(&sub);
            base += t;
        }
        cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

/// Host block map at row-tile height `t`, mirroring the production
/// router exactly: each expert emits ceil(r/t) fixed-height tiles (the
/// last m_end-clipped at the expert boundary) x n_per/64 column tiles,
/// then sentinel `(0,0,0,0)` padding up to the readback-free capacity
/// bound `ceil(S/t) + min(E, S)` the scratch allocates.
fn tiled_block_map(counts: &[usize], n_per: usize, t: usize) -> Vec<u32> {
    let s: usize = counts.iter().sum();
    let cap = s.div_ceil(t) + counts.len().min(s);
    let n_tiles = n_per / 64;
    let mut blocks: Vec<u32> = Vec::new();
    let mut base = 0u32;
    for (expert, &r) in counts.iter().enumerate() {
        for m0 in (0..r as u32).step_by(t) {
            for n0 in (0..n_per as u32).step_by(64) {
                blocks.extend_from_slice(&[
                    base + m0,
                    (expert * n_per) as u32 + n0,
                    n0,
                    base + r as u32,
                ]);
            }
        }
        base += r as u32;
    }
    blocks.resize(cap * n_tiles * 4, 0);
    blocks
}

/// Compares grouped-kernel tile variants with the CPU reference and each other.
/// Cases cover tile edges, sentinel padding, E > S, and skewed routing.
#[test]
fn gemm_q4_grouped_tiles_bit_identical_across_heights() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(29);
    let cases: [(&str, Vec<usize>, usize, usize); 4] = [
        // Ragged straddles of every tile height's edges.
        ("ragged", vec![0, 1, 15, 16, 17, 31, 63, 64, 65, 200], 128, 64),
        // K=2048 walks 32 K-tiles through the persistent f32
        // accumulator; two column tiles.
        ("deep_k", vec![0, 3, 17, 64, 65], 2048, 128),
        // More experts than routed rows (tail-chunk shape).
        ("e_gt_s", vec![0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0], 64, 64),
        // Heavy skew: one dominant expert plus dust.
        ("skewed", vec![200, 1, 0, 7, 33, 0, 2, 65], 192, 64),
    ];
    for (name, counts, k, n_per) in cases {
        let e = counts.len();
        let s: usize = counts.iter().sum();
        let a: Vec<f32> = (0..s * k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let (w, w_deq) = random_quant(&ctx, &mut rng, e * n_per, k);
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[s, k]).expect("a");

        let ar = cpu_ref::round_bf16(&a);
        let br = cpu_ref::round_bf16(&w_deq);
        let mut expected = vec![0f32; s * n_per];
        let mut base = 0usize;
        for (expert, &t) in counts.iter().enumerate() {
            if t == 0 {
                continue;
            }
            let sub = cpu_ref::gemm_nt(
                &ar[base * k..(base + t) * k],
                &br[expert * n_per * k..(expert + 1) * n_per * k],
                t,
                k,
                n_per,
            );
            expected[base * n_per..(base + t) * n_per].copy_from_slice(&sub);
            base += t;
        }

        let mut reference_bits: Option<Vec<u32>> = None;
        for tile in [MoeTile::T32Sg4, MoeTile::T64Sg4] {
            let blocks = tiled_block_map(&counts, n_per, tile.rows());
            let tblocks = Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&blocks),
                &[blocks.len()],
                DType::U32,
            )
            .expect("blocks");
            let tc = Tensor::zeros(&ctx, &[s, n_per], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_q4_grouped_nt(
                &ctx,
                &pass,
                &ta,
                &w,
                &tc,
                &tblocks,
                blocks.len() / 4,
                tile,
            )
            .expect("fused grouped gemm");
            pass.commit_wait().expect("commit");

            let got = tc.to_f32().expect("read");
            cpu_ref::assert_close(&got, &expected, 2e-2, 2e-2);
            let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            match &reference_bits {
                None => reference_bits = Some(bits),
                Some(reference) => assert_eq!(
                    reference,
                    &bits,
                    "{name}: tile {} output not bit-identical to 32x4",
                    tile.label()
                ),
            }
        }
    }
}

/// Checks all three MoE routes at their row-count boundaries.
#[test]
fn shipped_table_routes_by_chunk_m() {
    let table = moe_tile_mroute_table();
    assert_eq!(table.route_for(1), MoeRoute::SmallmGemv);
    assert_eq!(table.route_for(MOE_SMALLM_THRESHOLD), MoeRoute::SmallmGemv);
    assert_eq!(
        table.route_for(MOE_SMALLM_THRESHOLD + 1),
        MoeRoute::Grouped(MoeTile::T32Sg4)
    );
    assert_eq!(
        table.route_for(MOE_TILE_MROUTE_DEFAULT - 1),
        MoeRoute::Grouped(MoeTile::T32Sg4)
    );
    assert_eq!(
        table.route_for(MOE_TILE_MROUTE_DEFAULT),
        MoeRoute::Grouped(MoeTile::T64Sg4)
    );
}

/// Checks the inclusive small-M GEMV boundary and grouped-map allocation tile.
#[test]
fn moe_smallm_rail_routes_by_chunk_m() {
    let table = moe_tile_mroute_table();
    assert_eq!(table.route_for(1), MoeRoute::SmallmGemv);
    assert_eq!(table.route_for(8), MoeRoute::SmallmGemv);
    assert_eq!(table.route_for(9), MoeRoute::Grouped(MoeTile::T32Sg4));
    assert_eq!(table.route_for(3072), MoeRoute::Grouped(MoeTile::T64Sg4));
    assert_eq!(table.alloc_tile(), MoeTile::T32Sg4);
    assert_eq!(MoeRoute::SmallmGemv.label(), "gemv");
    assert_eq!(MoeRoute::Grouped(MoeTile::T64Sg4).label(), "64x4");
}

/// Backing-allocation bound: `cap(T) = ceil(S/T.rows()) + min(E, S)` is
/// monotone nonincreasing in T.rows(), so sizing at the policy's alloc
/// tile covers every per-chunk routed cap.
#[test]
fn alloc_tile_bounds_every_routed_cap() {
    let cap = |s: usize, t: MoeTile, e: usize| s.div_ceil(t.rows()) + e.min(s);

    let policy = moe_tile_mroute_table();
    assert_eq!(policy.alloc_tile(), MoeTile::T32Sg4);
    for e in [8, 256] {
        for m in [1, 6, 12, 128, 512, 2048, 3071, 3072, 3840, 4096] {
            let s = m * 8;
            let routed = cap(s, policy.tile_for(m), e);
            let bound = cap(s, policy.alloc_tile(), e);
            assert!(
                routed <= bound,
                "cap at routed tile exceeds the allocation bound \
                     (m={m} e={e}: {routed} > {bound})"
            );
        }
    }
}

#[test]
fn dequant_q4_matches_cpu_exactly() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(10);
    for (n, k) in [(1, 64), (3, 128), (33, 1024)] {
        let (w, expected) = random_quant(&ctx, &mut rng, n, k);
        let out = Tensor::zeros(&ctx, &[n, k], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        dequant_to_bf16(&ctx, &pass, &w, &out).expect("dequant");
        pass.commit_wait().expect("commit");
        // The kernel rounds the f32 dequant to bf16 exactly once — same
        // as rounding the reference.
        let expected_bf16: Vec<f32> =
            expected.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
        let got = out.to_f32().expect("read");
        assert_eq!(got, expected_bf16);
    }
}

#[test]
fn gemv_q8_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(14);
    for (n, k) in [(8, 128), (256, 2048)] {
        let words = k / 4;
        let groups = k / GROUP_SIZE;
        let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
        let scales: Vec<f32> = (0..n * groups)
            .map(|_| bf16::from_f32(rng.gen_range(0.005f32..0.1)).to_f32())
            .collect();
        let biases: Vec<f32> = (0..n * groups)
            .map(|_| bf16::from_f32(rng.gen_range(-1.0f32..0.0)).to_f32())
            .collect();
        let dequant =
            cpu_ref::dequant_affine(&codes, &scales, &biases, n, k, GROUP_SIZE, 8);
        let to_bf16 =
            |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
        let w = QuantWeights {
            codes: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&codes),
                &[n, words],
                DType::U32,
            )
            .expect("codes"),
            scales: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&to_bf16(&scales)),
                &[n, groups],
                DType::BF16,
            )
            .expect("scales"),
            biases: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&to_bf16(&biases)),
                &[n, groups],
                DType::BF16,
            )
            .expect("biases"),
            group_size: GROUP_SIZE,
            bits: 8,
        };
        let x: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[k]).expect("x");
        let expected = cpu_ref::gemm_nt(&cpu_ref::round_bf16(&x), &dequant, 1, k, n);
        for dtype in [DType::BF16, DType::F32] {
            let ty = Tensor::zeros(&ctx, &[n], dtype).expect("y");
            let pass = ctx.begin().expect("pass");
            gemv_quant(&ctx, &pass, &w, &tx, &ty).expect("gemv_q8");
            pass.commit_wait().expect("commit");
            cpu_ref::assert_close(&ty.to_f32().expect("read"), &expected, 2e-2, 2e-2);
        }
    }
}

#[test]
fn gemv_q4_matches_cpu_both_dtypes() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(11);
    for (n, k) in [(7, 64), (33, 1024), (1024, 3584)] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k);
        let x: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[k]).expect("x");
        let expected = cpu_ref::gemm_nt(&cpu_ref::round_bf16(&x), &dequant, 1, k, n);
        for dtype in [DType::BF16, DType::F32] {
            let ty = Tensor::zeros(&ctx, &[n], dtype).expect("y");
            let pass = ctx.begin().expect("pass");
            gemv_quant(&ctx, &pass, &w, &tx, &ty).expect("gemv_quant");
            pass.commit_wait().expect("commit");
            cpu_ref::assert_close(&ty.to_f32().expect("read"), &expected, 2e-2, 2e-2);
        }
    }
}

#[test]
fn gemm_q4_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(12);
    for (m, n, k) in [(5, 7, 64), (64, 33, 1024), (256, 512, 1024)] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k);
        let a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let scratch = Tensor::zeros(&ctx, &[n, k], DType::BF16).expect("scratch");
        let pass = ctx.begin().expect("pass");
        gemm_quant_bf16_nt(&ctx, &pass, &ta, &w, &tc, &scratch).expect("gemm_quant");
        pass.commit_wait().expect("commit");
        let dequant_bf16 = cpu_ref::round_bf16(&dequant);
        let expected =
            cpu_ref::gemm_nt(&cpu_ref::round_bf16(&a), &dequant_bf16, m, k, n);
        cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

#[test]
fn gather_kernels_match_dequant() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(13);
    let (vocab, k) = (50, 128);
    let (w, dequant) = random_quant(&ctx, &mut rng, vocab, k);
    let round = |v: &[f32]| -> Vec<f32> {
        v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
    };

    let ids: Vec<u32> = vec![3, 49, 0, 7];
    let tids =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[ids.len()], DType::U32)
            .expect("ids");
    let out = Tensor::zeros(&ctx, &[ids.len(), k], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");
    gather_rows_q4(&ctx, &pass, &w, &tids, &out).expect("gather rows");
    pass.commit_wait().expect("commit");

    let got = out.to_f32().expect("read");
    for (i, &id) in ids.iter().enumerate() {
        let expected = round(&dequant[id as usize * k..(id as usize + 1) * k]);
        assert_eq!(&got[i * k..(i + 1) * k], &expected[..], "row {id}");
    }
}
