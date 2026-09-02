const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/skinny.metal"),
    "\n",
    include_str!("../../metal/skinny_test.metal")
);

/// Test-only explicit staged variant at K-chunk `kc`.
fn gemm_skinny_q4_nt_staged(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    kc: usize,
) -> Result<()> {
    let (m, k, n) = validate_q4(a, w, c)?;
    if kc == 256 {
        return dispatch_q4_staged(ctx, pass, a, w, c, (m, k, n));
    }
    ensure!(m <= 8, "K-chunk {kc} is instantiated only for m <= 8 (m = {})", m);
    let fn_name = match kc {
        512 => "gemm_skinny_q4_bf16_m8_kc512",
        1024 => "gemm_skinny_q4_bf16_m8_kc1024",
        _ => anyhow::bail!("no test skinny q4 instantiation for K-chunk {kc}"),
    };
    let pipeline = ctx.pipeline(fn_name, TEST_SOURCE, MslVersion::V3_1)?;
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

/// Test-only register-A variant with an explicit rows-per-threadgroup width.
fn gemm_skinny_q4_nt_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    rows_per_tg: usize,
) -> Result<()> {
    let (m, k, n) = validate_q4(a, w, c)?;
    ensure!(m <= REG_MAX_M, "register-A kernels cover m <= 8 (m = {m})");
    ensure!(
        k.is_multiple_of(32) && w.group_size.is_multiple_of(32),
        "register-A q4 needs K % 32 == 0 and group size % 32 == 0 (k = {} gs = {})",
        k,
        w.group_size
    );
    ensure!((1..=32).contains(&rows_per_tg), "rows_per_tg {rows_per_tg} out of 1..=32");
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
        reg_grid(n, rows_per_tg),
    )
}

use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

const GROUP_SIZE: usize = 64;

/// M values covering register-A and both sides of the staged-A boundary.
const M_SWEEP: [usize; 6] = [1, 2, 5, 8, 9, 16];

/// Tolerance against an f32 reference over BF16-rounded operands.
const ATOL: f32 = 2e-2;
const RTOL: f32 = 2e-2;

fn random_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

/// A random 4-bit affine weight plus its exact f32 dequant image (the
/// quant.rs test-fixture recipe).
fn random_quant(
    ctx: &MetalContext,
    rng: &mut StdRng,
    n: usize,
    k: usize,
    gs: usize,
) -> (QuantWeights, Vec<f32>) {
    let words = k / 8;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(0.01f32..0.5)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(-2.0f32..0.0)).to_f32())
        .collect();
    let dequant = cpu_ref::dequant_q4(&codes, &scales, &biases, n, k, gs);
    (quant_tensors(ctx, codes, &scales, &biases, n, k, gs), dequant)
}

/// Uploads CPU-side q4 arrays as a [`QuantWeights`] (scales/biases are
/// already bf16-rounded f32).
fn quant_tensors(
    ctx: &MetalContext,
    codes: Vec<u32>,
    scales: &[f32],
    biases: &[f32],
    n: usize,
    k: usize,
    gs: usize,
) -> QuantWeights {
    let to_bf16 =
        |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, k / 8],
            DType::U32,
        )
        .expect("codes"),
        scales: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(scales)),
            &[n, k / gs],
            DType::BF16,
        )
        .expect("scales"),
        biases: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(biases)),
            &[n, k / gs],
            DType::BF16,
        )
        .expect("biases"),
        group_size: gs,
        bits: 4,
    }
}

/// Deterministic Q4 weights for tests that dequantize sampled rows on demand.
fn hash_quant_raw(
    ctx: &MetalContext,
    n: usize,
    k: usize,
    gs: usize,
) -> (QuantWeights, Vec<u32>, Vec<f32>, Vec<f32>) {
    let words = k / 8;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words)
        .map(|i| {
            let h1 = (i as u32).wrapping_mul(2654435761).wrapping_add(97);
            let h2 = (i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(0xA5A5_A5A5);
            (h1 & 0xFFFF_0000) | (h2 >> 16)
        })
        .collect();
    let unit = |i: usize, salt: u32| -> f32 {
        let h = (i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(salt);
        (h >> 8) as f32 / (1u32 << 24) as f32
    };
    let scales: Vec<f32> = (0..n * groups)
        .map(|i| bf16::from_f32(0.01 + 0.49 * unit(i, 5)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|i| bf16::from_f32(-2.0 + 2.0 * unit(i, 9)).to_f32())
        .collect();
    let w = quant_tensors(ctx, codes.clone(), &scales, &biases, n, k, gs);
    (w, codes, scales, biases)
}

/// Checks register/staged routing, N tails, and K-chunk tails against f32.
#[test]
fn gemm_skinny_q4_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(41);
    for (n, k) in [
        (16usize, 1024usize),
        (512, 2048),
        (2048, 512),
        (33, 320),
        (256, 64),
        (4, 256),
        (64, 256),
        (1024, 256),
        (256, 512),
        (1, 256),
        (2, 64),
    ] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in M_SWEEP {
            if m * n * k > 150_000_000 {
                continue;
            }
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
            pass.commit_wait().expect("commit");
            // The reference uses the kernel's BF16-rounded dequantized weights.
            let expected = cpu_ref::gemm_nt(
                &cpu_ref::round_bf16(&a),
                &cpu_ref::round_bf16(&dequant),
                m,
                k,
                n,
            );
            cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, ATOL, RTOL);
        }
    }
}

/// The 16-byte A staging path needs a 16B-aligned base; a bf16 view at an
/// odd 4-byte offset must take the scalar staging path and still agree.
#[test]
fn gemm_skinny_q4_unaligned_a_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(43);
    let (n, k, m) = (48usize, 320usize, 5usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let mut padded = vec![0.0f32; 2];
    padded.extend_from_slice(&a);
    let backing =
        Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + 2]).expect("backing");
    // Byte offset 4: valid for Metal (4B) but not 16B-aligned.
    let ta = backing.view(2, &[m, k]).expect("view");
    let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
    pass.commit_wait().expect("commit");
    let expected = cpu_ref::gemm_nt(
        &cpu_ref::round_bf16(&a),
        &cpu_ref::round_bf16(&dequant),
        m,
        k,
        n,
    );
    cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, ATOL, RTOL);
}

/// Samples boundary rows and a prime-stride walk across wide outputs.
fn sample_rows(n: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..16).chain(n - 16..n).collect();
    rows.extend((0..n).step_by(9973));
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Checks the wide-output register range, final-row tail, and staged fallback.
/// The f32 reference evaluates sampled rows at dispatch boundaries.
#[test]
fn gemm_skinny_q4_vocab_shape_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(44);
    let (k, gs) = (2048usize, 64usize);
    let (words, groups) = (k / 8, k / gs);
    for n in [248320usize, 248317] {
        let (w, codes, scales, biases) = hash_quant_raw(&ctx, n, k, gs);
        let rows = sample_rows(n);
        for m in 1..=9usize {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
            pass.commit_wait().expect("commit");
            let got_full = tc.to_f32().expect("read");
            let a_r = cpu_ref::round_bf16(&a);
            let mut got = Vec::new();
            let mut want = Vec::new();
            for &r in &rows {
                let deq = cpu_ref::dequant_q4(
                    &codes[r * words..(r + 1) * words],
                    &scales[r * groups..(r + 1) * groups],
                    &biases[r * groups..(r + 1) * groups],
                    1,
                    k,
                    gs,
                );
                let w_r = cpu_ref::round_bf16(&deq);
                for i in 0..m {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a_r[i * k + kk] * w_r[kk];
                    }
                    want.push(sum);
                    got.push(got_full[i * n + r]);
                }
            }
            cpu_ref::assert_close(&got, &want, ATOL, RTOL);
        }
    }
}

/// Wide N used to exercise register-A routing.
const WIDE_N: usize = 65536;

#[test]
fn reg_route_keeps_layer_widths_staged() {
    assert!(!reg_routes(1, WIDE_N - 1, true));
    assert!(reg_routes(1, WIDE_N, true));
}

#[test]
fn fused_stack_requires_one_route_for_stack_and_slices() {
    assert!(stack_route_uniform(1, WIDE_N - 1, &[1024, 2048], true));
    assert!(!stack_route_uniform(1, WIDE_N, &[WIDE_N - 1, 1], true));
    assert!(stack_route_uniform(REG_MAX_M + 1, WIDE_N, &[WIDE_N - 1, 1], true));
    assert!(stack_route_uniform(1, WIDE_N, &[WIDE_N], false));
}

/// The route boundary itself: the register body serves m up to its
/// instantiation ceiling and nothing past it.
#[test]
fn gemm_skinny_q4_wide_route_boundary_matches_reference() {
    assert!(reg_routes(1, WIDE_N, true));
    assert!(reg_routes(REG_MAX_M, WIDE_N, true));
    assert!(
        !reg_routes(REG_MAX_M + 1, WIDE_N, true),
        "m past the register instantiations must stage"
    );
    assert!(
        !reg_routes(REG_MAX_M, WIDE_N, false),
        "a failed block-walk precondition must stage"
    );

    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(45);
    let k = 256usize;
    let check = |w: &QuantWeights, dequant: &[f32], n: usize, m: usize| {
        let a = random_vec(&mut StdRng::seed_from_u64((n + m) as u64), m * k);
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemm_skinny_q4_nt(&ctx, &pass, &ta, w, &tc).expect("skinny q4");
        pass.commit_wait().expect("commit");
        let expected = cpu_ref::gemm_nt(
            &cpu_ref::round_bf16(&a),
            &cpu_ref::round_bf16(dequant),
            m,
            k,
            n,
        );
        cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, ATOL, RTOL);
    };
    for (n, ms) in [(WIDE_N - 4, &[8usize][..]), (WIDE_N, &[1usize, 8, 9][..])] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for &m in ms {
            check(&w, &dequant, n, m);
        }
    }
}

/// The register kernel's scalar-A fallback: a bf16 view at an odd 4-byte
/// offset is not 16B-aligned, so the wide route's per-lane A loads must
/// take the element path and still agree.
#[test]
fn gemm_skinny_q4_wide_unaligned_a_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(47);
    let (n, k, m) = (WIDE_N, 256usize, 3usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let mut padded = vec![0.0f32; 2];
    padded.extend_from_slice(&a);
    let backing =
        Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + 2]).expect("backing");
    let ta = backing.view(2, &[m, k]).expect("view");
    let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
    pass.commit_wait().expect("commit");
    let expected = cpu_ref::gemm_nt(
        &cpu_ref::round_bf16(&a),
        &cpu_ref::round_bf16(&dequant),
        m,
        k,
        n,
    );
    cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, ATOL, RTOL);
}

/// Checks explicit staged K-chunk and register threadgroup-width variants.
#[test]
fn gemm_skinny_q4_variants_match_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(48);
    for (n, k) in [(1024usize, 1280usize), (64, 320)] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in [1usize, 5, 8] {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let expected = cpu_ref::gemm_nt(
                &cpu_ref::round_bf16(&a),
                &cpu_ref::round_bf16(&dequant),
                m,
                k,
                n,
            );
            for kc in [512usize, 1024] {
                let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
                let pass = ctx.begin().expect("pass");
                gemm_skinny_q4_nt_staged(&ctx, &pass, &ta, &w, &tc, kc)
                    .expect("skinny q4 kc");
                pass.commit_wait().expect("commit");
                cpu_ref::assert_close(
                    &tc.to_f32().expect("read"),
                    &expected,
                    ATOL,
                    RTOL,
                );
            }
        }
    }
    let (n, k, m) = (WIDE_N, 256usize, 5usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
    let expected = cpu_ref::gemm_nt(
        &cpu_ref::round_bf16(&a),
        &cpu_ref::round_bf16(&dequant),
        m,
        k,
        n,
    );
    for rows in [2usize, 4, 8] {
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemm_skinny_q4_nt_reg(&ctx, &pass, &ta, &w, &tc, rows).expect("skinny q4 reg");
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, ATOL, RTOL);
    }
}

/// Creates a row-range view of a Q4 weight.
fn quant_view_rows(q: &QuantWeights, start: usize, rows: usize) -> QuantWeights {
    let words = q.codes.shape()[1];
    let groups = q.scales.shape()[1];
    QuantWeights {
        codes: q.codes.view(start * words, &[rows, words]).expect("codes view"),
        scales: q.scales.view(start * groups, &[rows, groups]).expect("scales"),
        biases: q.biases.view(start * groups, &[rows, groups]).expect("biases"),
        group_size: q.group_size,
        bits: q.bits,
    }
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: bit mismatch at {i} ({g} vs {w})"
        );
    }
}

/// Fused-stack and per-slice dispatches must be bit-identical, including
/// column splitting. Uneven widths and K=320 cover row and K-chunk tails.
#[test]
fn gemm_skinny_fused_stack_matches_per_slice_bits() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(49);
    let k = 320usize;
    // Starts must be even: gs = 64 gives an odd group count (5), and a
    // bf16 scale view at an odd row start would not be 4-byte aligned.
    let widths = [40usize, 24, 4, 4];
    let n_total: usize = widths.iter().sum();
    let (w, _) = random_quant(&ctx, &mut rng, n_total, k, GROUP_SIZE);

    for m in [1usize, 5, 8, 16] {
        let a_vals = random_vec(&mut rng, m * k);
        let a = Tensor::from_f32_as_bf16(&ctx, &a_vals, &[m, k]).expect("a");
        // The register kernels are instantiated for m <= 8 only.
        for use_reg in [false, true] {
            if use_reg && m > REG_MAX_M {
                continue;
            }
            let name = if use_reg { "reg2" } else { "routed" };
            let run = |wq: &QuantWeights, n: usize| -> Vec<f32> {
                let c = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
                let pass = ctx.begin().expect("pass");
                if use_reg {
                    gemm_skinny_q4_nt_reg(&ctx, &pass, &a, wq, &c, 2)
                        .expect("reg dispatch");
                } else {
                    gemm_skinny_q4_nt(&ctx, &pass, &a, wq, &c).expect("dispatch");
                }
                pass.commit_wait().expect("commit");
                c.to_f32().expect("read")
            };
            let fused = run(&w, n_total);
            let mut start = 0usize;
            let mut slice_outs: Vec<Vec<f32>> = Vec::new();
            for &rows in &widths {
                let ws = quant_view_rows(&w, start, rows);
                slice_outs.push(run(&ws, rows));
                start += rows;
            }
            let mut off = 0usize;
            for (si, slice) in slice_outs.iter().enumerate() {
                let mut seg = Vec::with_capacity(m * widths[si]);
                for i in 0..m {
                    for j in 0..widths[si] {
                        seg.push(fused[i * n_total + off + j]);
                    }
                }
                assert_bits_eq(
                    &seg,
                    slice,
                    &format!("{name} fused vs slice {si} at m={m}"),
                );
                off += widths[si];
            }
            // The production epilogue: split the fused output back into
            // the contiguous per-projection tensors, bit-exact.
            let fused_t =
                Tensor::from_f32_as_bf16(&ctx, &fused, &[m, n_total]).expect("c");
            let dsts: Vec<Tensor> = widths
                .iter()
                .map(|&rows| Tensor::zeros(&ctx, &[m, rows], DType::BF16))
                .collect::<Result<_>>()
                .expect("dsts");
            let pass = ctx.begin().expect("pass");
            crate::kernels::elementwise::split_cols_bf16(
                &ctx,
                &pass,
                &fused_t,
                &dsts.iter().collect::<Vec<_>>(),
            )
            .expect("split");
            pass.commit_wait().expect("commit");
            for (si, (d, slice)) in dsts.iter().zip(&slice_outs).enumerate() {
                assert_bits_eq(
                    &d.to_f32().expect("read"),
                    slice,
                    &format!("{name} split segment {si} at m={m}"),
                );
            }
        }
    }
}

/// Checks explicit register-A kernels across regular and boundary shapes.
#[test]
fn gemm_skinny_reg_layer_shapes_match_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(51);
    for (n, k) in [
        (16usize, 1024usize),
        (512, 2048),
        (2048, 512),
        (256, 64),
        (4, 256),
        (1, 256),
        (2, 64),
        (64, 320),
    ] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in 1..=REG_MAX_M {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let expected = cpu_ref::gemm_nt(
                &cpu_ref::round_bf16(&a),
                &cpu_ref::round_bf16(&dequant),
                m,
                k,
                n,
            );
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt_reg(&ctx, &pass, &ta, &w, &tc, 2).expect("q4 reg");
            pass.commit_wait().expect("commit");
            cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, ATOL, RTOL);
        }
    }
}

/// The fixed routing predicate includes its threshold.
#[test]
fn dense_smallm_routing_boundary() {
    assert!(dense_smallm_routes(1));
    assert!(dense_smallm_routes(DENSE_SMALLM_THRESHOLD));
    assert!(!dense_smallm_routes(DENSE_SMALLM_THRESHOLD + 1));
    assert!(!dense_smallm_routes(0));
}

#[test]
fn skinny_rejects_partial_group_tail() {
    let ctx = MetalContext::new().expect("metal context");
    let (m, n, k) = (1usize, 4usize, 96usize);
    let w = quant_tensors(&ctx, vec![0; n * k / 8], &[1.0; 4], &[0.0; 4], n, k, 64);
    let a = Tensor::zeros(&ctx, &[m, k], DType::BF16).expect("a");
    let c = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    let err = gemm_skinny_q4_nt(&ctx, &pass, &a, &w, &c).expect_err("K=96 must reject");
    assert!(err.to_string().contains("K % 64 == 0"), "{err:#}");
}
