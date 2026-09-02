use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;
use crate::tensor::DType;

#[test]
fn split_scratch_tracks_surviving_routes() {
    assert_eq!(sdpa_split_scratch_splits(0), 1);
    assert_eq!(sdpa_split_scratch_splits(1), 1);
    assert_eq!(sdpa_split_scratch_splits(257), 2);
    assert_eq!(sdpa_split_scratch_splits(8192), 32);
    assert_eq!(sdpa_split_scratch_splits(32767), 128);
    assert_eq!(sdpa_split_scratch_splits(32768), 128);
    assert_eq!(sdpa_split_scratch_splits(MAX_SEQ), 128);
}

#[test]
fn production_split_route_boundaries_are_pinned() {
    let route = |len| {
        sdpa_split_route(
            len,
            256,
            8,
            16,
            2,
            GQA_FOLD_MIN_CONTEXT,
            SDPA_FIXED_BLOCK_CROSSOVER,
        )
    };
    assert_eq!(route(8191), SdpaSplitRoute::PerHead);
    assert_eq!(route(8192), SdpaSplitRoute::Gqa);
    assert_eq!(route(32767), SdpaSplitRoute::Gqa);
    assert_eq!(route(32768), SdpaSplitRoute::FixedBlock);
}

#[test]
fn sdpa_decode_matches_cpu() {
    // Mono path (len below one split chunk) and the split-K path with a
    // ragged tail chunk.
    check_sdpa_decode(
        8,
        2,
        64,
        128,
        33,
        None,
        GQA_FOLD_MIN_CONTEXT,
        SDPA_FIXED_BLOCK_CROSSOVER,
    );
    check_sdpa_decode(
        8,
        2,
        64,
        1024,
        700,
        Some(3),
        GQA_FOLD_MIN_CONTEXT,
        SDPA_FIXED_BLOCK_CROSSOVER,
    );
    check_sdpa_decode(
        4,
        2,
        32,
        600,
        512,
        Some(2),
        GQA_FOLD_MIN_CONTEXT,
        SDPA_FIXED_BLOCK_CROSSOVER,
    );
    // Force the folded GQA route at a compact test shape: catches drift in
    // the grouped kernel's distinct buffer-11 chunk ABI without making the
    // GPU-heavy unit suite allocate and scan an 8192-row cache in parallel.
    check_sdpa_decode(8, 1, 256, 600, 512, Some(2), 0, SDPA_FIXED_BLOCK_CROSSOVER);
    // Force the fixed-block route at the same compact shape. Its 128
    // strided blocks cover ragged tails as well as the production 32K+
    // regime without making this CPU-oracle test scan a long cache.
    check_sdpa_decode(8, 1, 256, 600, 513, Some(SDPA_MLX_BLOCKS), usize::MAX, 0);
}

#[allow(clippy::too_many_arguments)]
fn check_sdpa_decode(
    nq: usize,
    kvh: usize,
    d: usize,
    max_seq: usize,
    len: usize,
    splits: Option<usize>,
    gqa_fold_min_context: usize,
    fixed_block_min_context: usize,
) {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(20 + len as u64);
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..nq * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let mut k = vec![0.0f32; kvh * max_seq * d];
    let mut v = vec![0.0f32; kvh * max_seq * d];
    for h in 0..kvh {
        for pos in 0..len {
            for i in 0..d {
                k[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
                v[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
            }
        }
    }

    let tq = Tensor::from_f32_as_bf16(&ctx, &q, &[nq, d]).expect("q");
    let tk = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
    let tv = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
    let out = Tensor::zeros(&ctx, &[nq, d], DType::BF16).expect("out");

    let scratch = splits.map(|s| {
        (
            Tensor::zeros(&ctx, &[nq, s, d], DType::F32).expect("partials"),
            Tensor::zeros(&ctx, &[nq, s, 2], DType::F32).expect("stats"),
        )
    });
    let pass = ctx.begin().expect("pass");
    sdpa_decode_inner(
        &ctx,
        &pass,
        &tq,
        &tk,
        &tv,
        &out,
        len,
        scale,
        scratch.as_ref().map(|(p, st)| (p, st)),
        gqa_fold_min_context,
        fixed_block_min_context,
    )
    .expect("sdpa");
    pass.commit_wait().expect("commit");

    // CPU reference.
    let (rq, rk, rv) =
        (cpu_ref::round_bf16(&q), cpu_ref::round_bf16(&k), cpu_ref::round_bf16(&v));
    let mut expected = vec![0.0f32; nq * d];
    let group = nq / kvh;
    for hq in 0..nq {
        let h = hq / group;
        let mut scores = vec![0.0f32; len];
        for (pos, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += rq[hq * d + i] * rk[(h * max_seq + pos) * d + i];
            }
            *score = dot * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for i in 0..d {
            let mut acc = 0.0f32;
            for (pos, s) in scores.iter().enumerate() {
                acc += s * rv[(h * max_seq + pos) * d + i];
            }
            expected[hq * d + i] = acc / sum;
        }
    }
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn rope_batched_matches_cpu_per_token() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(21);
    let (m, heads, d, rot, base_pos) = (5, 3, 64, 16, 7);
    let theta = 1e7f32;
    let x: Vec<f32> = (0..m * heads * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, heads, d]).expect("x");
    let pass = ctx.begin().expect("pass");
    rope_neox(&ctx, &pass, &tx, heads, rot, base_pos, theta).expect("rope");
    pass.commit_wait().expect("commit");

    let mut expected = cpu_ref::round_bf16(&x);
    for t in 0..m {
        cpu_ref::rope_neox(
            &mut expected[t * heads * d..(t + 1) * heads * d],
            d,
            rot,
            base_pos + t,
            theta,
        );
    }
    cpu_ref::assert_close(&tx.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

fn check_sdpa_prefill(
    m: usize,
    nq: usize,
    kvh: usize,
    d: usize,
    max_seq: usize,
    base_len: usize,
    seed: u64,
) {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = 1.0 / (d as f32).sqrt();

    // Cache pre-filled for base_len + m positions (the chunk's keys are
    // already scattered, as in the real prefill flow).
    let filled = base_len + m;
    let mut k = vec![0.0f32; kvh * max_seq * d];
    let mut v = vec![0.0f32; kvh * max_seq * d];
    for h in 0..kvh {
        for pos in 0..filled {
            for i in 0..d {
                k[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
                v[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
            }
        }
    }
    let q: Vec<f32> = (0..m * nq * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let tq = Tensor::from_f32_as_bf16(&ctx, &q, &[m, nq, d]).expect("q");
    let tk = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
    let tv = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
    let out = Tensor::zeros(&ctx, &[m, nq, d], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    sdpa_prefill(&ctx, &pass, &tq, &tk, &tv, &out, base_len, scale).expect("sdpa");
    pass.commit_wait().expect("commit");

    let (rq, rk, rv) =
        (cpu_ref::round_bf16(&q), cpu_ref::round_bf16(&k), cpu_ref::round_bf16(&v));
    let group = nq / kvh;
    let mut expected = vec![0.0f32; m * nq * d];
    for t in 0..m {
        let len = base_len + t + 1;
        for hq in 0..nq {
            let h = hq / group;
            let qv = &rq[(t * nq + hq) * d..(t * nq + hq + 1) * d];
            let mut scores = vec![0.0f32; len];
            for (pos, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += qv[i] * rk[(h * max_seq + pos) * d + i];
                }
                *score = dot * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for i in 0..d {
                let mut acc = 0.0f32;
                for (pos, s) in scores.iter().enumerate() {
                    acc += s * rv[(h * max_seq + pos) * d + i];
                }
                expected[(t * nq + hq) * d + i] = acc / sum;
            }
        }
    }
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn sdpa_prefill_matches_cpu_causal() {
    // Every case runs the flash kernel, which is compiled for FA_D, so
    // the head dim is fixed and the cases vary what the tiling has to get
    // right: tile-edge query rows, key tiles crossing the causal diagonal,
    // and a base_len that misaligns the key tiling.
    check_sdpa_prefill(6, 4, 2, FA_D, 64, 9, 22);
    check_sdpa_prefill(16, 8, 2, 256, 64, 0, 23);
    check_sdpa_prefill(40, 4, 2, 256, 128, 9, 24);
    check_sdpa_prefill(33, 8, 2, 256, 96, 47, 25);
}

#[test]
fn scatter_kv_places_rows() {
    let ctx = MetalContext::new().expect("metal context");
    let (kvh, max_seq, d) = (2, 8, 16);
    let cache = Tensor::zeros(&ctx, &[kvh, max_seq, d], DType::BF16).expect("cache");
    let row: Vec<f32> = (0..kvh * d).map(|i| i as f32).collect();
    let trow = Tensor::from_f32_as_bf16(&ctx, &row, &[kvh, d]).expect("row");

    let pass = ctx.begin().expect("pass");
    scatter_kv(&ctx, &pass, &cache, &trow, 3).expect("scatter");
    pass.commit_wait().expect("commit");

    let data = cache.to_f32().expect("read");
    for h in 0..kvh {
        for i in 0..d {
            assert_eq!(data[(h * max_seq + 3) * d + i], (h * d + i) as f32);
            assert_eq!(data[(h * max_seq + 2) * d + i], 0.0);
        }
    }
}
