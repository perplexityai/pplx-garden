const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/gdn.metal"),
    "\n",
    include_str!("../../metal/gdn_test.metal")
);

/// One GDN decode step over all v-heads. `state` is fp32 `[H, 128, 128]` and
/// is updated in place; `a_log`/`w` in the checkpoint are f32, everything
/// else bf16. GVA: q/k carry `num_k_heads` heads; v-head `h` uses q/k head
/// `h / (H / num_k_heads)`.
#[allow(clippy::too_many_arguments)]
fn gdn_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let vpk = num_heads / num_k_heads;
    for (name, t, len) in [
        ("q", q, num_k_heads * dim),
        ("k", k, num_k_heads * dim),
        ("v", v, num_heads * dim),
        ("b", b, num_heads),
        ("dt_bias", dt_bias, num_heads),
        ("out", out, num_heads * dim),
    ] {
        ensure!(t.numel() == len, "{name} numel {} != {len}", t.numel());
        ensure!(t.dtype() == DType::BF16, "{name} must be BF16");
    }
    ensure!(
        a_log.numel() == num_heads && a_log.dtype() == DType::F32,
        "a_log must be F32 [H]"
    );
    ensure!(state.numel() == num_heads * dim * dim, "state must be [H, {dim}, {dim}]");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    let pipeline = ctx.pipeline("gdn_step", TEST_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            q.binding(),
            k.binding(),
            v.binding(),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            state.binding(),
            out.binding(),
        ],
        &[&scale.to_ne_bytes(), &u32_bytes(vpk)],
        Grid::Threadgroups { groups: (num_heads, 1, 1), threadgroup: (dim, 1, 1) },
    )
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

fn random_vec(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(lo..hi)).collect()
}

/// Checks blockwise and sequential CPU recurrence across block boundaries.
#[test]
fn gdn_blockwise_matches_sequential_reference() {
    let mut rng = StdRng::seed_from_u64(9);
    let (dim, tokens) = (32, 45);
    let q = random_vec(&mut rng, tokens * dim, -1.0, 1.0);
    let k = random_vec(&mut rng, tokens * dim, -1.0, 1.0);
    let v = random_vec(&mut rng, tokens * dim, -1.0, 1.0);
    let decay = random_vec(&mut rng, tokens, 0.05, 0.995);
    let beta = random_vec(&mut rng, tokens, 0.05, 0.95);
    let scale = 1.0 / (dim as f32).sqrt();

    let mut st_seq = vec![0.0f32; dim * dim];
    let mut out_seq = Vec::with_capacity(tokens * dim);
    for t in 0..tokens {
        out_seq.extend(cpu_ref::gdn_step(
            &q[t * dim..(t + 1) * dim],
            &k[t * dim..(t + 1) * dim],
            &v[t * dim..(t + 1) * dim],
            &mut st_seq,
            &decay[t..t + 1],
            &beta[t..t + 1],
            scale,
            1,
            1,
            dim,
            dim,
        ));
    }

    for block in [1, 16, 64] {
        let mut st = vec![0.0f32; dim * dim];
        let out = cpu_ref::gdn_blockwise(
            &q, &k, &v, &mut st, &decay, &beta, scale, dim, block,
        );
        cpu_ref::assert_close(&out, &out_seq, 2e-3, 2e-3);
        cpu_ref::assert_close(&st, &st_seq, 2e-3, 2e-3);
    }
}

#[test]
fn gdn_step_matches_cpu_over_sequence() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let steps = 64;

    // nk == nv, plus GVA layouts (v-heads a multiple of k-heads) incl.
    // the 35B ratio.
    for (seed, hk, h) in [(10, 4, 4), (20, 2, 4), (21, 4, 8)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
        let t_state = Tensor::zeros(&ctx, &[h, dim, dim], DType::F32).expect("state");
        let mut ref_state = vec![0.0f32; h * dim * dim];

        for step in 0..steps {
            let q = random_vec(&mut rng, hk * dim, -1.0, 1.0);
            let k = random_vec(&mut rng, hk * dim, -1.0, 1.0);
            let v = random_vec(&mut rng, h * dim, -1.0, 1.0);
            let a = random_vec(&mut rng, h, -1.0, 1.0);
            let b = random_vec(&mut rng, h, -1.0, 1.0);

            let tq = Tensor::from_f32_as_bf16(&ctx, &q, &[hk, dim]).expect("q");
            let tk = Tensor::from_f32_as_bf16(&ctx, &k, &[hk, dim]).expect("k");
            let tv = Tensor::from_f32_as_bf16(&ctx, &v, &[h, dim]).expect("v");
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[h]).expect("a");
            let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[h]).expect("b");
            let out = Tensor::zeros(&ctx, &[h, dim], DType::BF16).expect("out");

            let pass = ctx.begin().expect("pass");
            gdn_step(
                &ctx, &pass, &tq, &tk, &tv, &ta, &tb, &t_a_log, &t_dt_bias, &t_state,
                &out, scale, hk,
            )
            .expect("gdn_step");
            pass.commit_wait().expect("commit");

            let (decay, beta) = cpu_ref::gdn_gates(
                &a_log,
                &cpu_ref::round_bf16(&a),
                &cpu_ref::round_bf16(&dt_bias),
                &cpu_ref::round_bf16(&b),
            );
            let expected = cpu_ref::gdn_step(
                &cpu_ref::round_bf16(&q),
                &cpu_ref::round_bf16(&k),
                &cpu_ref::round_bf16(&v),
                &mut ref_state,
                &decay,
                &beta,
                scale,
                h,
                hk,
                dim,
                dim,
            );

            let actual = out.to_f32().expect("read out");
            cpu_ref::assert_close(&actual, &expected, 2e-2, 2e-2);
            // The state is fp32 on both sides; only op order differs, so
            // drift across a long sequence must stay tiny.
            if step == steps - 1 {
                let state_actual = t_state.to_f32().expect("read state");
                cpu_ref::assert_close(&state_actual, &ref_state, 5e-3, 5e-3);
            }
        }
    }
}

#[test]
fn gdn_prefill_matches_looped_step() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let m = 33;

    for (seed, hk, h) in [(13, 4, 4), (23, 2, 4)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = (2 * hk + h) * dim;

        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
        let a = random_vec(&mut rng, m * h, -1.0, 1.0);
        let b = random_vec(&mut rng, m * h, -1.0, 1.0);

        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
        let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
        let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
        let t_state = Tensor::zeros(&ctx, &[h, dim, dim], DType::F32).expect("state");
        let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");

        // The default route is the register scan and needs staging.
        let (qk_norm, decay, beta) = staging_tensors(&ctx, m, hk, h);
        let staging =
            GdnRegscanStaging { qk_norm: &qk_norm, decay: &decay, beta: &beta };
        let pass = ctx.begin().expect("pass");
        gdn_prefill(
            &ctx, &pass, &t_qkv, &ta, &tb, &t_a_log, &t_dt_bias, &staging, &t_state,
            &out, scale, hk,
        )
        .expect("gdn_prefill");
        pass.commit_wait().expect("commit");

        // Reference: the single-token step looped over the sequence.
        let (rqkv, ra, rb) = (
            cpu_ref::round_bf16(&qkv),
            cpu_ref::round_bf16(&a),
            cpu_ref::round_bf16(&b),
        );
        let rdt = cpu_ref::round_bf16(&dt_bias);
        let mut ref_state = vec![0.0f32; h * dim * dim];
        let mut expected = Vec::new();
        for t in 0..m {
            let row = &rqkv[t * c..(t + 1) * c];
            let (decay, beta) = cpu_ref::gdn_gates(
                &a_log,
                &ra[t * h..(t + 1) * h],
                &rdt,
                &rb[t * h..(t + 1) * h],
            );
            expected.extend(cpu_ref::gdn_step(
                &row[..hk * dim],
                &row[hk * dim..2 * hk * dim],
                &row[2 * hk * dim..],
                &mut ref_state,
                &decay,
                &beta,
                scale,
                h,
                hk,
                dim,
                dim,
            ));
        }
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
        cpu_ref::assert_close(
            &t_state.to_f32().expect("state"),
            &ref_state,
            5e-3,
            5e-3,
        );
    }
}

#[test]
fn gdn_prefill_nonzero_state_matches_looped_step() {
    // Cover nonzero-state resume across tail and multi-block lengths.
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();

    for (seed, hk, h, m) in
        [(31, 2, 4, 7), (33, 4, 4, 16), (35, 4, 4, 33), (39, 2, 4, 64)]
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = (2 * hk + h) * dim;

        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
        let a = random_vec(&mut rng, m * h, -1.0, 1.0);
        let b = random_vec(&mut rng, m * h, -1.0, 1.0);
        let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
        let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
        let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
        let t_state =
            Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
        let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");

        // The default route is the register scan and needs staging.
        let (qk_norm, decay, beta) = staging_tensors(&ctx, m, hk, h);
        let staging =
            GdnRegscanStaging { qk_norm: &qk_norm, decay: &decay, beta: &beta };
        let pass = ctx.begin().expect("pass");
        gdn_prefill(
            &ctx, &pass, &t_qkv, &ta, &tb, &t_a_log, &t_dt_bias, &staging, &t_state,
            &out, scale, hk,
        )
        .expect("gdn_prefill");
        pass.commit_wait().expect("commit");

        let (rqkv, ra, rb) = (
            cpu_ref::round_bf16(&qkv),
            cpu_ref::round_bf16(&a),
            cpu_ref::round_bf16(&b),
        );
        let rdt = cpu_ref::round_bf16(&dt_bias);
        let mut ref_state = init_state.clone();
        let mut expected = Vec::new();
        for t in 0..m {
            let row = &rqkv[t * c..(t + 1) * c];
            let (decay, beta) = cpu_ref::gdn_gates(
                &a_log,
                &ra[t * h..(t + 1) * h],
                &rdt,
                &rb[t * h..(t + 1) * h],
            );
            expected.extend(cpu_ref::gdn_step(
                &row[..hk * dim],
                &row[hk * dim..2 * hk * dim],
                &row[2 * hk * dim..],
                &mut ref_state,
                &decay,
                &beta,
                scale,
                h,
                hk,
                dim,
                dim,
            ));
        }
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
        cpu_ref::assert_close(
            &t_state.to_f32().expect("state"),
            &ref_state,
            5e-3,
            5e-3,
        );
    }
}

/// Fresh regscan staging tensors for one chunk shape (the default route
/// needs them, like the production per-chunk scratch views).
fn staging_tensors(
    ctx: &MetalContext,
    m: usize,
    hk: usize,
    h: usize,
) -> (Tensor, Tensor, Tensor) {
    (
        Tensor::zeros(ctx, &[m, 2 * hk * GDN_HEAD_DIM], DType::BF16).expect("qk_norm"),
        Tensor::zeros(ctx, &[m, h], DType::F32).expect("decay"),
        Tensor::zeros(ctx, &[m, h], DType::F32).expect("beta"),
    )
}

#[test]
fn gdn_regscan_staging_stages_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(40);
    let (m, hk, h, dim) = (7, 2, 4, GDN_HEAD_DIM);
    let c = (2 * hk + h) * dim;
    let scale = 1.0 / (dim as f32).sqrt();
    let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
    let a = random_vec(&mut rng, m * h, -1.0, 1.0);
    let b = random_vec(&mut rng, m * h, -1.0, 1.0);
    let a_log = random_vec(&mut rng, h, -2.0, 0.5);
    let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);

    let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
    let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
    let t_dt_bias = Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
    let (qk_norm, decay, beta) = staging_tensors(&ctx, m, hk, h);

    let pass = ctx.begin().expect("pass");
    gdn_qk_l2norm(&ctx, &pass, &t_qkv, &qk_norm, scale, hk, h).expect("gdn_qk_l2norm");
    gdn_gates(&ctx, &pass, &ta, &tb, &t_a_log, &t_dt_bias, &decay, &beta)
        .expect("gdn_gates");
    pass.commit_wait().expect("commit");

    let rounded_qkv = cpu_ref::round_bf16(&qkv);
    let mut expected_qk = vec![0.0f32; m * 2 * hk * dim];
    for token in 0..m {
        let row = &rounded_qkv[token * c..(token + 1) * c];
        let out_row = token * 2 * hk * dim;
        for head in 0..hk {
            let mut q = row[head * dim..(head + 1) * dim].to_vec();
            let mut k = row[(hk + head) * dim..(hk + head + 1) * dim].to_vec();
            cpu_ref::l2_normalize(&mut q);
            cpu_ref::l2_normalize(&mut k);
            for value in &mut q {
                *value *= scale;
            }
            expected_qk[out_row + head * dim..out_row + (head + 1) * dim]
                .copy_from_slice(&q);
            expected_qk[out_row + (hk + head) * dim..out_row + (hk + head + 1) * dim]
                .copy_from_slice(&k);
        }
    }
    let expected_qk = cpu_ref::round_bf16(&expected_qk);
    cpu_ref::assert_close(
        &qk_norm.to_f32().expect("read qk_norm"),
        &expected_qk,
        2e-2,
        2e-2,
    );

    let (rounded_a, rounded_b, rounded_dt_bias) = (
        cpu_ref::round_bf16(&a),
        cpu_ref::round_bf16(&b),
        cpu_ref::round_bf16(&dt_bias),
    );
    let mut expected_decay = Vec::with_capacity(m * h);
    let mut expected_beta = Vec::with_capacity(m * h);
    for token in 0..m {
        let (token_decay, token_beta) = cpu_ref::gdn_gates(
            &a_log,
            &rounded_a[token * h..(token + 1) * h],
            &rounded_dt_bias,
            &rounded_b[token * h..(token + 1) * h],
        );
        expected_decay.extend(token_decay);
        expected_beta.extend(token_beta);
    }
    cpu_ref::assert_close(
        &decay.to_f32().expect("read decay"),
        &expected_decay,
        1e-5,
        1e-5,
    );
    cpu_ref::assert_close(
        &beta.to_f32().expect("read beta"),
        &expected_beta,
        1e-5,
        1e-5,
    );
}

/// One regscan prefill dispatch over host data; returns (out, state).
/// Staging tensors are allocated per call, like the production per-chunk
/// scratch views.
#[allow(clippy::too_many_arguments)]
fn run_regscan(
    ctx: &MetalContext,
    qkv: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    scale: f32,
    hk: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dim = GDN_HEAD_DIM;
    let m = a.len() / h;
    let c = (2 * hk + h) * dim;
    let t_qkv = Tensor::from_f32_as_bf16(ctx, qkv, &[m, c]).expect("qkv");
    let ta = Tensor::from_f32_as_bf16(ctx, a, &[m, h]).expect("a");
    let tb = Tensor::from_f32_as_bf16(ctx, b, &[m, h]).expect("b");
    let (qk_norm, decay, beta) = staging_tensors(ctx, m, hk, h);
    let staging = GdnRegscanStaging { qk_norm: &qk_norm, decay: &decay, beta: &beta };
    let out = Tensor::zeros(ctx, &[m, h, dim], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    gdn_prefill(
        ctx, &pass, &t_qkv, &ta, &tb, a_log, dt_bias, &staging, state, &out, scale, hk,
    )
    .expect("gdn_prefill");
    pass.commit_wait().expect("commit");
    (out.to_f32().expect("read out"), state.to_f32().expect("read state"))
}

#[test]
fn gdn_prefill_regscan_rejects_bf16_state() {
    let ctx = MetalContext::new().expect("metal context");
    let (m, hk, h, dim) = (1, 1, 1, GDN_HEAD_DIM);
    let qkv = Tensor::zeros(&ctx, &[m, (2 * hk + h) * dim], DType::BF16).expect("qkv");
    let a = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("a");
    let b = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("b");
    let a_log = Tensor::zeros(&ctx, &[h], DType::F32).expect("a_log");
    let dt_bias = Tensor::zeros(&ctx, &[h], DType::BF16).expect("dt_bias");
    let (qk_norm, decay, beta) = staging_tensors(&ctx, m, hk, h);
    let staging = GdnRegscanStaging { qk_norm: &qk_norm, decay: &decay, beta: &beta };
    let state = Tensor::zeros(&ctx, &[h, dim, dim], DType::BF16).expect("state");
    let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");

    let err = gdn_prefill(
        &ctx, &pass, &qkv, &a, &b, &a_log, &dt_bias, &staging, &state, &out, 1.0, hk,
    )
    .expect_err("BF16 state must be rejected before dispatch");
    assert!(err.to_string().contains("state must be F32"), "unexpected error: {err:#}");
}

#[test]
fn gdn_fused_decode_f32_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let (steps, hk, h, dim) = (16usize, 1usize, 2usize, GDN_HEAD_DIM);
    let scale = 1.0 / (dim as f32).sqrt();
    let eps = 1e-6f32;
    let mut rng = StdRng::seed_from_u64(0xf32dec0de);
    let a_log = random_vec(&mut rng, h, -2.0, 0.5);
    let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
    let norm_w = random_vec(&mut rng, dim, 0.5, 1.5);
    let init_state = random_vec(&mut rng, h * dim * dim, -0.25, 0.25);
    let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
    let t_dt_bias = Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
    let t_norm_w = Tensor::from_f32(&ctx, &norm_w, &[dim]).expect("norm_w");
    let t_state = Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
    let mut ref_state = init_state;

    for _ in 0..steps {
        let qkv = random_vec(&mut rng, (2 * hk + h) * dim, -1.0, 1.0);
        let a = random_vec(&mut rng, h, -1.0, 1.0);
        let b = random_vec(&mut rng, h, -1.0, 1.0);
        let z = random_vec(&mut rng, h * dim, -1.0, 1.0);
        let t_qkv =
            Tensor::from_f32_as_bf16(&ctx, &qkv, &[2 * hk + h, dim]).expect("qkv");
        let t_a = Tensor::from_f32_as_bf16(&ctx, &a, &[h]).expect("a");
        let t_b = Tensor::from_f32_as_bf16(&ctx, &b, &[h]).expect("b");
        let t_z = Tensor::from_f32_as_bf16(&ctx, &z, &[h, dim]).expect("z");
        let out = Tensor::zeros(&ctx, &[h, dim], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        gdn_step_gated_fused(
            &ctx, &pass, &t_qkv, &t_a, &t_b, &t_a_log, &t_dt_bias, &t_state, &t_z,
            &t_norm_w, &out, scale, hk, eps,
        )
        .expect("F32 fused decode");
        pass.commit_wait().expect("commit");

        let (rqkv, ra, rb, rz, rdt) = (
            cpu_ref::round_bf16(&qkv),
            cpu_ref::round_bf16(&a),
            cpu_ref::round_bf16(&b),
            cpu_ref::round_bf16(&z),
            cpu_ref::round_bf16(&dt_bias),
        );
        let (decay, beta) = cpu_ref::gdn_gates(&a_log, &ra, &rdt, &rb);
        let raw = cpu_ref::gdn_step(
            &rqkv[..hk * dim],
            &rqkv[hk * dim..2 * hk * dim],
            &rqkv[2 * hk * dim..],
            &mut ref_state,
            &decay,
            &beta,
            scale,
            h,
            hk,
            dim,
            dim,
        );
        let mut expected = vec![0.0f32; h * dim];
        for head in 0..h {
            let row = &raw[head * dim..(head + 1) * dim];
            let inv_rms = 1.0
                / (row.iter().map(|value| value * value).sum::<f32>() / dim as f32
                    + eps)
                    .sqrt();
            for d in 0..dim {
                let gate = rz[head * dim + d];
                expected[head * dim + d] =
                    norm_w[d] * row[d] * inv_rms * gate / (1.0 + (-gate).exp());
            }
        }
        cpu_ref::assert_close(&out.to_f32().expect("out"), &expected, 4e-2, 4e-2);
    }
    cpu_ref::assert_close(&t_state.to_f32().expect("state"), &ref_state, 2e-5, 2e-5);
}

/// Correctness matrix: the register scan vs the per-token cpu_ref from a
/// NONZERO initial state, over
/// M in {1, 7, 16, 33, 200, 2048} x vpk in {1, 2}, at the existing
/// tolerances. The full head geometry uses a shorter M list; small-head cells
/// cover the complete sequence-length matrix.
#[test]
fn gdn_prefill_regscan_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let m_full: &[usize] = &[1, 7, 16, 33, 200, 2048];
    let m_shape: &[usize] = &[33, 200];
    for (seed, hk, h, ms) in [
        (41u64, 4usize, 4usize, m_full),
        (43, 2, 4, m_full),
        (45, 16, 16, m_shape),
        (47, 16, 32, m_shape),
    ] {
        for &m in ms {
            let mut rng = StdRng::seed_from_u64(seed + m as u64);
            let c = (2 * hk + h) * dim;

            let a_log = random_vec(&mut rng, h, -2.0, 0.5);
            let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
            let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
            let a = random_vec(&mut rng, m * h, -1.0, 1.0);
            let b = random_vec(&mut rng, m * h, -1.0, 1.0);
            let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

            let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
            let t_dt_bias =
                Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
            let t_state =
                Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
            let (out_reg, state_reg) = run_regscan(
                &ctx, &qkv, &a, &b, &t_a_log, &t_dt_bias, &t_state, scale, hk, h,
            );

            // Per-token cpu_ref reference over the same bf16-rounded
            // inputs.
            let (rqkv, ra, rb) = (
                cpu_ref::round_bf16(&qkv),
                cpu_ref::round_bf16(&a),
                cpu_ref::round_bf16(&b),
            );
            let rdt = cpu_ref::round_bf16(&dt_bias);
            let mut ref_state = init_state.clone();
            let mut expected = Vec::new();
            for t in 0..m {
                let row = &rqkv[t * c..(t + 1) * c];
                let (decay, beta) = cpu_ref::gdn_gates(
                    &a_log,
                    &ra[t * h..(t + 1) * h],
                    &rdt,
                    &rb[t * h..(t + 1) * h],
                );
                expected.extend(cpu_ref::gdn_step(
                    &row[..hk * dim],
                    &row[hk * dim..2 * hk * dim],
                    &row[2 * hk * dim..],
                    &mut ref_state,
                    &decay,
                    &beta,
                    scale,
                    h,
                    hk,
                    dim,
                    dim,
                ));
            }
            cpu_ref::assert_close(&out_reg, &expected, 2e-2, 2e-2);
            cpu_ref::assert_close(&state_reg, &ref_state, 5e-3, 5e-3);
        }
    }
}

/// Chunk-boundary state carry through the regscan kernel: splitting a
/// sequence into chunks only round-trips the fp32 state through device
/// memory (which the kernel does at entry/exit anyway), so outputs and
/// the final state must be BIT-identical to the single-shot run.
#[test]
fn gdn_prefill_regscan_chunk_carry_bit_identical() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let (m, chunk) = (200usize, 64usize);

    for (seed, hk, h) in [(51u64, 4usize, 4usize), (53, 2, 4)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = (2 * hk + h) * dim;

        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
        let a = random_vec(&mut rng, m * h, -1.0, 1.0);
        let b = random_vec(&mut rng, m * h, -1.0, 1.0);
        let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");

        let state_full =
            Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
        let (out_full, final_full) = run_regscan(
            &ctx,
            &qkv,
            &a,
            &b,
            &t_a_log,
            &t_dt_bias,
            &state_full,
            scale,
            hk,
            h,
        );

        let state_chunked =
            Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
        let mut out_chunked = Vec::new();
        let mut final_chunked = Vec::new();
        for t0 in (0..m).step_by(chunk) {
            let t1 = (t0 + chunk).min(m);
            let (out, state) = run_regscan(
                &ctx,
                &qkv[t0 * c..t1 * c],
                &a[t0 * h..t1 * h],
                &b[t0 * h..t1 * h],
                &t_a_log,
                &t_dt_bias,
                &state_chunked,
                scale,
                hk,
                h,
            );
            out_chunked.extend(out);
            final_chunked = state;
        }
        assert_eq!(out_full, out_chunked, "chunked outputs diverged");
        assert_eq!(final_full, final_chunked, "chunked final state diverged");
    }
}

/// Chunk sequences covering single tokens, tile boundaries, a full
/// prefill chunk, and window carry across chunks including trailing
/// chunks shorter than the window (M < KD-1).
fn conv1d_chunk_cases() -> Vec<Vec<usize>> {
    let tile = CONV1D_PREFILL_TILE;
    vec![
        vec![1],
        vec![3],
        vec![4],
        vec![16],
        vec![tile - 1],
        vec![tile],
        vec![tile + 1],
        vec![2048],
        vec![tile + 1, 2],
        vec![2, 2],
        vec![5, tile],
    ]
}

#[test]
fn conv1d_prefill_matches_looped_step() {
    let ctx = MetalContext::new().expect("metal context");

    for kd in [4usize, 9] {
        for chunks in conv1d_chunk_cases() {
            let mut rng = StdRng::seed_from_u64(14 + kd as u64);
            let c = 96;
            let w = random_vec(&mut rng, kd * c, -1.0, 1.0);
            let win0 = random_vec(&mut rng, c * (kd - 1), -1.0, 1.0);
            // The inactive buffer starts as garbage: the final tile must
            // overwrite every slot of it.
            let junk = random_vec(&mut rng, c * (kd - 1), -9.0, 9.0);
            let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");
            let windows = [
                Tensor::from_f32_as_bf16(&ctx, &win0, &[c, kd - 1]).expect("window a"),
                Tensor::from_f32_as_bf16(&ctx, &junk, &[c, kd - 1]).expect("window b"),
            ];
            let mut slot = 0;
            let rw = cpu_ref::round_bf16(&w);
            let mut ref_window = cpu_ref::round_bf16(&win0);

            for m in chunks {
                let x = random_vec(&mut rng, m * c, -1.0, 1.0);
                let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, c]).expect("x");
                let out = Tensor::zeros(&ctx, &[m, c], DType::BF16).expect("out");

                let pass = ctx.begin().expect("pass");
                conv1d_prefill(
                    &ctx,
                    &pass,
                    &windows[slot],
                    &windows[1 - slot],
                    &tx,
                    &tw,
                    &out,
                )
                .expect("conv1d_prefill");
                pass.commit_wait().expect("commit");
                slot = 1 - slot;

                let rx = cpu_ref::round_bf16(&x);
                let mut expected = Vec::new();
                for t in 0..m {
                    expected.extend(cpu_ref::conv1d_step(
                        &mut ref_window,
                        &rx[t * c..(t + 1) * c],
                        &rw,
                        c,
                        kd,
                    ));
                }
                cpu_ref::assert_close(
                    &out.to_f32().expect("read"),
                    &expected,
                    2e-2,
                    2e-2,
                );
                cpu_ref::assert_close(
                    &windows[slot].to_f32().expect("window"),
                    &ref_window,
                    2e-2,
                    2e-2,
                );
            }
        }
    }
}

/// Serial prefill conv oracle; tiled output and carried window must match bits.
const CONV1D_SERIAL_REF: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv1d_prefill_serial_ref(device bfloat*       window [[buffer(0)]],
                                      device const bfloat* x      [[buffer(1)]],
                                      device const bfloat* w      [[buffer(2)]],
                                      device bfloat*       out    [[buffer(3)]],
                                      constant uint&       C      [[buffer(4)]],
                                      constant uint&       KD     [[buffer(5)]],
                                      constant uint&       M      [[buffer(6)]],
                                      uint c [[thread_position_in_grid]]) {
    uint taps = KD - 1;
    float win[8];
    for (uint t = 0; t < taps; ++t) {
        win[t] = float(window[c * taps + t]);
    }
    for (uint m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (uint t = 0; t < taps; ++t) {
            acc += win[t] * float(w[t * C + c]);
        }
        float xc = float(x[(ulong)m * C + c]);
        acc += xc * float(w[taps * C + c]);
        out[(ulong)m * C + c] = bfloat(acc / (1.0f + exp(-acc)));
        for (uint t = 0; t + 1 < taps; ++t) {
            win[t] = win[t + 1];
        }
        win[taps - 1] = xc;
    }
    for (uint t = 0; t < taps; ++t) {
        window[c * taps + t] = bfloat(win[t]);
    }
}
"#;

#[test]
fn conv1d_prefill_bitwise_matches_serial_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let tile = CONV1D_PREFILL_TILE;
    let ref_pipeline = ctx
        .pipeline("conv1d_prefill_serial_ref", CONV1D_SERIAL_REF, MslVersion::V3_1)
        .expect("ref pipeline");

    for (seed, c, kd, chunks) in [
        (31u64, 96usize, 4usize, vec![2048]),
        (32, 96, 4, vec![tile + 1, 2]),
        (33, 300, 4, vec![tile, 1, 3]),
        (34, 96, 2, vec![tile - 1, 16]),
        (35, 96, 9, vec![2, tile + 1, 4]),
    ] {
        let mut rng = StdRng::seed_from_u64(seed);
        let w = random_vec(&mut rng, kd * c, -1.0, 1.0);
        let win0 = random_vec(&mut rng, c * (kd - 1), -1.0, 1.0);
        let junk = random_vec(&mut rng, c * (kd - 1), -9.0, 9.0);
        let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");
        let win_new = [
            Tensor::from_f32_as_bf16(&ctx, &win0, &[c, kd - 1]).expect("win a"),
            Tensor::from_f32_as_bf16(&ctx, &junk, &[c, kd - 1]).expect("win b"),
        ];
        let mut slot = 0;
        let win_ref =
            Tensor::from_f32_as_bf16(&ctx, &win0, &[c, kd - 1]).expect("win ref");

        for m in chunks {
            let x = random_vec(&mut rng, m * c, -1.0, 1.0);
            let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, c]).expect("x");
            let out_new = Tensor::zeros(&ctx, &[m, c], DType::BF16).expect("out");
            let out_ref = Tensor::zeros(&ctx, &[m, c], DType::BF16).expect("ref");

            let pass = ctx.begin().expect("pass");
            conv1d_prefill(
                &ctx,
                &pass,
                &win_new[slot],
                &win_new[1 - slot],
                &tx,
                &tw,
                &out_new,
            )
            .expect("conv1d_prefill");
            pass.dispatch_at(
                &ref_pipeline,
                &[win_ref.binding(), tx.binding(), tw.binding(), out_ref.binding()],
                &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(m)],
                Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
            )
            .expect("serial ref dispatch");
            pass.commit_wait().expect("commit");
            slot = 1 - slot;

            assert_eq!(
                out_new.to_f32().expect("read out"),
                out_ref.to_f32().expect("read ref out"),
                "output diverged from the serial kernel (c={c} kd={kd} m={m})"
            );
            // The freshly written (now active) slot must carry exactly
            // the serial kernel's in-place window.
            assert_eq!(
                win_new[slot].to_f32().expect("read window"),
                win_ref.to_f32().expect("read ref window"),
                "window diverged from the serial kernel (c={c} kd={kd} m={m})"
            );
        }
    }
}

#[test]
fn conv1d_step_matches_cpu_over_steps() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(11);
    let (c, kd) = (96, 4);

    let w = random_vec(&mut rng, kd * c, -1.0, 1.0);
    let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");
    let t_window = Tensor::zeros(&ctx, &[c, kd - 1], DType::BF16).expect("window");
    let mut ref_window = vec![0.0f32; c * (kd - 1)];

    for _ in 0..8 {
        let x = random_vec(&mut rng, c, -1.0, 1.0);
        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[c]).expect("x");
        let out = Tensor::zeros(&ctx, &[c], DType::BF16).expect("out");

        let pass = ctx.begin().expect("pass");
        conv1d_step(&ctx, &pass, &t_window, &tx, &tw, &out).expect("conv1d_step");
        pass.commit_wait().expect("commit");

        let expected = cpu_ref::conv1d_step(
            &mut ref_window,
            &cpu_ref::round_bf16(&x),
            &cpu_ref::round_bf16(&w),
            c,
            kd,
        );
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

#[test]
fn gated_rmsnorm_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(12);
    let (rows, d) = (16, 128);
    let eps = 1e-6;

    let x = random_vec(&mut rng, rows * d, -2.0, 2.0);
    let gate = random_vec(&mut rng, rows * d, -2.0, 2.0);
    let w = random_vec(&mut rng, d, 0.5, 1.5);

    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[rows, d]).expect("x");
    let tg = Tensor::from_f32_as_bf16(&ctx, &gate, &[rows, d]).expect("gate");
    let tw = Tensor::from_f32(&ctx, &w, &[d]).expect("w");
    let out = Tensor::zeros(&ctx, &[rows, d], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    gated_rmsnorm(&ctx, &pass, &tx, &tg, &tw, &out, eps).expect("gated_rmsnorm");
    pass.commit_wait().expect("commit");

    let expected = cpu_ref::gated_rmsnorm(
        &cpu_ref::round_bf16(&x),
        &cpu_ref::round_bf16(&gate),
        &w,
        d,
        eps,
    );
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}
