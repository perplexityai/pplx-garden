use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

#[test]
fn rmsnorm_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(4);
    let (m, h) = (3, 1024);
    let x: Vec<f32> = (0..m * h).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let w: Vec<f32> = (0..h).map(|_| rng.gen_range(0.5f32..1.5)).collect();
    let eps = 1e-6;

    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, h]).expect("x");
    let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[h]).expect("w");
    let out = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    rmsnorm_bf16(&ctx, &pass, &tx, &tw, &out, eps, 0.0).expect("rmsnorm");
    pass.commit_wait().expect("commit");

    let expected =
        cpu_ref::rmsnorm(&cpu_ref::round_bf16(&x), &cpu_ref::round_bf16(&w), h, eps);
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn add_rmsnorm_matches_unfused_pair() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(11);
    let (m, h) = (3, RESIDUAL_H);
    let x: Vec<f32> = (0..m * h).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let b: Vec<f32> = (0..m * h).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let w: Vec<f32> = (0..h).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
    let eps = 1e-6;
    let w_bias = 1.0;

    let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[h]).expect("w");

    // Reference: the unfused pair (standalone add kernel, then rmsnorm).
    let rx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, h]).expect("rx");
    let rb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("rb");
    let r_out = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("r_out");
    let pass = ctx.begin().expect("pass");
    crate::kernels::elementwise::add_bf16(&ctx, &pass, &rx, &rb, &rx).expect("add");
    rmsnorm_bf16(&ctx, &pass, &rx, &tw, &r_out, eps, w_bias).expect("rmsnorm");
    pass.commit_wait().expect("commit");

    // Fused kernel over the same inputs.
    let fx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, h]).expect("fx");
    let fb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("fb");
    let f_out = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("f_out");
    let pass = ctx.begin().expect("pass");
    add_rmsnorm_bf16(&ctx, &pass, &fx, &fb, &tw, &f_out, eps, w_bias).expect("fused");
    pass.commit_wait().expect("commit");

    // The stored residual must be bit-identical (it feeds the next layer
    // and the host-visible state). The normalized output is bit-identical
    // on g15/g16/g17 but 1 bf16 ulp off on g14, whose compiler keeps the
    // unrounded fp32 sum in the reduction accumulator — assert tight
    // closeness rather than equality.
    assert_eq!(
        fx.to_f32().expect("fx"),
        rx.to_f32().expect("rx"),
        "fused residual sum diverged from the standalone add"
    );
    cpu_ref::assert_close(
        &f_out.to_f32().expect("f_out"),
        &r_out.to_f32().expect("r_out"),
        1e-2,
        1e-2,
    );
}
