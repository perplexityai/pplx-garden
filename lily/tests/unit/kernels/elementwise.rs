const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/elementwise.metal"),
    "\n",
    include_str!("../../metal/elementwise_test.metal")
);

fn mul_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op_source(ctx, pass, "mul_bf16", DType::BF16, a, b, out, TEST_SOURCE)
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

#[test]
fn elementwise_ops_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(3);
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let (ra, rb) = (cpu_ref::round_bf16(&a), cpu_ref::round_bf16(&b));

    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[n]).expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[n]).expect("b");

    let pair = |f: &dyn Fn(f32, f32) -> f32| -> Vec<f32> {
        ra.iter().zip(&rb).map(|(&x, &y)| f(x, y)).collect()
    };
    let cases = [
        ("add", pair(&|x, y| x + y)),
        ("mul", pair(&|x, y| x * y)),
        ("silu_mul", pair(&|x, y| cpu_ref::silu(x) * y)),
    ];
    for (name, expected) in cases {
        let out = Tensor::zeros(&ctx, &[n], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        match name {
            "add" => add_bf16(&ctx, &pass, &ta, &tb, &out).expect("add"),
            "mul" => mul_bf16(&ctx, &pass, &ta, &tb, &out).expect("mul"),
            _ => silu_mul_bf16(&ctx, &pass, &ta, &tb, &out).expect("silu_mul"),
        }
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

#[test]
fn argmax_f32_stages_match_first_max_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(8);
    let n = 4097;
    let mut x: Vec<f32> = (0..n).map(|_| rng.gen_range(-8.0f32..8.0)).collect();
    x[513] = 9.5;
    x[3501] = 9.5;
    let expected = 513u32;

    let tx = Tensor::from_f32(&ctx, &x, &[n]).expect("x");
    let partials =
        Tensor::zeros(&ctx, &[ARGMAX_GROUPS * 2], DType::F32).expect("partials");
    let out = Tensor::zeros(&ctx, &[1], DType::U32).expect("out");

    let partial_pass = ctx.begin().expect("partial pass");
    argmax_f32_partial(&ctx, &partial_pass, &tx, &partials)
        .expect("argmax_f32_partial");
    partial_pass.commit_wait().expect("partial commit");

    let chunk = n.div_ceil(ARGMAX_GROUPS);
    let bytes = partials.raw_bytes();
    for group in 0..ARGMAX_GROUPS {
        let start = group * chunk;
        let end = ((group + 1) * chunk).min(n);
        let (expected_value, expected_index) = x[start..end].iter().enumerate().fold(
            (f32::NEG_INFINITY, 0usize),
            |(best_value, best_index), (offset, &value)| {
                let index = start + offset;
                if value > best_value {
                    (value, index)
                } else {
                    (best_value, best_index)
                }
            },
        );
        let base = group * 8;
        let actual_value = f32::from_ne_bytes(
            bytes[base..base + 4].try_into().expect("partial value"),
        );
        let actual_index = u32::from_ne_bytes(
            bytes[base + 4..base + 8].try_into().expect("partial index"),
        );
        assert_eq!(actual_value, expected_value, "group {group} value");
        assert_eq!(actual_index as usize, expected_index, "group {group} index");
    }

    let final_pass = ctx.begin().expect("final pass");
    argmax_f32_final(&ctx, &final_pass, &partials, &out).expect("argmax_f32_final");
    final_pass.commit_wait().expect("final commit");
    assert_eq!(out.to_u32().expect("read staged result")[0], expected);

    out.zero_fill();
    let combined_pass = ctx.begin().expect("combined pass");
    argmax_f32(&ctx, &combined_pass, &tx, &partials, &out).expect("argmax_f32");
    combined_pass.commit_wait().expect("combined commit");
    assert_eq!(out.to_u32().expect("read combined result")[0], expected);
}
