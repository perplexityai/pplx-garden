use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;
use crate::tensor::DType;

fn random_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn check_gemm(seed: u64) {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(seed);
    // Tile-exact and partial-tile shapes around both kernels' tile sizes
    // (32x32xK16 and 64x64xK32): multiples, non-multiples (incl. odd K
    // exercising the scalar-load path), single-row M, the M/N/K tails one
    // past a tile boundary, and the model's real shapes.
    for (m, k, n) in [
        (5, 64, 7),
        (32, 64, 32),
        (128, 1024, 512),
        (129, 100, 33),
        (1, 17, 40),
        (33, 16, 31),
        (63, 24, 65),
        (65, 64, 63),
        (257, 512, 40),
        (64, 1024, 3584),
        (256, 3584, 1024),
        (1, 1024, 6144),
    ] {
        let a = random_vec(&mut rng, m * k);
        let b = random_vec(&mut rng, n * k);

        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[n, k]).expect("b");
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");

        let pass = ctx.begin().expect("pass");
        gemm_bf16_nt(&ctx, &pass, &ta, &tb, &tc).expect("gemm");
        pass.commit_wait().expect("commit");

        let expected = cpu_ref::gemm_nt(
            &cpu_ref::round_bf16(&a),
            &cpu_ref::round_bf16(&b),
            m,
            k,
            n,
        );
        cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

#[test]
fn gemm_bf16_nt_matches_cpu() {
    check_gemm(0);
}
