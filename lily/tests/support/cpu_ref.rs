//! Test-only plain-Rust f32 reference implementations used by kernel tests. Inputs are
//! pre-rounded to bf16 (see `round_bf16`) so comparisons isolate accumulation
//! error rather than input-rounding error.

use half::bf16;

pub fn round_bf16(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// C[M,N] = A[M,K] . B[K,N].
pub fn gemm_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

/// C[M,N] = A[M,K] . B[N,K]^T (weight-natural layout).
pub fn gemm_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[col * k + i];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

/// Row-wise weighted RMSNorm over the last dimension.
pub fn rmsnorm(x: &[f32], w: &[f32], h: usize, eps: f32) -> Vec<f32> {
    let m = x.len() / h;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..m {
        let slice = &x[row * h..(row + 1) * h];
        let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / h as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..h {
            out[row * h + i] = slice[i] * inv_rms * w[i];
        }
    }
    out
}

/// Decay/update gates for one GDN step: `decay = exp(-exp(A_log) *
/// softplus(a + dt_bias))`, `beta = sigmoid(b)` (softplus switches to identity
/// above 20 for stability, for numerical stability).
pub fn gdn_gates(
    a_log: &[f32],
    a: &[f32],
    dt_bias: &[f32],
    b: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let decay = a_log
        .iter()
        .zip(a)
        .zip(dt_bias)
        .map(|((log_a, av), bias)| {
            let x = av + bias;
            let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            (-log_a.exp() * softplus).exp()
        })
        .collect();
    let beta = b.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    (decay, beta)
}

/// `F.normalize(p=2)`: v / max(||v||, eps) with eps = 1e-12.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// One GDN decode step (the Gated DeltaNet recurrence): per
/// head, L2-normalize q/k, decay the state, apply the delta rule, and read the
/// output. Mutates `state` (`[H, K, V]` fp32) and returns `o` (`[H, V]`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    decay: &[f32],
    beta: &[f32],
    scale: f32,
    num_heads: usize,
    num_k_heads: usize,
    dim_k: usize,
    dim_v: usize,
) -> Vec<f32> {
    assert_eq!(num_heads % num_k_heads, 0, "v-heads must be a multiple of k-heads");
    let vpk = num_heads / num_k_heads;
    let mut out = vec![0.0f32; num_heads * dim_v];
    for h in 0..num_heads {
        // GVA: v-head h reads q/k head h / vpk.
        let hk = h / vpk;
        let mut qn = q[hk * dim_k..(hk + 1) * dim_k].to_vec();
        let mut kn = k[hk * dim_k..(hk + 1) * dim_k].to_vec();
        l2_normalize(&mut qn);
        l2_normalize(&mut kn);
        for x in qn.iter_mut() {
            *x *= scale;
        }

        let st = &mut state[h * dim_k * dim_v..(h + 1) * dim_k * dim_v];
        for x in st.iter_mut() {
            *x *= decay[h];
        }
        for vi in 0..dim_v {
            let mut kv = 0.0f32;
            for ki in 0..dim_k {
                kv += kn[ki] * st[ki * dim_v + vi];
            }
            let v_new = (v[h * dim_v + vi] - kv) * beta[h];
            let mut o = 0.0f32;
            for ki in 0..dim_k {
                st[ki * dim_v + vi] += kn[ki] * v_new;
                o += qn[ki] * st[ki * dim_v + vi];
            }
            out[h * dim_v + vi] = o;
        }
    }
    out
}

/// One causal depthwise conv1d + SiLU decode step. `window` is `[C, KD-1]`
/// (channel-major recent inputs, oldest first), `w` is `[KD, C]` (tap-major);
/// shifts the window in place.
pub fn conv1d_step(
    window: &mut [f32],
    x: &[f32],
    w: &[f32],
    c: usize,
    kd: usize,
) -> Vec<f32> {
    let taps = kd - 1;
    let mut out = vec![0.0f32; c];
    for ch in 0..c {
        let mut acc = 0.0f32;
        for t in 0..taps {
            acc += window[ch * taps + t] * w[t * c + ch];
        }
        acc += x[ch] * w[taps * c + ch];
        out[ch] = silu(acc);
        for t in 0..taps.saturating_sub(1) {
            window[ch * taps + t] = window[ch * taps + t + 1];
        }
        window[ch * taps + taps - 1] = x[ch];
    }
    out
}

/// Gated RMS norm : per row of size `d`,
/// `out = w * x * rsqrt(mean(x^2) + eps) * silu(gate)`.
pub fn gated_rmsnorm(
    x: &[f32],
    gate: &[f32],
    w: &[f32],
    d: usize,
    eps: f32,
) -> Vec<f32> {
    let rows = x.len() / d;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let xs = &x[row * d..(row + 1) * d];
        let mean_sq = xs.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..d {
            out[row * d + i] = w[i] * xs[i] * inv_rms * silu(gate[row * d + i]);
        }
    }
    out
}

/// Blockwise delta rule over a whole sequence for ONE head (
/// `blockwise_delta_rule` reference), mathematically equivalent to looping
/// [`gdn_step`] token by token: per block, cumulative log-space decays, the
/// unit-lower-triangular `I + tril(beta * exp(Gamma) * K K^T)` inverted by
/// forward substitution, then block-level matmuls for the corrected values,
/// intra/inter outputs, and the decayed state update. `q`/`k`/`v` are raw
/// `[tokens, dim]` rows (normalization and the q scale happen here, matching
/// `gdn_step`); mutates `state` (`[dim, dim]` fp32) and returns `[tokens,
/// dim]` outputs.
#[allow(clippy::too_many_arguments)]
pub fn gdn_blockwise(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    decay: &[f32],
    beta: &[f32],
    scale: f32,
    dim: usize,
    block: usize,
) -> Vec<f32> {
    let tokens = q.len() / dim;
    let mut out = vec![0.0f32; tokens * dim];

    for b0 in (0..tokens).step_by(block) {
        let n = block.min(tokens - b0);

        // Normalized per-row q (scaled) and k for the block.
        let mut qh = q[b0 * dim..(b0 + n) * dim].to_vec();
        let mut kh = k[b0 * dim..(b0 + n) * dim].to_vec();
        for i in 0..n {
            l2_normalize(&mut qh[i * dim..(i + 1) * dim]);
            l2_normalize(&mut kh[i * dim..(i + 1) * dim]);
        }
        for x in qh.iter_mut() {
            *x *= scale;
        }

        // Cumulative log decays: exp(cu[i] - cu[j]) is the decay product over
        // tokens j+1..=i (the epsilon guards log of a zero decay).
        let mut cu = vec![0.0f32; n];
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += (decay[b0 + i] + 1e-10).ln();
            cu[i] = acc;
        }

        // Unit-lower L = I + strict_tril(beta_i * exp(Gamma_ij) * k_i.k_j),
        // inverted in place by forward substitution, then column-scaled by
        // beta to form T.
        let dot =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let mut l = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..i {
                l[i * n + j] = beta[b0 + i]
                    * (cu[i] - cu[j]).exp()
                    * dot(&kh[i * dim..(i + 1) * dim], &kh[j * dim..(j + 1) * dim]);
            }
        }
        let mut t = vec![0.0f32; n * n];
        for i in 0..n {
            t[i * n + i] = 1.0;
            for j in 0..i {
                let mut s = 0.0f32;
                for p in j..i {
                    s += l[i * n + p] * t[p * n + j];
                }
                t[i * n + j] = -s;
            }
        }
        for i in 0..n {
            for j in 0..=i {
                t[i * n + j] *= beta[b0 + j];
            }
        }

        // new_v = T.v - (T.(exp(cu).k)).S_prev
        let mut new_v = vec![0.0f32; n * dim];
        let mut w = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..=i {
                let tv = t[i * n + j];
                let e = tv * cu[j].exp();
                for c in 0..dim {
                    new_v[i * dim + c] += tv * v[(b0 + j) * dim + c];
                    w[i * dim + c] += e * kh[j * dim + c];
                }
            }
        }
        for i in 0..n {
            for c in 0..dim {
                let mut s = 0.0f32;
                for ki in 0..dim {
                    s += w[i * dim + ki] * state[ki * dim + c];
                }
                new_v[i * dim + c] -= s;
            }
        }

        // o = exp(cu_i).q_i.S_prev + sum_{j<=i} (q_i.k_j) exp(cu_i-cu_j) new_v_j
        for i in 0..n {
            let ecu = cu[i].exp();
            for c in 0..dim {
                let mut s = 0.0f32;
                for ki in 0..dim {
                    s += qh[i * dim + ki] * state[ki * dim + c];
                }
                out[(b0 + i) * dim + c] = ecu * s;
            }
            for j in 0..=i {
                let m = dot(&qh[i * dim..(i + 1) * dim], &kh[j * dim..(j + 1) * dim])
                    * (cu[i] - cu[j]).exp();
                for c in 0..dim {
                    out[(b0 + i) * dim + c] += m * new_v[j * dim + c];
                }
            }
        }

        // S = exp(cu_last).S_prev + sum_i exp(cu_last - cu_i) k_i (x) new_v_i
        let cul = cu[n - 1];
        let ecul = cul.exp();
        for ki in 0..dim {
            for c in 0..dim {
                state[ki * dim + c] *= ecul;
            }
        }
        for i in 0..n {
            let e = (cul - cu[i]).exp();
            for ki in 0..dim {
                let kk = e * kh[i * dim + ki];
                for c in 0..dim {
                    state[ki * dim + c] += kk * new_v[i * dim + c];
                }
            }
        }
    }
    out
}

/// In-place partial NeoX RoPE on one token's `[heads, d]`: rotate the first
/// `rot` dims with half-split pairing at position `pos`.
pub fn rope_neox(x: &mut [f32], d: usize, rot: usize, pos: usize, theta: f32) {
    let heads = x.len() / d;
    let half = rot / 2;
    for h in 0..heads {
        for j in 0..half {
            let inv_freq = theta.powf(-2.0 * j as f32 / rot as f32);
            let ang = pos as f32 * inv_freq;
            let (sin, cos) = ang.sin_cos();
            let lo = x[h * d + j];
            let hi = x[h * d + half + j];
            x[h * d + j] = lo * cos - hi * sin;
            x[h * d + half + j] = hi * cos + lo * sin;
        }
    }
}

/// Dequantizes MLX affine 4-bit codes: element `c` of row `r` is nibble
/// `c % 8` (low nibble first) of `codes[r][c / 8]`, dequantized as
/// `scales[r][c/group_size] * q + biases[r][c/group_size]` in f32 — the exact
/// arithmetic `mx.dequantize` performs before its final rounding (pinned by
/// `tests/goldens/quant_fixture.json`).
pub fn dequant_q4(
    codes: &[u32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<f32> {
    dequant_affine(codes, scales, biases, rows, cols, group_size, 4)
}

/// [`dequant_q4`] generalized over the bit width (4 or 8; MoE routers are
/// 8-bit): element `c` is bits `bits*(c % per_word)` of `codes[r][c /
/// per_word]`, low element first.
#[allow(clippy::too_many_arguments)]
pub fn dequant_affine(
    codes: &[u32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    let per_word = 32 / bits;
    let mask = (1u32 << bits) - 1;
    assert_eq!(cols % per_word, 0, "cols must pack whole u32 words");
    assert_eq!(cols % group_size, 0, "cols must be a multiple of group_size");
    let words = cols / per_word;
    let groups = cols / group_size;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let word = codes[r * words + c / per_word];
            let q = (word >> (bits * (c % per_word))) & mask;
            let g = r * groups + c / group_size;
            out[r * cols + c] = scales[g] * q as f32 + biases[g];
        }
    }
    out
}

/// Asserts element-wise `|a - e| <= atol + rtol * |e|`, reporting the first
/// offending index.
pub fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tol = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "mismatch at {i}: actual={a} expected={e} (tol {tol})"
        );
    }
}
