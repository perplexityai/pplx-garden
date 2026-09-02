// Gated DeltaNet recurrence, causal depthwise conv1d, and gated RMSNorm.
#include <metal_stdlib>
using namespace metal;

#define DIM 128

// One GDN decode step per value head; each thread owns one state column.
// Value head h uses key head h / vpk; state is FP32 [H, DIM, DIM].
template <typename StateT>
static inline float gdn_step_body(device const bfloat* q,
                                 device const bfloat* k,
                                 device const bfloat* v,
                                 device const bfloat* a,
                                 device const bfloat* b,
                                 device const float* a_log,
                                 device const bfloat* dt_bias,
                                 device StateT* state,
                                 float scale,
                                 uint vpk,
                                 uint h,
                                 uint tid,
                                 uint sg,
                                 uint lane,
                                 threadgroup float* q_norm,
                                 threadgroup float* k_norm,
                                 threadgroup float* part_q,
                                 threadgroup float* part_k,
                                 threadgroup float* gates) {

    const uint hk = h / vpk;
    float qv = float(q[hk * DIM + tid]);
    float kv = float(k[hk * DIM + tid]);
    float sq = simd_sum(qv * qv);
    float sk = simd_sum(kv * kv);
    if (lane == 0) {
        part_q[sg] = sq;
        part_k[sg] = sk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tq = 0.0f;
        float tk = 0.0f;
        for (uint i = 0; i < DIM / 32; ++i) {
            tq += part_q[i];
            tk += part_k[i];
        }
        // Keep zero rows finite.
        gates[0] = 1.0f / max(sqrt(tq), 1e-12f);
        gates[1] = 1.0f / max(sqrt(tk), 1e-12f);
        // Stable softplus.
        float x = float(a[h]) + float(dt_bias[h]);
        float softplus = x > 20.0f ? x : log(1.0f + exp(x));
        gates[2] = exp(-exp(a_log[h]) * softplus);
        gates[3] = 1.0f / (1.0f + exp(-float(b[h])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q_norm[tid] = qv * gates[0] * scale;
    k_norm[tid] = kv * gates[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float decay = gates[2];
    float beta = gates[3];
    device StateT* st = state + (ulong)h * DIM * DIM;

    // Predict from the fully decayed state before applying the delta update.
    float kv_pred = 0.0f;
    for (uint ki = 0; ki < DIM; ++ki) {
        kv_pred += k_norm[ki] * st[ki * DIM + tid] * decay;
    }
    float v_new = (float(v[h * DIM + tid]) - kv_pred) * beta;
    float o = 0.0f;
    for (uint ki = 0; ki < DIM; ++ki) {
        float updated = st[ki * DIM + tid] * decay + k_norm[ki] * v_new;
        st[ki * DIM + tid] = StateT(updated);
        o += q_norm[ki] * updated;
    }
    return o;
}

#define GDN_STEP_GATED_WRAPPER(NAME, STATE_T)                                 \
kernel void NAME(device const bfloat* q       [[buffer(0)]],                  \
                 device const bfloat* k       [[buffer(1)]],                  \
                 device const bfloat* v       [[buffer(2)]],                  \
                 device const bfloat* a       [[buffer(3)]],                  \
                 device const bfloat* b       [[buffer(4)]],                  \
                 device const float*  a_log   [[buffer(5)]],                  \
                 device const bfloat* dt_bias [[buffer(6)]],                  \
                 device STATE_T*      state   [[buffer(7)]],                  \
                 device const bfloat* z       [[buffer(8)]],                  \
                 device const float*  norm_w  [[buffer(9)]],                  \
                 device bfloat*       out     [[buffer(10)]],                 \
                 constant float&      scale   [[buffer(11)]],                 \
                 constant uint&       vpk     [[buffer(12)]],                 \
                 constant float&      eps     [[buffer(13)]],                 \
                 uint h    [[threadgroup_position_in_grid]],                  \
                 uint tid  [[thread_index_in_threadgroup]],                   \
                 uint sg   [[simdgroup_index_in_threadgroup]],                \
                 uint lane [[thread_index_in_simdgroup]]) {                   \
    threadgroup float q_norm[DIM];                                            \
    threadgroup float k_norm[DIM];                                            \
    threadgroup float part_q[DIM / 32];                                       \
    threadgroup float part_k[DIM / 32];                                       \
    threadgroup float gates[4];                                               \
    threadgroup bfloat raw[DIM];                                              \
    threadgroup float norm_part[DIM / 32];                                    \
    threadgroup float inv_rms = 0.0f;                                         \
    const float o = gdn_step_body(q, k, v, a, b, a_log, dt_bias, state,      \
                                  scale, vpk, h, tid, sg, lane, q_norm,       \
                                  k_norm, part_q, part_k, gates);             \
    raw[tid] = bfloat(o);                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const float rv = float(raw[tid]);                                         \
    const float ss = simd_sum(rv * rv);                                       \
    if (lane == 0) {                                                          \
        norm_part[sg] = ss;                                                   \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    if (tid == 0) {                                                           \
        float total = 0.0f;                                                   \
        for (uint i = 0; i < DIM / 32; ++i) {                                \
            total += norm_part[i];                                            \
        }                                                                     \
        inv_rms = rsqrt(total / float(DIM) + eps);                            \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const float gate = float(z[h * DIM + tid]);                               \
    const float silu = gate / (1.0f + exp(-gate));                            \
    out[h * DIM + tid] = bfloat(norm_w[tid] * rv * inv_rms * silu);           \
}

GDN_STEP_GATED_WRAPPER(gdn_step_gated, float)
#undef GDN_STEP_GATED_WRAPPER

// Normalizes q/k rows for the register scan.
kernel void gdn_qk_l2norm(device const bfloat* qkv   [[buffer(0)]],  // [M, C]
                          device bfloat*       qk    [[buffer(1)]],  // [M, 2*HK*DIM]
                          constant float&      scale [[buffer(2)]],
                          constant uint&       HK    [[buffer(3)]],
                          constant uint&       H     [[buffer(4)]],
                          uint3 gid  [[thread_position_in_grid]],
                          uint  lane [[thread_index_in_simdgroup]]) {
    const uint hk = gid.y;
    const ulong row = (ulong)gid.z * (2 * HK + H) * DIM;
    const ulong out_row = (ulong)gid.z * 2 * HK * DIM;
    float qv[DIM / 32], kv[DIM / 32];
    float sq = 0.0f, sk = 0.0f;
    for (uint i = 0; i < DIM / 32; ++i) {
        uint d = lane + 32 * i;
        qv[i] = float(qkv[row + hk * DIM + d]);
        kv[i] = float(qkv[row + (HK + hk) * DIM + d]);
        sq += qv[i] * qv[i];
        sk += kv[i] * kv[i];
    }
    float inv_q = 1.0f / max(sqrt(simd_sum(sq)), 1e-12f);
    float inv_k = 1.0f / max(sqrt(simd_sum(sk)), 1e-12f);
    for (uint i = 0; i < DIM / 32; ++i) {
        uint d = lane + 32 * i;
        qk[out_row + hk * DIM + d] = bfloat(qv[i] * inv_q * scale);
        qk[out_row + (HK + hk) * DIM + d] = bfloat(kv[i] * inv_k);
    }
}

// Computes FP32 decay and beta gates.
kernel void gdn_gates(device const bfloat* a       [[buffer(0)]],  // [M, H]
                      device const bfloat* b       [[buffer(1)]],  // [M, H]
                      device const float*  a_log   [[buffer(2)]],  // [H]
                      device const bfloat* dt_bias [[buffer(3)]],  // [H]
                      device float*        decay   [[buffer(4)]],  // [M, H]
                      device float*        beta    [[buffer(5)]],  // [M, H]
                      constant uint&       H       [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
    uint h = gid % H;
    float x = float(a[gid]) + float(dt_bias[h]);
    float softplus = x > 20.0f ? x : log(1.0f + exp(x));
    decay[gid] = exp(-exp(a_log[h]) * softplus);
    beta[gid] = 1.0f / (1.0f + exp(-float(b[gid])));
}

// Register-resident prefill scan over tokens, one simdgroup per value column.
template <typename StateT>
static inline void gdn_prefill_regscan_body(device const bfloat* qkv,
                                            device const bfloat* qk,
                                            device const float* decay,
                                            device const float* beta,
                                            device StateT* state,
                                            device bfloat* out,
                                            uint M,
                                            uint H,
                                            uint vpk,
                                            uint2 tg,
                                            uint sg,
                                            uint lane) {
    const uint NK = DIM / 32;  // state rows per lane
    const uint h = tg.x;
    const uint dv = tg.y * 4 + sg;
    const uint HK = H / vpk;
    const uint hk = h / vpk;
    const uint C = (2 * HK + H) * DIM;

    // Each lane owns NK rows of one state column.
    device StateT* st = state + ((ulong)h * DIM + NK * lane) * DIM + dv;
    float s[NK];
    for (uint i = 0; i < NK; ++i) {
        s[i] = st[i * DIM];
    }

    for (uint t = 0; t < M; ++t) {
        device const bfloat* qrow =
            qk + (ulong)t * 2 * HK * DIM + hk * DIM + NK * lane;
        device const bfloat* krow = qrow + HK * DIM;
        float g = decay[t * H + h];
        float kh[NK];
        float kv = 0.0f;
        for (uint i = 0; i < NK; ++i) {
            kh[i] = float(krow[i]);
            s[i] *= g;
            kv += kh[i] * s[i];
        }
        kv = simd_sum(kv);
        float v_new = (float(qkv[(ulong)t * C + (2 * HK + h) * DIM + dv]) - kv)
            * beta[t * H + h];
        float o = 0.0f;
        for (uint i = 0; i < NK; ++i) {
            s[i] += kh[i] * v_new;
            o += float(qrow[i]) * s[i];
        }
        o = simd_sum(o);
        if (lane == 0) {
            out[((ulong)t * H + h) * DIM + dv] = bfloat(o);
        }
    }
    for (uint i = 0; i < NK; ++i) {
        st[i * DIM] = StateT(s[i]);
    }
}

#define GDN_REGSCAN_WRAPPER(NAME, STATE_T)                                    \
kernel void NAME(device const bfloat* qkv   [[buffer(0)]],                    \
                 device const bfloat* qk    [[buffer(1)]],                    \
                 device const float*  decay [[buffer(2)]],                    \
                 device const float*  beta  [[buffer(3)]],                    \
                 device STATE_T*      state [[buffer(4)]],                    \
                 device bfloat*       out   [[buffer(5)]],                    \
                 constant uint&       M     [[buffer(6)]],                    \
                 constant uint&       H     [[buffer(7)]],                    \
                 constant uint&       vpk   [[buffer(8)]],                    \
                 uint2 tg   [[threadgroup_position_in_grid]],                 \
                 uint  sg   [[simdgroup_index_in_threadgroup]],               \
                 uint  lane [[thread_index_in_simdgroup]]) {                  \
    gdn_prefill_regscan_body(qkv, qk, decay, beta, state, out, M, H, vpk, tg,\
                             sg, lane);                                       \
}

GDN_REGSCAN_WRAPPER(gdn_prefill_regscan, float)
#undef GDN_REGSCAN_WRAPPER

// Causal depthwise conv1d + SiLU; window stores KD-1 inputs oldest first.
kernel void conv1d_step_bf16(device bfloat*       window [[buffer(0)]],  // [C, KD-1]
                             device const bfloat* x      [[buffer(1)]],  // [C]
                             device const bfloat* w      [[buffer(2)]],  // [KD, C]
                             device bfloat*       out    [[buffer(3)]],  // [C]
                             constant uint&       C      [[buffer(4)]],
                             constant uint&       KD     [[buffer(5)]],
                             uint c [[thread_position_in_grid]]) {
    uint taps = KD - 1;
    float acc = 0.0f;
    for (uint t = 0; t < taps; ++t) {
        acc += float(window[c * taps + t]) * float(w[t * C + c]);
    }
    float xc = float(x[c]);
    acc += xc * float(w[taps * C + c]);
    out[c] = bfloat(acc / (1.0f + exp(-acc)));
    for (uint t = 0; t + 1 < taps; ++t) {
        window[c * taps + t] = window[c * taps + t + 1];
    }
    window[c * taps + taps - 1] = bfloat(xc);
}

// Tiled conv1d prefill uses separate input/output windows to avoid races.
#define CONV_TILE 64  // must match CONV1D_PREFILL_TILE in kernels/gdn.rs

kernel void conv1d_prefill_bf16(device const bfloat* win_in  [[buffer(0)]],  // [C, KD-1]
                                device bfloat*       win_out [[buffer(1)]],  // [C, KD-1]
                                device const bfloat* x       [[buffer(2)]],  // [M, C]
                                device const bfloat* w       [[buffer(3)]],  // [KD, C]
                                device bfloat*       out     [[buffer(4)]],  // [M, C]
                                constant uint&       C       [[buffer(5)]],
                                constant uint&       KD      [[buffer(6)]],
                                constant uint&       M       [[buffer(7)]],
                                uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint m0 = gid.y * CONV_TILE;
    uint m1 = min(m0 + CONV_TILE, M);
    uint taps = KD - 1;
    // Prime the register window with inputs preceding this tile.
    float win[8];  // supports KD <= 9; Qwen3.5 uses KD = 4
    for (uint t = 0; t < taps; ++t) {
        uint i = m0 + t;
        win[t] = i >= taps ? float(x[(ulong)(i - taps) * C + c])
                           : float(win_in[c * taps + i]);
    }
    for (uint m = m0; m < m1; ++m) {
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
    // The final tile writes the post-chunk window.
    if (m1 == M) {
        for (uint t = 0; t < taps; ++t) {
            win_out[c * taps + t] = bfloat(win[t]);
        }
    }
}

// Gated RMSNorm, one threadgroup per row.
#define TG 256

kernel void gated_rmsnorm_bf16(device const bfloat* x    [[buffer(0)]],  // [rows, D]
                               device const bfloat* gate [[buffer(1)]],  // [rows, D]
                               device const float*  w    [[buffer(2)]],  // [D]
                               device bfloat*       out  [[buffer(3)]],  // [rows, D]
                               constant uint&       D    [[buffer(4)]],
                               constant float&      eps  [[buffer(5)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;

    float acc = 0.0f;
    for (uint i = tid; i < D; i += TG) {
        float v = float(x[row * D + i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < D; i += TG) {
        float g = float(gate[row * D + i]);
        float s = g / (1.0f + exp(-g));
        out[row * D + i] = bfloat(w[i] * float(x[row * D + i]) * inv_rms * s);
    }
}
