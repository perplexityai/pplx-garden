// Sparse-MoE routing, expert projection, and weighted combine.

#include <metal_stdlib>
using namespace metal;

constant uint MOE_MAX_E = 1024;
constant uint MOE_MAX_K = 16;

// Routes logits [E] to K experts; ties select the lowest expert id.
// Scores use full softmax unless renorm restricts it to selected logits.
kernel void moe_router_topk(device const float* logits  [[buffer(0)]],
                            device uint*        indices [[buffer(1)]],
                            device float*       scores  [[buffer(2)]],
                            constant uint&      E       [[buffer(3)]],
                            constant uint&      K       [[buffer(4)]],
                            constant uint&      renorm  [[buffer(5)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint tg_size [[threads_per_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float probs[MOE_MAX_E];
    threadgroup float red[32];
    threadgroup float sel_val[MOE_MAX_K];
    threadgroup uint  sel_idx[MOE_MAX_K];

    // Full softmax cancels when selected logits are renormalized.
    if (renorm != 0) {
        for (uint i = tid; i < E; i += tg_size) {
            probs[i] = logits[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {

    float local_max = -INFINITY;
    for (uint i = tid; i < E; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    local_max = simd_max(local_max);
    if (lane == 0) {
        red[sg] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            m = max(m, red[i]);
        }
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gmax = red[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < E; i += tg_size) {
        float e = exp(logits[i] - gmax);
        probs[i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        red[sg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            s += red[i];
        }
        red[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / red[0];
    for (uint i = tid; i < E; i += tg_size) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Repeated argmax masks each selected expert before the next round.
    threadgroup float win_val[32];
    threadgroup uint  win_idx[32];
    for (uint round = 0; round < K; ++round) {
        float best = -INFINITY;
        uint best_i = 0xFFFFFFFF;
        for (uint i = tid; i < E; i += tg_size) {
            float p = probs[i];
            if (p > best || (p == best && i < best_i)) {
                best = p;
                best_i = i;
            }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best, off);
            uint oi = simd_shuffle_down(best_i, off);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            win_val[sg] = best;
            win_idx[sg] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = -INFINITY;
            uint bi = 0xFFFFFFFF;
            for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
                if (win_val[i] > b || (win_val[i] == b && win_idx[i] < bi)) {
                    b = win_val[i];
                    bi = win_idx[i];
                }
            }
            sel_val[round] = b;
            sel_idx[round] = bi;
            // NaN cannot collide with a valid -inf logit on the raw path.
            probs[bi] = NAN;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        if (renorm != 0) {
            float mx = sel_val[0];
            float denom = 0.0f;
            for (uint j = 0; j < K; ++j) {
                denom += exp(sel_val[j] - mx);
            }
            for (uint j = 0; j < K; ++j) {
                indices[j] = sel_idx[j];
                scores[j] = exp(sel_val[j] - mx) / denom;
            }
        } else {
            for (uint j = 0; j < K; ++j) {
                indices[j] = sel_idx[j];
                scores[j] = sel_val[j];
            }
        }
    }
}

// Fuses selected-expert gate/up projections and writes SwiGLU output.
kernel void moe_gather_gemv_q4_gate_up(
    device const uint* gate_codes [[buffer(0)]],
    device const bfloat* gate_scales [[buffer(1)]],
    device const bfloat* gate_biases [[buffer(2)]],
    device const uint* up_codes [[buffer(3)]],
    device const bfloat* up_scales [[buffer(4)]],
    device const bfloat* up_biases [[buffer(5)]],
    device const bfloat* x [[buffer(6)]],
    device const uint* indices [[buffer(7)]],
    device bfloat* y [[buffer(8)]],
    constant uint& K [[buffer(9)]],
    constant uint& GS [[buffer(10)]],
    constant uint& N [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint row = tg.x;
    const uint slot = tg.y;
    const ulong grow = (ulong)indices[slot] * N + row;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    device const uint4* gate_wrow =
        (device const uint4*)(gate_codes + grow * words);
    device const uint4* up_wrow =
        (device const uint4*)(up_codes + grow * words);
    device const bfloat4* xv = (device const bfloat4*)x;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint i = lane; i < blocks; i += 32) {
        uint g = i / bpg;
        uint4 gate_w4 = gate_wrow[i];
        uint4 up_w4 = up_wrow[i];
        float gate_dot = 0.0f;
        float up_dot = 0.0f;
        float x_sum = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            float4 xlo = float4(xv[2 * (4 * i + j)]);
            float4 xhi = float4(xv[2 * (4 * i + j) + 1]);
            uint gw = gate_w4[j];
            uint uw = up_w4[j];
            float4 gqlo = float4(float((gw >> 0) & 0xF), float((gw >> 4) & 0xF),
                                 float((gw >> 8) & 0xF), float((gw >> 12) & 0xF));
            float4 gqhi = float4(float((gw >> 16) & 0xF), float((gw >> 20) & 0xF),
                                 float((gw >> 24) & 0xF), float((gw >> 28) & 0xF));
            float4 uqlo = float4(float((uw >> 0) & 0xF), float((uw >> 4) & 0xF),
                                 float((uw >> 8) & 0xF), float((uw >> 12) & 0xF));
            float4 uqhi = float4(float((uw >> 16) & 0xF), float((uw >> 20) & 0xF),
                                 float((uw >> 24) & 0xF), float((uw >> 28) & 0xF));
            gate_dot += dot(gqlo, xlo) + dot(gqhi, xhi);
            up_dot += dot(uqlo, xlo) + dot(uqhi, xhi);
            x_sum += dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
        }
        gate_sum += float(gate_scales[grow * groups + g]) * gate_dot
                  + float(gate_biases[grow * groups + g]) * x_sum;
        up_sum += float(up_scales[grow * groups + g]) * up_dot
                + float(up_biases[grow * groups + g]) * x_sum;
    }
    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);
    if (lane == 0) {
        gate_sum = float(bfloat(gate_sum));
        up_sum = float(bfloat(up_sum));
        y[(ulong)slot * N + row] =
            bfloat((gate_sum / (1.0f + exp(-gate_sum))) * up_sum);
    }
}

kernel void moe_gather_gemv_q4_down_combine(
    device const uint*   codes       [[buffer(0)]],
    device const bfloat* scales      [[buffer(1)]],
    device const bfloat* biases      [[buffer(2)]],
    device const bfloat* x           [[buffer(3)]],
    device const uint*   indices     [[buffer(4)]],
    device const float*  scores      [[buffer(5)]],
    device const bfloat* shared_out  [[buffer(6)]],
    device const float*  shared_gate [[buffer(7)]],
    device bfloat*       out         [[buffer(8)]],
    constant uint&       K           [[buffer(9)]],
    constant uint&       GS          [[buffer(10)]],
    constant uint&       H           [[buffer(11)]],
    constant uint&       S           [[buffer(12)]],
    constant uint&       has_shared  [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_threadgroup]]) {
    const uint hlane = lane % 16;
    const uint row = group * 2 + lane / 16;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    float combined = 0.0f;
    for (uint slot = 0; slot < S; ++slot) {
        const ulong grow = (ulong)indices[slot] * H + row;
        device const uint4* wrow =
            (device const uint4*)(codes + grow * words);
        device const bfloat4* xv =
            (device const bfloat4*)(x + (ulong)slot * K);
        float sum = 0.0f;
        for (uint i = hlane; i < blocks; i += 16) {
            const uint g = i / bpg;
            const float scale = float(scales[grow * groups + g]);
            const float bias = float(biases[grow * groups + g]);
            const uint4 w4 = wrow[i];
            float qx = 0.0f;
            float xs = 0.0f;
            for (uint j = 0; j < 4; ++j) {
                const uint word = w4[j];
                const float4 xlo = float4(xv[2 * (4 * i + j)]);
                const float4 xhi = float4(xv[2 * (4 * i + j) + 1]);
                const float4 qlo =
                    float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                           float((word >> 8) & 0xF), float((word >> 12) & 0xF));
                const float4 qhi =
                    float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                           float((word >> 24) & 0xF), float((word >> 28) & 0xF));
                qx += dot(qlo, xlo) + dot(qhi, xhi);
                xs += dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
            }
            sum += scale * qx + bias * xs;
        }
        for (uint off = 8; off > 0; off >>= 1) {
            sum += simd_shuffle_down(sum, off);
        }
        if (hlane == 0) {
            combined += scores[slot] * float(bfloat(sum));
        }
    }
    if (hlane == 0) {
        if (has_shared != 0) {
            const float gate = 1.0f / (1.0f + exp(-shared_gate[0]));
            combined += gate * float(shared_out[row]);
        }
        out[row] = bfloat(combined);
    }
}

// Small-M GEMV reuses each selected expert across its routed rows.

// Masked Q4 nibbles are rescaled before affine dequantization.
static inline float2 qdot_word_masked(uint word, float4 xlo, float4 xhi) {
    const float4 inv =
        float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    const uint wl = word & 0xFFFFu;
    const uint wh = word >> 16;
    float4 ql = float4(float(wl & 0x000Fu), float(wl & 0x00F0u),
                       float(wl & 0x0F00u), float(wl & 0xF000u));
    float4 qh = float4(float(wh & 0x000Fu), float(wh & 0x00F0u),
                       float(wh & 0x0F00u), float(wh & 0xF000u));
    return float2(dot(ql, xlo * inv) + dot(qh, xhi * inv),
                  dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f)));
}

constant uint MOE_SMALLM_MAX_S = 256;

constant uint MOE_SMALLM_MAX_ROWS = 16;

// Reads expert ids and routed rows from the union map.
#define MOE_GEMV_SMALLM_EM_PROLOGUE(MAXR)                                      \
    const uint u = tg.y;                                                       \
    if (u >= min(umap[0], S)) {                                                \
        return;                                                                \
    }                                                                          \
    const uint e = umap[1 + u];                                                \
    const uint nr = min(umap[1 + MOE_SMALLM_MAX_S + u], (uint)MAXR);           \
    device const uint* upairs =                                                \
        umap + 1 + 2 * MOE_SMALLM_MAX_S + u * MOE_SMALLM_MAX_ROWS;             \
    uint pair[MAXR];                                                           \
    uint xbase[MAXR];                                                          \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint ri = 0; ri < MAXR; ++ri) {                                       \
        if (ri < nr) {                                                         \
            const uint jj = upairs[ri];                                        \
            pair[ri] = jj;                                                     \
            xbase[ri] = ((XPP != 0) ? jj : jj / TOPK) * (K / 4);               \
        }                                                                      \
    }

#define MOE_GEMV_SMALLM_EM_BODY(MAXR, R4)                               \
    MOE_GEMV_SMALLM_EM_PROLOGUE(MAXR)                                          \
    MOE_GEMV_SMALLM_CORE(MAXR, R4)

#define MOE_GEMV_SMALLM_CORE(MAXR, R4)                                  \
    const uint sg = lane / 32;                                                 \
    const uint sl = lane % 32;                                                 \
    const uint row0 = (tg.x * 2 + sg) * R4;                                    \
    const uint words = K / 8;                                                  \
    const uint blocks = K / 16;                                                \
    const uint bpg = GS / 16;                                                  \
    const uint groups = K / GS;                                                \
    device const bfloat4* xw = (device const bfloat4*)x;                       \
    device const uint2* wrow[R4];                                              \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint r4 = 0; r4 < R4; ++r4) {                                         \
        wrow[r4] =                                                             \
            (device const uint2*)(codes + ((ulong)e * N + row0 + r4) * words); \
    }                                                                          \
    float acc[MAXR][R4];                                                       \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint ri = 0; ri < MAXR; ++ri) {                                       \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint r4 = 0; r4 < R4; ++r4) {                                     \
            acc[ri][r4] = 0.0f;                                                \
        }                                                                      \
    }                                                                          \
    for (uint i = sl; i < blocks; i += 32) {                                   \
        const uint g = i / bpg;                                                \
        uint2 w2[R4];                                                          \
        float s[R4];                                                           \
        float b[R4];                                                           \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint r4 = 0; r4 < R4; ++r4) {                                     \
            const ulong grow = (ulong)e * N + row0 + r4;                       \
            w2[r4] = wrow[r4][i];                                              \
            s[r4] = float(scales[grow * groups + g]);                          \
            b[r4] = float(biases[grow * groups + g]);                          \
        }                                                                      \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint ri = 0; ri < MAXR; ++ri) {                                   \
            if (ri >= nr) {                                                    \
                break;                                                         \
            }                                                                  \
            device const bfloat4* xv = xw + xbase[ri] + 4 * i;                 \
            const float4 x0 = float4(xv[0]);                                   \
            const float4 x1 = float4(xv[1]);                                   \
            const float4 x2 = float4(xv[2]);                                   \
            const float4 x3 = float4(xv[3]);                                   \
            _Pragma("clang loop unroll(full)")                                 \
            for (uint r4 = 0; r4 < R4; ++r4) {                                 \
                float2 d = qdot_word_masked(w2[r4].x, x0, x1);                 \
                d += qdot_word_masked(w2[r4].y, x2, x3);                       \
                acc[ri][r4] += s[r4] * d.x + b[r4] * d.y;                      \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("clang loop unroll(full)")                                         \
    for (uint ri = 0; ri < MAXR; ++ri) {                                       \
        if (ri >= nr) {                                                        \
            break; /* nr is threadgroup-uniform, so reductions stay whole */   \
        }                                                                      \
        _Pragma("clang loop unroll(full)")                                     \
        for (uint r4 = 0; r4 < R4; ++r4) {                                     \
            const float r = simd_sum(acc[ri][r4]);                             \
            if (sl == 0) {                                                     \
                y[(ulong)pair[ri] * N + row0 + r4] = bfloat(r);                \
            }                                                                  \
        }                                                                      \
    }

// Builds an expert-major union map; S must not exceed MOE_SMALLM_MAX_S.
// One pair per thread avoids a macOS 27 AGX compiler crash.
kernel void moe_union_experts(device const uint* indices [[buffer(0)]],
                              device uint*       umap    [[buffer(1)]],
                              constant uint&     S       [[buffer(2)]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint pairs[MOE_SMALLM_MAX_S];
    threadgroup uint first[MOE_SMALLM_MAX_S];
    threadgroup uint slot[MOE_SMALLM_MAX_S];
    if (tid < S) {
        pairs[tid] = indices[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < S) {
        uint f = 1;
        for (uint jj = 0; jj < tid; ++jj) {
            if (pairs[jj] == pairs[tid]) {
                f = 0;
                break;
            }
        }
        first[tid] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint u = 0;
        for (uint j = 0; j < S; ++j) {
            slot[j] = u;
            u += first[j];
        }
        umap[0] = u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < S && first[tid] != 0) {
        const uint u = slot[tid];
        const uint e = pairs[tid];
        uint nr = 0;
        for (uint jj = tid; jj < S && nr < MOE_SMALLM_MAX_ROWS; ++jj) {
            if (pairs[jj] == e) {
                umap[1 + 2 * MOE_SMALLM_MAX_S + u * MOE_SMALLM_MAX_ROWS + nr] = jj;
                ++nr;
            }
        }
        umap[1 + u] = e;
        umap[1 + MOE_SMALLM_MAX_S + u] = nr;
    }
}

#define MOE_GEMV_SMALLM_EM_KERNEL(NAME, MAXR, R4)                              \
    kernel void NAME(device const uint*   codes   [[buffer(0)]],               \
                     device const bfloat* scales  [[buffer(1)]],               \
                     device const bfloat* biases  [[buffer(2)]],               \
                     device const bfloat* x       [[buffer(3)]],               \
                     device const uint*   umap    [[buffer(4)]],               \
                     device bfloat*       y       [[buffer(5)]],               \
                     constant uint&       K       [[buffer(6)]],               \
                     constant uint&       GS      [[buffer(7)]],               \
                     constant uint&       N       [[buffer(8)]],               \
                     constant uint&       S       [[buffer(9)]],               \
                     constant uint&       TOPK    [[buffer(10)]],              \
                     constant uint&       XPP     [[buffer(11)]],              \
                     uint2 tg  [[threadgroup_position_in_grid]],               \
                     uint lane [[thread_index_in_threadgroup]]) {              \
        MOE_GEMV_SMALLM_EM_BODY(MAXR, R4)                                    \
    }


MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r8, 8, 4)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r8_w, 8, 8)

// Adds sigmoid(gate[r]) * src[r, :] to each destination row.
kernel void moe_row_gate_add(device const bfloat* src  [[buffer(0)]],  // [m, h]
                             device const bfloat* gate [[buffer(1)]],  // [m]
                             device bfloat*       dst  [[buffer(2)]],  // [m, h]
                             constant uint&       H    [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint r = gid.y;
    if (col >= H) {
        return;
    }
    float g = 1.0f / (1.0f + exp(-float(gate[r])));
    ulong d = (ulong)r * H + col;
    dst[d] = bfloat(float(dst[d]) + g * float(src[d]));
}

// Combines expert outputs in ascending slot order.
kernel void moe_combine_rows(device const bfloat* ed     [[buffer(0)]],  // [S, H]
                             device const uint*   slots  [[buffer(1)]],  // [m, K]
                             device const float*  scores [[buffer(2)]],  // [m, K]
                             device bfloat*       out    [[buffer(3)]],  // [m, H]
                             constant uint&       H      [[buffer(4)]],
                             constant uint&       TOPK   [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const uint c4 = gid.x;
    const uint row = gid.y;
    if (c4 * 4 >= H) {
        return;
    }
    float4 acc = float4(0.0f);
    for (uint k = 0; k < TOPK; ++k) {
        uint slot = slots[row * TOPK + k];
        float s = scores[row * TOPK + k];
        acc += s * float4(((device const bfloat4*)(ed + (ulong)slot * H))[c4]);
    }
    ((device bfloat4*)(out + (ulong)row * H))[c4] = bfloat4(acc);
}

// Routes each BF16 logits row [E] to K experts.
kernel void moe_router_topk_rows(device const bfloat* logits  [[buffer(0)]],
                                 device uint*         indices [[buffer(1)]],
                                 device float*        scores  [[buffer(2)]],
                                 constant uint&       E       [[buffer(3)]],
                                 constant uint&       K       [[buffer(4)]],
                                 constant uint&       renorm  [[buffer(5)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float probs[MOE_MAX_E];
    threadgroup float red[32];
    threadgroup float sel_val[MOE_MAX_K];
    threadgroup uint  sel_idx[MOE_MAX_K];
    device const bfloat* row_logits = logits + (ulong)row * E;

    if (renorm != 0) {
        for (uint i = tid; i < E; i += tg_size) {
            probs[i] = float(row_logits[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {

    float local_max = -INFINITY;
    for (uint i = tid; i < E; i += tg_size) {
        local_max = max(local_max, float(row_logits[i]));
    }
    local_max = simd_max(local_max);
    if (lane == 0) {
        red[sg] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            m = max(m, red[i]);
        }
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gmax = red[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < E; i += tg_size) {
        float e = exp(float(row_logits[i]) - gmax);
        probs[i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        red[sg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
            s += red[i];
        }
        red[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / red[0];
    for (uint i = tid; i < E; i += tg_size) {
        probs[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float win_val[32];
    threadgroup uint  win_idx[32];
    for (uint round = 0; round < K; ++round) {
        float best = -INFINITY;
        uint best_i = 0xFFFFFFFF;
        for (uint i = tid; i < E; i += tg_size) {
            float p = probs[i];
            if (p > best || (p == best && i < best_i)) {
                best = p;
                best_i = i;
            }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best, off);
            uint oi = simd_shuffle_down(best_i, off);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            win_val[sg] = best;
            win_idx[sg] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = -INFINITY;
            uint bi = 0xFFFFFFFF;
            for (uint i = 0; i < (tg_size + 31) / 32; ++i) {
                if (win_val[i] > b || (win_val[i] == b && win_idx[i] < bi)) {
                    b = win_val[i];
                    bi = win_idx[i];
                }
            }
            sel_val[round] = b;
            sel_idx[round] = bi;
            probs[bi] = NAN;  // collision-free selection mask
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        if (renorm != 0) {
            float mx = sel_val[0];
            float denom = 0.0f;
            for (uint j = 0; j < K; ++j) {
                denom += exp(sel_val[j] - mx);
            }
            for (uint j = 0; j < K; ++j) {
                indices[(ulong)row * K + j] = sel_idx[j];
                scores[(ulong)row * K + j] = exp(sel_val[j] - mx) / denom;
            }
        } else {
            for (uint j = 0; j < K; ++j) {
                indices[(ulong)row * K + j] = sel_idx[j];
                scores[(ulong)row * K + j] = sel_val[j];
            }
        }
    }
}

kernel void fill_zero_u32(device uint* dst [[buffer(0)]],
                          constant uint& N [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < N) {
        dst[i] = 0;
    }
}

kernel void moe_histogram(device const uint*   indices [[buffer(0)]],  // [m*K]
                          device atomic_uint*  counts  [[buffer(1)]],  // [E]
                          constant uint&       S       [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < S) {
        atomic_fetch_add_explicit(&counts[indices[i]], 1u, memory_order_relaxed);
    }
}

// Scans slot and row-tile offsets; T must match the grouped GEMM tile height.
kernel void moe_scan_offsets(device const uint* counts       [[buffer(0)]],  // [E]
                             device uint*       offsets      [[buffer(1)]],  // [E+1]
                             device uint*       tile_offsets [[buffer(2)]],  // [E+1]
                             constant uint&     E            [[buffer(3)]],
                             constant uint&     T            [[buffer(4)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    uint acc = 0, tacc = 0;
    for (uint e = 0; e < E; ++e) {
        offsets[e] = acc;
        tile_offsets[e] = tacc;
        acc += counts[e];
        tacc += (counts[e] + T - 1) / T;
    }
    offsets[E] = acc;
    tile_offsets[E] = tacc;
}

// Scatters slots into expert-major order; within-expert order is unspecified.
kernel void moe_scatter_slots(device const uint*  indices   [[buffer(0)]],  // [m, K]
                              device const uint*  offsets   [[buffer(1)]],  // [E+1]
                              device atomic_uint* cursors   [[buffer(2)]],  // [E]
                              device uint*        ids_sorted [[buffer(3)]], // [S]
                              device uint*        slot_of   [[buffer(4)]],  // [m, K]
                              constant uint&      K         [[buffer(5)]],
                              constant uint&      S         [[buffer(6)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= S) {
        return;
    }
    uint e = indices[i];
    uint pos = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
    uint slot = offsets[e] + pos;
    ids_sorted[slot] = i / K;
    slot_of[i] = slot;
}

// Builds grouped-GEMM blocks; T must match the offset scan.
kernel void moe_build_blocks(device const uint* offsets      [[buffer(0)]],  // [E+1]
                             device const uint* tile_offsets [[buffer(1)]],  // [E+1]
                             device uint4*      blocks       [[buffer(2)]],
                             constant uint&     E            [[buffer(3)]],
                             constant uint&     N_PER        [[buffer(4)]],
                             constant uint&     T            [[buffer(5)]],
                             constant uint&     ORDER        [[buffer(6)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.x;
    const uint n_tile = gid.y;
    const uint n_tiles = N_PER / 64;
    const uint total = tile_offsets[E];
    ulong out = (ulong)t * n_tiles + n_tile;
    if (t >= total) {
        blocks[out] = uint4(0);
        return;
    }
    // Find the expert containing tile t.
    uint lo = 0, hi = E;
    while (lo + 1 < hi) {
        uint mid = (lo + hi) / 2;
        if (tile_offsets[mid] <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const uint e = lo;
    const uint tile_in_e = t - tile_offsets[e];
    if (ORDER == 1) {
        // ORDER=1 makes routed M tiles fastest-varying.
        const uint expert_tiles = tile_offsets[e + 1] - tile_offsets[e];
        out = (ulong)tile_offsets[e] * n_tiles
            + (ulong)n_tile * expert_tiles + tile_in_e;
    }
    blocks[out] = uint4(offsets[e] + tile_in_e * T,
                        e * N_PER + n_tile * 64,
                        n_tile * 64,
                        offsets[e + 1]);
}
