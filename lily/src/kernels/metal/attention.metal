// RoPE, q/gate splitting, KV-cache append, and scaled dot-product attention.
#include <metal_stdlib>
using namespace metal;

#define TG 256
#define MAX_T 4096

// In-place partial NeoX RoPE over [M, H, D]; grid is (H * rot/2, M).
kernel void rope_neox_bf16(device bfloat*  x        [[buffer(0)]],
                           constant uint&  D        [[buffer(1)]],
                           constant uint&  rot      [[buffer(2)]],
                           constant uint&  base_pos [[buffer(3)]],
                           constant float& theta    [[buffer(4)]],
                           constant uint&  row_elems [[buffer(5)]],  // H * D
                           uint2 gid [[thread_position_in_grid]]) {
    uint half_rot = rot / 2;
    uint h = gid.x / half_rot;
    uint j = gid.x % half_rot;
    float inv_freq = pow(theta, -2.0f * float(j) / float(rot));
    float ang = float(base_pos + gid.y) * inv_freq;
    float c = cos(ang);
    float s = sin(ang);
    ulong base = (ulong)gid.y * row_elems + (ulong)h * D;
    float lo = float(x[base + j]);
    float hi = float(x[base + half_rot + j]);
    x[base + j] = bfloat(lo * c - hi * s);
    x[base + half_rot + j] = bfloat(hi * c + lo * s);
}

// Splits per-head [q(D) | gate(D)] rows into separate tensors.
kernel void split_q_gate_bf16(device const bfloat* qg   [[buffer(0)]],
                              device bfloat*       q    [[buffer(1)]],
                              device bfloat*       gate [[buffer(2)]],
                              constant uint&       D    [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    uint h = gid / D;
    uint d = gid % D;
    q[gid] = qg[(ulong)h * 2 * D + d];
    gate[gid] = qg[(ulong)h * 2 * D + D + d];
}

// Appends [M, KVH, D] rows to cache [KVH, MAX, D] at base_pos.
kernel void scatter_kv_bf16(device bfloat*       cache [[buffer(0)]],
                            device const bfloat* rows  [[buffer(1)]],
                            constant uint&       D     [[buffer(2)]],
                            constant uint&       max_seq [[buffer(3)]],
                            constant uint&       base_pos [[buffer(4)]],
                            constant uint&       kvh   [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    uint d = gid % D;
    uint h = (gid / D) % kvh;
    uint m = gid / (D * kvh);
    cache[((ulong)h * max_seq + base_pos + m) * D + d] = rows[gid];
}

kernel void q_norm_rope_split_decode_bf16(
    device const bfloat* qg   [[buffer(0)]],
    device const bfloat* w    [[buffer(1)]],
    device bfloat*       q    [[buffer(2)]],
    device bfloat*       gate [[buffer(3)]],
    constant uint&       D    [[buffer(4)]],
    constant uint&       rot  [[buffer(5)]],
    constant uint&       pos  [[buffer(6)]],
    constant float&      theta [[buffer(7)]],
    constant float&      eps   [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    threadgroup float inv_rms = 0.0f;
    threadgroup bfloat normed[256];
    const ulong src = (ulong)head * 2 * D;
    const ulong dst = (ulong)head * D;
    const float value = float(qg[src + tid]);
    float acc = value * value;
    gate[dst + tid] = qg[src + D + tid];
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = float(pos) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        q[dst + tid] = bfloat(lo * c - hi * s);
        q[dst + half_rot + tid] = bfloat(hi * c + lo * s);
    } else if (tid >= rot) {
        q[dst + tid] = normed[tid];
    }
}

kernel void k_norm_rope_scatter_decode_bf16(
    device const bfloat* k     [[buffer(0)]],
    device const bfloat* w     [[buffer(1)]],
    device bfloat*       cache [[buffer(2)]],
    constant uint&       D     [[buffer(3)]],
    constant uint&       rot   [[buffer(4)]],
    constant uint&       pos   [[buffer(5)]],
    constant float&      theta [[buffer(6)]],
    constant float&      eps   [[buffer(7)]],
    constant uint&       max_seq [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    threadgroup float inv_rms = 0.0f;
    threadgroup bfloat normed[256];
    const ulong src = (ulong)head * D;
    const ulong dst = ((ulong)head * max_seq + pos) * D;
    const float value = float(k[src + tid]);
    float acc = value * value;
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = bfloat(value * inv_rms * (1.0f + float(w[tid])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot / 2;
    if (tid < half_rot) {
        const float inv_freq = pow(theta, -2.0f * float(tid) / float(rot));
        const float angle = float(pos) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = float(normed[tid]);
        const float hi = float(normed[half_rot + tid]);
        cache[dst + tid] = bfloat(lo * c - hi * s);
        cache[dst + half_rot + tid] = bfloat(hi * c + lo * s);
    } else if (tid >= rot) {
        cache[dst + tid] = normed[tid];
    }
}

// Causal prefill attention with one query tile and head per threadgroup.
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

constant constexpr int FA_D = 256;   // head dim (the model's, fixed)
constant constexpr int FA_SG = 4;    // simdgroups per threadgroup

template <int BQ, int BK, int SG = FA_SG, bool QDEV = false>
static void sdpa_nax_body(device bfloat* q,
                          device bfloat* k_cache,
                          device bfloat* v_cache,
                          device bfloat* out,
                          uint max_seq,
                          uint base_len,
                          uint M,
                          uint nq,
                          uint group,
                          float scale,
                          uint parallel_softmax,
                          uint descending_blocks,
                          threadgroup bfloat* q_tile,
                          threadgroup float* s_tile,
                          threadgroup bfloat* p_tile,
                          threadgroup float* row_max,
                          threadgroup float* row_sum,
                          threadgroup float* row_alpha,
                          uint2 tg,
                          uint tid,
                          uint2 tg_size) {
    constexpr uint LANES = uint(32 * SG) / uint(BQ);
    static_assert(BQ * int(LANES) == 32 * SG,
                  "the threadgroup must divide into BQ equal row groups");
    static_assert(BK % int(LANES) == 0,
                  "each lane must own a whole number of key columns");
    static_assert(LANES <= 32, "a row's lanes must sit inside one simdgroup");

    using namespace mpp::tensor_ops;

    const uint qb = descending_blocks != 0u
        ? (uint(M + BQ - 1) / BQ) - 1u - tg.x
        : tg.x;
    const int q0 = int(qb) * BQ;
    const uint hq = tg.y;
    const int len_total = int(base_len) + int(M);

    // Tile layout requires SG * 32 threads.
    if (tg_size.x != uint(32 * SG)) {
        return;
    }
    if (!QDEV) {
        for (uint i = tid; i < uint(BQ * FA_D); i += 32 * SG) {
            uint r = i / FA_D, d = i % FA_D;
            q_tile[i] = (uint(q0) + r) < M
                ? q[((ulong)(q0 + r) * nq + hq) * FA_D + d]
                : bfloat(0.0f);
        }
    }
    if (tid < uint(BQ)) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device bfloat* k_head = k_cache + (ulong)(hq / group) * max_seq * FA_D;
    device bfloat* v_head = v_cache + (ulong)(hq / group) * max_seq * FA_D;
    auto tQ = [&]() {
        if constexpr (QDEV) {
            const array<int, 2> qs{1, int(nq * uint(FA_D))};
            return tensor(q + (ulong)(uint(q0) * nq + hq) * FA_D,
                          dextents<int32_t, 2>(FA_D, min(BQ, int(M) - q0)), qs);
        } else {
            return tensor(q_tile, dextents<int32_t, 2>(FA_D, BQ));
        }
    }();
    auto tK = tensor(k_head, dextents<int32_t, 2>(FA_D, len_total));
    auto tV = tensor(v_head, dextents<int32_t, 2>(FA_D, len_total));
    auto tS = tensor(s_tile, dextents<int32_t, 2>(BK, BQ));
    auto tP = tensor(p_tile, dextents<int32_t, 2>(BK, BQ));

    constexpr auto qk_desc = matmul2d_descriptor(
        BQ, BK, FA_D, false, /*transpose_right=*/true, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        BQ, FA_D, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, metal::execution_simdgroups<SG>> qk_op;
    matmul2d<pv_desc, metal::execution_simdgroups<SG>> pv_op;

    using KSlice = decltype(tK.slice(0, 0));
    using VSlice = decltype(tV.slice(0, 0));

    auto acc = pv_op.template get_destination_cooperative_tensor<decltype(tP), VSlice, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    const int k_end = min(len_total, int(base_len) + q0 + BQ);
    for (int k0 = 0; k0 < k_end; k0 += BK) {
        const int kk = k0;
        auto kSlice = tK.slice(0, kk);
        auto vSlice = tV.slice(0, kk);
        auto sT = qk_op.template get_destination_cooperative_tensor<decltype(tQ), KSlice, float>();
        qk_op.run(tQ, kSlice, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The parallel reduction is not bit-identical to the serial path.
        if (parallel_softmax != 0u) {
            const uint row = tid / LANES;
            const uint lane8 = tid % LANES;
            const uint cols = uint(BK) / LANES;
            const uint j0 = lane8 * cols;
            const int grow = q0 + int(row);
            const int limit =
                min(min(int(base_len) + grow + 1, len_total) - k0, BK);
            const float prev_max = row_max[row];
            float local = prev_max;
            for (uint j = j0; j < j0 + cols; ++j) {
                if (int(j) < limit) {
                    local = max(local, s_tile[row * BK + j] * scale);
                }
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                local = max(local, simd_shuffle_xor(local, off));
            }
            const float mx = local;
            const float alpha = exp(prev_max - mx);
            float psum = 0.0f;
            for (uint j = j0; j < j0 + cols; ++j) {
                const float p = int(j) < limit
                    ? exp(s_tile[row * BK + j] * scale - mx)
                    : 0.0f;
                p_tile[row * BK + j] = bfloat(p);
                psum += p;
            }
            for (uint off = 1u; off < LANES; off <<= 1) {
                psum += simd_shuffle_xor(psum, off);
            }
            if (lane8 == 0u) {
                row_sum[row] = row_sum[row] * alpha + psum;
                row_max[row] = mx;
                row_alpha[row] = alpha;
            }
        } else if (tid < uint(BQ)) {
            int grow = q0 + int(tid);
            int limit = min(min(int(base_len) + grow + 1, len_total) - k0, BK);
            float mx = row_max[tid];
            for (int j = 0; j < limit; ++j) {
                mx = max(mx, s_tile[tid * BK + j] * scale);
            }
            float alpha = exp(row_max[tid] - mx);
            float s = row_sum[tid] * alpha;
            for (int j = 0; j < BK; ++j) {
                float p = j < limit
                    ? exp(s_tile[tid * BK + j] * scale - mx)
                    : 0.0f;
                p_tile[tid * BK + j] = bfloat(p);
                s += j < limit ? p : 0.0f;
            }
            row_max[tid] = mx;
            row_sum[tid] = s;
            row_alpha[tid] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) {
                    auto ix = acc.get_multidimensional_index(i);
                    acc[i] *= row_alpha[ix[1]];
                }
            }
        }
        pv_op.run(tP, vSlice, acc);
        // Finish tensor ops before reusing p_tile and s_tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            int row = ix[1], col = ix[0];
            if (q0 + row < int(M)) {
                out[((ulong)(q0 + row) * nq + hq) * FA_D + col] =
                    bfloat(acc[i] / row_sum[row]);
            }
        }
    }
}
// Device-Q variant; threadgroup memory holds only softmax state.
#define SDPA_NAX_KERNEL_QDEV(NAME, BQ, BK, SG)                                  \
kernel void NAME(device bfloat* q       [[buffer(0)]],                                  \
                 device bfloat* k_cache [[buffer(1)]],                                  \
                 device bfloat* v_cache [[buffer(2)]],                                  \
                 device bfloat* out     [[buffer(3)]],                                  \
                 constant uint& max_seq  [[buffer(4)]],                                 \
                 constant uint& base_len [[buffer(5)]],                                 \
                 constant uint& M        [[buffer(6)]],                                 \
                 constant uint& nq       [[buffer(7)]],                                 \
                 constant uint& group    [[buffer(8)]],                                 \
                 constant float& scale   [[buffer(9)]],                                 \
                 constant uint& parallel_softmax [[buffer(10)]],                        \
                 constant uint& descending_blocks [[buffer(11)]],                       \
                 uint2 tg  [[threadgroup_position_in_grid]],                             \
                 uint  tid [[thread_index_in_threadgroup]],                              \
                 uint2 tg_size [[threads_per_threadgroup]]) {                            \
    threadgroup float  s_tile[(BQ) * (BK)];                                              \
    threadgroup bfloat p_tile[(BQ) * (BK)];                                              \
    threadgroup float  row_max[(BQ)], row_sum[(BQ)], row_alpha[(BQ)];                    \
    threadgroup bfloat q_tile[1];                                                        \
    sdpa_nax_body<(BQ), (BK), (SG), true>(                                     \
        q, k_cache, v_cache, out, max_seq, base_len, M, nq, group, scale,               \
        parallel_softmax, descending_blocks, q_tile, s_tile, p_tile,                    \
        row_max, row_sum, row_alpha, tg, tid, tg_size);                                  \
}

SDPA_NAX_KERNEL_QDEV(sdpa_prefill_nax_qdev128, 16, 128, 4)

#endif

// Single-token GQA SDPA with one threadgroup per query head.
kernel void sdpa_decode_bf16(device const bfloat* q       [[buffer(0)]],  // [NQ, D]
                             device const bfloat* k_cache [[buffer(1)]],  // [KVH, MAX, D]
                             device const bfloat* v_cache [[buffer(2)]],  // [KVH, MAX, D]
                             device bfloat*       out     [[buffer(3)]],  // [NQ, D]
                             constant uint&       D       [[buffer(4)]],
                             constant uint&       max_seq [[buffer(5)]],
                             constant uint&       len     [[buffer(6)]],  // valid positions
                             constant uint&       group   [[buffer(7)]],  // NQ / KVH
                             constant float&      scale   [[buffer(8)]],
                             uint hq   [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[MAX_T];
    threadgroup float q_s[TG];  // D <= TG assumed
    threadgroup float part[TG / 32];
    threadgroup float red;

    device const bfloat* k_head = k_cache + (ulong)(hq / group) * max_seq * D;
    device const bfloat* v_head = v_cache + (ulong)(hq / group) * max_seq * D;

    if (tid < D) {
        q_s[tid] = float(q[hq * D + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint pos = tid; pos < len; pos += TG) {
        float dot = 0.0f;
        for (uint d = 0; d < D; ++d) {
            dot += q_s[d] * float(k_head[(ulong)pos * D + d]);
        }
        float s = dot * scale;
        scores[pos] = s;
        local_max = max(local_max, s);
    }

    local_max = simd_max(local_max);
    if (lane == 0) { part[sg] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < TG / 32; ++i) { m = max(m, part[i]); }
        red = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = red;

    float local_sum = 0.0f;
    for (uint pos = tid; pos < len; pos += TG) {
        float e = exp(scores[pos] - global_max);
        scores[pos] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { part[sg] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) { s += part[i]; }
        red = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / red;

    for (uint d = tid; d < D; d += TG) {
        float acc = 0.0f;
        for (uint pos = 0; pos < len; ++pos) {
            acc += scores[pos] * float(v_head[(ulong)pos * D + d]);
        }
        out[hq * D + d] = bfloat(acc * inv_sum);
    }
}

// Split-K decode emits local softmax statistics and weighted-V partials.
#define SDPA_SPLIT 256

// CPT chunks per threadgroup; STRIDED distributes them across the grid.
template <uint CPT, bool STRIDED = false>
kernel void sdpa_decode_split_t(device const bfloat* q        [[buffer(0)]],  // [NQ, D]
                                   device const bfloat* k_cache  [[buffer(1)]],
                                   device const bfloat* v_cache  [[buffer(2)]],
                                   device float*        partials [[buffer(3)]],  // [NQ, S, D]
                                   device float*        stats    [[buffer(4)]],  // [NQ, S, 2]
                                   constant uint&       D        [[buffer(5)]],
                                   constant uint&       max_seq  [[buffer(6)]],
                                   constant uint&       len      [[buffer(7)]],
                                   constant uint&       splits   [[buffer(8)]],
                                   constant uint&       group    [[buffer(9)]],
                                   constant float&      scale    [[buffer(10)]],
                                   uint2 tg  [[threadgroup_position_in_grid]],
                                   uint tid  [[thread_index_in_threadgroup]],
                                   uint sg   [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[SDPA_SPLIT];
    threadgroup float q_s[TG];
    threadgroup float part[TG / 32];
    threadgroup float red;

    const uint hq = tg.x;
    device const bfloat* k_head = k_cache + (ulong)(hq / group) * max_seq * D;
    device const bfloat* v_head = v_cache + (ulong)(hq / group) * max_seq * D;

    for (uint cc = 0; cc < CPT; ++cc) {
    const uint split = STRIDED ? (tg.y + cc * ((splits + CPT - 1) / CPT))
                               : (tg.y * CPT + cc);
    if (split >= splits) { continue; }
    const uint base = split * SDPA_SPLIT;
    const uint count = base < len ? min(uint(SDPA_SPLIT), len - base) : 0;

    if (count == 0) {
        if (tid == 0) {
            stats[(hq * splits + split) * 2] = -INFINITY;
            stats[(hq * splits + split) * 2 + 1] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }

    if (tid < D) {
        q_s[tid] = float(q[hq * D + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    if (D % 8 == 0) {
        for (uint p = sg; p < count; p += TG / 32) {
            device const uint4* krow =
                (device const uint4*)(k_head + (ulong)(base + p) * D);
            float dot = 0.0f;
            for (uint i = lane; i * 8 < D; i += 32) {
                uint4 kq = krow[i];
                float4 qa(q_s[i * 8], q_s[i * 8 + 1], q_s[i * 8 + 2],
                          q_s[i * 8 + 3]);
                float4 qb(q_s[i * 8 + 4], q_s[i * 8 + 5], q_s[i * 8 + 6],
                          q_s[i * 8 + 7]);
                dot += metal::dot(float4(as_type<bfloat4>(kq.xy)), qa);
                dot += metal::dot(float4(as_type<bfloat4>(kq.zw)), qb);
            }
            dot = simd_sum(dot);
            float s = dot * scale;
            if (lane == 0) {
                scores[p] = s;
            }
            local_max = max(local_max, s);
        }
    } else {
        if (tid < count) {
            float dot = 0.0f;
            for (uint d = 0; d < D; ++d) {
                dot += q_s[d] * float(k_head[(ulong)(base + tid) * D + d]);
            }
            float s = dot * scale;
            scores[tid] = s;
            local_max = s;
        }
    }
    local_max = simd_max(local_max);
    if (lane == 0) { part[sg] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint i = 0; i < TG / 32; ++i) { m = max(m, part[i]); }
        red = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float chunk_max = red;

    float e = 0.0f;
    if (tid < count) {
        e = exp(scores[tid] - chunk_max);
        scores[tid] = e;
    }
    float local_sum = simd_sum(e);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { part[sg] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) { s += part[i]; }
        stats[(hq * splits + split) * 2] = chunk_max;
        stats[(hq * splits + split) * 2 + 1] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce per-simdgroup weighted-V accumulators.
    if (D % 8 == 0 && D <= 8 * 32) {
        threadgroup float v_stage[(TG / 32) * 256];
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        for (uint p = sg; p < count; p += TG / 32) {
            float wgt = scores[p];
            device const uint4* vrow =
                (device const uint4*)(v_head + (ulong)(base + p) * D);
            if (lane * 8 < D) {
                uint4 vq = vrow[lane];
                acc0 += wgt * float4(as_type<bfloat4>(vq.xy));
                acc1 += wgt * float4(as_type<bfloat4>(vq.zw));
            }
        }
        if (lane * 8 < D) {
            for (uint j = 0; j < 4; ++j) {
                v_stage[sg * D + lane * 8 + j] = acc0[j];
                v_stage[sg * D + lane * 8 + 4 + j] = acc1[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = tid; d < D; d += TG) {
            float acc = 0.0f;
            for (uint s = 0; s < TG / 32; ++s) {
                acc += v_stage[s * D + d];
            }
            partials[((ulong)hq * splits + split) * D + d] = acc;
        }
    } else {
        for (uint d = tid; d < D; d += TG) {
            float acc = 0.0f;
            for (uint p = 0; p < count; ++p) {
                acc += scores[p] *
                    float(v_head[(ulong)(base + p) * D + d]);
            }
            partials[((ulong)hq * splits + split) * D + d] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("sdpa_decode_split_bf16")]] kernel void
sdpa_decode_split_t<1>(
    device const bfloat*, device const bfloat*, device const bfloat*, device float*,
    device float*, constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant float&, uint2, uint, uint, uint);


// Folded GQA reuses each K/V row across query heads.
#define MAX_GQA 8

// HPT caps query heads; FIXED requires group % HPP == 0.
template <uint HPP, uint GCPT = 1, uint HPT = 0, uint RIF = 1,
          uint SERIAL = 0, uint FIXED = 0>
kernel void sdpa_decode_split_gqa_t(device const bfloat* q        [[buffer(0)]],
                                       device const bfloat* k_cache  [[buffer(1)]],
                                       device const bfloat* v_cache  [[buffer(2)]],
                                       device float*        partials [[buffer(3)]],
                                       device float*        stats    [[buffer(4)]],
                                       constant uint&       D        [[buffer(5)]],
                                       constant uint&       max_seq  [[buffer(6)]],
                                       constant uint&       len      [[buffer(7)]],
                                       constant uint&       splits   [[buffer(8)]],
                                       constant uint&       group    [[buffer(9)]],
                                       constant float&      scale    [[buffer(10)]],
                                       // <= SDPA_SPLIT
                                       constant uint&       chunk    [[buffer(11)]],
                                       uint2 tg  [[threadgroup_position_in_grid]],
                                       uint tid  [[thread_index_in_threadgroup]],
                                       uint sg   [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[HPP][SDPA_SPLIT];
    threadgroup float part[HPP][TG / 32];
    threadgroup float red[HPP];
    threadgroup float v_stage[(TG / 32) * 256];

    const uint gfull = min(group, uint(MAX_GQA));
    const uint sub = HPT ? max(1u, gfull / uint(HPT)) : 1u;
    const uint kh = tg.x / sub;
    const uint hbase = (tg.x % sub) * (HPT ? uint(HPT) : gfull);
    const uint g = HPT ? min(uint(HPT), gfull - hbase) : gfull;
    for (uint gcc = 0; gcc < GCPT; ++gcc) {
    const uint split = tg.y + gcc * ((splits + GCPT - 1) / GCPT);
    if (split >= splits) { continue; }
    const uint c = min(chunk, uint(SDPA_SPLIT));
    const uint base = split * c;
    const uint count = base < len ? min(c, len - base) : 0;
    device const bfloat* k_head = k_cache + (ulong)kh * max_seq * D;
    device const bfloat* v_head = v_cache + (ulong)kh * max_seq * D;

    if (count == 0) {
        if (tid == 0) {
            for (uint h = 0; h < g; ++h) {
                const uint hq = kh * gfull + hbase + h;
                stats[(hq * splits + split) * 2] = -INFINITY;
                stats[(hq * splits + split) * 2 + 1] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }

    // Each lane holds eight Q dimensions per head.
    for (uint h0 = 0; h0 < g; h0 += HPP) {
    const uint gh = FIXED ? uint(HPP) : min(uint(HPP), g - h0);
    float4 qa[HPP];
    float4 qb[HPP];
    for (uint h = 0; h < gh; ++h) {
        const uint hq = kh * gfull + hbase + h0 + h;
        device const bfloat* qh = q + (ulong)hq * D;
        const uint o = lane * 8;
        qa[h] = float4(float(qh[o]), float(qh[o + 1]), float(qh[o + 2]),
                       float(qh[o + 3]));
        qb[h] = float4(float(qh[o + 4]), float(qh[o + 5]), float(qh[o + 6]),
                       float(qh[o + 7]));
    }

    float local_max[HPP];
    for (uint h = 0; h < gh; ++h) { local_max[h] = -INFINITY; }

    // Uniform tail guards preserve full-simdgroup reductions.
    const uint kstride = TG / 32;
    for (uint p0 = sg; p0 < count; p0 += RIF * kstride) {
#define LILY_KROW(P) ((device const uint4*)(k_head + (ulong)(base + (P)) * D))
        uint4 r0 = LILY_KROW(p0)[lane];
        uint4 r1, r2, r3;
        const bool h1 = RIF > 1 && p0 + kstride < count;
        const bool h2 = RIF > 2 && p0 + 2 * kstride < count;
        const bool h3 = RIF > 3 && p0 + 3 * kstride < count;
        const uint dep1 = SERIAL ? uint(r0.x == 0xFFFFFFFFu) : 0u;
        if (h1) { r1 = LILY_KROW(p0 + kstride + dep1)[lane]; }
        const uint dep2 = SERIAL ? uint(r1.x == 0xFFFFFFFFu) : 0u;
        if (h2) { r2 = LILY_KROW(p0 + 2 * kstride + dep2)[lane]; }
        const uint dep3 = SERIAL ? uint(r2.x == 0xFFFFFFFFu) : 0u;
        if (h3) { r3 = LILY_KROW(p0 + 3 * kstride + dep3)[lane]; }
#undef LILY_KROW
#define LILY_K_CONSUME(P, ROW)                                              \
        {                                                                   \
            float4 ka = float4(as_type<bfloat4>((ROW).xy));                 \
            float4 kb = float4(as_type<bfloat4>((ROW).zw));                 \
            for (uint h = 0; h < gh; ++h) {                                 \
                float dot = 0.0f;                                           \
                dot += metal::dot(ka, qa[h]);                               \
                dot += metal::dot(kb, qb[h]);                               \
                dot = simd_sum(dot);                                        \
                float s = dot * scale;                                      \
                if (lane == 0) { scores[h][(P)] = s; }                      \
                local_max[h] = max(local_max[h], s);                        \
            }                                                               \
        }
        LILY_K_CONSUME(p0, r0)
        if (h1) { LILY_K_CONSUME(p0 + kstride, r1) }
        if (h2) { LILY_K_CONSUME(p0 + 2 * kstride, r2) }
        if (h3) { LILY_K_CONSUME(p0 + 3 * kstride, r3) }
#undef LILY_K_CONSUME
    }

    for (uint h = 0; h < gh; ++h) {
        float m = simd_max(local_max[h]);
        if (lane == 0) { part[h][sg] = m; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint h = 0; h < gh; ++h) {
            float m = -INFINITY;
            for (uint i = 0; i < TG / 32; ++i) { m = max(m, part[h][i]); }
            red[h] = m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 0; h < gh; ++h) {
        const float chunk_max = red[h];
        float e = 0.0f;
        if (tid < count) {
            e = exp(scores[h][tid] - chunk_max);
            scores[h][tid] = e;
        }
        float local_sum = simd_sum(e);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) { part[h][sg] = local_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s = 0.0f;
            for (uint i = 0; i < TG / 32; ++i) { s += part[h][i]; }
            const uint hq = kh * gfull + hbase + h0 + h;
            stats[(hq * splits + split) * 2] = chunk_max;
            stats[(hq * splits + split) * 2 + 1] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage one head at a time while reusing V rows.
    float4 acc0[HPP];
    float4 acc1[HPP];
    for (uint h = 0; h < gh; ++h) {
        acc0[h] = float4(0.0f);
        acc1[h] = float4(0.0f);
    }
    const uint vstride = TG / 32;
    for (uint p0 = sg; p0 < count; p0 += RIF * vstride) {
#define LILY_VROW(P) ((device const uint4*)(v_head + (ulong)(base + (P)) * D))
        uint4 s0 = LILY_VROW(p0)[lane];
        uint4 s1, s2, s3;
        const bool g1 = RIF > 1 && p0 + vstride < count;
        const bool g2 = RIF > 2 && p0 + 2 * vstride < count;
        const bool g3 = RIF > 3 && p0 + 3 * vstride < count;
        const uint vdep1 = SERIAL ? uint(s0.x == 0xFFFFFFFFu) : 0u;
        if (g1) { s1 = LILY_VROW(p0 + vstride + vdep1)[lane]; }
        const uint vdep2 = SERIAL ? uint(s1.x == 0xFFFFFFFFu) : 0u;
        if (g2) { s2 = LILY_VROW(p0 + 2 * vstride + vdep2)[lane]; }
        const uint vdep3 = SERIAL ? uint(s2.x == 0xFFFFFFFFu) : 0u;
        if (g3) { s3 = LILY_VROW(p0 + 3 * vstride + vdep3)[lane]; }
#undef LILY_VROW
#define LILY_V_WEIGH(P, ROW)                                                \
        {                                                                   \
            float4 va = float4(as_type<bfloat4>((ROW).xy));                 \
            float4 vb = float4(as_type<bfloat4>((ROW).zw));                 \
            for (uint h = 0; h < gh; ++h) {                                 \
                float wgt = scores[h][(P)];                                 \
                acc0[h] += wgt * va;                                        \
                acc1[h] += wgt * vb;                                        \
            }                                                               \
        }
        LILY_V_WEIGH(p0, s0)
        if (g1) { LILY_V_WEIGH(p0 + vstride, s1) }
        if (g2) { LILY_V_WEIGH(p0 + 2 * vstride, s2) }
        if (g3) { LILY_V_WEIGH(p0 + 3 * vstride, s3) }
#undef LILY_V_WEIGH
    }
    for (uint h = 0; h < gh; ++h) {
        for (uint j = 0; j < 4; ++j) {
            v_stage[sg * D + lane * 8 + j] = acc0[h][j];
            v_stage[sg * D + lane * 8 + 4 + j] = acc1[h][j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hq = kh * gfull + hbase + h0 + h;
        for (uint d = tid; d < D; d += TG) {
            float acc = 0.0f;
            for (uint s = 0; s < TG / 32; ++s) {
                acc += v_stage[s * D + d];
            }
            partials[((ulong)hq * splits + split) * D + d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }
    }
}








template [[host_name("sdpa_decode_split_gqa_h4u_bf16")]] kernel void
sdpa_decode_split_gqa_t<4, 1, 4, 1, 0, 1>(
    device const bfloat*, device const bfloat*, device const bfloat*, device float*,
    device float*, constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant float&, constant uint&, uint2, uint, uint, uint);

#define LILY_MLX_DECL(j)                                                       \
    float4 qa##j, qb##j, aa##j, ab##j;                                        \
    float m##j, l##j;                                                         \
    {                                                                         \
        device const bfloat* qh = q + (ulong)(hq0 + j) * D + lane * 8;        \
        qa##j = float4(float(qh[0]), float(qh[1]), float(qh[2]), float(qh[3]));\
        qb##j = float4(float(qh[4]), float(qh[5]), float(qh[6]), float(qh[7]));\
        aa##j = float4(0.0f);                                                 \
        ab##j = float4(0.0f);                                                 \
        m##j = -INFINITY;                                                     \
        l##j = 0.0f;                                                          \
    }

#define LILY_MLX_SCORE2(j)                                                    \
    float w##j##_0, w##j##_1;                                                 \
    {                                                                         \
        float s0 = simd_sum(metal::dot(ka0, qa##j) + metal::dot(kb0, qb##j))   \
                 * scale;                                                     \
        float s1 = have1                                                      \
                 ? simd_sum(metal::dot(ka1, qa##j) + metal::dot(kb1, qb##j))  \
                   * scale                                                    \
                 : -INFINITY;                                                 \
        float nm = max(m##j, max(s0, s1));                                    \
        float c = exp(m##j - nm);                                             \
        w##j##_0 = exp(s0 - nm);                                              \
        w##j##_1 = have1 ? exp(s1 - nm) : 0.0f;                              \
        l##j = l##j * c + w##j##_0 + w##j##_1;                               \
        m##j = nm;                                                            \
        aa##j *= c;                                                           \
        ab##j *= c;                                                           \
    }

#define LILY_MLX_ACC2(j)                                                      \
    aa##j += w##j##_0 * va0 + w##j##_1 * va1;                                \
    ab##j += w##j##_0 * vb0 + w##j##_1 * vb1;

#define LILY_MLX_STORE(j)                                                     \
    {                                                                         \
        const uint hq = hq0 + j;                                              \
        device float* o = partials + ((ulong)hq * splits + block) * D         \
                        + lane * 8;                                           \
        o[0] = aa##j.x; o[1] = aa##j.y; o[2] = aa##j.z; o[3] = aa##j.w;       \
        o[4] = ab##j.x; o[5] = ab##j.y; o[6] = ab##j.z; o[7] = ab##j.w;       \
        if (lane == 0) {                                                      \
            stats[(hq * splits + block) * 2] = m##j;                          \
            stats[(hq * splits + block) * 2 + 1] = l##j;                      \
        }                                                                     \
    }

#define LILY_MLX_EACH2(M) M(0) M(1)

#define LILY_MLX_KERNEL_R2(NAME)                                              \
    kernel void NAME(device const bfloat* q        [[buffer(0)]],             \
                     device const bfloat* k_cache  [[buffer(1)]],             \
                     device const bfloat* v_cache  [[buffer(2)]],             \
                     device float*        partials [[buffer(3)]],             \
                     device float*        stats    [[buffer(4)]],             \
                     constant uint&       D        [[buffer(5)]],             \
                     constant uint&       max_seq  [[buffer(6)]],             \
                     constant uint&       len      [[buffer(7)]],             \
                     constant uint&       splits   [[buffer(8)]],             \
                     constant uint&       group    [[buffer(9)]],             \
                     constant float&      scale    [[buffer(10)]],            \
                     constant uint&       chunk    [[buffer(11)]],            \
                     uint2 tg   [[threadgroup_position_in_grid]],             \
                     uint sg    [[simdgroup_index_in_threadgroup]],           \
                     uint lane  [[thread_index_in_simdgroup]]) {              \
        const uint kh = tg.x;                                                 \
        const uint block = tg.y;                                              \
        const uint hq0 = kh * group + sg * 2;                                 \
        LILY_MLX_EACH2(LILY_MLX_DECL)                                         \
        device const bfloat* kb_ = k_cache + (ulong)kh * max_seq * D          \
                                 + lane * 8;                                  \
        device const bfloat* vb_ = v_cache + (ulong)kh * max_seq * D          \
                                 + lane * 8;                                  \
        for (uint p = block; p < len; p += 256) {                             \
            const uint p1 = p + 128;                                          \
            const bool have1 = p1 < len;                                      \
            uint4 kq0 = *(device const uint4*)(kb_ + (ulong)p * D);           \
            uint4 kq1 = *(device const uint4*)(kb_                            \
                        + (ulong)(have1 ? p1 : p) * D);                       \
            float4 ka0 = float4(as_type<bfloat4>(kq0.xy));                    \
            float4 kb0 = float4(as_type<bfloat4>(kq0.zw));                    \
            float4 ka1 = float4(as_type<bfloat4>(kq1.xy));                    \
            float4 kb1 = float4(as_type<bfloat4>(kq1.zw));                    \
            LILY_MLX_EACH2(LILY_MLX_SCORE2)                                   \
            uint4 vq0 = *(device const uint4*)(vb_ + (ulong)p * D);           \
            uint4 vq1 = *(device const uint4*)(vb_                            \
                        + (ulong)(have1 ? p1 : p) * D);                       \
            float4 va0 = float4(as_type<bfloat4>(vq0.xy));                    \
            float4 vb0 = float4(as_type<bfloat4>(vq0.zw));                    \
            float4 va1 = float4(as_type<bfloat4>(vq1.xy));                    \
            float4 vb1 = float4(as_type<bfloat4>(vq1.zw));                    \
            LILY_MLX_EACH2(LILY_MLX_ACC2)                                     \
        }                                                                     \
        LILY_MLX_EACH2(LILY_MLX_STORE)                                        \
    }

LILY_MLX_KERNEL_R2(sdpa_decode_mlx_b128h2r2)

#undef LILY_MLX_KERNEL_R2
#undef LILY_MLX_EACH2
#undef LILY_MLX_STORE
#undef LILY_MLX_ACC2
#undef LILY_MLX_SCORE2
#undef LILY_MLX_DECL

kernel void sdpa_decode_combine(device const float* partials [[buffer(0)]],  // [NQ, S, D]
                                device const float* stats    [[buffer(1)]],  // [NQ, S, 2]
                                device bfloat*      out      [[buffer(2)]],  // [NQ, D]
                                constant uint&      D        [[buffer(3)]],
                                constant uint&      splits   [[buffer(4)]],
                                uint hq  [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
    // Combine split statistics without per-split scratch.
    threadgroup float m_max;
    threadgroup float inv_l;

    if (tid == 0) {
        float m = -INFINITY;
        for (uint s = 0; s < splits; ++s) {
            m = max(m, stats[(hq * splits + s) * 2]);
        }
        float l = 0.0f;
        for (uint s = 0; s < splits; ++s) {
            float w = stats[(hq * splits + s) * 2 + 1] *
                exp(stats[(hq * splits + s) * 2] - m);
            l += w;
        }
        m_max = m;
        inv_l = 1.0f / l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float m = m_max;
    for (uint d = tid; d < D; d += TG) {
        float acc = 0.0f;
        for (uint s = 0; s < splits; ++s) {
            acc += exp(stats[(hq * splits + s) * 2] - m) *
                partials[((ulong)hq * splits + s) * D + d];
        }
        out[hq * D + d] = bfloat(acc * inv_l);
    }
}
