// Elementwise, embedding, and argmax kernels.
#include <metal_stdlib>
using namespace metal;

kernel void add_bf16(device const bfloat* a   [[buffer(0)]],
                     device const bfloat* b   [[buffer(1)]],
                     device bfloat*       out [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = bfloat(float(a[gid]) + float(b[gid]));
}

// SwiGLU: out = silu(gate) * up.
kernel void silu_mul_bf16(device const bfloat* gate [[buffer(0)]],
                          device const bfloat* up   [[buffer(1)]],
                          device bfloat*       out  [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    float g = float(gate[gid]);
    float s = g / (1.0f + exp(-g));
    out[gid] = bfloat(s * float(up[gid]));
}

// Attention output gate: out = x * sigmoid(gate).
kernel void sigmoid_mul_bf16(device const bfloat* gate [[buffer(0)]],
                             device const bfloat* x    [[buffer(1)]],
                             device bfloat*       out  [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    float g = float(gate[gid]);
    out[gid] = bfloat(float(x[gid]) / (1.0f + exp(-g)));
}

// Splits [m, n_total] into up to four contiguous column segments.
kernel void split_cols_bf16(device const bfloat* src [[buffer(0)]],
                            device bfloat*       d0  [[buffer(1)]],
                            device bfloat*       d1  [[buffer(2)]],
                            device bfloat*       d2  [[buffer(3)]],
                            device bfloat*       d3  [[buffer(4)]],
                            constant uint4&      w   [[buffer(5)]],
                            constant uint&  n_total  [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    const uint row = gid.y;
    const bfloat v = src[(ulong)row * n_total + col];
    if (col < w.x) {
        d0[(ulong)row * w.x + col] = v;
        return;
    }
    col -= w.x;
    if (col < w.y) {
        d1[(ulong)row * w.y + col] = v;
        return;
    }
    col -= w.y;
    if (col < w.z) {
        d2[(ulong)row * w.z + col] = v;
        return;
    }
    col -= w.z;
    if (col < w.w) {
        d3[(ulong)row * w.w + col] = v;
    }
}

// Embedding lookup: copies row `row` of a [rows, H] bf16 table.
kernel void gather_row_bf16(device const bfloat* table [[buffer(0)]],
                            device bfloat*       out   [[buffer(1)]],
                            constant uint&       row   [[buffer(2)]],
                            constant uint&       H     [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    out[gid] = table[(ulong)row * H + gid];
}

// Batched embedding lookup: out[i, :] = table[ids[i], :].
kernel void gather_rows_bf16(device const bfloat* table [[buffer(0)]],
                             device const uint*   ids   [[buffer(1)]],
                             device bfloat*       out   [[buffer(2)]],
                             constant uint&       H     [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint i = gid / H;
    uint d = gid % H;
    out[gid] = table[(ulong)ids[i] * H + d];
}

#define ARGMAX_TG 256

struct ArgMaxPair {
    float v;
    uint  i;
};

static inline void argmax_merge(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

kernel void argmax_f32_partial(device const float* x        [[buffer(0)]],
                               device ArgMaxPair*  partials [[buffer(1)]],
                               constant uint&      n        [[buffer(2)]],
                               constant uint&      chunk    [[buffer(3)]],
                               uint g    [[threadgroup_position_in_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part_v[ARGMAX_TG / 32];
    threadgroup uint  part_i[ARGMAX_TG / 32];

    float bv = -INFINITY;
    uint  bi = 0;
    uint end = min((g + 1) * chunk, n);
    for (uint i = g * chunk + tid; i < end; i += ARGMAX_TG) {
        argmax_merge(bv, bi, x[i], i);
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        uint  oi = simd_shuffle_down(bi, off);
        argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) {
        part_v[sg] = bv;
        part_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < ARGMAX_TG / 32; ++s) {
            argmax_merge(bv, bi, part_v[s], part_i[s]);
        }
        partials[g].v = bv;
        partials[g].i = bi;
    }
}

// Reduces partial argmax results to one index.
kernel void argmax_f32_final(device const ArgMaxPair* partials [[buffer(0)]],
                             device uint*             out      [[buffer(1)]],
                             constant uint&           groups   [[buffer(2)]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part_v[ARGMAX_TG / 32];
    threadgroup uint  part_i[ARGMAX_TG / 32];

    float bv = -INFINITY;
    uint  bi = 0;
    for (uint g = tid; g < groups; g += ARGMAX_TG) {
        argmax_merge(bv, bi, partials[g].v, partials[g].i);
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        uint  oi = simd_shuffle_down(bi, off);
        argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) {
        part_v[sg] = bv;
        part_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < ARGMAX_TG / 32; ++s) {
            argmax_merge(bv, bi, part_v[s], part_i[s]);
        }
        out[0] = bi;
    }
}
