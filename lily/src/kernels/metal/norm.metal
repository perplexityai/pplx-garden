// Row-wise weighted RMSNorm with fp32 reduction.
#include <metal_stdlib>
using namespace metal;

#define TG 256

// H=2048 fused residual add + RMSNorm; rounded residuals are written to x.
constant constexpr uint RESIDUAL_H = 2048;

kernel void add_rmsnorm_bf16(
                             device bfloat*       x   [[buffer(0)]],
                             device const bfloat* b   [[buffer(1)]],
                             device const bfloat* w   [[buffer(2)]],
                             device bfloat*       out [[buffer(3)]],
                             constant float&      eps [[buffer(4)]],
                             constant float&      w_bias [[buffer(5)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    bfloat cached[8];

    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        const uint i = tid + j * TG;
        const bfloat y = bfloat(float(x[row * RESIDUAL_H + i]) +
                                float(b[row * RESIDUAL_H + i]));
        cached[j] = y;
        x[row * RESIDUAL_H + i] = y;
        const float v = float(y);
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
        inv_rms = rsqrt(total / float(RESIDUAL_H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 8; ++j) {
        const uint i = tid + j * TG;
        const float gain = w_bias + float(w[i]);
        out[row * RESIDUAL_H + i] = bfloat(float(cached[j]) * inv_rms * gain);
    }
}

// RMSNorm with gain w_bias + w and FP32 accumulation.
kernel void rmsnorm_bf16(device const bfloat* x   [[buffer(0)]],
                         device const bfloat* w   [[buffer(1)]],
                         device bfloat*       out [[buffer(2)]],
                         constant uint&       H   [[buffer(3)]],
                         constant float&      eps [[buffer(4)]],
                         constant float&      w_bias [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;

    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        float v = float(x[row * H + i]);
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
        inv_rms = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < H; i += TG) {
        float gain = w_bias + float(w[i]);
        out[row * H + i] = bfloat(float(x[row * H + i]) * inv_rms * gain);
    }
}
