// Affine Q4/Q8 kernels with BF16 activations and FP32 accumulation.
// Q4 codes pack eight low-nibble-first values per u32; w = scale*q + bias.

#include <metal_stdlib>
using namespace metal;

// Affine-dequantized dot product for one Q4 code word.
static inline float dot_word_q4(uint word, float s, float b,
                                device const bfloat4* xv, uint w) {
    float4 xlo = float4(xv[2 * w]);
    float4 xhi = float4(xv[2 * w + 1]);
    float4 qlo = float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                        float((word >> 8) & 0xF), float((word >> 12) & 0xF));
    float4 qhi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                        float((word >> 24) & 0xF), float((word >> 28) & 0xF));
    float qx = dot(qlo, xlo) + dot(qhi, xhi);
    float xs = dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
    return s * qx + b * xs;
}

// Q4 GEMV uses one simdgroup per row; GS must be divisible by 64.
#define GEMV_Q4_BODY(OUT_STORE)                                                \
    const uint words = K / 8;                                                  \
    const uint blocks = words / 4;                                             \
    const uint bpg = GS / 32;                                                  \
    const uint groups = K / GS;                                                \
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words); \
    device const bfloat4* xv = (device const bfloat4*)x;                       \
    float sum = 0.0f;                                                          \
    for (uint i = lane; i < blocks; i += 32) {                                 \
        uint g = i / bpg;                                                      \
        float s = float(scales[row * groups + g]);                             \
        float b = float(biases[row * groups + g]);                             \
        uint4 w4 = wrow[i];                                                    \
        sum += dot_word_q4(w4.x, s, b, xv, 4 * i)                              \
            + dot_word_q4(w4.y, s, b, xv, 4 * i + 1)                           \
            + dot_word_q4(w4.z, s, b, xv, 4 * i + 2)                           \
            + dot_word_q4(w4.w, s, b, xv, 4 * i + 3);                          \
    }                                                                          \
    sum = simd_sum(sum);                                                       \
    if (lane == 0) { OUT_STORE; }

kernel void gemv_q4_bf16(device const uint*   codes  [[buffer(0)]],
                         device const bfloat* scales [[buffer(1)]],
                         device const bfloat* biases [[buffer(2)]],
                         device const bfloat* x      [[buffer(3)]],
                         device bfloat*       y      [[buffer(4)]],
                         constant uint&       K      [[buffer(5)]],
                         constant uint&       GS     [[buffer(6)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    GEMV_Q4_BODY(y[row] = bfloat(sum))
}

kernel void gemv_q4_bf16_f32out(device const uint*   codes  [[buffer(0)]],
                                device const bfloat* scales [[buffer(1)]],
                                device const bfloat* biases [[buffer(2)]],
                                device const bfloat* x      [[buffer(3)]],
                                device float*        y      [[buffer(4)]],
                                constant uint&       K      [[buffer(5)]],
                                constant uint&       GS     [[buffer(6)]],
                                uint row  [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]]) {
    GEMV_Q4_BODY(y[row] = sum)
}

template <uint ROWS>
static inline float gemv_q4_packed_sum(device const uint* codes,
                                       device const bfloat* scales,
                                       device const bfloat* biases,
                                       device const bfloat* x,
                                       uint row, uint sublane,
                                       uint K, uint GS) {
    constexpr uint LANES = 32 / ROWS;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    device const uint4* wrow =
        (device const uint4*)(codes + (ulong)row * words);
    device const bfloat4* xv = (device const bfloat4*)x;
    float sum = 0.0f;
    for (uint i = sublane; i < blocks; i += LANES) {
        uint g = i / bpg;
        float s = float(scales[row * groups + g]);
        float b = float(biases[row * groups + g]);
        uint4 w4 = wrow[i];
        sum += dot_word_q4(w4.x, s, b, xv, 4 * i)
            + dot_word_q4(w4.y, s, b, xv, 4 * i + 1)
            + dot_word_q4(w4.z, s, b, xv, 4 * i + 2)
            + dot_word_q4(w4.w, s, b, xv, 4 * i + 3);
    }
    for (uint off = LANES / 2; off > 0; off >>= 1) {
        sum += simd_shuffle_down(sum, off);
    }
    return sum;
}

#define GEMV_Q4_PACKED_KERNEL(NAME, ROWS)                                     \
    kernel void NAME(device const uint*   codes  [[buffer(0)]],               \
                     device const bfloat* scales [[buffer(1)]],               \
                     device const bfloat* biases [[buffer(2)]],               \
                     device const bfloat* x      [[buffer(3)]],               \
                     device bfloat*       y      [[buffer(4)]],               \
                     constant uint&       K      [[buffer(5)]],               \
                     constant uint&       GS     [[buffer(6)]],               \
                     uint group [[threadgroup_position_in_grid]],             \
                     uint lane  [[thread_index_in_threadgroup]]) {            \
        constexpr uint LANES = 32 / ROWS;                                     \
        const uint sublane = lane % LANES;                                    \
        const uint row = group * ROWS + lane / LANES;                         \
        const float sum = gemv_q4_packed_sum<ROWS>(                            \
            codes, scales, biases, x, row, sublane, K, GS);                   \
        if (sublane == 0) {                                                    \
            y[row] = bfloat(sum);                                              \
        }                                                                      \
    }

GEMV_Q4_PACKED_KERNEL(gemv_q4_bf16_2row, 2)

// 8-bit variant: one code word holds 4 elements (low byte first).
static inline float dot_word_q8(uint word, float s, float b,
                                device const bfloat4* xv, uint w) {
    float4 x = float4(xv[w]);
    float4 q = float4(float(word & 0xFF), float((word >> 8) & 0xFF),
                      float((word >> 16) & 0xFF), float((word >> 24) & 0xFF));
    return s * dot(q, x) + b * dot(x, float4(1.0f));
}

// Q8 GEMV uses uint4 blocks of 16 elements; GS must be divisible by 16.
#define GEMV_Q8_BODY(OUT_STORE)                                                \
    const uint words = K / 4;                                                  \
    const uint blocks = words / 4;                                             \
    const uint bpg = GS / 16;                                                  \
    const uint groups = K / GS;                                                \
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words); \
    device const bfloat4* xv = (device const bfloat4*)x;                       \
    float sum = 0.0f;                                                          \
    for (uint i = lane; i < blocks; i += 32) {                                 \
        uint g = i / bpg;                                                      \
        float s = float(scales[row * groups + g]);                             \
        float b = float(biases[row * groups + g]);                             \
        uint4 w4 = wrow[i];                                                    \
        sum += dot_word_q8(w4.x, s, b, xv, 4 * i)                              \
            + dot_word_q8(w4.y, s, b, xv, 4 * i + 1)                           \
            + dot_word_q8(w4.z, s, b, xv, 4 * i + 2)                           \
            + dot_word_q8(w4.w, s, b, xv, 4 * i + 3);                          \
    }                                                                          \
    sum = simd_sum(sum);                                                       \
    if (lane == 0) { OUT_STORE; }

kernel void gemv_q8_bf16(device const uint*   codes  [[buffer(0)]],
                         device const bfloat* scales [[buffer(1)]],
                         device const bfloat* biases [[buffer(2)]],
                         device const bfloat* x      [[buffer(3)]],
                         device bfloat*       y      [[buffer(4)]],
                         constant uint&       K      [[buffer(5)]],
                         constant uint&       GS     [[buffer(6)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    GEMV_Q8_BODY(y[row] = bfloat(sum))
}

kernel void gemv_q8_bf16_f32out(device const uint*   codes  [[buffer(0)]],
                                device const bfloat* scales [[buffer(1)]],
                                device const bfloat* biases [[buffer(2)]],
                                device const bfloat* x      [[buffer(3)]],
                                device float*        y      [[buffer(4)]],
                                constant uint&       K      [[buffer(5)]],
                                constant uint&       GS     [[buffer(6)]],
                                uint row  [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]]) {
    GEMV_Q8_BODY(y[row] = sum)
}

// Materializes Q8 weights with one thread per code word.
kernel void dequant_q8_bf16(device const uint*   codes  [[buffer(0)]],
                            device const bfloat* scales [[buffer(1)]],
                            device const bfloat* biases [[buffer(2)]],
                            device bfloat*       out    [[buffer(3)]],
                            constant uint&       K      [[buffer(4)]],
                            constant uint&       GS     [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint words = K / 4;
    const uint w = gid.x;
    const uint row = gid.y;
    if (w >= words) {
        return;
    }
    const uint groups = K / GS;
    uint g = w / (GS / 4);
    float s = float(scales[row * groups + g]);
    float b = float(biases[row * groups + g]);
    uint word = codes[(ulong)row * words + w];
    float4 q = float4(float(word & 0xFF), float((word >> 8) & 0xFF),
                      float((word >> 16) & 0xFF), float((word >> 24) & 0xFF));
    ((device bfloat4*)(out + (ulong)row * K))[w] = bfloat4(q * s + b);
}

// Dequantizes one Q4 code word to eight BF16 values.
static inline void store_word_q4(uint word, float s, float b,
                                 device bfloat4* out, uint w) {
    float4 lo = float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                       float((word >> 8) & 0xF), float((word >> 12) & 0xF));
    float4 hi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                       float((word >> 24) & 0xF), float((word >> 28) & 0xF));
    out[2 * w] = bfloat4(lo * s + b);
    out[2 * w + 1] = bfloat4(hi * s + b);
}

// Materializes Q4 weights with one thread per code word.
kernel void dequant_q4_bf16(device const uint*   codes  [[buffer(0)]],
                            device const bfloat* scales [[buffer(1)]],
                            device const bfloat* biases [[buffer(2)]],
                            device bfloat*       out    [[buffer(3)]],
                            constant uint&       K      [[buffer(4)]],
                            constant uint&       GS     [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint words = K / 8;
    const uint w = gid.x;
    const uint row = gid.y;
    if (w >= words) {
        return;
    }
    const uint groups = K / GS;
    uint g = w / (GS / 8);
    float s = float(scales[row * groups + g]);
    float b = float(biases[row * groups + g]);
    uint word = codes[(ulong)row * words + w];
    store_word_q4(word, s, b, (device bfloat4*)(out + (ulong)row * K), w);
}

// Gathers and dequantizes embedding rows.
kernel void gather_rows_q4_bf16(device const uint*   codes  [[buffer(0)]],
                                device const bfloat* scales [[buffer(1)]],
                                device const bfloat* biases [[buffer(2)]],
                                device const uint*   ids    [[buffer(3)]],
                                device bfloat*       out    [[buffer(4)]],
                                constant uint&       K      [[buffer(5)]],
                                constant uint&       GS     [[buffer(6)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const uint words = K / 8;
    const uint w = gid.x;
    if (w >= words) {
        return;
    }
    const uint row = ids[gid.y];
    const uint groups = K / GS;
    uint g = w / (GS / 8);
    float s = float(scales[row * groups + g]);
    float b = float(biases[row * groups + g]);
    uint word = codes[(ulong)row * words + w];
    store_word_q4(word, s, b, (device bfloat4*)(out + (ulong)gid.y * K), w);
}

// Dequantizes one Q4 word into a threadgroup B tile.
static inline void store_word_q4_tg(uint word, float s, float b,
                                    threadgroup bfloat4* out, uint w) {
    float4 lo = float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                       float((word >> 8) & 0xF), float((word >> 12) & 0xF));
    float4 hi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                       float((word >> 24) & 0xF), float((word >> 28) & 0xF));
    out[2 * w] = bfloat4(lo * s + b);
    out[2 * w + 1] = bfloat4(hi * s + b);
}

#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Grouped Q4 GEMM dequantizes B tiles to BF16 and accumulates in FP32.
constant constexpr uint QNAX_BN = 64;
constant constexpr uint QNAX_BK = 64;

// BM must match the block-map row tile.
template <uint BM, int SG>
static void gemm_q4_nt_nax_grouped_body(device const uint* codes,
                                        device const bfloat* scales,
                                        device const bfloat* biases,
                                        device bfloat* a,
                                        device bfloat* c,
                                        device const uint4* blocks,
                                        uint K, uint N, uint GS,
                                        threadgroup bfloat* b_tile,
                                        uint bid, uint tid) {
    using namespace mpp::tensor_ops;

    const uint4 blk = blocks[bid];
    if (blk.x >= blk.w) {
        return;  // sentinel entry from the GPU-built block map
    }
    const uint m0 = blk.x, b_row0 = blk.y, n0 = blk.z, m_end = blk.w;
    const uint words = K / 8;
    const uint groups = K / GS;

    auto ta = metal::tensor(a, metal::dextents<int32_t, 2>(K, int(m_end)));
    auto tb = metal::tensor(b_tile, metal::dextents<int32_t, 2>(QNAX_BK, QNAX_BN));
    constexpr auto desc = matmul2d_descriptor(
        BM, QNAX_BN, QNAX_BK, /*transpose_left=*/false,
        /*transpose_right=*/true, /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroups<SG>> op;

    using ASlice = decltype(ta.slice(0, 0));
    auto acc = op.template get_destination_cooperative_tensor<ASlice,
                                                              decltype(tb), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    for (uint k0 = 0; k0 < K; k0 += QNAX_BK) {
        // Dequantize one 64x64 B tile across 32*SG threads.
        for (uint i = tid; i < QNAX_BN * QNAX_BK / 8; i += 32 * SG) {
            uint r = i / (QNAX_BK / 8);
            uint wcol = i % (QNAX_BK / 8);
            uint k = k0 + wcol * 8;
            ulong row = b_row0 + r;
            uint word = codes[row * words + k / 8];
            uint g = k / GS;
            float s = float(scales[row * groups + g]);
            float b = float(biases[row * groups + g]);
            store_word_q4_tg(word, s, b,
                             (threadgroup bfloat4*)(b_tile + r * QNAX_BK), wcol);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a_slice = ta.slice(int(k0), int(m0));
        op.run(a_slice, tb, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto ix = acc.get_multidimensional_index(i);
            uint row = m0 + uint(ix[1]);
            uint col = n0 + uint(ix[0]);
            if (row < m_end) {
                c[(ulong)row * N + col] = bfloat(acc[i]);
            }
        }
    }
}

#define GEMM_Q4_NT_NAX_GROUPED(NAME, BM, SG)                                   \
    kernel void NAME(device const uint*   codes  [[buffer(0)]],                \
                     device const bfloat* scales [[buffer(1)]],                \
                     device const bfloat* biases [[buffer(2)]],                \
                     device bfloat*       a      [[buffer(3)]],                \
                     device bfloat*       c      [[buffer(4)]],                \
                     device const uint4*  blocks [[buffer(5)]],                \
                     constant uint&       K      [[buffer(6)]],                \
                     constant uint&       N      [[buffer(7)]],                \
                     constant uint&       GS     [[buffer(8)]],                \
                     uint bid [[threadgroup_position_in_grid]],                \
                     uint tid [[thread_index_in_threadgroup]]) {               \
        threadgroup bfloat b_tile[QNAX_BN * QNAX_BK];                          \
        gemm_q4_nt_nax_grouped_body<BM, SG>(codes, scales, biases, a, c,       \
                                            blocks, K, N, GS, b_tile, bid,     \
                                            tid);                              \
    }

// Dispatch instantiated tiles with 32*SG threads.
GEMM_Q4_NT_NAX_GROUPED(gemm_q4_nt_nax_grouped_t32x4, 32, 4)
GEMM_Q4_NT_NAX_GROUPED(gemm_q4_nt_nax_grouped_t64x4, 64, 4)
#endif
