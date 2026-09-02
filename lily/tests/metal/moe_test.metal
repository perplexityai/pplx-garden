// Unit-test-only pair-major and extended-row small-M variants.
#define MOE_GEMV_SMALLM_BODY(MAXR, R4)                                        \
    threadgroup uint tg_pairs[MOE_SMALLM_MAX_S];                              \
    for (uint jj = lane; jj < S; jj += 64) {                                  \
        tg_pairs[jj] = indices[jj];                                           \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    const uint j = tg.y;                                                      \
    const uint e = tg_pairs[j];                                               \
    for (uint jj = 0; jj < j; ++jj) {                                         \
        if (tg_pairs[jj] == e) {                                              \
            return;                                                           \
        }                                                                     \
    }                                                                         \
    uint pair[MAXR];                                                          \
    uint xbase[MAXR];                                                         \
    uint nr = 0;                                                              \
    for (uint jj = j; jj < S; ++jj) {                                         \
        if (tg_pairs[jj] != e || nr >= MAXR) {                                \
            continue;                                                         \
        }                                                                     \
        const uint xr = (XPP != 0) ? jj : jj / TOPK;                          \
        _Pragma("clang loop unroll(full)")                                    \
        for (uint ri = 0; ri < MAXR; ++ri) {                                  \
            if (ri == nr) {                                                   \
                pair[ri] = jj;                                                \
                xbase[ri] = xr * (K / 4);                                     \
            }                                                                 \
        }                                                                     \
        ++nr;                                                                 \
    }                                                                         \
    MOE_GEMV_SMALLM_CORE(MAXR, R4)

#define MOE_GEMV_SMALLM_KERNEL(NAME, MAXR, R4)                                \
    kernel void NAME(device const uint*   codes   [[buffer(0)]],              \
                     device const bfloat* scales  [[buffer(1)]],              \
                     device const bfloat* biases  [[buffer(2)]],              \
                     device const bfloat* x       [[buffer(3)]],              \
                     device const uint*   indices [[buffer(4)]],              \
                     device bfloat*       y       [[buffer(5)]],              \
                     constant uint&       K       [[buffer(6)]],              \
                     constant uint&       GS      [[buffer(7)]],              \
                     constant uint&       N       [[buffer(8)]],              \
                     constant uint&       S       [[buffer(9)]],              \
                     constant uint&       TOPK    [[buffer(10)]],             \
                     constant uint&       XPP     [[buffer(11)]],             \
                     uint2 tg  [[threadgroup_position_in_grid]],              \
                     uint lane [[thread_index_in_threadgroup]]) {             \
        MOE_GEMV_SMALLM_BODY(MAXR, R4)                                        \
    }

MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r8, 8, 4)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r16, 16, 2)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r8_w, 8, 8)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r16_w, 16, 4)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r8_n2, 8, 1)
MOE_GEMV_SMALLM_KERNEL(moe_gemv_smallm_q4_r16_n2, 16, 1)

MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r16, 16, 2)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r16_w, 16, 4)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r8_n2, 8, 1)
MOE_GEMV_SMALLM_EM_KERNEL(moe_gemv_smallm_q4_em_r16_n2, 16, 1)
