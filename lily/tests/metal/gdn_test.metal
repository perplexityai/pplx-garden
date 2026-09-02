// Unit-test-only entry point for the unfused recurrent-step oracle. The shared
// body comes from gdn.metal, which is concatenated before this file by gdn.rs
// under cfg(test).
#define GDN_STEP_WRAPPER(NAME, STATE_T)                                       \
kernel void NAME(device const bfloat* q       [[buffer(0)]],                  \
                 device const bfloat* k       [[buffer(1)]],                  \
                 device const bfloat* v       [[buffer(2)]],                  \
                 device const bfloat* a       [[buffer(3)]],                  \
                 device const bfloat* b       [[buffer(4)]],                  \
                 device const float*  a_log   [[buffer(5)]],                  \
                 device const bfloat* dt_bias [[buffer(6)]],                  \
                 device STATE_T*      state   [[buffer(7)]],                  \
                 device bfloat*       out     [[buffer(8)]],                  \
                 constant float&      scale   [[buffer(9)]],                  \
                 constant uint&       vpk     [[buffer(10)]],                 \
                 uint h    [[threadgroup_position_in_grid]],                  \
                 uint tid  [[thread_index_in_threadgroup]],                   \
                 uint sg   [[simdgroup_index_in_threadgroup]],                \
                 uint lane [[thread_index_in_simdgroup]]) {                   \
    threadgroup float q_norm[DIM];                                            \
    threadgroup float k_norm[DIM];                                            \
    threadgroup float part_q[DIM / 32];                                       \
    threadgroup float part_k[DIM / 32];                                       \
    threadgroup float gates[4];                                               \
    const float o = gdn_step_body(q, k, v, a, b, a_log, dt_bias, state,      \
                                  scale, vpk, h, tid, sg, lane, q_norm,       \
                                  k_norm, part_q, part_k, gates);             \
    out[h * DIM + tid] = bfloat(o);                                           \
}

GDN_STEP_WRAPPER(gdn_step, float)
#undef GDN_STEP_WRAPPER
