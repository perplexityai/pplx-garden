#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cassert>

template <size_t V>
class Fixed {
public:
    __device__ Fixed(size_t value) {}

    __device__ operator size_t() const { return V; }

    static constexpr size_t Value = V;
};

class NotFixed {
public:
    __device__ NotFixed(size_t value) : value_(value) {}

    __device__ operator size_t() const { return value_; }
private:
    size_t value_;
};

#define _LAUNCH_TYPE(kind, var, value, ...) \
    case kind: { \
        using var = value; \
        { __VA_ARGS__; } \
        break; \
    }

#define _LAUNCH_VAL(kind, var, value, ...) \
    case kind: { \
        static constexpr decltype(value) var = value; \
        { __VA_ARGS__; } \
        break; \
    }

// Generic dtype dispatch macro
#ifndef LAUNCH_DTYPE
#define LAUNCH_DTYPE(dtype, var, ...) \
    switch (dtype) { \
        _LAUNCH_TYPE(DTYPE_FLOAT16, var, half, __VA_ARGS__) \
        _LAUNCH_TYPE(DTYPE_BFLOAT16, var, __nv_bfloat16, __VA_ARGS__) \
        _LAUNCH_TYPE(DTYPE_FLOAT32, var, float, __VA_ARGS__) \
        default: { \
            assert(false && "Unsupported dtype"); \
            break; \
        } \
    }
#endif

// Static specialization for hidden dimension
#ifndef LAUNCH_HIDDEN_DIM
#define LAUNCH_HIDDEN_DIM(dtype, var, ...) \
    switch (dtype) { \
        _LAUNCH_TYPE(2048, var, Fixed<2048>, __VA_ARGS__) \
        _LAUNCH_TYPE(4096, var, Fixed<4096>, __VA_ARGS__) \
        _LAUNCH_TYPE(7168, var, Fixed<7168>, __VA_ARGS__) \
        default: { \
            using var = NotFixed; \
            { __VA_ARGS__; } \
            break; \
        } \
    }
#endif

// Static specialization for the token dimension
#ifndef LAUNCH_TOKEN_DIM_DISPATCH
#define LAUNCH_TOKEN_DIM_DISPATCH(dim, var, ...) \
    switch (dim) { \
        _LAUNCH_TYPE(2048, var, Fixed<2048>, __VA_ARGS__) \
        _LAUNCH_TYPE(4096, var, Fixed<4096>, __VA_ARGS__) \
        _LAUNCH_TYPE(7168, var, Fixed<7168>, __VA_ARGS__) \
        default: { \
            using var = NotFixed; \
            { __VA_ARGS__; } \
            break; \
        } \
    }
#endif


// Static specialization for the token dimension
#ifndef LAUNCH_TOKEN_DIM_COMBINE
#define LAUNCH_TOKEN_DIM_COMBINE(dim, var, ...) \
    switch (dim) { \
        _LAUNCH_TYPE(7168 * 2, var, Fixed<7168 * 2>, __VA_ARGS__) \
        default: { \
            using var = NotFixed; \
            { __VA_ARGS__; } \
            break; \
        } \
    }
#endif


// Static specialization for number of experts per token
#ifndef LAUNCH_NUM_EXPERTS_PER_TOKEN
#define LAUNCH_NUM_EXPERTS_PER_TOKEN(dtype, var, ...) \
    switch (dtype) { \
        _LAUNCH_TYPE(8, var, Fixed<8>, __VA_ARGS__) \
        default: { \
            using var = NotFixed; \
            { __VA_ARGS__; } \
            break; \
        } \
    }
#endif

// Static specialization for the hidden dim scale.
#ifndef LAUNCH_HIDDEN_DIM_SCALE
#define LAUNCH_HIDDEN_DIM_SCALE(dtype, var, ...) \
    switch (dtype) { \
        _LAUNCH_TYPE(8, var, Fixed<8>, __VA_ARGS__) \
        _LAUNCH_TYPE(32, var, Fixed<32>, __VA_ARGS__) \
        _LAUNCH_TYPE(56, var, Fixed<56>, __VA_ARGS__) \
        default: { \
            using var = NotFixed; \
            { __VA_ARGS__; } \
            break; \
        } \
    }
#endif

// Static specialization for the hidden dim scale.
#ifndef LAUNCH_HIDDEN_DIM_SCALE_BYTES
#define LAUNCH_HIDDEN_DIM_SCALE_BYTES(dtype, var, ...) \
    switch (dtype) { \
        _LAUNCH_TYPE(32, var, Fixed<32>, __VA_ARGS__) \
        _LAUNCH_TYPE(128, var, Fixed<128>, __VA_ARGS__) \
        _LAUNCH_TYPE(224, var, Fixed<224>, __VA_ARGS__) \
        default: { \
            using var = NotFixed; \
            { __VA_ARGS__; } \
            break; \
        } \
    }
#endif

// Static specialization for the world size.
#ifndef LAUNCH_WORLD_SIZE
#define LAUNCH_WORLD_SIZE(world_size, var, ...) \
    switch (world_size) { \
        _LAUNCH_VAL(1, var, 1, __VA_ARGS__) \
        _LAUNCH_VAL(2, var, 2, __VA_ARGS__) \
        _LAUNCH_VAL(4, var, 4, __VA_ARGS__) \
        _LAUNCH_VAL(8, var, 8, __VA_ARGS__) \
        default: { \
            assert(false && "Unsupported world size"); \
            break; \
        } \
    }
#endif

// Static specialization for the DP group size
#ifndef LAUNCH_DP_SIZE
#define LAUNCH_DP_SIZE(dp_size, var, ...) \
    switch (dp_size) { \
        _LAUNCH_VAL(1, var, 1, __VA_ARGS__) \
        _LAUNCH_VAL(2, var, 2, __VA_ARGS__) \
        _LAUNCH_VAL(4, var, 4, __VA_ARGS__) \
        _LAUNCH_VAL(8, var, 8, __VA_ARGS__) \
        default: { \
            assert(false && "Unsupported DP size"); \
            break; \
        } \
    }
#endif

// Static specialization for the DP group size
#ifndef LAUNCH_ACCUMULATE
#define LAUNCH_ACCUMULATE(accumulate, var, ...) \
    switch (accumulate) { \
        _LAUNCH_VAL(true, var, true, __VA_ARGS__) \
        _LAUNCH_VAL(false, var, false, __VA_ARGS__) \
    }
#endif



#ifndef LAUNCH_BASIC_FLOAT
#define LAUNCH_BASIC_FLOAT(dtype, var, ...) \
    switch (dtype) { \
        _LAUNCH_TYPE(torch_lib::ScalarType::F16, var, half, __VA_ARGS__) \
        _LAUNCH_TYPE(torch_lib::ScalarType::BF16, var, __nv_bfloat16, __VA_ARGS__) \
        _LAUNCH_TYPE(torch_lib::ScalarType::F32, var, float, __VA_ARGS__) \
        default: { \
            assert(false && "Unsupported dtype"); \
            break; \
        } \
    }
#endif
