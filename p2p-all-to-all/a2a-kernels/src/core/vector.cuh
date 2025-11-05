#pragma once

#include "core/memory.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace rose {

template <typename T>
struct VecStorageSize;

template <>
struct VecStorageSize<float> {
    static constexpr int SIZE = 4;
};

template <>
struct VecStorageSize<half> {
    static constexpr int SIZE = 8;
};

template <>
struct VecStorageSize<__nv_bfloat16> {
    static constexpr int SIZE = 8;
};

template <typename T>
struct Vec {
    static constexpr int SIZE = VecStorageSize<T>::SIZE;
    float value[SIZE];

    __forceinline__ __device__ Vec() {}

    __forceinline__ __device__ explicit Vec(const float (&vals)[SIZE]) {
        #pragma unroll
        for (int i = 0; i < SIZE; ++i) {
            value[i] = vals[i];
        }
    }

    __forceinline__ __device__ static Vec load(const T *ptr);

    __forceinline__ __device__ void store(T *ptr) const;
};

namespace detail {

struct HalfPairConverter {
    __forceinline__ __device__ static half2 apply(float x, float y) {
        return __floats2half2_rn(x, y);
    }
};

struct BFloatPairConverter {
    __forceinline__ __device__ static __nv_bfloat162 apply(float x, float y) {
        return __floats2bfloat162_rn(x, y);
    }
};

template <typename PairType, typename Converter, int NUM_PAIRS>
__forceinline__ __device__ uint4 pack_float_pairs(const float *values) {
    uint4 data;
    uint32_t *raw = reinterpret_cast<uint32_t *>(&data);
    union {
        uint32_t raw;
        PairType pair;
    } convert;

    #pragma unroll
    for (int i = 0; i < NUM_PAIRS; ++i) {
        convert.pair = Converter::apply(values[2 * i], values[2 * i + 1]);
        raw[i] = convert.raw;
    }

    return data;
}

} // namespace detail

template <>
__forceinline__ __device__ Vec<float> Vec<float>::load(const float *ptr) {
    Vec<float> vec;
    float x, y, z, w;
    asm volatile(
        "{ ld.global.v4.f32 {%0, %1, %2, %3}, [%4]; }"
        : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
        : "l"(ptr)
    );
    vec.value[0] = x;
    vec.value[1] = y;
    vec.value[2] = z;
    vec.value[3] = w;
    return vec;
}

template <>
__forceinline__ __device__ void Vec<float>::store(float *ptr) const {
    asm volatile(
        "{ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }"
        :
        : "l"(ptr)
        , "f"(value[0])
        , "f"(value[1])
        , "f"(value[2])
        , "f"(value[3])
    );
}

template <>
__forceinline__ __device__ Vec<half> Vec<half>::load(const half *ptr) {
    Vec<half> vec;
    uint4 data = ld_global_uint4(ptr);

    union {
        uint32_t raw;
        half2 h;
    } convert;

    const uint32_t *raw = reinterpret_cast<const uint32_t *>(&data);

    #pragma unroll
    for (int i = 0; i < SIZE / 2; ++i) {
        convert.raw = raw[i];
        float2 f = __half22float2(convert.h);
        vec.value[2 * i] = f.x;
        vec.value[2 * i + 1] = f.y;
    }

    return vec;
}

template <>
__forceinline__ __device__ void Vec<half>::store(half *ptr) const {
    uint4 data = detail::pack_float_pairs<half2, detail::HalfPairConverter, SIZE / 2>(value);
    st_global_uint4(ptr, data);
}

template <>
__forceinline__ __device__ Vec<__nv_bfloat16> Vec<__nv_bfloat16>::load(const __nv_bfloat16 *ptr) {
    Vec<__nv_bfloat16> vec;
    uint4 data = ld_global_uint4(ptr);

    union {
        uint32_t raw;
        __nv_bfloat162 b;
    } convert;

    const uint32_t *raw = reinterpret_cast<const uint32_t *>(&data);

    #pragma unroll
    for (int i = 0; i < SIZE / 2; ++i) {
        convert.raw = raw[i];
        float2 f = __bfloat1622float2(convert.b);
        vec.value[2 * i] = f.x;
        vec.value[2 * i + 1] = f.y;
    }

    return vec;
}

template <>
__forceinline__ __device__ void Vec<__nv_bfloat16>::store(__nv_bfloat16 *ptr) const {
    uint4 data = detail::pack_float_pairs<__nv_bfloat162, detail::BFloatPairConverter, SIZE / 2>(value);
    st_global_uint4(ptr, data);
}

template <typename T>
struct FloatConvert;

template <>
struct FloatConvert<float> {
    __forceinline__ __device__ static float apply(float value) {
        return value;
    }
};

template <>
struct FloatConvert<half> {
    __forceinline__ __device__ static half apply(float value) {
        return __float2half_rn(value);
    }
};

template <>
struct FloatConvert<__nv_bfloat16> {
    __forceinline__ __device__ static __nv_bfloat16 apply(float value) {
        return __float2bfloat16_rn(value);
    }
};

} // namespace rose
