#pragma once

#include "core/memory.cuh"
#include "core/device_utils.cuh"
#include "core/vector.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace rose {

namespace detail {

__forceinline__ __device__ uint32_t pack_float_2(float a, float b) {
    uint32_t value;
    asm volatile(
        "{ cvt.rn.bf16x2.f32 %0, %1, %2; }"
        : "=r"(value)
        : "f"(b)
        , "f"(a)
    );
    return value;
}

}

template <typename T, size_t SIZE>
struct Arg {
    float v[SIZE];

    __forceinline__ Arg() = default;

    __forceinline__ __device__ Arg(const T *ptr) {
        #pragma unroll
        for (unsigned i = 0; i < SIZE; ++i) {
            v[i] = static_cast<float>(ptr[i]);
        }
    }

    template <typename U>
    __forceinline__ __device__ void store(U *ptr) const {
        #pragma unroll
        for (unsigned i = 0; i < SIZE; ++i) {
            ptr[i] = static_cast<U>(v[i]);
        }
    }

    __forceinline__ __device__ void store(__nv_bfloat16 *ptr) const {
        st_global_nc_uint4(ptr, make_uint4(
            detail::pack_float_2(v[0], v[1]),
            detail::pack_float_2(v[2], v[3]),
            detail::pack_float_2(v[4], v[5]),
            detail::pack_float_2(v[6], v[7])
        ));
    }
};

template <>
__forceinline__ __device__ Arg<__nv_bfloat16, 8>::Arg(const __nv_bfloat16 *ptr) {
    auto from_uint32 = [](uint32_t value) -> float2{
        union {
            uint32_t value;
            __nv_bfloat162 bvalue;
        } temp;
        temp.value = value;
        return __bfloat1622float2(temp.bvalue);
    };

    uint4 data = ld_global_nc_uint4(ptr);
    auto v0 = from_uint32(data.x);
    v[0] = v0.x;
    v[1] = v0.y;
    auto v1 = from_uint32(data.y);
    v[2] = v1.x;
    v[3] = v1.y;

    auto v2 = from_uint32(data.z);
    v[4] = v2.x;
    v[5] = v2.y;
    auto v3 = from_uint32(data.w);
    v[6] = v3.x;
    v[7] = v3.y;
};

template <typename T, size_t SIZE>
struct Acc {
    Arg<T, SIZE> v;

    __forceinline__ __device__ Acc() {
        #pragma unroll
        for (unsigned i = 0; i < SIZE; ++i) {
            v.v[i] = 0.0f;
        }
    }

    template <typename U>
    __forceinline__ __device__ Acc(const Arg<U, SIZE> &arg) {
        #pragma unroll
        for (unsigned i = 0; i < SIZE; ++i) {
            v.v[i] = arg.v[i];
        }
    }

    __forceinline__ __device__ void store(T *ptr) {
        v.store(ptr);
    }

    template <typename U>
    __forceinline__ __device__ void add(float weight, const Arg<U, SIZE> &arg) {
        #pragma unroll
        for (unsigned i = 0; i < SIZE; ++i) {
            v.v[i] += arg.v[i] * weight;
        }
    }

    template <typename U>
    __forceinline__ __device__ void add(const Arg<U, SIZE> &arg) {
        #pragma unroll
        for (unsigned i = 0; i < SIZE; ++i) {
            v.v[i] += arg.v[i];
        }
    }
};

template <typename T, typename U> struct CombineVec {
    static constexpr size_t SIZE = VecStorageSize<T>::SIZE;
    using DstTy = Arg<U, SIZE>;
    using SrcTy = Arg<T, SIZE>;
    using AccTy = Acc<U, SIZE>;
};

} // namespace rose
