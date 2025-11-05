#pragma once

#include <climits>
#include <cstdint>

#ifdef __CUDA_ARCH__
#define ROSE_HOST_DEVICE __host__ __device__
#else
#define ROSE_HOST_DEVICE
#endif

namespace rose {

/// The fixed warp size.
constexpr size_t WARP_SIZE = 32;

/// Return the next power of 2 following the given number.
ROSE_HOST_DEVICE inline uint32_t next_pow_2(const uint32_t num) {
  if (num <= 1) {
    return num;
  }
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename T> ROSE_HOST_DEVICE T ceil_div(T x, T y) { return (x + y - 1) / y; }

template <typename T> ROSE_HOST_DEVICE T round_up(T x, T y) { return ceil_div<T>(x, y) * y; }

template <typename T> ROSE_HOST_DEVICE T min(T x, T y) { return x < y ? x : y; }

template <typename T> ROSE_HOST_DEVICE T max(T x, T y) { return x > y ? x : y; }

} // namespace rose
