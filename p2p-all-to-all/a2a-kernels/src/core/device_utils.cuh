#pragma once

#include "core/common_utils.h"
#include <math_constants.h>

#define ROSE_ENABLE_DEVICE_ASSERT 0

#if ROSE_ENABLE_DEVICE_ASSERT == 1
#define ROSE_DEVICE_ASSERT(cond)                                                                   \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                         \
      asm("trap;");                                                                                \
    }                                                                                              \
  } while (0)
#else
#define ROSE_DEVICE_ASSERT(cond)
#endif

namespace rose {
namespace device {

// A wrapper for the kernels that is used to guard against compilation on
// architectures that will never use the kernel.
template <typename Kernel> struct enable_sm90_or_later : Kernel {
  template <typename... Args> __device__ void operator()(Args &&...args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

__forceinline__ __device__ unsigned warp_sum(unsigned value) {
  value += __shfl_xor_sync(0xffffffff, value, 16);
  value += __shfl_xor_sync(0xffffffff, value, 8);
  value += __shfl_xor_sync(0xffffffff, value, 4);
  value += __shfl_xor_sync(0xffffffff, value, 2);
  value += __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ bool warp_and(bool value) {
  value &= __shfl_xor_sync(0xffffffff, value, 16);
  value &= __shfl_xor_sync(0xffffffff, value, 8);
  value &= __shfl_xor_sync(0xffffffff, value, 4);
  value &= __shfl_xor_sync(0xffffffff, value, 2);
  value &= __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ float half_warp_reduce_max(float value) {
  auto mask = __activemask();
  value = max(value, __shfl_xor_sync(mask, value, 8));
  value = max(value, __shfl_xor_sync(mask, value, 4));
  value = max(value, __shfl_xor_sync(mask, value, 2));
  value = max(value, __shfl_xor_sync(mask, value, 1));
  return value;
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__forceinline__ __device__ uint32_t elect_one_sync() {
#if __CUDA_ARCH__ >= 900
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;
#endif
}

__forceinline__ __device__ int last_active_lane(uint32_t mask) {
    return mask ? (31 - __clz(mask)) : 0;
}

__forceinline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__forceinline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float block_reduce_max(float val, float* smem, int tid, int block_size) {
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int num_warps = (block_size + 31) / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : -CUDART_INF_F;
        val = warp_reduce_max(val);
    }

    if (tid == 0) {
        smem[0] = val;
    }
    __syncthreads();

    return smem[0];
}

__device__ inline float block_reduce_sum(float val, float* smem, int tid, int block_size) {
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int num_warps = (block_size + 31) / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    if (tid == 0) {
        smem[0] = val;
    }
    __syncthreads();

    return smem[0];
}

__device__ inline void build_cdf_tiled(
    const float* probs,
    float* cdf,
    size_t vocab_size,
    float* smem_workspace,
    int tid,
    int block_size
) {
    const int num_warps = (block_size + 31) / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    const size_t chunk_size = (vocab_size + num_warps - 1) / num_warps;
    const size_t chunk_start = static_cast<size_t>(warp_id) * chunk_size;
    const size_t chunk_end = rose::min(chunk_start + chunk_size, vocab_size);

    float running_sum = 0.0f;

    for (size_t idx = chunk_start + lane; idx < chunk_end; idx += 32) {
        uint32_t mask = __activemask();
        float warp_sum = probs[idx];

        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float n = __shfl_up_sync(mask, warp_sum, offset);
            if (lane >= offset) {
                warp_sum += n;
            }
        }

        warp_sum += running_sum;
        cdf[idx] = warp_sum;

        int last_lane = last_active_lane(mask);
        running_sum = __shfl_sync(mask, warp_sum, last_lane);
    }

    if (lane == 0) {
        smem_workspace[warp_id] = running_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        uint32_t active_mask = __ballot_sync(0xffffffff, lane < num_warps);
        if (lane < num_warps) {
            float warp_total = smem_workspace[lane];

            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                float n = __shfl_up_sync(active_mask, warp_total, offset);
                if (lane >= offset) {
                    warp_total += n;
                }
            }

            smem_workspace[lane] = warp_total;
        }
    }
    __syncthreads();

    if (warp_id > 0) {
        float prefix = smem_workspace[warp_id - 1];
        for (size_t idx = chunk_start + lane; idx < chunk_end; idx += 32) {
            cdf[idx] += prefix;
        }
    }
    __syncthreads();
}

__device__ inline int binary_search_cdf(
    const float* cdf,
    size_t vocab_size,
    float sample
) {
    int left = 0;
    int right = static_cast<int>(vocab_size) - 1;

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (cdf[mid] < sample) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return rose::min(rose::max(left, 0), static_cast<int>(vocab_size) - 1);
}

} // namespace device
} // namespace rose
