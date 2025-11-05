#include "a2a/a2a_kernels.h"
#include "core/memory.cuh"
#include "core/device_utils.cuh"
#include "core/combine_utils.cuh"
#include "core/launch_utils.cuh"

#include <cuda.h>
#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>

#include <cassert>
#include <cstdint>

#include <type_traits>

using namespace rose;
using namespace rose::device;


template <unsigned NUM_WARPS, unsigned NODE_SIZE, typename T, typename U, typename NumExpertsPerToken>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 1) void a2a_combine_recv_kernel(
    const size_t token_dim,
    size_t hidden_dim,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t rank,
    size_t world_size,
    size_t num_tokens,
    const int32_t *bound_m_ptr,
    const int32_t *indices_ptr,
    const size_t indices_stride,
    const float *weights_ptr,
    const size_t weights_stride,
    U *out_tokens_ptr,
    size_t out_tokens_stride,
    uint8_t accumulate,
    std::byte *recv_buffer,
    uint32_t *token_offset,
    uint32_t *expert_offsets,
    uint8_t *combine_recv_flag,
    uint8_t *combine_recv_done,
    uint32_t *sync_counter,
    uint32_t **sync_ptrs
) {
    extern __shared__ std::byte shared_memory[];

    auto grid = cooperative_groups::this_grid();
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    const unsigned lane_id = get_lane_id();

    // Determine the number of tokens to combine on the current rank.
    const size_t num_send_tokens = bound_m_ptr ? *bound_m_ptr : num_tokens;

    // In a first pass, copy the positions into shared memory.
    // This block processes tokens blockIdx.x + i * gridDim.x.
    uint32_t *positions = reinterpret_cast<uint32_t *>(shared_memory);
    {
        uint32_t i = threadIdx.x;
        for (;;) {
            const uint32_t local_token = i / num_experts_per_token;
            const uint32_t token = blockIdx.x + local_token * gridDim.x;
            const uint32_t route = i % num_experts_per_token;
            if (token >= num_send_tokens) {
                break;
            }

            const uint32_t global_slot = token * num_experts_per_token + route;
            const uint32_t local_slot = local_token * num_experts_per_token + route;

            const uint32_t expert = indices_ptr[token * indices_stride + route];
            const uint32_t offset = token_offset[global_slot];
            const uint32_t position = (expert > 0 ? expert_offsets[expert - 1] : 0) + offset;
            positions[local_slot] = position;
            i += blockDim.x;
        }
        __syncthreads();
    }

    // Wait for NVLink transfers to complete and the EFA recv flag to be set.
    auto counter = *sync_counter;
    if (warp_id == 0) {
        if (elect_one_sync()) {
            while (ld_mmio_b8(combine_recv_flag) == 0);
        }
    } else if (warp_id == 1 && NODE_SIZE > 1) {
        auto local_rank = rank % NODE_SIZE;
        if (lane_id < NODE_SIZE) {
            auto *flag_ptr = &sync_ptrs[local_rank][lane_id + NODE_SIZE];
            while (ld_acquire_u32(flag_ptr) != counter);
        }
    }
    __syncthreads();

    for (unsigned token = blockIdx.x, local_token = 0; token < num_send_tokens; token += gridDim.x, local_token++) {
        U *dstPtr = out_tokens_ptr + token * out_tokens_stride;

        NumExpertsPerToken experts_per_token_bound(num_experts_per_token);
        using VecTy = CombineVec<T, U>;
        using DstTy = typename VecTy::DstTy;
        using SrcTy = typename VecTy::SrcTy;
        using AccTy = typename VecTy::AccTy;
        constexpr unsigned VEC_SIZE = VecTy::SIZE;

        if constexpr (std::is_same<NumExpertsPerToken, NotFixed>::value) {
            for (unsigned j = threadIdx.x * VEC_SIZE; j < hidden_dim; j += blockDim.x * VEC_SIZE) {
                AccTy acc = accumulate ? DstTy(dstPtr + j) : AccTy();

                #pragma unroll(8)
                for (unsigned k = 0; k < experts_per_token_bound; ++k) {
                    const float weight = weights_ptr[token * weights_stride + k];
                    const uint32_t position = positions[local_token * num_experts_per_token + k];

                    T *buffer = (T*)(recv_buffer + position * token_dim);
                    acc.add(weight, SrcTy(buffer + j));
                }

                acc.store(dstPtr + j);
            }
        } else {
            static constexpr size_t NUM_EXPERTS = NumExpertsPerToken::Value;
            T *tokens[NUM_EXPERTS];
            float weights[NUM_EXPERTS];

            #pragma unroll(NUM_EXPERTS)
            for (unsigned k = 0; k < NUM_EXPERTS; ++k) {
                const uint32_t position = positions[local_token * num_experts_per_token + k];
                tokens[k] = (T*)(recv_buffer + position * token_dim);
                weights[k] = weights_ptr[token * weights_stride + k];
            }

            for (unsigned j = threadIdx.x * VEC_SIZE; j < hidden_dim; j += blockDim.x * VEC_SIZE) {
                AccTy acc = accumulate ? DstTy(dstPtr + j) : AccTy();

                SrcTy srcs[NUM_EXPERTS];
                #pragma unroll(NUM_EXPERTS)
                for (unsigned k = 0; k < NUM_EXPERTS; ++k) {
                    srcs[k] = SrcTy(tokens[k] + j);
                }

                #pragma unroll(NUM_EXPERTS)
                for (unsigned k = 0; k < NUM_EXPERTS; ++k) {
                    acc.add(weights[k], srcs[k]);
                }

                acc.store(dstPtr + j);
            }
        }
    }

    grid.sync();

    if (blockIdx.x == 0) {
        if (warp_id == 0) {
            if (elect_one_sync()) {
                st_mmio_b8(combine_recv_done, 1);
                *combine_recv_flag = 0;
                *sync_counter = counter + 1;
            }
        } else if (warp_id == 1 && NODE_SIZE > 1) {
            auto local_rank = rank % NODE_SIZE;
            unsigned peer = lane_id;
            if (peer < NODE_SIZE) {
                st_volatile_u32(&sync_ptrs[local_rank][peer], counter + 1);
            }
        }
    }
}


int a2a_kernels::a2a_combine_recv(
    size_t num_blocks,
    size_t hidden_dim,
    size_t x_elemsize,
    ScalarType in_dtype,
    ScalarType out_dtype,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t rank,
    size_t node_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t *bound_m_ptr,
    const int32_t *indices_ptr,
    size_t indices_stride,
    const float *weights_ptr,
    size_t weights_stride,
    uint8_t *out_tokens_ptr,
    size_t out_tokens_stride,
    bool accumulate,
    uint8_t *recv_buffer,
    uint32_t *token_offset,
    uint32_t *expert_offsets,
    uint8_t *combine_recv_flag,
    uint8_t *combine_recv_done,
    uint32_t *sync_counter,
    uint32_t **sync_ptrs,
    uint64_t stream
) {
    const size_t token_dim = round_up<size_t>(hidden_dim * x_elemsize, sizeof(int4));
    const size_t tokens_per_block = ceil_div<size_t>(num_tokens, num_blocks);

    void *args[] = {
        const_cast<size_t *>(&token_dim),
        &hidden_dim,
        &num_experts,
        &num_experts_per_token,
        &rank,
        &world_size,
        &num_tokens,
        &bound_m_ptr,
        &indices_ptr,
        &indices_stride,
        &weights_ptr,
        &weights_stride,
        &out_tokens_ptr,
        &out_tokens_stride,
        &accumulate,
        &recv_buffer,
        &token_offset,
        &expert_offsets,
        &combine_recv_flag,
        &combine_recv_done,
        &sync_counter,
        &sync_ptrs,
    };

    constexpr size_t NUM_WARPS = 16;
    constexpr size_t WARP_SIZE = 32;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * WARP_SIZE, 1, 1);

    const size_t shared_memory = tokens_per_block * num_experts_per_token * sizeof(uint32_t);

    cudaError_t status;
    nvtxRangePush("combine_recv");
    LAUNCH_BASIC_FLOAT(in_dtype, InTy, {
        LAUNCH_BASIC_FLOAT(out_dtype, OutTy, {
            LAUNCH_NUM_EXPERTS_PER_TOKEN(num_experts_per_token, NumExpertsPerToken, {
                LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
                    status = cudaLaunchCooperativeKernel(
                        (void *)&a2a_combine_recv_kernel<NUM_WARPS, NODE_SIZE, InTy, OutTy, NumExpertsPerToken>,
                        dimGrid,
                        dimBlock,
                        args,
                        shared_memory,
                        (cudaStream_t)stream
                    );
                });
            })
        })
    });
    nvtxRangePop();
    return status;
}
