#include "a2a/a2a_kernels.h"
#include "core/device_utils.cuh"
#include "core/launch_utils.cuh"
#include "core/memory.cuh"

#include <cuda.h>
#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>

#include <cassert>
#include <cstdint>

using namespace rose;
using namespace rose::device;

struct ExpertAndOffset {
    uint32_t expert;
    uint32_t offset;
    uint32_t position;
    float weight;
};


/// Wrapper class to efficiently access the expert indices and offsets.
template<typename NumExpertsPerTokenTy>
class ExpertIterator {
public:
    __forceinline__ __device__ ExpertIterator(
        NumExpertsPerTokenTy num_experts_per_token,
        const int32_t *indices,
        const size_t indices_stride,
        const float *weights,
        const size_t weights_stride,
        const uint32_t *token_offset,
        const uint32_t *expert_offsets,
        unsigned token,
        unsigned experts_per_rank
    ) : num_experts_per_token_(num_experts_per_token),
        indices_(indices),
        indices_stride_(indices_stride),
        weights_(weights),
        weights_stride_(weights_stride),
        token_offset_(token_offset),
        expert_offsets_(expert_offsets),
        token_(token),
        experts_per_rank(experts_per_rank)
    {
    }

    __forceinline__ __device__ ExpertAndOffset operator[](unsigned i) {
        const uint32_t expert = indices_[token_ * indices_stride_ + i];
        const float weight = weights_[token_ * weights_stride_ + i];
        const uint32_t offset = token_offset_[token_ * num_experts_per_token_ + i];
        const uint32_t position = (expert > 0 ? expert_offsets_[expert - 1] : 0) + offset;
        const uint32_t dst_rank = expert / experts_per_rank;
        const uint32_t rank_offset = dst_rank > 0 ? expert_offsets_[dst_rank * experts_per_rank - 1] : 0;
        return {expert, position - rank_offset, position, weight};
    }

private:
    NumExpertsPerTokenTy num_experts_per_token_;
    const int32_t *indices_;
    const size_t indices_stride_;
    const float *weights_;
    const size_t weights_stride_;
    const uint32_t *token_offset_;
    const uint32_t *expert_offsets_;
    unsigned token_;
    unsigned experts_per_rank;
};

template <size_t N>
class ExpertIterator<Fixed<N>> {
public:
    __forceinline__ __device__ ExpertIterator(
        Fixed<N> num_experts_per_token,
        const int32_t *indices,
        const size_t indices_stride,
        const float *weights,
        const size_t weights_stride,
        const uint32_t *token_offset,
        const uint32_t *expert_offsets,
        unsigned token,
        unsigned experts_per_rank
    ) {
        #pragma unroll(N)
        for (unsigned i = 0; i < N; i++) {
            const auto expert = indices[token * indices_stride + i];
            const auto weight = weights[token * weights_stride + i];
            const auto offset = token_offset[token * N + i];
            const uint32_t position = (expert > 0 ? expert_offsets[expert - 1] : 0) + offset;
            const uint32_t dst_rank = expert / experts_per_rank;
            const uint32_t rank_offset = dst_rank > 0 ? expert_offsets[dst_rank * experts_per_rank - 1] : 0;
            experts_[i] = expert;
            weights_[i] = weight;
            offsets_[i] = position - rank_offset;
            positions_[i] = position;
        }
    }

    __forceinline__ __device__ ExpertAndOffset operator[](unsigned i) {
        return {experts_[i], offsets_[i], positions_[i], weights_[i]};
    }

private:
    uint32_t experts_[N];
    float weights_[N];
    uint32_t offsets_[N];
    uint32_t positions_[N];
};

template<bool QUICK, size_t NUM_WARPS, size_t NODE_SIZE, typename TokenDimTy, typename HiddenDimScaleTy, typename NumExpertsPerTokenTy>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 1) void a2a_dispatch_send_kernel(
    const size_t token_dim,
    const size_t token_scale_dim,
    const size_t token_stride,
    size_t hidden_dim,
    size_t hidden_dim_scale,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t max_private_tokens,
    size_t rank,
    size_t dp_size,
    size_t node_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t * __restrict__ bound_m_ptr,
    const std::byte * __restrict__ x_ptr,
    size_t x_elemsize,
    size_t x_stride,
    const float * __restrict__ x_scale_ptr,
    size_t x_scale_elemsize,
    size_t x_scale_stride_elem,
    size_t x_scale_stride_token,
    const int32_t * __restrict__ indices,
    size_t indices_stride,
    const float *__restrict__ weights,
    size_t weights_stride,
    uint32_t * __restrict__ token_offset,
    uint32_t * __restrict__ num_routed,
    uint32_t * __restrict__ expert_offsets,
    uint8_t * __restrict__ dispatch_route_done,
    uint8_t * __restrict__ dispatch_send_done,
    uint8_t * __restrict__ tx_ready,
    std::byte * __restrict__ send_buffer,
    uint32_t * __restrict__ grid_counter,
    uint32_t * __restrict__ sync_counter,
    uint32_t ** __restrict__ sync_ptrs,
    std::byte ** __restrict__ recv_ptrs
) {
    TokenDimTy token_dim_bound(token_dim);
    HiddenDimScaleTy hidden_dim_scale_bound(hidden_dim_scale);
    NumExpertsPerTokenTy num_experts_per_token_bound(num_experts_per_token);

    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    extern __shared__ std::byte shared_memory[];
    constexpr size_t NUM_THREADS = NUM_WARPS * WARP_SIZE;
    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = get_lane_id();

    uint32_t counter = *sync_counter;

    const size_t node_rank = rank / NODE_SIZE;
    const size_t node_group = rank / dp_size;
    const size_t dp_group = rank / dp_size;
    const size_t experts_per_rank = ceil_div<size_t>(num_experts, world_size);
    const size_t first_expert = rank * experts_per_rank;
    const size_t last_expert = min<size_t>(first_expert + experts_per_rank, num_experts);

    const size_t num_send_tokens = bound_m_ptr ? *bound_m_ptr : num_tokens;

    // In the first phase, count how many tokens are sent to each other rank
    // and assign a unique offset to each token within the ranks.
    if (blockIdx.x == 0) {
        uint32_t *tokens_per_expert = (uint32_t*)shared_memory;
        for (uint32_t i = threadIdx.x; i <num_experts; i += blockDim.x) {
            tokens_per_expert[i] = 0;
        }
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < num_send_tokens * num_experts_per_token_bound; i += blockDim.x) {
            const uint32_t token = i / num_experts_per_token_bound;
            const uint32_t index = i % num_experts_per_token_bound;
            const uint32_t expert = __ldg(&indices[token * indices_stride + index]);

            // Assign an offset to the token within the current rank and expert.
            token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
        }
        __syncthreads();

        // Find the start offset of each rank by computing a cumulative sum within tokens_per_rank.
        // Compute sums within each warp and store the sums in shared memory.
        const uint32_t i = threadIdx.x;
        const uint32_t num_warps = ceil_div<size_t>(num_experts, WARP_SIZE);
        uint32_t *expert_sums = (uint32_t*)shared_memory;

        uint32_t *local_num_routed = num_routed + dp_group * num_experts;
        uint32_t expert_offset = 0;
        if (i < num_experts) {
            expert_offset = tokens_per_expert[i];
            local_num_routed[i] = expert_offset;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            st_mmio_b8(dispatch_route_done, 1);
        }
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            unsigned warp_sum_expert = __shfl_up_sync(0xFFFFFFFF, expert_offset, offset);
            if (lane_id >= offset) {
                expert_offset += warp_sum_expert;
            }
        }
        if (lane_id == WARP_SIZE - 1) {
            expert_sums[warp_id] = expert_offset;
        }
        __syncthreads();

        // Sum up the warp sums in the first warp.
        if (warp_id == 0) {
            uint32_t total_expert_sum = (lane_id < num_warps) ? expert_sums[lane_id] : 0;
            for (int offset = 1; offset < num_warps; offset <<= 1) {
                unsigned warp_sum = __shfl_up_sync(0xFFFFFFFF, total_expert_sum, offset);
                if (lane_id >= offset) {
                    total_expert_sum += warp_sum;
                }
            }
            if (lane_id < num_warps) {
                expert_sums[lane_id] = total_expert_sum;
            }
        }
        __syncthreads();

        // Add the sums to the token counts to find the start offset of each expert.
        if (i < num_experts) {
            if (warp_id > 0) {
                expert_offsets[i] = expert_sums[warp_id - 1] + expert_offset;
            } else {
                expert_offsets[i] = expert_offset;
            }
        }
    }
    __syncthreads();

    // NVLink barrier set on the end of combine.
    if (NODE_SIZE > 1) {
        if (blockIdx.x == 0) {
            auto local_rank = rank % NODE_SIZE;
            for (unsigned peer = threadIdx.x; peer < NODE_SIZE; peer += blockDim.x) {
                while (ld_volatile_u32(&sync_ptrs[local_rank][peer]) != counter);
            }
        }
    }

    // Wait for all transactions using the send buffer to finish before writing to it.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (ld_mmio_b8(tx_ready) == 0);
    }

    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *sync_counter = counter + 1;
    }

    if constexpr (QUICK) {
        unsigned token = blockIdx.x;
        if (token < num_send_tokens) {
            uint4 *x_token_src = (uint4*)(x_ptr + token * x_stride);
            float *x_scale_src = (float*)(x_scale_ptr + token * x_scale_stride_token);

            ExpertIterator<NumExpertsPerTokenTy> expert_iterator(
                num_experts_per_token_bound,
                indices,
                indices_stride,
                weights,
                weights_stride,
                token_offset,
                expert_offsets,
                token,
                experts_per_rank
            );

            if constexpr (std::is_same_v<TokenDimTy, NotFixed>) {
                // Copy to shared memory and sync up threads.
                for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim_bound; i += NUM_THREADS) {
                    const bool has_scale = x_scale_ptr && i < hidden_dim_scale_bound;

                    uint4 val = ld_global_nc_uint4(&x_token_src[i]);
                    float scale_val;
                    if (has_scale) {
                        scale_val =  *(float*)(x_scale_src + i * x_scale_stride_elem);
                    }

                    // Copy from shared memory to the send buffer, ensuring a contiguous layout per rank.
                    #pragma unroll
                    for (unsigned e = 0; e < num_experts_per_token_bound; e++) {
                        auto route = expert_iterator[e];
                        const uint32_t dst_rank = route.expert / experts_per_rank;
                        const uint32_t dst_node = dst_rank / NODE_SIZE;

                        // If the destination is within the same node, write using NVLink.
                        if (dst_node == node_rank && dst_rank != rank && route.offset < max_private_tokens) {
                            if (dst_rank % dp_size == rank % dp_size) {
                                // Write to the private recv buffer directly using NVLink.
                                const uint32_t local_peer = dst_rank % NODE_SIZE;
                                std::byte *token_ptr = recv_ptrs[local_peer] + (node_group * max_private_tokens + route.offset) * token_stride;
                                uint4 *x_token_dst = (uint4*)token_ptr;
                                st_global_nc_uint4(&x_token_dst[i], val);
                                if (has_scale) {
                                    *((float*)(token_ptr + token_dim_bound) + i) = scale_val;
                                }
                            }
                        } else {
                            // Always write into the send buffer for local copies.
                            std::byte *token_ptr = send_buffer + route.position * token_stride;
                            uint4 *x_token_dst = (uint4*)token_ptr;
                            st_global_nc_uint4(&x_token_dst[i], val);
                            if (has_scale) {
                                *((float*)(token_ptr + token_dim_bound) + i) = scale_val;
                            }
                        }
                    }
                }

                if (threadIdx.x == 0) {
                    auto counter = add_release_gpu_u32(grid_counter, 1) + 1;
                    if (counter == num_send_tokens) {
                        st_mmio_b8(dispatch_send_done, 1);
                        *grid_counter = 0;
                    }
                }
            } else {
                constexpr size_t TOKEN_DIM = TokenDimTy::Value;
                constexpr size_t NUM_STEPS = (TOKEN_DIM + NUM_THREADS - 1) / NUM_THREADS;

                uint4 vals[NUM_STEPS];
                float scales[NUM_STEPS];

                #pragma unroll(NUM_STEPS)
                for (unsigned i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM; i += NUM_THREADS, s++) {
                    const bool has_scale = x_scale_ptr && i < hidden_dim_scale_bound;
                    vals[s] = ld_global_nc_uint4(&x_token_src[i]);
                    if (has_scale) {
                        scales[s] = *(float*)(x_scale_src + i * x_scale_stride_elem);
                    }
                }

                // Copy from shared memory to the send buffer, ensuring a contiguous layout per rank.
                #pragma unroll
                for (unsigned e = 0; e < num_experts_per_token_bound; e++) {
                    auto route = expert_iterator[e];
                    const uint32_t dst_rank = route.expert / experts_per_rank;
                    const uint32_t dst_node = dst_rank / NODE_SIZE;

                    // If the destination is within the same node, write using NVLink.
                    if (dst_node != node_rank || dst_rank == rank || route.offset >= max_private_tokens) {
                        // Always write into the send buffer for local copies.
                        std::byte *token_ptr = send_buffer + route.position * token_stride;
                        uint4 *x_token_dst = (uint4*)token_ptr;
                        for (unsigned i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM; i += NUM_THREADS, s++) {
                            const bool has_scale = x_scale_ptr && i < hidden_dim_scale_bound;
                            st_global_nc_uint4(&x_token_dst[i], vals[s]);
                            if (has_scale) {
                                *((float*)(token_ptr + token_dim_bound) + i) = scales[s];
                            }
                        }
                    }
                }

                __syncthreads();

                if (threadIdx.x == 0) {
                    auto counter = add_release_gpu_u32(grid_counter, 1) + 1;
                    if (counter == num_send_tokens) {
                        st_mmio_b8(dispatch_send_done, 1);
                        *grid_counter = 0;
                    }
                }

                grid.sync();

                #pragma unroll
                for (unsigned e = 0; e < num_experts_per_token_bound; e++) {
                    auto route = expert_iterator[e];
                    const uint32_t dst_rank = route.expert / experts_per_rank;
                    const uint32_t dst_node = dst_rank / NODE_SIZE;

                    // If the destination is within the same node, write using NVLink.
                    if (dst_node == node_rank && dst_rank != rank && route.offset < max_private_tokens) {
                        // Write to the private recv buffer directly using NVLink.
                        const uint32_t local_peer = dst_rank % NODE_SIZE;
                        std::byte *token_ptr = recv_ptrs[local_peer] + (node_group * max_private_tokens + route.offset) * token_stride;
                        uint4 *x_token_dst = (uint4*)token_ptr;
                        for (unsigned i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM; i += NUM_THREADS, s++) {
                            const bool has_scale = x_scale_ptr && i < hidden_dim_scale_bound;
                            st_global_nc_uint4(&x_token_dst[i], vals[s]);
                            if (has_scale) {
                                *((float*)(token_ptr + token_dim_bound) + i) = scales[s];
                            }
                        }
                    }
                }
            }
        } else {
            if constexpr (!std::is_same_v<TokenDimTy, NotFixed>) {
                grid.sync();
            }
        }
    } else {
        // Copy the tokens to their corresponding position in the send buffer via shared memory.
        unsigned num_local_tokens = 0;
        for (unsigned token = blockIdx.x; token < num_send_tokens; token += gridDim.x, num_local_tokens++) {
            uint4 *x_token_src = (uint4*)(x_ptr + token * x_stride);
            float *x_scale_src = (float*)(x_scale_ptr + token * x_scale_stride_token);

            ExpertIterator<NumExpertsPerTokenTy> expert_iterator(
                num_experts_per_token_bound,
                indices,
                indices_stride,
                weights,
                weights_stride,
                token_offset,
                expert_offsets,
                token,
                experts_per_rank
            );


            // Copy to shared memory and sync up threads.
            for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim_bound; i += blockDim.x) {
                const bool has_scale = x_scale_ptr && i < hidden_dim_scale_bound;

                uint4 val = ld_global_nc_uint4(&x_token_src[i]);
                float scale_val;
                if (has_scale) {
                    scale_val =  *(float*)(x_scale_src + i * x_scale_stride_elem);
                }

                // Copy from shared memory to the send buffer, ensuring a contiguous layout per rank.
                #pragma unroll
                for (unsigned e = 0; e < num_experts_per_token_bound; e++) {
                    auto route = expert_iterator[e];
                    const uint32_t dst_rank = route.expert / experts_per_rank;
                    const uint32_t dst_node = dst_rank / NODE_SIZE;

                    // If the destination is within the same node, write using NVLink.
                    if (dst_node == node_rank && dst_rank != rank && route.offset < max_private_tokens) {
                        continue;
                    } else {
                        // Always write into the send buffer for local copies.
                        std::byte *token_ptr = send_buffer + route.position * token_stride;
                        uint4 *x_token_dst = (uint4*)token_ptr;
                        st_global_nc_uint4(&x_token_dst[i], val);
                        if (has_scale) {
                            *((float*)(token_ptr + token_dim_bound) + i) = scale_val;
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            auto counter = add_release_gpu_u32(grid_counter, num_local_tokens) + num_local_tokens;
            if (counter == num_send_tokens) {
                st_mmio_b8(dispatch_send_done, 1);
                *grid_counter = 0;
            }
        }

        if (NODE_SIZE >= 1) {
            for (unsigned token = blockIdx.x; token < num_send_tokens; token += gridDim.x) {
                uint4 *x_token_src = (uint4*)(x_ptr + token * x_stride);
                float *x_scale_src = (float*)(x_scale_ptr + token * x_scale_stride_token);

                ExpertIterator<NumExpertsPerTokenTy> expert_iterator(
                    num_experts_per_token_bound,
                    indices,
                    indices_stride,
                    weights,
                    weights_stride,
                    token_offset,
                    expert_offsets,
                    token,
                    experts_per_rank
                );

                // Copy to shared memory and sync up threads.
                for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim_bound; i += blockDim.x) {
                    const bool has_scale = x_scale_ptr && i < hidden_dim_scale_bound;

                    uint4 val = ld_global_nc_uint4(&x_token_src[i]);
                    float scale_val;
                    if (has_scale) {
                        scale_val =  *(float*)(x_scale_src + i * x_scale_stride_elem);
                    }

                    // Copy from shared memory to the send buffer, ensuring a contiguous layout per rank.
                    #pragma unroll
                    for (unsigned e = 0; e < num_experts_per_token_bound; e++) {
                        auto route = expert_iterator[e];
                        const uint32_t dst_rank = route.expert / experts_per_rank;
                        const uint32_t dst_node = dst_rank / NODE_SIZE;

                        // If the destination is within the same node, write using NVLink.
                        if (dst_node == node_rank && dst_rank != rank && route.offset < max_private_tokens) {
                            if (dst_rank % dp_size == rank % dp_size) {
                                // Write to the private recv buffer directly using NVLink.
                                const uint32_t local_peer = dst_rank % NODE_SIZE;
                                std::byte *token_ptr = recv_ptrs[local_peer] + (node_group * max_private_tokens + route.offset) * token_stride;
                                uint4 *x_token_dst = (uint4*)token_ptr;
                                st_global_nc_uint4(&x_token_dst[i], val);
                                if (has_scale) {
                                    *((float*)(token_ptr + token_dim_bound) + i) = scale_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (NODE_SIZE > 1) {
        grid.sync();

        if (blockIdx.x == 0) {
            auto local_rank = rank % NODE_SIZE;
            if (threadIdx.x < NODE_SIZE) {
                auto *flag = &sync_ptrs[threadIdx.x][local_rank + NODE_SIZE];
                st_release_u32(flag, counter + 1);
            }
        }
    }
}


int a2a_kernels::a2a_dispatch_send(
    size_t num_blocks,
    size_t hidden_dim,
    size_t hidden_dim_scale,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t max_private_tokens,
    size_t rank,
    size_t dp_size,
    size_t node_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t *bound_m_ptr,
    const uint8_t *x_ptr,
    size_t x_elemsize,
    size_t x_stride,
    const uint8_t *x_scale_ptr,
    size_t x_scale_elemsize,
    size_t x_scale_stride_elem,
    size_t x_scale_stride_token,
    const int32_t *indices,
    size_t indices_stride,
    const float *weights,
    size_t weights_stride,
    uint32_t *token_offset,
    uint32_t *num_routed,
    uint32_t *expert_offsets,
    uint8_t *dispatch_route_done,
    uint8_t *dispatch_send_done,
    uint8_t *tx_ready,
    uint8_t *send_buffer,
    uint32_t *grid_counter,
    uint32_t *sync_counter,
    uint32_t **sync_ptrs,
    uint8_t **recv_ptrs,
    uint64_t stream
) {
    constexpr size_t NUM_WARPS = 16;
    constexpr size_t NUM_THREADS = NUM_WARPS * WARP_SIZE;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    // There should be enough warps to do a horizontal reduction across ranks.
    assert(world_size <= NUM_THREADS);
    assert(num_experts <= NUM_THREADS);

    const size_t token_dim = round_up<size_t>(hidden_dim * x_elemsize, sizeof(int4));
    const size_t token_scale_dim = round_up<size_t>(hidden_dim_scale * x_scale_elemsize, sizeof(int4));
    const size_t token_stride = token_dim + token_scale_dim + 16;
    assert(token_stride % sizeof(int4) == 0);

    void *args[] = {
        const_cast<size_t *>(&token_dim),
        const_cast<size_t *>(&token_scale_dim),
        const_cast<size_t *>(&token_stride),
        &hidden_dim,
        &hidden_dim_scale,
        &num_experts,
        &num_experts_per_token,
        &max_private_tokens,
        &rank,
        &dp_size,
        &node_size,
        &world_size,
        &num_tokens,
        &bound_m_ptr,
        &x_ptr,
        &x_elemsize,
        &x_stride,
        &x_scale_ptr,
        &x_scale_elemsize,
        &x_scale_stride_elem,
        &x_scale_stride_token,
        &indices,
        &indices_stride,
        &weights,
        &weights_stride,
        &token_offset,
        &num_routed,
        &expert_offsets,
        &dispatch_route_done,
        &dispatch_send_done,
        &tx_ready,
        &send_buffer,
        &grid_counter,
        &sync_counter,
        &sync_ptrs,
        &recv_ptrs,
    };

    const size_t shared_memory_send = std::max(num_experts, NUM_WARPS) * sizeof(uint32_t);

    nvtxRangePush("dispatch_send");
    cudaError_t status;
    LAUNCH_TOKEN_DIM_DISPATCH(token_dim, TokenDim, {
        LAUNCH_NUM_EXPERTS_PER_TOKEN(num_experts_per_token, NumExpertsPerToken, {
            LAUNCH_HIDDEN_DIM_SCALE(hidden_dim_scale, HiddenDimScale, {
                LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
                    if (num_blocks >= num_tokens) {
                        status = cudaLaunchCooperativeKernel(
                            (void *)&a2a_dispatch_send_kernel<
                                true,
                                NUM_WARPS,
                                NODE_SIZE,
                                TokenDim,
                                HiddenDimScale,
                                NumExpertsPerToken
                            >,
                            dimGrid,
                            dimBlock,
                            args,
                            shared_memory_send,
                            (cudaStream_t)stream
                        );
                    } else {
                        status = cudaLaunchCooperativeKernel(
                            (void *)&a2a_dispatch_send_kernel<
                                false,
                                NUM_WARPS,
                                NODE_SIZE,
                                TokenDim,
                                HiddenDimScale,
                                NumExpertsPerToken
                            >,
                            dimGrid,
                            dimBlock,
                            args,
                            shared_memory_send,
                            (cudaStream_t)stream
                        );
                    }
                });
            });
        });
    });
    nvtxRangePop();
    return status;
}
