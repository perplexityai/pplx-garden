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

template<unsigned NUM_WARPS, unsigned NODE_SIZE, typename TokenDimTy, typename HiddenDimScaleTy>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE, 1)
void a2a_dispatch_recv_kernel(
    const size_t token_dim,
    const size_t token_scale_dim,
    const size_t token_stride,
    size_t hidden_dim,
    size_t hidden_dim_scale,
    size_t x_elemsize,
    size_t x_scale_elemsize,
    size_t num_experts,
    size_t rank,
    size_t world_size,
    int32_t * __restrict__ out_num_tokens_ptr,
    std::byte * __restrict__ out_x_ptr,
    size_t out_x_stride,
    float * __restrict__ out_x_scale_ptr,
    size_t out_x_scale_stride_elem,
    size_t out_x_scale_stride_token,
    uint32_t * __restrict__ tokens_per_expert,
    std::byte * __restrict__ send_buffer,
    std::byte * __restrict__ recv_buffer,
    uint32_t * __restrict__ source_rank,
    uint32_t * __restrict__ source_offset,
    uint32_t * __restrict__ padded_index,
    uint32_t * __restrict__ num_routed,
    uint32_t * __restrict__ num_recv_tokens_ptr,
    uint8_t * __restrict__ num_recv_tokens_flag,
    uint8_t * __restrict__ dispatch_recv_flag,
    uint8_t * __restrict__ dispatch_recv_done,
    uint32_t * __restrict__ grid_counter,
    uint32_t * __restrict__ sync_counter,
    uint32_t ** __restrict__ sync_ptrs,
    std::byte **send_ptrs
) {
    TokenDimTy token_dim_bound(token_dim);
    HiddenDimScaleTy hidden_dim_scale_bound(hidden_dim_scale);

    struct SharedStage {
        uint32_t src_index;
        uint32_t dst_index;
    };
    struct LocalStage {
        uint4 *x_token_src;
        uint4 *x_token_dst;
        float *x_scale_src;
        float *x_scale_dst;
    };
    constexpr size_t NUM_STAGES = 8;

    __shared__ SharedStage shared_stage[NUM_STAGES];
    LocalStage local_stage[NUM_STAGES];

    auto shared_to_local = [&]{
        #pragma unroll(NUM_STAGES)
        for (unsigned i = 0; i < NUM_STAGES; ++i) {
            uint32_t src_index = shared_stage[i].src_index;
            uint32_t dst_index = shared_stage[i].dst_index;
            local_stage[i].x_token_src = (uint4*)(recv_buffer + src_index * token_stride);
            local_stage[i].x_scale_src = (float*)(recv_buffer + src_index * token_stride + token_dim_bound);
            local_stage[i].x_token_dst = (uint4*)(out_x_ptr + dst_index * out_x_stride);
            local_stage[i].x_scale_dst = (float*)(out_x_scale_ptr + dst_index * out_x_scale_stride_token);
        }
        __syncthreads();
    };

    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    const unsigned lane_id = get_lane_id();

    const size_t experts_per_rank = ceil_div<size_t>(num_experts, world_size);
    const size_t first_expert = rank * experts_per_rank;
    const size_t last_expert = min<size_t>(first_expert + experts_per_rank, num_experts);

    // Wait for NVLink transfers to complete.
    auto counter = *sync_counter;
    if (warp_id == 0) {
        if (elect_one_sync()) {
            while (ld_mmio_b8(num_recv_tokens_flag) == 0);
        }
    } else if (warp_id == 1) {
        if constexpr (NODE_SIZE > 1) {
            auto local_rank = rank % NODE_SIZE;
            if (lane_id < NODE_SIZE) {
                auto *flag_ptr = &sync_ptrs[local_rank][lane_id + NODE_SIZE];
                while (ld_acquire_u32(flag_ptr) != counter);
            }
        }
    }
    __syncthreads();

    // Wait for the worker to indicate the number of tokens received.
    const unsigned num_recv_tokens = ld_volatile_u32(num_recv_tokens_ptr);
    const unsigned num_efa_tokens = ld_volatile_u32(num_recv_tokens_ptr + 1);

    // Pre-populate token information into the pipeline.
    auto next_token = blockIdx.x + threadIdx.x * gridDim.x;
    if (threadIdx.x < NUM_STAGES && next_token < num_efa_tokens) {
        shared_stage[threadIdx.x].src_index = source_offset[next_token];
        shared_stage[threadIdx.x].dst_index = padded_index[next_token];
    }
    __syncthreads();

    // Wait for the worker to indicate receipt of payloads. Since the EFA payloads might
    // arrive later, time is spent here copying from local ranks and nodes.
    for (unsigned token = num_efa_tokens + blockIdx.x; token < num_recv_tokens; token += gridDim.x) {
        auto padded_token = padded_index[token];
        auto token_rank = source_rank[token];

        // Token originates from the local rank.
        auto local_rank = token_rank % NODE_SIZE;
        auto position = source_offset[token];
        uint4 *x_token_src;
        if (token_rank == rank) {
            x_token_src = (uint4*)(send_buffer + position * token_stride);
        } else if (position & (1u << 31)) {
            x_token_src = (uint4*)(send_ptrs[local_rank] + (position & ~(1u << 31)) * token_stride);
        } else {
            x_token_src = (uint4*)(recv_buffer + position * token_stride);
        }

        // Token originates from the local node - copy it from an NVLink buffer.
        uint4 *x_token_dst = (uint4*)(out_x_ptr + padded_token * out_x_stride);
        float *x_scale_src = (float*)((std::byte*)x_token_src + token_dim);
        float *x_scale_dst = (float*)(out_x_scale_ptr + padded_token * out_x_scale_stride_token);
        for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim; i += blockDim.x) {
            const bool has_scale = out_x_scale_ptr && i < hidden_dim_scale_bound;
            auto val = ld_global_nc_uint4(&x_token_src[i]);

            float scale;
            if (has_scale) {
                scale = x_scale_src[i];
            }

            st_global_nc_uint4(&x_token_dst[i], val);
            if (has_scale) {
                x_scale_dst[i * out_x_scale_stride_elem] = scale;
            }
        }
    }

    // NVLink barrier to avoid combine overwriting the buffers.
    if constexpr (NODE_SIZE > 1) {
        grid.sync();
        if (blockIdx.x == 0) {
            if (threadIdx.x == 0) {
                *sync_counter = counter + 1;
            }
            auto local_rank = rank % NODE_SIZE;
            for (unsigned peer = threadIdx.x; peer < NODE_SIZE; peer += blockDim.x) {
                st_volatile_u32(&sync_ptrs[peer][local_rank], counter + 1);
            }
        }
    }

    shared_to_local();

    // Wait for the worker to indicate that all tokens have been received over EFA.
    if (warp_id == 0) {
        if (elect_one_sync()) {
            while (ld_mmio_b8(dispatch_recv_flag) == 0);
        }
    }
    __syncthreads();

    // Copy the tokens into the output buffer.
    unsigned num_local_tokens = 0;
    unsigned token = blockIdx.x;
    while (token < num_efa_tokens) {
        auto next_token = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
        if (threadIdx.x < NUM_STAGES && next_token < num_efa_tokens) {
            shared_stage[threadIdx.x].src_index = source_offset[next_token];
            shared_stage[threadIdx.x].dst_index = padded_index[next_token];
        }
        __syncthreads();

        for (unsigned s = 0; s < NUM_STAGES && token < num_efa_tokens; s++) {
            // Copy the token.
            uint4 *x_token_src = local_stage[s].x_token_src;
            uint4 *x_token_dst = local_stage[s].x_token_dst;
            float *x_scale_dst = local_stage[s].x_scale_dst;
            float *x_scale_src = local_stage[s].x_scale_src;

            for (unsigned i = threadIdx.x; i * sizeof(uint4) < token_dim_bound; i += blockDim.x) {
                const bool has_scale = out_x_scale_ptr && i < hidden_dim_scale_bound;
                auto val = ld_global_nc_uint4(&x_token_src[i]);
                float scale;
                if (has_scale) {
                    scale = x_scale_src[i];
                }
                st_global_nc_uint4(&x_token_dst[i], val);
                if (has_scale) {
                    x_scale_dst[i * out_x_scale_stride_elem] = scale;
                }
            }

            token += gridDim.x;
            num_local_tokens++;
        }

        shared_to_local();
    }

    if (blockIdx.x == 0) {
        for (unsigned expert = threadIdx.x; expert < last_expert - first_expert; expert += blockDim.x) {
            out_num_tokens_ptr[expert] = tokens_per_expert[expert];
        }
    }

    if (threadIdx.x == 0) {
        auto counter = add_release_gpu_u32(grid_counter, num_local_tokens) + num_local_tokens;
        if (counter == num_efa_tokens) {
            st_mmio_b8(dispatch_recv_done, 1);
            // Reset the state.
            *num_recv_tokens_flag = 0;
            *dispatch_recv_flag = 0;
            *grid_counter = 0;
        }
    }
}


int a2a_kernels::a2a_dispatch_recv(
    size_t num_blocks,
    size_t hidden_dim,
    size_t hidden_dim_scale,
    size_t x_elemsize,
    size_t x_scale_elemsize,
    size_t num_experts,
    size_t rank,
    size_t node_size,
    size_t world_size,
    int32_t *out_num_tokens_ptr,
    uint8_t *out_x_ptr,
    size_t out_x_stride,
    uint8_t *out_x_scale_ptr,
    size_t out_x_scale_stride_elem,
    size_t out_x_scale_stride_token,
    uint32_t *tokens_per_expert,
    uint8_t *send_buffer,
    uint8_t *recv_buffer,
    uint32_t *source_rank,
    uint32_t *source_offset,
    uint32_t *padded_index,
    uint32_t *num_routed,
    uint32_t *num_recv_tokens_ptr,
    uint8_t *num_recv_tokens_flag,
    uint8_t *dispatch_recv_flag,
    uint8_t *dispatch_recv_done,
    uint32_t *grid_counter,
    uint32_t *sync_counter,
    uint32_t **sync_ptrs,
    uint8_t **send_ptrs,
    uint64_t stream
) {
    constexpr size_t NUM_WARPS = 16;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * WARP_SIZE, 1, 1);

    const size_t token_dim = round_up<size_t>(hidden_dim * x_elemsize, sizeof(float4));
    const size_t token_scale_dim = round_up<size_t>(hidden_dim_scale * x_scale_elemsize, sizeof(float4));
    const size_t token_stride = token_dim + token_scale_dim + 16;
    assert(token_stride % sizeof(float4) == 0);

    void *args[] = {
        const_cast<size_t *>(&token_dim),
        const_cast<size_t *>(&token_scale_dim),
        const_cast<size_t *>(&token_stride),
        &hidden_dim,
        &hidden_dim_scale,
        &x_elemsize,
        &x_scale_elemsize,
        &num_experts,
        &rank,
        &world_size,
        &out_num_tokens_ptr,
        &out_x_ptr,
        &out_x_stride,
        &out_x_scale_ptr,
        &out_x_scale_stride_elem,
        &out_x_scale_stride_token,
        &tokens_per_expert,
        &send_buffer,
        &recv_buffer,
        &source_rank,
        &source_offset,
        &padded_index,
        &num_routed,
        &num_recv_tokens_ptr,
        &num_recv_tokens_flag,
        &dispatch_recv_flag,
        &dispatch_recv_done,
        &grid_counter,
        &sync_counter,
        &sync_ptrs,
        &send_ptrs,
    };

    nvtxRangePush("dispatch_recv");
    cudaError_t status;
    LAUNCH_WORLD_SIZE(node_size, NODE_SIZE, {
        LAUNCH_TOKEN_DIM_DISPATCH(token_dim, TokenDim, {
            LAUNCH_HIDDEN_DIM_SCALE(hidden_dim_scale, HiddenDimScale, {
                status = cudaLaunchCooperativeKernel(
                    (void *)&a2a_dispatch_recv_kernel<NUM_WARPS, NODE_SIZE, TokenDim, HiddenDimScale>,
                    dimGrid,
                    dimBlock,
                    args,
                    0,
                    (cudaStream_t)stream
                );
            });
        });
    });
    nvtxRangePop();
    return status;
}
