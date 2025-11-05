#pragma once

#include <cstdbool>
#include <cstdint>
#include <cstdlib>

#include "torch-lib/src/torch_lib.h"

namespace a2a_kernels {
using ScalarType = torch_lib::ScalarType;
} // namespace a2a_kernels

#include "a2a-kernels/src/lib.rs.h"

namespace a2a_kernels {

int a2a_dispatch_send(
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
);

int a2a_dispatch_recv(
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
);

int a2a_combine_send(
    size_t num_blocks,
    size_t hidden_dim,
    size_t x_elemsize,
    size_t rank,
    size_t node_size,
    size_t dp_size,
    const uint8_t *expert_x_ptr,
    size_t expert_x_stride,
    uint8_t *tx_ready,
    uint8_t *send_buffer,
    uint8_t *recv_buffer,
    uint32_t *source_rank,
    uint32_t *combine_send_offset,
    uint32_t *padded_index,
    uint32_t *num_recv_tokens_ptr,
    uint8_t *combine_send_done,
    uint32_t *token_counter,
    uint32_t *sync_counter,
    uint32_t **sync_ptrs,
    uint8_t **recv_ptrs,
    uint64_t stream
);

int a2a_combine_recv(
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
);

} // namespace a2a_kernels
