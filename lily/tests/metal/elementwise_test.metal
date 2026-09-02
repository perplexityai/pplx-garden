// Unit-test-only primitive used to validate the elementwise binary-op harness.
kernel void mul_bf16(device const bfloat* a   [[buffer(0)]],
                     device const bfloat* b   [[buffer(1)]],
                     device bfloat*       out [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = bfloat(float(a[gid]) * float(b[gid]));
}
