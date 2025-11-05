use cuda_sys::CUDA_ERROR_OUT_OF_MEMORY;

use crate::driver::CudaDriverError;
use proc_lib::gpu_test;

#[gpu_test]
#[test]
fn CudaDriverError_display() {
    let e = CudaDriverError::new(CUDA_ERROR_OUT_OF_MEMORY, "some test context");
    assert_eq!(
        format!("{}", e),
        "CudaDriverError: code 2 (\"out of memory\"), context: some test context"
    );
}
