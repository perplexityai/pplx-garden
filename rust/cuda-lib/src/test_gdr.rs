use crate::gdr::{GdrCopyContext, GdrFlag};
use proc_lib::gpu_test;

#[gpu_test]
#[test]
fn gdr_copy_flag() {
    // Set the current device.
    unsafe {
        let mut device: i32 = 0;
        cuda_sys::cuInit(0);
        cuda_sys::cuDeviceGet(&mut device, 0);

        let mut dev_ctx: cuda_sys::CUcontext = { std::ptr::null_mut() };
        cuda_sys::cuDevicePrimaryCtxRetain(&mut dev_ctx, device);
        cuda_sys::cuCtxSetCurrent(dev_ctx);
    }

    // Create the GDR copy context.
    let gdr_context = GdrCopyContext::new().unwrap();

    // Allocate a flag.
    let flag = GdrFlag::new(&gdr_context).unwrap();

    // Set the value of the flag.
    flag.set(true);
}
