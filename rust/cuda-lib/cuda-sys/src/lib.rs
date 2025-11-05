#![allow(warnings)]
include!(concat!(env!("OUT_DIR"), "/cuda-bindings.rs"));

pub unsafe fn cuMemAlloc(dptr: *mut u64, bytesize: usize) -> CUresult {
    cuMemAlloc_v2(dptr, bytesize)
}

pub unsafe fn cuMemFree(dptr: u64) -> CUresult {
    cuMemFree_v2(dptr)
}
