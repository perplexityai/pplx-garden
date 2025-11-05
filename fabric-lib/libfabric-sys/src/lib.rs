#![allow(warnings)]
include!(concat!(env!("OUT_DIR"), "/libfabric-bindings.rs"));

pub const FI_ADDR_UNSPEC: fi_addr_t = u64::MAX;

pub fn make_fi_version(major: u16, minor: u16) -> u32 {
    ((major as u32) << 16) | (minor as u32)
}

pub unsafe fn fi_close(fid: *mut fid) {
    (*(*fid).ops).close.unwrap_unchecked()(fid);
}
