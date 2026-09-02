//! Runtime-compiled Metal kernels and Rust dispatch wrappers.

pub mod attention;
pub mod elementwise;
pub mod gdn;
pub mod gemm;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod skinny;

pub fn u32_bytes(v: usize) -> [u8; 4] {
    (v as u32).to_ne_bytes()
}

pub fn f32_bytes(v: f32) -> [u8; 4] {
    v.to_ne_bytes()
}
