//! Lily: Lightweight Local Inference.
//!
//! A Metal (Apple Silicon) LLM inference engine built on the `objc2` bindings.
//! Kernels are compiled from MSL source at runtime, so building or running
//! this crate needs no offline `.metal` -> `.metallib` step (which would
//! require a non-hermetic Xcode Metal toolchain that Bazel does not model).

pub mod chat;
pub mod config;
#[cfg(target_os = "macos")]
pub mod generate;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod model;
pub mod safetensors;
#[cfg(target_os = "macos")]
pub mod serve;
#[cfg(target_os = "macos")]
pub mod tensor;
#[cfg(target_os = "macos")]
pub mod tokenizer;
#[cfg(target_os = "macos")]
pub mod weights;

#[cfg(all(target_os = "macos", test))]
#[path = "../tests/support/cpu_ref.rs"]
pub mod cpu_ref;

#[cfg(not(target_os = "macos"))]
use anyhow::bail;
