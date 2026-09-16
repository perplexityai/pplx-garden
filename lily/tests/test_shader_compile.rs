//! Compiles every shipped `.metal` source through the Metal framework
//! compiler and instantiates a compute pipeline for every kernel function it
//! contains. This shifts shader syntax/type errors from first-use at runtime
//! to test time, using the exact compiler the app uses (no Xcode offline
//! toolchain needed). Kernel functions are enumerated from the compiled
//! library, so new kernels are covered automatically.
#![cfg(target_os = "macos")]

use anyhow::{Context as _, Result, ensure};
use lily::metal::{MetalContext, MslVersion};
use objc2_metal::MTLLibrary as _;

const SHADERS: [(&str, &str); 8] = [
    ("attention.metal", include_str!("../src/kernels/metal/attention.metal")),
    ("elementwise.metal", include_str!("../src/kernels/metal/elementwise.metal")),
    ("gdn.metal", include_str!("../src/kernels/metal/gdn.metal")),
    ("gemm.metal", include_str!("../src/kernels/metal/gemm.metal")),
    ("moe.metal", include_str!("../src/kernels/metal/moe.metal")),
    ("norm.metal", include_str!("../src/kernels/metal/norm.metal")),
    ("quant.metal", include_str!("../src/kernels/metal/quant.metal")),
    ("skinny.metal", include_str!("../src/kernels/metal/skinny.metal")),
];

fn build_all_pipelines(
    ctx: &MetalContext,
    name: &str,
    source: &str,
    version: MslVersion,
    require_kernel: bool,
) -> Result<usize> {
    let library = ctx
        .compile_library(source, version)
        .with_context(|| format!("compiling {name}"))?;
    let functions = library.functionNames();
    if require_kernel {
        ensure!(!functions.is_empty(), "{name} contains no kernel functions");
    }
    for function in &functions {
        let fn_name = function.to_string();
        ctx.pipeline_from_library(&library, &fn_name)
            .with_context(|| format!("building pipeline for {name}:{fn_name}"))?;
    }
    Ok(functions.len())
}

fn print_device_capabilities(ctx: &MetalContext) {
    println!(
        "Apple GPU family {}; native tensor acceleration: {}",
        ctx.apple_gpu_family(),
        ctx.has_native_tensor_acceleration()
    );
}

#[test]
fn all_shaders_compile_and_all_kernels_build_pipelines() -> Result<()> {
    let ctx = MetalContext::new()?;
    print_device_capabilities(&ctx);
    for (name, source) in SHADERS {
        // gemm.metal intentionally contains only the MSL 4 tensor-op kernel.
        let count = build_all_pipelines(
            &ctx,
            name,
            source,
            MslVersion::V3_1,
            name != "gemm.metal",
        )?;
        println!("{name}: {count} kernels OK at MSL 3.1");
    }
    Ok(())
}

/// The `#if __METAL_VERSION__ >= 400` kernels (the neural-accelerator GEMM,
/// flash SDPA, and blockwise GDN) only exist in the 4.0 compile; cover them
/// at the production language version. On Apple GPU families 7-9 this also
/// verifies that Metal can build the optimized-shader TensorOps path.
#[test]
fn shaders_compile_at_msl4() -> Result<()> {
    let ctx = MetalContext::new()?;
    print_device_capabilities(&ctx);
    for (name, source) in SHADERS {
        let count = build_all_pipelines(&ctx, name, source, MslVersion::V4_0, true)?;
        println!("{name}: {count} kernels OK at MSL 4.0");
    }
    Ok(())
}
