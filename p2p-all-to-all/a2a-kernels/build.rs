use std::{env, path::PathBuf};

use build_utils::emit_rerun_if_changed_files;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let cuda_archs =
        env::var("PPLX_GARDEN_CUDA_ARCHS").unwrap_or_else(|_| "90a;100a".to_string());

    // Generate bindings
    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .debug(false)
        .cuda(true)
        .flag("-t0")
        .flag("-O3")
        .flag("-cudart=shared")
        .flag(format!("-I{}/src", manifest_dir.display()));

    for arch in cuda_archs.split(';').filter(|value| !value.is_empty()) {
        build.flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }

    build
        .file("src/a2a/a2a_dispatch_recv.cu")
        .file("src/a2a/a2a_combine_send.cu")
        .file("src/a2a/a2a_combine_recv.cu")
        .file("src/a2a/a2a_dispatch_send.cu")
        .compile("liba2a_kernels.a");

    emit_rerun_if_changed_files("src", &["cu", "cuh", "h"]);

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
