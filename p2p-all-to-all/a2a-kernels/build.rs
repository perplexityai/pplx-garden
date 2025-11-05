use std::{env, path::PathBuf};

use build_utils::emit_rerun_if_changed_files;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Generate bindings
    cxx_build::bridge("src/lib.rs")
        .debug(false)
        .cuda(true)
        .flag("-t0")
        .flag("-O3")
        .flag("-cudart=shared")
        .flag("-gencode=arch=compute_90a,code=sm_90a")
        .flag("-gencode=arch=compute_100a,code=sm_100a")
        .flag(format!("-I{}/src", manifest_dir.display()))
        .file("src/a2a/a2a_dispatch_recv.cu")
        .file("src/a2a/a2a_combine_send.cu")
        .file("src/a2a/a2a_combine_recv.cu")
        .file("src/a2a/a2a_dispatch_send.cu")
        .compile("liba2a_kernels.a");

    emit_rerun_if_changed_files("src", &["cu", "cuh", "h"]);

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
