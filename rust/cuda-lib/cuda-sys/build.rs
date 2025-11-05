use std::{env, path::PathBuf};

use build_utils::find_package;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_home = find_package("CUDA_HOME", &["/usr/local/cuda"], "include/cuda.h");
    let bindings = bindgen::Builder::default()
        .header(cuda_home.join("include/cuda.h").to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"(cu|CU).*")
        .derive_default(true)
        .generate()
        .expect("Unable to generate cuda driver bindings");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("cuda-bindings.rs"))
        .expect("Couldn't write cuda driver bindings!");

    // Dynamic link dependencies
    println!("cargo:rustc-link-search=native={}/lib64/stubs", cuda_home.display());
    println!("cargo:rustc-link-lib=cuda");

    Ok(())
}
