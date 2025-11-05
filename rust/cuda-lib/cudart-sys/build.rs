use std::{env, path::PathBuf};

use build_utils::find_package;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_home = find_package("CUDA_HOME", &["/usr/local/cuda"], "include/cuda.h");
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", cuda_home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"cuda.*")
        .derive_default(true)
        .generate()
        .expect("Unable to generate cuda runtime bindings");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("cudart-bindings.rs"))
        .expect("Couldn't write cuda runtime bindings!");

    // Dynamic link dependencies
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home.display());
    println!("cargo:rustc-link-lib=cudart");

    Ok(())
}
