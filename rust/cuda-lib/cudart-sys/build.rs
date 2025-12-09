use bindgen::callbacks::{ItemInfo, ParseCallbacks};
use build_utils::find_package;
use std::{env, path::PathBuf};

#[derive(Debug)]
struct RenameCallback;

impl ParseCallbacks for RenameCallback {
    fn item_name(&self, item_info: ItemInfo) -> Option<String> {
        match item_info.name {
            // CUDA 12 defines cudaGetDeviceProperties as cudaGetDeviceProperties_v2.
            // CUDA 13 dropped the _v2 suffix.
            "cudaGetDeviceProperties_v2" => Some("cudaGetDeviceProperties".into()),

            // No rename needed.
            _ => None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_home = find_package("CUDA_HOME", &["/usr/local/cuda"], "include/cuda.h");
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", cuda_home.display()))
        .parse_callbacks(Box::new(RenameCallback))
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
