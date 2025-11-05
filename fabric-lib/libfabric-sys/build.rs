use std::{env, path::PathBuf};

use build_utils::find_package;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let libfabric_home = find_package(
        "LIBFABRIC_HOME",
        &["/opt/amazon/efa", "/usr"],
        "include/rdma/fabric.h",
    );

    // Generate bindings
    // https://rust-lang.github.io/rust-bindgen/tutorial-3.html
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", libfabric_home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"(fi|FI)_.*")
        .derive_default(true)
        .generate()
        .expect("Unable to generate libfabric bindings");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("libfabric-bindings.rs"))
        .expect("Couldn't write libfabric bindings!");

    // Dynamic link libfabric
    println!("cargo:rustc-link-search=native={}/lib", libfabric_home.display());
    println!("cargo:rustc-link-lib=fabric");

    Ok(())
}
