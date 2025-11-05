use std::{env, path::PathBuf};

use build_utils::find_package;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let libibverbs_home =
        find_package("LIBIBVERBS_HOME", &["/usr"], "include/infiniband/verbs.h");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate bindings
    // https://rust-lang.github.io/rust-bindgen/tutorial-3.html
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", libibverbs_home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"(ibv_|IBV_|ib_|IB_).*")
        .derive_debug(false)
        .derive_default(true)
        // Some functions use static inline functions to lookup vtable.
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_dir.join("wrap_static_fns.c"))
        // Some structs includes pthread types. Let's treat them as opaque.
        .allowlist_item(r"pthread_.*")
        .opaque_type(r"pthread_.*")
        .no_default(r"pthread_.*")
        .generate()
        .expect("Unable to generate libibverbs bindings");
    bindings
        .write_to_file(out_dir.join("libibverbs-bindings.rs"))
        .expect("Couldn't write libibverbs bindings!");

    // Compile wrap_static_fns.c
    cc::Build::new()
        .file(out_dir.join("wrap_static_fns.c"))
        .include(libibverbs_home.join("include"))
        .include(env!("CARGO_MANIFEST_DIR"))
        .compile("wrap_static_fns");

    // Dynamic link dependencies
    println!("cargo:rustc-link-search=native={}/lib", libibverbs_home.display());
    println!("cargo:rustc-link-lib=ibverbs");

    Ok(())
}
