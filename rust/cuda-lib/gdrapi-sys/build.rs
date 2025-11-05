use std::{env, path::PathBuf};

use build_utils::find_package;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gdrapi_home = find_package("GDRAPI_HOME", &["/usr"], "include/gdrapi.h");
    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include <gdrapi.h>")
        .clang_arg(format!("-I{}/include", gdrapi_home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"gdr.*")
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .expect("Unable to generate gdrapi bindings");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("gdrapi-bindings.rs"))
        .expect("Couldn't write gdrapi bindings!");

    // Dynamic link dependencies
    println!("cargo:rustc-link-lib=gdrapi");
    println!("cargo:rustc-link-search=native={}/lib", gdrapi_home.display());

    Ok(())
}
