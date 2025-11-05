use std::path::Path;
use std::process::Command;

fn main() {
    let cmake_prefix_path = match std::env::var("TORCH_CMAKE_PREFIX_PATH") {
        Ok(path) => path,
        Err(_) => {
            let output = Command::new("python3")
                .arg("-W")
                .arg("ignore")
                .arg("-c")
                .arg("import torch; print(torch.utils.cmake_prefix_path)")
                .output()
                .expect("failed to find Torch CMake prefix path");

            if !output.stderr.is_empty() {
                let stderr_str = String::from_utf8_lossy(&output.stderr);
                println!(
                    "cargo:warning=error getting torch prefix path: {}",
                    stderr_str.trim()
                );
            }

            String::from_utf8(output.stdout).unwrap()
        }
    };

    let torch_path = Path::new(&cmake_prefix_path).parent().unwrap().parent().unwrap();
    let torch_include = torch_path.join("include");
    let torch_lib = torch_path.join("lib");

    let config = pkg_config::Config::new().probe("python3").unwrap();

    cxx_build::bridge("src/lib.rs")
        .file("src/torch_lib.cc")
        .flag("-Wno-unused-parameter")
        .includes(config.include_paths)
        .include(torch_include)
        .include("/usr/local/cuda/include")
        .std("c++20")
        .compile("torch-lib");

    println!("cargo:rerun-if-changed=src/torch_lib.cc");
    println!("cargo:rerun-if-changed=src/torch_lib.h");

    println!("cargo:rustc-link-search=native={}", torch_lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", torch_lib.display());
    println!("cargo:rustc-link-lib=torch_python");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=c10");
}
