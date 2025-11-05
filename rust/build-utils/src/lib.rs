use std::{
    env,
    path::{Path, PathBuf},
};

/// Finds the path to a package directory by checking an environment variable and a list of default paths.
///
/// The function checks if the environment variable `env_var` is set and points to a directory containing `check_file`.
/// If not, it searches each path in `default_paths` for the presence of `check_file`.
/// Returns the first directory containing `check_file`, or panics if none is found.
///
/// # Arguments
/// * `env_var` - The name of the environment variable to check.
/// * `default_paths` - A slice of default directory paths to search.
/// * `check_file` - The relative path to the file that must exist in the directory.
///
/// # Panics
/// Panics if neither the environment variable nor any of the default paths contain `check_file`.
pub fn find_package(
    env_var: &str,
    default_paths: &[&str],
    check_file: &str,
) -> PathBuf {
    println!("cargo:rerun-if-env-changed={}", env_var);
    env::var_os(env_var)
        .map(PathBuf::from)
        .into_iter()
        .chain(default_paths.iter().map(PathBuf::from))
        .find(|dir| dir.join(check_file).is_file())
        .unwrap_or_else(|| {
            panic!(
                "find_package: {} is not set and {} is not found in the default paths",
                env_var, check_file
            )
        })
}

/// Recursively emits `cargo:rerun-if-changed` for all files under `src_dir`
/// with one of the given `extensions`.
///
/// Example:
/// ```no_run
/// use build_utils::emit_rerun_if_changed_files;
/// emit_rerun_if_changed_files("src", &["cu", "cuh", "h"]);
/// ```
pub fn emit_rerun_if_changed_files(src_dir: &str, extensions: &[&str]) {
    fn visit_dir(dir: &Path, extensions: &[&str]) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dir(&path, extensions)?;
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str())
                && extensions.contains(&ext)
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
        Ok(())
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join(src_dir);

    if let Err(err) = visit_dir(&root, extensions) {
        eprintln!("cargo:warning=Failed to scan {}: {}", root.display(), err);
    }

    // Also watch the directory itself so new files trigger rebuilds
    println!("cargo:rerun-if-changed={}", root.display());
}
