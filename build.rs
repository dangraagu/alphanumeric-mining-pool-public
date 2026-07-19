//! Build script: compile the CUDA search kernel with nvcc.
//!
//! Adapts the nvcc invocation the user's Midstate GPU miner documents in its
//! `kernel/*.cu` header comments:
//!     nvcc -O3 -arch=sm_120 -o alphanumeric_search.exe alphanumeric_search.cu
//! into a cargo build step, and records the resulting exe path in the
//! `ALPHANUMERIC_SEARCH_EXE` env for `main.rs` (`env!(...)`).
//!
//! NON-FATAL by design: if nvcc is missing or the compile fails, the Rust host
//! still builds cleanly (it has no CUDA link/FFI dependency -- it only *spawns*
//! the kernel exe at runtime). A `cargo:warning` explains how to build the
//! kernel manually. This matches the reality that many dev/CI boxes have no
//! CUDA toolchain, while the mining box (RTX 5070 Ti, nvcc 13.3) does.
//!
//! ⚠️ ORCHESTRATOR: `-arch=sm_120` targets the 5070 Ti (Blackwell). VERIFY this
//! matches the actual GPU + that the installed nvcc supports sm_120 (needs CUDA
//! >= 12.8; the box's nvcc 13.3 is fine). Override with ALPHANUMERIC_NVCC_ARCH.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = manifest_dir.join("kernel").join("alphanumeric_search.cu");

    println!("cargo:rerun-if-changed={}", kernel_src.display());
    println!("cargo:rerun-if-changed=kernel/alphanumeric_blake3.cu");
    println!("cargo:rerun-if-env-changed=ALPHANUMERIC_NVCC_ARCH");
    println!("cargo:rerun-if-env-changed=ALPHANUMERIC_SKIP_NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // nvcc appends .exe on Windows; name it explicitly so the host locates it
    // deterministically on every platform.
    let exe_name = if cfg!(windows) { "alphanumeric_search.exe" } else { "alphanumeric_search" };
    let exe_path = out_dir.join(exe_name);

    // ALWAYS record where the host should look, even if we skip/fail the build.
    // The host checks existence at runtime and errors with build instructions.
    println!("cargo:rustc-env=ALPHANUMERIC_SEARCH_EXE={}", exe_path.display());

    if env::var("ALPHANUMERIC_SKIP_NVCC").is_ok() {
        println!("cargo:warning=ALPHANUMERIC_SKIP_NVCC set -- skipping CUDA kernel build.");
        return;
    }

    let arch = env::var("ALPHANUMERIC_NVCC_ARCH").unwrap_or_else(|_| "sm_120".to_string());
    let nvcc = locate_nvcc();

    let status = Command::new(&nvcc)
        .arg("-O3")
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&exe_path)
        .arg(&kernel_src)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=Built CUDA kernel: {}", exe_path.display());
        }
        Ok(s) => {
            println!(
                "cargo:warning=nvcc exited {s} building {}. Host still built; kernel exe absent. \
                 Build manually: nvcc -O3 -arch={arch} -o {exe_name} kernel/alphanumeric_search.cu",
                kernel_src.display()
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=could not run nvcc ({e}) -- CUDA toolkit not found? Host still built; \
                 kernel exe absent. Install CUDA and build manually: \
                 nvcc -O3 -arch={arch} -o {exe_name} kernel/alphanumeric_search.cu (or set \
                 ALPHANUMERIC_SKIP_NVCC=1 to silence this)."
            );
        }
    }
}

/// Prefer `$CUDA_PATH/bin/nvcc` if present; otherwise rely on `nvcc` being on
/// PATH (spawn will surface a clear error if it is not).
fn locate_nvcc() -> PathBuf {
    if let Ok(cuda) = env::var("CUDA_PATH") {
        let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
        let p = PathBuf::from(cuda).join("bin").join(exe);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("nvcc")
}
