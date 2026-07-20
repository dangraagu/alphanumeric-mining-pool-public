//! `alphanumeric-gpu-miner` -- CLI entry point: parse arguments, connect to the
//! pool, hand off to [`alphanumeric_gpu_miner::gpu::run`].
//!
//! Argument parsing is hand-rolled (matching the CPU miner's style): the flag
//! set is small, so a CLI-parsing crate would outweigh what it saves. The
//! `--pool`/`--address`/`--worker` flags are IDENTICAL to the CPU miner; the
//! GPU-specific `--device`/`--batch`/`--kernel` flags are additive.

use std::path::PathBuf;
use std::process::ExitCode;

use alphanumeric_gpu_miner::gpu::{self, GpuConfig, MinerConfig};
use alphanumeric_gpu_miner::protocol::is_valid_address;
use alphanumeric_gpu_miner::update::UpdateOptions;

// Hardcoded, endpoint-LOCKED pool address: this miner mines ONLY the official
// alphanumeric G-pool. There is deliberately no `--pool` flag, so a copy can't
// be repointed at a competing pool (see LICENSE: PolyForm Noncommercial 1.0.0).
const POOL_ENDPOINT: &str = "alphanumeric.yamaduo.no:3777";
const DEFAULT_WORKER: &str = "gpu-worker";
/// Nonces per GPU dispatch. 2^22 amortises subprocess-spawn overhead while
/// staying well under the kernel's `u32` count limit. Tune per GPU / batch time.
// 64M nonces/batch. With the persistent kernel + constant-memory hot path,
// this runs in ~19ms on an RTX 5070 Ti (~3.4 GH/s) -- big enough to amortise
// the per-batch pipe round-trip and kernel launch, small enough to stay
// responsive to new jobs / vardiff target changes (checked between batches).
const DEFAULT_BATCH: u64 = 67_108_864;

/// Where `build.rs` placed the compiled kernel exe (always set by build.rs,
/// even when it skipped the nvcc compile -- the file may therefore not exist).
const BUILT_KERNEL_PATH: &str = env!("ALPHANUMERIC_SEARCH_EXE");

struct Args {
    pool: String,
    address: String,
    worker: String,
    device: Option<usize>,
    batch: u64,
    kernel: Option<String>,
    /// `--auto-update`: apply a found update (sha256-verified) and re-exec.
    /// Default is off -- notify only.
    auto_update: bool,
    /// `--no-update-check`: disable the update check entirely.
    no_update_check: bool,
}

fn print_usage() {
    eprintln!(
        "Usage: alphanumeric-gpu-miner --address <40-hex-address> [--worker <label>]\n\
         \x20                            [--device <gpu-index>] [--batch <nonces>] [--kernel <path>]\n\
         \n\
         Mines the official alphanumeric G-pool ({POOL_ENDPOINT}) -- endpoint is fixed.\n\
         \n\
         --address   required. Your 40-character lowercase hex alphanumeric payout address.\n\
         --worker    optional. Worker label sent on authorize, default: {DEFAULT_WORKER}\n\
         --device    optional. CUDA device index (via CUDA_VISIBLE_DEVICES), default: 0.\n\
         --batch     optional. Nonces per GPU dispatch (1..=4294967295), default: {DEFAULT_BATCH}.\n\
         --threads   optional. Alias for --batch.\n\
         --kernel    optional. Path to the compiled alphanumeric_search kernel exe.\n\
         \x20                   Default (from build.rs): {BUILT_KERNEL_PATH}\n\
         --auto-update      optional. Apply a newer version automatically (sha256-verified, then\n\
         \x20                         re-exec). Default: OFF -- just notify you one is available.\n\
         --no-update-check  optional. Disable the periodic update check entirely."
    );
}

fn parse_args() -> Result<Args, String> {
    // Endpoint-locked: not overridable at runtime (no `--pool` flag).
    let pool = POOL_ENDPOINT.to_string();
    let mut worker = DEFAULT_WORKER.to_string();
    let mut address: Option<String> = None;
    let mut device: Option<usize> = None;
    let mut batch: u64 = DEFAULT_BATCH;
    let mut kernel: Option<String> = None;
    let mut auto_update = false;
    let mut no_update_check = false;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--address" => address = Some(args.next().ok_or("--address requires a value")?),
            "--worker" => worker = args.next().ok_or("--worker requires a value")?,
            "--device" => {
                let v = args.next().ok_or("--device requires a value")?;
                device = Some(v.parse().map_err(|_| format!("--device must be a non-negative integer, got: {v}"))?);
            }
            "--batch" | "--threads" => {
                let v = args.next().ok_or("--batch requires a value")?;
                batch = v.parse().map_err(|_| format!("--batch must be a positive integer, got: {v}"))?;
            }
            "--kernel" => kernel = Some(args.next().ok_or("--kernel requires a value")?),
            "--auto-update" => auto_update = true,
            "--no-update-check" => no_update_check = true,
            "-h" | "--help" => return Err(String::new()), // triggers usage print, no error text
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    if batch == 0 || batch > u32::MAX as u64 {
        return Err(format!("--batch must be in 1..=4294967295 (the kernel's u32 count), got: {batch}"));
    }

    let address = address.ok_or("--address is required")?;
    Ok(Args { pool, address, worker, device, batch, kernel, auto_update, no_update_check })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}\n");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    if !is_valid_address(&args.address) {
        eprintln!(
            "error: --address must be exactly 40 lowercase hex characters, got: {:?} (len={})",
            args.address,
            args.address.len()
        );
        return ExitCode::FAILURE;
    }

    let kernel_path = PathBuf::from(args.kernel.as_deref().unwrap_or(BUILT_KERNEL_PATH));
    if !kernel_path.exists() {
        eprintln!(
            "error: CUDA kernel not found at {}.\n\
             Build it first (needs the CUDA toolkit / nvcc):\n  \
             nvcc -O3 -arch=sm_120 -o alphanumeric_search.exe kernel/alphanumeric_search.cu\n\
             then pass it with --kernel <path>, or run `cargo build` on a box with nvcc on PATH\n\
             so build.rs compiles it automatically.",
            kernel_path.display()
        );
        return ExitCode::FAILURE;
    }

    let config = MinerConfig { address: args.address, worker: args.worker };
    let gpu_config = GpuConfig { kernel_path, device: args.device, batch: args.batch };
    // Safe default: check for updates and NOTIFY only. `--auto-update` opts into
    // the sha256-verified auto-apply; `--no-update-check` disables the check.
    let update = UpdateOptions {
        check_enabled: !args.no_update_check,
        auto_update: args.auto_update,
    };
    // Mine forever, reconnecting across pool restarts (deploys drop every
    // miner's socket -- see `gpu::run_reconnecting`). This call never returns:
    // it redials on every connection drop, so the process ends only when killed.
    // The config-error exits above still return non-zero -- a bad address or a
    // missing kernel is fatal, but the pool being temporarily down is exactly
    // what we now ride out instead of exiting on.
    gpu::run_reconnecting(&args.pool, &config, &gpu_config, update)
}
