//! `alphanumeric-gpu-miner` -- CLI entry point: parse arguments, connect to the
//! pool, hand off to [`alphanumeric_gpu_miner::gpu::run`].
//!
//! Argument parsing is hand-rolled (matching the CPU miner's style): the flag
//! set is small, so a CLI-parsing crate would outweigh what it saves. The
//! `--pool`/`--address`/`--worker` flags are IDENTICAL to the CPU miner; the
//! GPU-specific `--device`/`--batch`/`--kernel` flags are additive.

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;

use alphanumeric_gpu_miner::gpu::{self, GpuConfig, MinerConfig};
use alphanumeric_gpu_miner::protocol::is_valid_address;

const DEFAULT_POOL_ADDR: &str = "127.0.0.1:3777";
const DEFAULT_WORKER: &str = "gpu-worker";
/// Nonces per GPU dispatch. 2^22 amortises subprocess-spawn overhead while
/// staying well under the kernel's `u32` count limit. Tune per GPU / batch time.
const DEFAULT_BATCH: u64 = 4_194_304;

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
}

fn print_usage() {
    eprintln!(
        "Usage: alphanumeric-gpu-miner --address <40-hex-address> [--pool <host:port>] [--worker <label>]\n\
         \x20                            [--device <gpu-index>] [--batch <nonces>] [--kernel <path>]\n\
         \n\
         --address   required. Your 40-character lowercase hex alphanumeric payout address.\n\
         --pool      optional. Pool TCP address, default: {DEFAULT_POOL_ADDR}\n\
         --worker    optional. Worker label sent on authorize, default: {DEFAULT_WORKER}\n\
         --device    optional. CUDA device index (via CUDA_VISIBLE_DEVICES), default: 0.\n\
         --batch     optional. Nonces per GPU dispatch (1..=4294967295), default: {DEFAULT_BATCH}.\n\
         --threads   optional. Alias for --batch.\n\
         --kernel    optional. Path to the compiled alphanumeric_search kernel exe.\n\
         \x20                   Default (from build.rs): {BUILT_KERNEL_PATH}"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut pool = DEFAULT_POOL_ADDR.to_string();
    let mut worker = DEFAULT_WORKER.to_string();
    let mut address: Option<String> = None;
    let mut device: Option<usize> = None;
    let mut batch: u64 = DEFAULT_BATCH;
    let mut kernel: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--pool" => pool = args.next().ok_or("--pool requires a value")?,
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
            "-h" | "--help" => return Err(String::new()), // triggers usage print, no error text
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    if batch == 0 || batch > u32::MAX as u64 {
        return Err(format!("--batch must be in 1..=4294967295 (the kernel's u32 count), got: {batch}"));
    }

    let address = address.ok_or("--address is required")?;
    Ok(Args { pool, address, worker, device, batch, kernel })
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

    println!("[connect] dialing pool at {}...", args.pool);
    let stream = match TcpStream::connect(&args.pool) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not connect to pool at {}: {e}", args.pool);
            return ExitCode::FAILURE;
        }
    };
    println!(
        "[connect] connected. subscribing + authorizing as address={} worker={} (gpu device={:?} batch={} kernel={})",
        args.address, args.worker, args.device, args.batch, kernel_path.display()
    );

    let config = MinerConfig { address: args.address, worker: args.worker };
    let gpu_config = GpuConfig { kernel_path, device: args.device, batch: args.batch };
    match gpu::run(stream, &config, &gpu_config) {
        Ok(()) => {
            println!("[connect] pool closed the connection.");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: connection to pool ended with an I/O error: {e}");
            ExitCode::FAILURE
        }
    }
}
