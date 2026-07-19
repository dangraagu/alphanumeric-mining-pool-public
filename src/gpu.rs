//! The TCP client loop, GPU edition: connect, `subscribe`, `authorize`, then
//! grind nonces on the GPU in batches against whatever job the pool pushes and
//! `submit` any hit.
//!
//! ── Relationship to the CPU miner ───────────────────────────────────────────
//! The connection lifecycle, the `subscribe`/`authorize` handshake, the
//! per-job vardiff target re-read, the resubscribe-on-reject recovery, and the
//! defensive line-handling helpers are all REUSED (largely verbatim) from the
//! alphanumeric CPU miner's `src/miner.rs`. The ONLY substantive change is the
//! grind: the CPU miner walks `nonce = 0, 1, 2, ...` one BLAKE3 at a time; this
//! miner hands a whole batch `[nonce_base, nonce_base + count)` to a CUDA kernel
//! per dispatch and reads back any nonces whose hash met the target.
//!
//! ── GPU dispatch backend (subprocess) ───────────────────────────────────────
//! Dispatch mirrors the user's Midstate GPU miner's proven architecture: the
//! CUDA kernel is a standalone nvcc-built executable (`alphanumeric_search.exe`,
//! see `kernel/alphanumeric_search.cu`) that takes `(header_hex, target_hex,
//! nonce_start, count)` and prints `FOUND <nonce> <hash_hex>` lines on stdout;
//! the host spawns it once per batch and parses those lines. This keeps the
//! Rust host free of any CUDA link/FFI dependency (it builds even without a
//! CUDA toolchain) and reuses Midstate's exact stdout result plumbing.
//!   TODO(perf): a single 92-byte BLAKE3 per nonce is extremely fast, so
//!   process-spawn-per-batch overhead is proportionally larger here than in
//!   Midstate (whose 1,000,000-iteration chain dwarfed spawn cost). For
//!   production throughput, replace `dispatch_batch` with a persistent kernel
//!   process (keep it running, feed batches over a pipe) or a proper CUDA FFI
//!   launcher. The rest of this file does not care which backend is used.
//!
//! ── Safety net: every GPU hit is re-verified on the CPU ──────────────────────
//! The CUDA kernel is UNVALIDATED (see the crate README + `tests/bit_exact_TODO.md`).
//! So before ANY nonce is submitted, the host recomputes the hash with the
//! reference `pow::header_hash` and re-checks `pow::meets_target`. A GPU-reported
//! nonce that fails this check is logged loudly (it indicates a kernel bug) and
//! dropped -- an invalid share can never reach the pool because of a bad kernel.

use std::collections::HashSet;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::pow;
use crate::protocol::{self, DecodedJob, IncomingLine, JobNotify};

/// What this miner identifies itself as when it authorizes.
pub struct MinerConfig {
    pub address: String,
    pub worker: String,
}

/// GPU dispatch configuration (the knobs the CPU miner doesn't have).
pub struct GpuConfig {
    /// Path to the compiled `alphanumeric_search` kernel executable.
    pub kernel_path: PathBuf,
    /// Optional CUDA device index. Applied to the kernel subprocess via
    /// `CUDA_VISIBLE_DEVICES` (so device 2 appears to the child as device 0).
    pub device: Option<usize>,
    /// Nonces per GPU dispatch. The kernel launches one thread per nonce, so
    /// this is also the thread count for the launch. Must be `1..=u32::MAX`
    /// (the kernel's `count` argument is a `u32`).
    pub batch: u64,
}

/// A nonce the GPU reported as meeting the target, plus the 32-byte hash it
/// reported for it (used only for logging / cross-checking; the host recomputes
/// the hash itself before trusting it).
type GpuHit = (u64, [u8; 32]);

/// Drive one pool connection to completion. Returns `Ok(())` when the pool
/// closes the connection cleanly, `Err` on a genuine I/O error. Malformed or
/// unexpected messages from the pool are logged and skipped -- see
/// [`handle_line`] -- they never end the connection or panic, matching the
/// pool server's own defensive posture.
pub fn run(stream: TcpStream, config: &MinerConfig, gpu: &GpuConfig) -> io::Result<()> {
    // One persistent kernel process for the whole life of this connection --
    // the CUDA context init is paid once here, not per batch (see
    // `KernelServer`). If the pool drops us, `run_reconnecting` calls `run`
    // again and a fresh server is spawned; the old one is reaped on drop.
    let mut kernel = KernelServer::spawn(gpu)?;

    let mut writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);

    let mut next_id: u64 = 1;
    let subscribe_id = next_id;
    next_id += 1;
    write_line(&mut writer, &protocol::subscribe_request(subscribe_id).to_line())?;

    let authorize_id = next_id;
    next_id += 1;
    write_line(
        &mut writer,
        &protocol::authorize_request(authorize_id, config.address.clone(), config.worker.clone()).to_line(),
    )?;

    let mut current_job: Option<DecodedJob> = None;
    let mut submit_ids: HashSet<u64> = HashSet::new();

    'outer: loop {
        // Block, reading and dispatching messages, until we have a job to
        // grind against (the initial post-subscribe push, or a later job
        // handed to us after the previous one's nonce space ran out).
        let job = loop {
            if let Some(job) = &current_job {
                break job.clone();
            }
            match read_line(&mut reader)? {
                None => return Ok(()), // pool closed the connection
                Some(line) => handle_line(&line, subscribe_id, authorize_id, &submit_ids, &mut current_job),
            }
        };

        println!(
            "[job] grinding job_id={} number={} difficulty={} on GPU (batch={})",
            job.job_id, job.number, job.difficulty, gpu.batch
        );

        // Header field bytes are FIXED for the life of a job_id (only the pool's
        // vardiff `target` can change, never the header itself -- same invariant
        // the CPU miner relies on). Build the 92-byte header once, with the
        // nonce field zeroed; the kernel splices each thread's own nonce in at
        // `pow::NONCE_OFFSET`. Hex-encode it once for the subprocess CLI.
        let header_prefix = pow::header_bytes(
            job.number,
            job.previous_hash,
            job.timestamp,
            0, // nonce placeholder -- kernel overwrites [44..52) per thread
            job.difficulty,
            job.merkle_root,
        );
        let header_hex = hex::encode(header_prefix);

        let mut nonce_base: u64 = 0;
        loop {
            // Re-read whatever target `current_job` says RIGHT NOW for this same
            // job_id (a per-connection vardiff difficulty change updates the
            // target without changing the job_id -- see the CPU miner's miner.rs
            // for the full rationale). Every OTHER header field stays pinned to
            // the stable `job` captured above.
            let live_target = current_job
                .as_ref()
                .filter(|j| j.job_id == job.job_id)
                .map(|j| j.target)
                .unwrap_or(job.target);
            let target_hex = hex::encode(live_target);

            // Cap this batch so `nonce_base + count` never wraps u64 and `count`
            // always fits the kernel's u32 argument. `gpu.batch` is already
            // validated to be in `1..=u32::MAX` by the CLI parser.
            //
            // `remaining` = nonces STRICTLY AFTER nonce_base (never overflows);
            // `+1` (saturating) makes it the count INCLUDING nonce_base.
            let remaining = u64::MAX - nonce_base;
            let count = gpu.batch.min(remaining.saturating_add(1));
            if count == 0 {
                // Exhausted the full u64 nonce space for this job with no hit.
                if current_job.as_ref().map(|j| &j.job_id) == Some(&job.job_id) {
                    current_job = None;
                }
                continue 'outer;
            }

            let hits = kernel.dispatch(&header_hex, &target_hex, nonce_base, count)?;

            for (nonce, gpu_hash) in hits {
                // ── Re-verify on the CPU before trusting the (unvalidated) GPU.
                let ref_hash = pow::header_hash(
                    job.number,
                    job.previous_hash,
                    job.timestamp,
                    nonce,
                    job.difficulty,
                    job.merkle_root,
                );
                if ref_hash != gpu_hash {
                    eprintln!(
                        "[gpu-bug] kernel hash != reference for nonce={nonce} job_id={} \
                         (gpu={} ref={}) -- DROPPING. The CUDA kernel is not bit-exact; \
                         run the bit-exact gate (tests/bit_exact_TODO.md).",
                        job.job_id,
                        hex::encode(gpu_hash),
                        hex::encode(ref_hash)
                    );
                    continue;
                }
                // Re-read the target RIGHT NOW, per hit -- not the batch-time
                // `live_target` snapshot. vardiff tightens this connection's
                // target on every accepted share, and that update arrives via a
                // pool message consumed by `handle_line` during the PREVIOUS
                // hit's response drain (below). Without re-reading here, once the
                // first hit of a batch is accepted, every remaining hit is
                // checked against the now-stale easier target, passes locally,
                // and is submitted only for the pool to reject it against the
                // new harder target -- the "accepted, then a reject storm on the
                // same job" seen on the first live-rig canary. Re-reading makes
                // us skip those locally instead of wasting a submit+reject round
                // trip on each.
                let submit_target = current_job
                    .as_ref()
                    .filter(|j| j.job_id == job.job_id)
                    .map(|j| j.target)
                    .unwrap_or(live_target);
                if !pow::meets_target(&ref_hash, &submit_target) {
                    continue;
                }

                let submit_id = next_id;
                next_id += 1;
                submit_ids.insert(submit_id);
                println!("[share] found nonce={nonce} for job_id={} -- submitting", job.job_id);
                write_line(&mut writer, &protocol::submit_request(submit_id, job.job_id.clone(), nonce).to_line())?;

                // Drain lines until we've actually seen the response to THIS
                // submission specifically (mirrors the CPU miner exactly).
                let mut submit_was_rejected = false;
                loop {
                    match read_line(&mut reader)? {
                        None => return Ok(()),
                        Some(line) => {
                            let is_this_submits_response = response_id(&line) == Some(submit_id);
                            if is_this_submits_response {
                                submit_was_rejected = response_is_error(&line);
                            }
                            handle_line(&line, subscribe_id, authorize_id, &submit_ids, &mut current_job);
                            if is_this_submits_response {
                                break;
                            }
                        }
                    }
                }

                if submit_was_rejected {
                    // See the CPU miner's "Recovering from a rejected submit":
                    // nothing pushes a fresh job notify on a rejection, so ask
                    // for one instead of grinding a dead job_id forever.
                    println!("[submit] rejected -- resubscribing to recover a fresh job");
                    current_job = None;
                    write_line(&mut writer, &protocol::subscribe_request(subscribe_id).to_line())?;
                    continue 'outer;
                }

                // A difficulty change can be pushed immediately behind a submit's
                // own response; drain anything already queued so a same-job_id
                // target update takes effect promptly.
                drain_immediately_available(&mut reader, subscribe_id, authorize_id, &submit_ids, &mut current_job)?;
                if current_job.as_ref().map(|j| &j.job_id) != Some(&job.job_id) {
                    continue 'outer;
                }
            }

            // Opportunistically pick up any job/target update the pool pushed
            // while we were grinding this batch (a brand-new job means abandon
            // the rest of this nonce space and go grind it instead).
            drain_immediately_available(&mut reader, subscribe_id, authorize_id, &submit_ids, &mut current_job)?;
            if current_job.as_ref().map(|j| &j.job_id) != Some(&job.job_id) {
                continue 'outer;
            }

            match nonce_base.checked_add(count) {
                Some(next) => nonce_base = next,
                None => {
                    // Reached the very top of the u64 nonce space.
                    if current_job.as_ref().map(|j| &j.job_id) == Some(&job.job_id) {
                        current_job = None;
                    }
                    continue 'outer;
                }
            }
        }
    }
}

// ── Auto-reconnect across pool restarts (reused from the CPU miner) ──────────
//
// The pool process is restarted for deploys. When it goes down every miner's
// TCP socket drops: the blocking read returns a clean EOF (`Ok(None)` from
// `read_line`, which `run` surfaces as `Ok(())`) or a read/write fails with a
// real I/O error (`Err`). Either way `run` RETURNS -- and the original entry
// point then exited (or sat idle) the instant the pool bounced. That is exactly
// the observed "pool restarted, miner froze / never came back" failure.
//
// A miner has exactly ONE pool and must never give up on it. `run_reconnecting`
// wraps the entire dial -> subscribe -> authorize -> mine lifecycle (`run`
// itself does the last three) in an outer loop that, on ANY connection end,
// waits a short backoff and redials -- forever, until the process is killed.
// `run` is left completely unchanged: it still drives exactly one connection to
// completion; this just keeps handing it fresh ones. Identical policy to the CPU
// miner's `miner.rs`.

/// Reconnect backoff bounds, in seconds. Starts small so a quick pool bounce is
/// barely noticed; caps low because a miner has one pool -- there is never a
/// reason to wait longer than this between retries.
const RECONNECT_BACKOFF_MIN_SECS: u64 = 2;
const RECONNECT_BACKOFF_MAX_SECS: u64 = 30;

/// Pure backoff policy: given the number of consecutive failed connection
/// attempts (1-based; 1 = the first retry after a drop), return how many
/// seconds to wait before the next dial. Capped exponential: 2, 4, 8, 16, 30,
/// 30, ... -- doubles each time up to the cap, then holds. Kept a pure
/// `attempt -> secs` function so the policy is unit-testable without any
/// sockets or real sleeping (see this module's tests).
fn backoff_secs(attempt: u32) -> u64 {
    // Clamp the shift BEFORE shifting so `1 << shift` can never overflow, then
    // clamp the result to the max. `attempt` is 1-based, so attempt 1 => shift 0.
    let shift = attempt.saturating_sub(1).min(20);
    let delay = RECONNECT_BACKOFF_MIN_SECS.saturating_mul(1u64 << shift);
    delay.min(RECONNECT_BACKOFF_MAX_SECS)
}

/// Connect to `pool_addr` and mine on the GPU, reconnecting forever whenever the
/// connection drops (see this section's comment for why). Never returns: the
/// only way out is the process being killed.
///
/// `update` controls the safe, sha256-pinned self-update (see [`crate::update`]):
/// a leftover `<exe>.old` from a previous update is cleaned up once at startup,
/// then every outer iteration ticks the update check -- throttled to ~6h -- which
/// logs, and under `--auto-update` verifies + installs + re-execs. Any update
/// failure is logged and mining continues; it can never stop the miner.
pub fn run_reconnecting(
    pool_addr: &str,
    config: &MinerConfig,
    gpu: &GpuConfig,
    update: crate::update::UpdateOptions,
) -> ! {
    // Consecutive failed/short connections since the last successful dial. Drives
    // the backoff and is reset to 0 the moment a dial succeeds -- a successful
    // connect means the pool is back up and accepted us, and `run` writes
    // `subscribe` + `authorize` immediately, so a landed dial IS the
    // reconnect+authorize milestone. Resetting here means the NEXT drop after a
    // healthy session starts from the minimum backoff again, while a pool that
    // is still down (connect keeps failing) keeps growing it.
    let mut consecutive_failures: u32 = 0;

    // Self-update bookkeeping. Clean up a leftover `.old` from a prior update,
    // then track when we last checked so `tick` can throttle to ~6h. `None`
    // means "never checked", so the FIRST outer iteration checks on startup.
    crate::update::cleanup_old_binary();
    let mut last_update_check: Option<std::time::Instant> = None;

    loop {
        // Runs on startup (last_update_check == None) and ~every 6h thereafter.
        // On a successful `--auto-update` this re-execs and never returns.
        last_update_check = crate::update::tick(update, last_update_check, env!("CARGO_PKG_VERSION"));

        println!("[connect] dialing pool at {pool_addr}...");
        match TcpStream::connect(pool_addr) {
            Ok(stream) => {
                consecutive_failures = 0;
                println!(
                    "[connect] connected. subscribing + authorizing as address={} worker={} \
                     (gpu device={:?} batch={} kernel={})",
                    config.address,
                    config.worker,
                    gpu.device,
                    gpu.batch,
                    gpu.kernel_path.display()
                );
                match run(stream, config, gpu) {
                    Ok(()) => println!("[reconnect] pool closed the connection"),
                    Err(e) => eprintln!("[reconnect] connection dropped: {e}"),
                }
            }
            Err(e) => eprintln!("[reconnect] could not reach pool at {pool_addr}: {e}"),
        }

        // The connection ended (clean close, I/O error, or the dial never
        // landed). Back off, then redial -- a miner never abandons its pool.
        consecutive_failures = consecutive_failures.saturating_add(1);
        let secs = backoff_secs(consecutive_failures);
        println!("[reconnect] connection lost, retrying in {secs}s...");
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
}

/// Launch the CUDA search kernel for one batch and parse the nonces it reports.
///
/// A PERSISTENT kernel process running in `serve` mode.
///
/// The original design spawned the kernel exe once per batch and read its
/// full stdout (`Command::output()`). That paid the CUDA context +
/// buffer-allocation cost (~250ms measured) on EVERY batch, throttling a
/// single-block BLAKE3 -- which the GPU can do in sub-milliseconds for
/// millions of nonces -- to ~16 MH/s of almost-pure context-init overhead.
///
/// This keeps ONE kernel process alive (`kernel serve`), holding the CUDA
/// context and device buffers, and streams batches to it over stdin,
/// reading results back over stdout. The ~250ms init is paid once at
/// startup; every subsequent batch is just a line write + the kernel's
/// memcpy/launch/sync, so throughput is bounded by the GPU, not by process
/// spawning. Same `search` kernel and hashing primitives as the one-shot
/// path (still exercised by `selftest`), so bit-exactness is unchanged.
struct KernelServer {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl KernelServer {
    /// Spawn `kernel serve` with piped stdin/stdout. The ~250ms CUDA init
    /// happens lazily on the first `dispatch`, not here.
    fn spawn(gpu: &GpuConfig) -> io::Result<Self> {
        let mut cmd = Command::new(&gpu.kernel_path);
        cmd.arg("serve").stdin(Stdio::piped()).stdout(Stdio::piped());
        if let Some(dev) = gpu.device {
            // Make ONLY this device visible to the child (it sees it as
            // device 0). Same as before; needs no cudaSetDevice in the kernel.
            cmd.env("CUDA_VISIBLE_DEVICES", dev.to_string());
        }
        let mut child = cmd.spawn().map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "failed to launch CUDA kernel at {} in serve mode: {e}. Build it \
                     with nvcc (see README) or pass --kernel <path>.",
                    gpu.kernel_path.display()
                ),
            )
        })?;
        let stdin = BufWriter::new(child.stdin.take().expect("piped stdin"));
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(KernelServer { child, stdin, stdout })
    }

    /// Hash one batch: write the request line, then read `FOUND ...` lines
    /// until the `DONE ...` marker that ends this batch. Reusing the live
    /// context, so this is GPU-bound, not spawn-bound.
    fn dispatch(
        &mut self,
        header_hex: &str,
        target_hex: &str,
        nonce_base: u64,
        count: u64,
    ) -> io::Result<Vec<GpuHit>> {
        writeln!(self.stdin, "{header_hex} {target_hex} {nonce_base} {count}")?;
        self.stdin.flush()?;

        let mut hits = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line)?;
            if n == 0 {
                // Kernel process closed stdout / died -- surface it so run()
                // returns Err and the reconnect loop respawns everything.
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "CUDA kernel serve process ended unexpectedly",
                ));
            }
            let mut it = line.split_whitespace();
            match it.next() {
                Some("DONE") => break, // this batch is complete
                Some("FOUND") => {
                    let (Some(nonce_tok), Some(hash_tok)) = (it.next(), it.next()) else {
                        eprintln!("[warn] malformed FOUND line from kernel, ignoring: {}", line.trim());
                        continue;
                    };
                    let Ok(nonce) = nonce_tok.parse::<u64>() else {
                        eprintln!("[warn] FOUND line has non-numeric nonce, ignoring: {}", line.trim());
                        continue;
                    };
                    let Ok(bytes) = hex::decode(hash_tok) else {
                        eprintln!("[warn] FOUND line has non-hex hash, ignoring: {}", line.trim());
                        continue;
                    };
                    if bytes.len() != 32 {
                        eprintln!("[warn] FOUND hash is {} bytes (want 32), ignoring: {}", bytes.len(), line.trim());
                        continue;
                    }
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&bytes);
                    hits.push((nonce, hash));
                }
                // Anything else (a stray stderr echo, a warning line) is
                // ignored -- only FOUND/DONE are protocol.
                _ => {}
            }
        }
        Ok(hits)
    }
}

impl Drop for KernelServer {
    fn drop(&mut self) {
        // Dropping stdin closes it -> the kernel's fgets loop hits EOF and
        // exits cleanly; then reap it so we never leak a zombie GPU process.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Everything below is REUSED VERBATIM from the alphanumeric CPU miner's
// `src/miner.rs` -- protocol-level line handling that is backend-agnostic.
// ─────────────────────────────────────────────────────────────────────────────

/// How long [`drain_immediately_available`] waits for one more line before
/// giving up and assuming nothing else is queued.
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);

/// Opportunistically reads and handles any additional line(s) ALREADY sitting
/// in the socket right behind the one just read. Temporarily switches the
/// underlying socket to a short read timeout so "nothing more queued" becomes a
/// bounded, non-fatal timeout instead of a block; restores blocking mode before
/// returning on every exit path.
fn drain_immediately_available(
    reader: &mut BufReader<TcpStream>,
    subscribe_id: u64,
    authorize_id: u64,
    submit_ids: &HashSet<u64>,
    current_job: &mut Option<DecodedJob>,
) -> io::Result<()> {
    reader.get_ref().set_read_timeout(Some(DRAIN_TIMEOUT))?;
    let result = drain_loop(reader, subscribe_id, authorize_id, submit_ids, current_job);
    reader.get_ref().set_read_timeout(None)?; // always restore blocking mode
    result
}

fn drain_loop(
    reader: &mut BufReader<TcpStream>,
    subscribe_id: u64,
    authorize_id: u64,
    submit_ids: &HashSet<u64>,
    current_job: &mut Option<DecodedJob>,
) -> io::Result<()> {
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return Ok(()), // clean EOF -- nothing more to drain
            Ok(_) => {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    handle_line(trimmed, subscribe_id, authorize_id, submit_ids, current_job);
                }
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

/// Best-effort peek at a line's `id` field, without acting on the line.
fn response_id(line: &str) -> Option<u64> {
    serde_json::from_str::<IncomingLine>(line).ok().and_then(|p| p.id.as_u64())
}

/// Best-effort peek at whether a just-read line is an ERROR response.
fn response_is_error(line: &str) -> bool {
    serde_json::from_str::<IncomingLine>(line).ok().and_then(|p| p.error).is_some()
}

/// Parse and react to one line from the pool. Never panics: a line that fails
/// to parse, a `job` notify with unexpected/invalid params, or a response with
/// an `id` we don't recognize is logged and ignored rather than treated as fatal.
fn handle_line(
    line: &str,
    subscribe_id: u64,
    authorize_id: u64,
    submit_ids: &HashSet<u64>,
    current_job: &mut Option<DecodedJob>,
) {
    let parsed: IncomingLine = match serde_json::from_str(line) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[warn] malformed line from pool, ignoring ({e}): {line}");
            return;
        }
    };

    if parsed.notify.as_deref() == Some("job") {
        let Some(params) = parsed.params else {
            eprintln!("[warn] job notify with no params, ignoring");
            return;
        };
        let notify: JobNotify = match serde_json::from_value(params) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[warn] job notify params do not match the expected shape, ignoring: {e}");
                return;
            }
        };
        match notify.decode() {
            Ok(decoded) => {
                println!(
                    "[job] received job_id={} number={} difficulty={}",
                    decoded.job_id, decoded.number, decoded.difficulty
                );
                *current_job = Some(decoded);
            }
            Err(e) => eprintln!("[warn] job notify has invalid hex fields, ignoring this job: {e}"),
        }
        return;
    }

    let id = parsed.id.as_u64();
    if id == Some(subscribe_id) {
        match &parsed.error {
            Some(err) => eprintln!("[warn] subscribe failed: {err}"),
            None => println!("[subscribe] ok: {}", parsed.result.unwrap_or(serde_json::Value::Null)),
        }
    } else if id == Some(authorize_id) {
        match &parsed.error {
            // The pool's own submit handler doesn't gate on authorization, so a
            // failed authorize is logged but not treated as fatal.
            Some(err) => eprintln!("[warn] authorize failed: {err}"),
            None => println!("[authorize] ok: {}", parsed.result.unwrap_or(serde_json::Value::Null)),
        }
    } else if id.map(|i| submit_ids.contains(&i)).unwrap_or(false) {
        match &parsed.error {
            Some(err) => eprintln!("[submit] rejected: {err}"),
            None => println!("[submit] result: {}", parsed.result.unwrap_or(serde_json::Value::Null)),
        }
    } else {
        eprintln!("[warn] unrecognized message from pool, ignoring: {line}");
    }
}

fn write_line(writer: &mut TcpStream, line: &str) -> io::Result<()> {
    writer.write_all(line.as_bytes())
}

/// Read one non-blank line, discarding blank lines. Returns `Ok(None)` on a
/// clean EOF (pool closed the connection).
fn read_line(reader: &mut BufReader<TcpStream>) -> io::Result<Option<String>> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        return Ok(Some(trimmed.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_is_capped_exponential_and_never_exceeds_the_max() {
        // 1-based attempt: doubles from the minimum, then holds at the cap.
        assert_eq!(backoff_secs(1), 2);
        assert_eq!(backoff_secs(2), 4);
        assert_eq!(backoff_secs(3), 8);
        assert_eq!(backoff_secs(4), 16);
        // 2 * 2^4 = 32 would exceed the 30s cap -- clamped.
        assert_eq!(backoff_secs(5), RECONNECT_BACKOFF_MAX_SECS);
        assert_eq!(backoff_secs(6), RECONNECT_BACKOFF_MAX_SECS);
        // Huge attempt counts must stay clamped and never overflow or panic.
        assert_eq!(backoff_secs(1_000), RECONNECT_BACKOFF_MAX_SECS);
        assert_eq!(backoff_secs(u32::MAX), RECONNECT_BACKOFF_MAX_SECS);
    }

    #[test]
    fn backoff_first_retry_is_the_configured_minimum() {
        assert_eq!(backoff_secs(1), RECONNECT_BACKOFF_MIN_SECS);
    }
}
