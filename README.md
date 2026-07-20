# alphanumeric-gpu-miner

CUDA GPU miner for the **alphanumeric mining pool ("a#", pool #3)**
share-distribution protocol. It speaks the byte-identical wire protocol as the
CPU miner (`alphanumeric-mining-pool-public`) but grinds nonces on an NVIDIA GPU
instead of the CPU.

## Quick start — join the pool

This miner is **endpoint-locked** to the official alphanumeric G-pool
(`alphanumeric.yamaduo.no:3777`) — there is no `--pool` flag, so it always mines
the real pool. Rewards pay to **your own** address (`--address`, required).

**1. Build the CUDA kernel** (needs the NVIDIA CUDA Toolkit / `nvcc` — and on
**Windows** the MSVC C++ compiler; build from the *"x64 Native Tools Command
Prompt for VS 2022"* so `cl.exe` is on PATH, else `nvcc` fails — see Prerequisites):

```
nvcc -O3 -arch=sm_XX -o kernel/alphanumeric_search.exe kernel/alphanumeric_search.cu
```

Set `sm_XX` to your GPU's compute capability (e.g. `sm_86` RTX 30xx, `sm_89`
RTX 40xx, `sm_120` RTX 50xx). **Verify bit-exactness before mining:**

```
kernel/alphanumeric_search.exe selftest      # prints GPU hashes A/B/C
cargo run --release --example reference_vectors   # prints reference A/B/C -- MUST match
```

**2. Build the miner:** `cargo build --release`

**3. Mine** (use YOUR 40-char lowercase-hex payout address):

```
# if you built the kernel manually in step 1, also pass:
#   --kernel kernel/alphanumeric_search.exe   (drop the .exe on Linux)
./target/release/alphanumeric-gpu-miner --address <your-40-hex-address> --worker rig1
```

Optional flags: `--batch <nonces>` (default 67108864; **bigger = higher GPU
util** — try `--batch 1073741824` on a fast card for ~full utilization),
`--device <gpu-index>`, `--kernel <path>`.

> Correctness net: the Rust host re-verifies every nonce the GPU reports against
> the CPU BLAKE3 reference (`pow::header_hash`) before submitting, so a mis-built
> kernel can never leak an invalid share — it just mines nothing. Always run
> `selftest` first.

## What it is (and what hash family)

- **PoW = a single BLAKE3 of a fixed 92-byte header.** Not a chain, not a VDF.
- Adapted from the user's **Midstate GPU miner** (also BLAKE3) — the BLAKE3
  `compress` primitive is copied **verbatim** from Midstate's M1-validated
  kernel. This is the right lineage: **NOT** the CSD/`compute-substrate` miner
  (that is SHA256d — a completely different, wrong hash for this pool).
- Wire protocol (`subscribe` / `authorize` / job `notify` / `submit`, newline-
  delimited JSON over TCP) and the PoW math are **copied verbatim** from the
  alphanumeric CPU miner, so the two are protocol- and consensus-identical.

### The 92-byte header (little-endian)

```
number(4) | previous_hash(32) | timestamp(8) | nonce(8) | difficulty(8) | merkle_root(32) = 92 bytes
```

The nonce is at bytes `[44..52)`. The GPU varies the nonce, computes
`blake3(header)`, and a hash `<= target` is a share/block. The header spans two
64-byte BLAKE3 blocks (64 + 28) — the key place the new kernel must be validated.

## Architecture (why a subprocess)

This mirrors the Midstate GPU miner's proven design rather than inventing a Rust
↔ CUDA FFI layer:

```
 Rust host (this crate)                    CUDA kernel (nvcc-built exe)
 ─────────────────────                     ───────────────────────────
 connect / subscribe / authorize
 receive job  ──build 92-byte header──►
 per batch:  spawn ──(header_hex, target_hex, nonce_start, count)──►  alphanumeric_search.exe
             ◄──────── "FOUND <nonce> <hash>" lines on stdout ───────
 re-verify each hit on CPU, submit
```

- The Rust host has **no CUDA link/FFI dependency** — it builds cleanly even on
  a box with no CUDA toolchain (it only *spawns* the kernel exe at runtime).
- `build.rs` compiles `kernel/alphanumeric_search.cu` with `nvcc` when a
  toolchain is present, and is a no-op (with a warning) when it is not.
- Result plumbing (`FOUND <nonce> <hash>` stdout lines) is reused verbatim from
  Midstate's `midstate_search.cu`.

**Performance:** the host keeps ONE kernel process alive (`kernel serve`),
holding the CUDA context + device buffers, and streams batches to it — so the
GPU stays busy instead of paying process-spawn/context-init cost per batch. Push
`--batch` up (e.g. `1073741824`) to amortise the per-batch host round-trip and
saturate a fast card.

## Build

Prerequisites:
- **Rust** (stable) — builds the host + copied protocol/PoW logic and tests.
- **CUDA Toolkit / `nvcc`** — only needed to build the GPU kernel. On the mining
  box: nvcc 13.3, RTX 5070 Ti (Blackwell, compute capability **sm_120**),
  driver 596.49.
- **Windows only: MSVC C++ Build Tools** (`cl.exe`) — `nvcc` needs a host C++
  compiler. Install "Visual Studio Build Tools" (Desktop development with C++)
  and build from the **"x64 Native Tools Command Prompt for VS 2022"** (or run
  `vcvars64.bat` first). Without `cl.exe` on PATH, `nvcc` fails — and note that
  `cargo build --release` still prints `Finished` (the kernel error is only a
  cargo warning), so confirm you saw the `Built CUDA kernel` line, or you'll hit
  "CUDA kernel not found" at runtime.

```
cargo build --release          # builds host; build.rs runs nvcc if available
# If nvcc is not on PATH, build.rs warns and you build the kernel manually:
nvcc -O3 -arch=sm_120 -o alphanumeric_search.exe kernel/alphanumeric_search.cu
```

`build.rs` places the kernel exe in `OUT_DIR` and records its path for the host.
Override the arch with `ALPHANUMERIC_NVCC_ARCH=sm_XX`, or skip the nvcc step
entirely with `ALPHANUMERIC_SKIP_NVCC=1`. Override the exe at runtime with
`--kernel <path>`.

> **sm_120 is unverified for your exact toolchain.** `sm_120` requires CUDA
> ≥ 12.8. Confirm your GPU's compute capability and that `nvcc` accepts the flag
> (see the orchestrator checklist in `tests/bit-exact-check.md` and the notes
> below).

## Validate (do this before mining)

See **[`tests/bit-exact-check.md`](tests/bit-exact-check.md)**. In short:

```
cargo test                                        # copied pow/protocol tests + header_bytes
cargo run --release --example reference_vectors   # prints reference hashes A/B/C
nvcc -O3 -arch=sm_120 -o alphanumeric_search.exe kernel/alphanumeric_search.cu
./alphanumeric_search.exe selftest                # prints GPU hashes A/B/C
# A/B/C must match byte-for-byte. Then: throughput check -> real-rig canary.
```

## Run

```
alphanumeric-gpu-miner --address <40-hex-address> \
    [--worker <label>] [--device <gpu-index>] [--batch <nonces>] [--kernel <path>]
```

- `--address` (required): your 40-char lowercase-hex alphanumeric payout address.
- The pool endpoint is **fixed** to `alphanumeric.yamaduo.no:3777` (no `--pool` flag).
- `--worker` (default `gpu-worker`) — label shown per rig.
- `--device` — CUDA device index (applied via `CUDA_VISIBLE_DEVICES`).
- `--batch` / `--threads` — nonces per GPU dispatch (`1..=4294967295`, default
  67108864). **Bigger = higher GPU utilization**; `1073741824` ~saturates a fast card.
- `--kernel` — path to the compiled `alphanumeric_search` exe (defaults to the
  build.rs-compiled path).

## File map

```
Cargo.toml                         host deps (= CPU miner's) + example wiring
build.rs                           nvcc compile of the search kernel (non-fatal)
src/main.rs                        CLI (adds --device/--batch/--kernel)
src/lib.rs                         module wiring
src/pow.rs                         COPIED verbatim from CPU miner + header_bytes()
src/protocol.rs                    COPIED verbatim from CPU miner
src/gpu.rs                         host loop (CPU miner's client loop + GPU batch dispatch)
kernel/alphanumeric_blake3.cu      frozen BLAKE3 compress (verbatim) + isolated sanity harness
kernel/alphanumeric_search.cu      NEW search kernel (single 92-byte-header BLAKE3) + CLI
examples/reference_vectors.rs      reference hashes for the bit-exact gate
tests/bit-exact-check.md            the validation procedure (DO THIS FIRST)
```
