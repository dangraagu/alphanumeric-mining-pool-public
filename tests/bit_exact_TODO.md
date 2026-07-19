# BIT-EXACT VALIDATION — MUST PASS BEFORE MINING

**Status: NOT DONE. The CUDA kernel `hash_header_92` is UNVALIDATED.**

The GPU kernel computes BLAKE3 of the 92-byte alphanumeric header. The BLAKE3
`compress` primitive is copied verbatim from the user's Midstate GPU miner and
is already M1-validated bit-exact — but the **two-block (64 + 28) single-chunk
assembly** in `hash_header_92` is NEW and has never been checked against a
reference. It must match the Rust reference `pow::header_hash`
(== `blake3::hash(header_92_bytes)`) exactly, or every block/share the GPU finds
is worthless.

## The reference (source of truth)

`src/pow.rs::header_hash(number, previous_hash, timestamp, nonce, difficulty,
merkle_root)` = `blake3::hash` of the 92-byte little-endian header:

```
number(4) | previous_hash(32) | timestamp(8) | nonce(8) | difficulty(8) | merkle_root(32)
```

The nonce field is at byte offset **44..52** (`pow::NONCE_OFFSET = 44`).

## The exact check

Three canonical vectors are defined identically on both sides:

| Vector | Header | Nonce |
|--------|--------|-------|
| A | all-zero 92 bytes | 0 |
| B | all-zero 92 bytes | 1 |
| C | number=0x11111111, prev=0x22×32, timestamp=0x33×8, difficulty=0x55×8, merkle=0x66×32 | 0x4444444444444444 |

Steps:

1. **Print the reference hashes (Rust):**
   ```
   cargo run --release --example reference_vectors
   ```
   (Also run `cargo test` — `pow::header_bytes_then_blake3_equals_header_hash`
   and the copied `pow`/`protocol` tests must pass.)

2. **Build + run the kernel selftest (needs nvcc + a CUDA GPU):**
   ```
   nvcc -O3 -arch=sm_120 -o alphanumeric_search.exe kernel/alphanumeric_search.cu
   ./alphanumeric_search.exe selftest
   ```
   Also build the isolated primitive harness and confirm the empty-input line
   prints `... MATCH = OK` (proves the copied `compress` survived the copy):
   ```
   nvcc -O3 -arch=sm_120 -o alphanumeric_blake3_test.exe kernel/alphanumeric_blake3.cu
   ./alphanumeric_blake3_test.exe
   ```

3. **Compare:** the kernel `selftest` lines A / B / C must equal the Rust
   `reference_vectors` lines A / B / C **byte-for-byte**. Vector A additionally
   equals the `hdr_zero_n0` line from `alphanumeric_blake3_test.exe`.

   - If **A** differs → the base two-block assembly (flags / block_len / block
     chaining) is wrong. Suspects, in order:
     - block 0 flags must be `CHUNK_START (0x01)` only (NOT `0x0B`).
     - block 1 flags must be `CHUNK_END | ROOT (0x0A)`.
     - block 0 `block_len = 64`, block 1 `block_len = 28`.
     - block 1's `cv` must be block 0's output (the chaining value), NOT `IV`.
     - counter `t = 0` for both blocks (same chunk).
   - If **A** matches but **B** differs → the nonce splice (`m0[11]`/`m0[12]`)
     is wrong.
   - If **A** and **B** match but **C** differs → a field-to-word packing bug in
     block 0 words other than the nonce, or in block 1 (bytes 64..92 = the tail
     28 bytes of `merkle_root`).

4. **Only after all three match:** proceed to a throughput sanity check, then a
   REAL-RIG canary against the live pool (submit a low-difficulty share and
   confirm the pool ACCEPTS it) before trusting the miner. The Rust host already
   re-verifies every reported nonce on the CPU (`gpu::run`), so a still-broken
   kernel cannot leak an invalid share upstream — but it will silently mine
   nothing.

## Independent cross-check (recommended, no GPU needed)

You can validate `hash_header_92`'s block-splitting logic on the CPU without a
GPU by confirming a Rust `blake3::Hasher` fed the same 92 bytes equals
`pow::header_hash` (the `pow` unit test
`header_bytes_then_blake3_equals_header_hash` already does exactly this). That
proves the *reference* is self-consistent; the GPU selftest is what proves the
*kernel* reproduces it.
