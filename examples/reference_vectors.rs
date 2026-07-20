//! Prints the REFERENCE BLAKE3 hashes for the three canonical bit-exact test
//! vectors, using the exact same `pow::header_hash` the pool + CPU miner use.
//! This is the single source of truth the CUDA kernel's `selftest` output must
//! match byte-for-byte. Modelled on Midstate's `examples/capture_golden.rs`.
//!
//! Run:  cargo run --release --example reference_vectors
//!
//! The three vectors are byte-identical to those in
//! `kernel/alphanumeric_search.cu::run_selftest`. See `tests/bit-exact-check.md`.

use alphanumeric_gpu_miner::pow::header_hash;

fn main() {
    // Vector A: all-zero 92-byte header, nonce 0.
    let a = header_hash(0, [0u8; 32], 0, 0, 0, [0u8; 32]);
    // Vector B: all-zero 92-byte header, nonce 1 (exercises the nonce splice).
    let b = header_hash(0, [0u8; 32], 0, 1, 0, [0u8; 32]);
    // Vector C: distinct byte pattern per field (exercises both blocks + splice).
    let c = header_hash(
        0x1111_1111,                 // number
        [0x22u8; 32],                // previous_hash
        0x3333_3333_3333_3333,       // timestamp
        0x4444_4444_4444_4444,       // nonce
        0x5555_5555_5555_5555,       // difficulty
        [0x66u8; 32],                // merkle_root
    );

    println!("A zero-header nonce0        = {}", hex::encode(a));
    println!("B zero-header nonce1        = {}", hex::encode(b));
    println!("C mixed-header nonce0x44..  = {}", hex::encode(c));
    println!();
    println!("These must equal `alphanumeric_search.exe selftest` lines A/B/C byte-for-byte.");
}
