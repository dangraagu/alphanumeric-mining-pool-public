//! Library crate for `alphanumeric-gpu-miner`: a CUDA GPU miner for the
//! alphanumeric mining pool's share-distribution protocol.
//!
//! Structure mirrors the alphanumeric CPU miner (`alphanumeric-mining-pool-public`)
//! so the two share an identical wire protocol and PoW definition:
//!   * [`pow`]      -- the 92-byte header hash + target compare (copied verbatim
//!                     from the CPU miner, plus a `header_bytes` helper the GPU
//!                     host needs to hand the raw header to the kernel).
//!   * [`protocol`] -- the newline-delimited-JSON pool messages (copied verbatim).
//!   * [`gpu`]      -- the client loop: connect/subscribe/authorize/resubscribe/
//!                     vardiff logic reused from the CPU miner's `miner.rs`, with
//!                     the per-nonce CPU grind swapped for GPU batch dispatch.

pub mod gpu;
pub mod pow;
pub mod protocol;
