//! PoW primitives for the alphanumeric pool's share-distribution protocol:
//! the 92-byte header hash and the target comparison a miner needs to grind
//! nonces client-side.
//!
//! This is a deliberate duplication of the pool backend's own `src/pow.rs`
//! (a separate, private repo) rather than a shared dependency -- this crate
//! is a standalone public binary and must never depend on the private pool
//! backend's crate. The logic is simple pure math + BLAKE3, so keeping two
//! independent copies in sync by inspection is an acceptable v1 trade-off.
//!
//! ── PROVENANCE ──────────────────────────────────────────────────────────────
//! `header_hash` and `meets_target` are COPIED VERBATIM from the alphanumeric
//! CPU miner (`alphanumeric-mining-pool-public/src/pow.rs`). `header_bytes` is
//! NEW (added for the GPU host): it materialises the exact 92-byte header the
//! CPU miner hashes, so the GPU host can hex-encode it and hand it to the CUDA
//! kernel. `header_hash` is left byte-for-byte identical to the CPU miner's copy
//! and is the REFERENCE the CUDA kernel must reproduce bit-for-bit
//! (see `tests/bit-exact-check.md`); the GPU host uses it to re-verify every
//! nonce the kernel reports before submitting.

/// Byte offsets of each field inside the 92-byte little-endian header, and the
/// total header length. Kept as named constants so the GPU host, the CUDA
/// kernel, and this module all agree on exactly one layout.
///
/// Layout (all little-endian):
///   number(4)   @ [0..4)
///   prev_hash(32) @ [4..36)
///   timestamp(8) @ [36..44)
///   nonce(8)     @ [44..52)   <-- the field the GPU varies per thread
///   difficulty(8) @ [52..60)
///   merkle(32)   @ [60..92)
pub const HEADER_LEN: usize = 92;
/// Byte offset of the 8-byte little-endian nonce field within the header.
/// The CUDA kernel splices `nonce_base + thread_id` in at exactly this offset.
pub const NONCE_OFFSET: usize = 44;

/// Materialise the exact 92-byte little-endian header the alphanumeric chain
/// hashes: number(4) | previous_hash(32) | timestamp(8) | nonce(8) |
/// difficulty(8) | merkle_root(32).
///
/// NEW (not in the CPU miner). The GPU host calls this once per job with the
/// nonce field zeroed, hex-encodes the result, and passes it to the kernel;
/// the kernel then overwrites bytes `[NONCE_OFFSET..NONCE_OFFSET+8)` with each
/// thread's own nonce. It is verified against `header_hash` below so the two
/// can never drift.
pub fn header_bytes(
    number: u32,
    previous_hash: [u8; 32],
    timestamp: u64,
    nonce: u64,
    difficulty: u64,
    merkle_root: [u8; 32],
) -> [u8; HEADER_LEN] {
    let mut header = [0u8; HEADER_LEN];
    let mut off = 0;
    header[off..off + 4].copy_from_slice(&number.to_le_bytes());
    off += 4;
    header[off..off + 32].copy_from_slice(&previous_hash);
    off += 32;
    header[off..off + 8].copy_from_slice(&timestamp.to_le_bytes());
    off += 8;
    header[off..off + 8].copy_from_slice(&nonce.to_le_bytes());
    off += 8;
    header[off..off + 8].copy_from_slice(&difficulty.to_le_bytes());
    off += 8;
    header[off..off + 32].copy_from_slice(&merkle_root);
    header
}

/// BLAKE3 over the 92-byte little-endian header layout used by the
/// alphanumeric chain's own miner hot loop: number(4) | previous_hash(32) |
/// timestamp(8) | nonce(8) | difficulty(8) | merkle_root(32).
pub fn header_hash(
    number: u32,
    previous_hash: [u8; 32],
    timestamp: u64,
    nonce: u64,
    difficulty: u64,
    merkle_root: [u8; 32],
) -> [u8; 32] {
    let mut header = [0u8; 92];
    let mut off = 0;
    header[off..off + 4].copy_from_slice(&number.to_le_bytes());
    off += 4;
    header[off..off + 32].copy_from_slice(&previous_hash);
    off += 32;
    header[off..off + 8].copy_from_slice(&timestamp.to_le_bytes());
    off += 8;
    header[off..off + 8].copy_from_slice(&nonce.to_le_bytes());
    off += 8;
    header[off..off + 8].copy_from_slice(&difficulty.to_le_bytes());
    off += 8;
    header[off..off + 32].copy_from_slice(&merkle_root);
    *blake3::hash(&header).as_bytes()
}

/// Lexicographic byte compare == numeric compare for fixed-width big-endian
/// values -- a hash "meets" a target when it is numerically <= it.
pub fn meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    hash <= target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_hash_matches_hand_built_92_byte_layout() {
        // Field order/widths (all little-endian): number:u32(4) |
        // previous_hash(32) | timestamp:u64(8) | nonce:u64(8) |
        // difficulty:u64(8) | merkle_root(32) = 92 bytes, then blake3::hash.
        // Same self-consistency pattern the pool backend's own pow.rs tests
        // use: hand-build the header, hash it, compare against the function
        // under test.
        let number: u32 = 42;
        let previous_hash = [0x11u8; 32];
        let timestamp: u64 = 1_783_600_000;
        let nonce: u64 = 7_377;
        let difficulty: u64 = 464;
        let merkle_root = [0x22u8; 32];

        let mut expected = [0u8; 92];
        let mut off = 0;
        expected[off..off + 4].copy_from_slice(&number.to_le_bytes());
        off += 4;
        expected[off..off + 32].copy_from_slice(&previous_hash);
        off += 32;
        expected[off..off + 8].copy_from_slice(&timestamp.to_le_bytes());
        off += 8;
        expected[off..off + 8].copy_from_slice(&nonce.to_le_bytes());
        off += 8;
        expected[off..off + 8].copy_from_slice(&difficulty.to_le_bytes());
        off += 8;
        expected[off..off + 32].copy_from_slice(&merkle_root);
        let expected_hash = *blake3::hash(&expected).as_bytes();

        let actual_hash = header_hash(number, previous_hash, timestamp, nonce, difficulty, merkle_root);

        assert_eq!(actual_hash, expected_hash);
    }

    #[test]
    fn header_hash_changes_when_nonce_changes() {
        // A sanity check that grinding nonces actually explores the hash
        // space (a bug that ignored `nonce` would pass the hand-built test
        // above only by coincidence if the test always used the same nonce
        // value it computed against -- this catches that class of bug).
        let base = |nonce: u64| header_hash(1, [0u8; 32], 1_000, nonce, 464, [0u8; 32]);
        assert_ne!(base(0), base(1));
    }

    #[test]
    fn meets_target_is_lexicographic_less_or_equal() {
        let target = [0x80u8; 32];

        let mut under = target;
        under[31] -= 1;
        assert!(meets_target(&under, &target));

        assert!(meets_target(&target, &target));

        let mut over = target;
        over[0] += 1;
        assert!(!meets_target(&over, &target));
    }

    #[test]
    fn meets_target_max_target_accepts_any_hash() {
        // difficulty 0 on the pool side means target = [0xff; 32] -- every
        // possible hash meets it, matching the pool's own max-target
        // semantics.
        let max_target = [0xffu8; 32];
        assert!(meets_target(&[0u8; 32], &max_target));
        assert!(meets_target(&[0xffu8; 32], &max_target));
    }

    // ── NEW: header_bytes tests (guard the GPU host <-> kernel contract) ─────

    #[test]
    fn header_bytes_is_92_bytes_with_nonce_at_offset_44() {
        let h = header_bytes(0, [0u8; 32], 0, 0xAABB_CCDD_1122_3344, 0, [0u8; 32]);
        assert_eq!(h.len(), HEADER_LEN);
        assert_eq!(NONCE_OFFSET, 44);
        // nonce is little-endian at [44..52): LSB first.
        assert_eq!(&h[44..52], &0xAABB_CCDD_1122_3344u64.to_le_bytes());
    }

    #[test]
    fn header_bytes_then_blake3_equals_header_hash() {
        // The GPU host builds `header_bytes` and (indirectly, via the kernel)
        // hashes it; this proves hashing those exact bytes reproduces the
        // reference `header_hash`, so the kernel target is unambiguous.
        let number: u32 = 42;
        let previous_hash = [0x11u8; 32];
        let timestamp: u64 = 1_783_600_000;
        let nonce: u64 = 7_377;
        let difficulty: u64 = 464;
        let merkle_root = [0x22u8; 32];

        let bytes = header_bytes(number, previous_hash, timestamp, nonce, difficulty, merkle_root);
        let via_bytes = *blake3::hash(&bytes).as_bytes();
        let via_ref = header_hash(number, previous_hash, timestamp, nonce, difficulty, merkle_root);
        assert_eq!(via_bytes, via_ref);
    }
}
