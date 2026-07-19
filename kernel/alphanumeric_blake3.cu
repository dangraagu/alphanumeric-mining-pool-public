// alphanumeric PoW — frozen BLAKE3 primitive + isolated correctness sanity.
//
// ⚠️⚠️  UNVALIDATED SCAFFOLD — DO NOT MINE UNTIL THE BIT-EXACT GATE PASSES. ⚠️⚠️
//   The `compress` primitive below is COPIED VERBATIM from the user's Midstate
//   GPU miner, where it is M1-validated bit-exact against the `blake3` crate:
//     Documents/Claude/Projects/DGR/Midstate/gpu-miner/kernel/midstate_blake3.cu
//   DO NOT MODIFY `compress`. It is the consensus-frozen BLAKE3 compression.
//
//   What is NEW and UNVALIDATED here is `hash_header_92`: the two-block
//   (64 + 28) single-chunk BLAKE3 of the alphanumeric 92-byte header. Midstate
//   only ever hashed <=64-byte single blocks (its `FLAGS_ROOTBLOCK = 0x0B`
//   fused CHUNK_START|CHUNK_END|ROOT onto one block); a 92-byte message is TWO
//   blocks, so the flags/block_len split below (CHUNK_START on block 0;
//   CHUNK_END|ROOT on block 1) has NEVER been checked against a reference.
//   MUST be validated per tests/bit_exact_TODO.md before use.
//
// This file is the ISOLATED sanity harness (compress + one header hash). The
// actual mining kernel lives in alphanumeric_search.cu (self-contained copy).
//
// Build (VS2022 x64 dev env):
//   nvcc -O3 -arch=sm_120 -o alphanumeric_blake3_test.exe alphanumeric_blake3.cu
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// FROZEN BLAKE3 compress — copied verbatim from midstate_blake3.cu. Do not edit.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __constant__ uint32_t IVc[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ __forceinline__ void G(uint32_t* v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    v[a] = v[a] + v[b] + mx; v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + my; v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 7);
}

// Single-block BLAKE3 compression, counter t=0. out[8] = first 8 output words
// (32-byte digest AND the chaining value passed to the next block).
__device__ __forceinline__ void compress(const uint32_t cv[8], const uint32_t msg[16],
                                         uint32_t block_len, uint32_t flags, uint32_t out[8]) {
    uint32_t v[16];
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    v[8] = IVc[0]; v[9] = IVc[1]; v[10] = IVc[2]; v[11] = IVc[3];
    v[12] = 0u; v[13] = 0u; v[14] = block_len; v[15] = flags;

    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = msg[i];

    const int PERM[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};
#pragma unroll
    for (int r = 0; r < 7; r++) {
        G(v, 0, 4, 8,  12, m[0],  m[1]);
        G(v, 1, 5, 9,  13, m[2],  m[3]);
        G(v, 2, 6, 10, 14, m[4],  m[5]);
        G(v, 3, 7, 11, 15, m[6],  m[7]);
        G(v, 0, 5, 10, 15, m[8],  m[9]);
        G(v, 1, 6, 11, 12, m[10], m[11]);
        G(v, 2, 7, 8,  13, m[12], m[13]);
        G(v, 3, 4, 9,  14, m[14], m[15]);
        if (r < 6) {
            uint32_t pm[16];
#pragma unroll
            for (int i = 0; i < 16; i++) pm[i] = m[PERM[i]];
#pragma unroll
            for (int i = 0; i < 16; i++) m[i] = pm[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

// BLAKE3 domain-separation flags (standard values).
#define FLAG_CHUNK_START 0x01u
#define FLAG_CHUNK_END   0x02u
#define FLAG_ROOT        0x08u
// Midstate's single-block flag: CHUNK_START|CHUNK_END|ROOT (a <=64-byte message).
#define FLAGS_ROOTBLOCK  0x0Bu

// ─────────────────────────────────────────────────────────────────────────────
// NEW (UNVALIDATED): BLAKE3 of the 92-byte alphanumeric header.
//
// 92 bytes is a single BLAKE3 chunk spanning TWO 64-byte blocks:
//   block 0: header[0..64)   block_len=64  flags = CHUNK_START            cv = IV
//   block 1: header[64..92)  block_len=28  flags = CHUNK_END | ROOT       cv = block0 out
// The 32-byte digest is block 1's `out` (first 8 state words, little-endian per
// word, exactly as blake3::hash().as_bytes()).
//
// `nonce` is spliced into the header at byte offset 44 (= word 11 low / word 12
// high of block 0), matching pow::NONCE_OFFSET. The 8 nonce bytes lie entirely
// within block 0 and on 4-byte word boundaries (44 % 4 == 0), so:
//   m0[11] = nonce & 0xFFFFFFFF ;  m0[12] = nonce >> 32
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void hash_header_92(const uint8_t* header, uint64_t nonce, uint32_t out[8]) {
    // ── block 0: header[0..64) as 16 little-endian words, nonce spliced in.
    uint32_t m0[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        m0[i] = (uint32_t)header[4*i]
              | ((uint32_t)header[4*i+1] << 8)
              | ((uint32_t)header[4*i+2] << 16)
              | ((uint32_t)header[4*i+3] << 24);
    }
    // nonce field @ bytes [44..52) = words 11 (44..48) and 12 (48..52).
    m0[11] = (uint32_t)(nonce & 0xFFFFFFFFu);
    m0[12] = (uint32_t)(nonce >> 32);

    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = IVc[i];
    uint32_t cv1[8];
    compress(cv, m0, 64u, FLAG_CHUNK_START, cv1);   // block 0 -> chaining value

    // ── block 1: header[64..92) = 28 bytes -> 7 words, remaining words zero.
    uint32_t m1[16];
#pragma unroll
    for (int i = 0; i < 16; i++) m1[i] = 0u;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        m1[i] = (uint32_t)header[64 + 4*i]
              | ((uint32_t)header[64 + 4*i+1] << 8)
              | ((uint32_t)header[64 + 4*i+2] << 16)
              | ((uint32_t)header[64 + 4*i+3] << 24);
    }
    compress(cv1, m1, 28u, FLAG_CHUNK_END | FLAG_ROOT, out);  // block 1 -> root digest
}

// blake3("") = af1349b9... (universal known vector; localizes compress bugs).
__global__ void hash_empty(uint32_t* out) {
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = IVc[i];
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = 0u;
    uint32_t x[8];
    compress(cv, m, 0u, FLAGS_ROOTBLOCK, x);   // empty message: single 0-length block
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = x[i];
}

// Hash one 92-byte header (device buffer) with the given nonce.
__global__ void hash_header(const uint8_t* header, uint64_t nonce, uint32_t* out) {
    uint32_t x[8];
    hash_header_92(header, nonce, x);
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = x[i];
}

static void words_to_hex(const uint32_t* w, int nwords, char* hex) {
    const char* H = "0123456789abcdef";
    int p = 0;
    for (int i = 0; i < nwords; i++)
        for (int b = 0; b < 4; b++) {
            uint8_t byte = (w[i] >> (8*b)) & 0xff;   // little-endian, matches blake3 as_bytes()
            hex[p++] = H[byte >> 4]; hex[p++] = H[byte & 0xf];
        }
    hex[p] = 0;
}

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ printf("CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

int main() {
    // 1) empty-input sanity (localizes compress vs everything-else bugs).
    uint32_t* d_e; CK(cudaMalloc(&d_e, 32));
    hash_empty<<<1,1>>>(d_e); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint32_t h_e[8]; CK(cudaMemcpy(h_e, d_e, 32, cudaMemcpyDeviceToHost));
    char hex[256]; words_to_hex(h_e, 8, hex);
    int empty_ok = (strcmp(hex, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262") == 0);
    printf("empty_blake3   = %s\n", hex);
    printf("empty_expected = af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262\n");
    printf("empty MATCH    = %s\n", empty_ok ? "OK (compress primitive is bit-exact)" : "FAIL");

    // 2) hash the all-zero 92-byte header at nonce 0.
    //    NOTE: there is NO hardcoded golden here on purpose -- it must be
    //    compared against the Rust reference (pow::header_hash / the
    //    `reference_vectors` example), which is the single source of truth.
    //    See tests/bit_exact_TODO.md.
    uint8_t h_hdr[92]; memset(h_hdr, 0, sizeof(h_hdr));
    uint8_t* d_hdr; CK(cudaMalloc(&d_hdr, 92)); CK(cudaMemcpy(d_hdr, h_hdr, 92, cudaMemcpyHostToDevice));
    uint32_t* d_h; CK(cudaMalloc(&d_h, 32));
    hash_header<<<1,1>>>(d_hdr, 0ull, d_h); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint32_t h_h[8]; CK(cudaMemcpy(h_h, d_h, 32, cudaMemcpyDeviceToHost));
    char hhex[256]; words_to_hex(h_h, 8, hhex);
    printf("hdr_zero_n0    = %s\n", hhex);
    printf("hdr_zero_n0 golden = <FILL FROM RUST: `cargo run --example reference_vectors`>\n");

    return empty_ok ? 0 : 1;
}
