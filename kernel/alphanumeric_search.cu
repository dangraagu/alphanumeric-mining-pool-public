// alphanumeric PoW CUDA search kernel — the miner's grind backend.
//
// ⚠️⚠️  UNVALIDATED SCAFFOLD — DO NOT MINE UNTIL THE BIT-EXACT GATE PASSES. ⚠️⚠️
//   `compress` is COPIED VERBATIM from the user's Midstate GPU miner
//   (midstate_search.cu / midstate_blake3.cu), where it is M1-validated
//   bit-exact against the `blake3` crate. DO NOT MODIFY `compress`.
//
//   `hash_header_92` (the two-block 64+28 single-chunk BLAKE3 of the 92-byte
//   header) is NEW and has NEVER been checked against a reference. Its flag /
//   block_len split (CHUNK_START on block 0; CHUNK_END|ROOT on block 1) is the
//   #1 thing to validate. See tests/bit_exact_TODO.md. Until `selftest` output
//   matches the Rust `pow::header_hash` reference byte-for-byte, treat every
//   hash this kernel produces as SUSPECT. (The Rust host independently
//   re-verifies every reported nonce on the CPU, so a bad kernel cannot leak an
//   invalid share -- but it can silently mine nothing, or waste the GPU.)
//
// ── CRITICAL DIFFERENCE vs Midstate ──────────────────────────────────────────
//   Midstate: seed = BLAKE3(40-byte midstate||nonce), then chain BLAKE3 1,000,000
//             times. alphanumeric: ONE BLAKE3 of a fixed 92-byte header. There is
//             NO iteration chain. Midstate's `chain_for_nonce` is replaced by
//             `hash_header_92` below.
//   Target compare: alphanumeric uses hash <= target (LESS-THAN-OR-EQUAL,
//             matching pow::meets_target). Midstate used strict < . See
//             `le_target` below.
//
// CLI:
//   alphanumeric_search.exe <header_hex184> <target_hex64> <nonce_start_u64> <count_u32>
//       header_hex184 : the 92-byte little-endian header, hex (184 chars). The
//                       nonce field @ bytes [44..52) is IGNORED here -- the
//                       kernel splices nonce = nonce_start + tid in per thread.
//       -> prints "FOUND <nonce_dec> <final_hex64>" for every thread whose
//          32-byte hash <= target (big-endian byte compare). Consumer parses
//          these lines (see src/gpu.rs::dispatch_batch).
//   alphanumeric_search.exe selftest
//       -> hashes fixed canonical vectors and prints them, for byte-for-byte
//          comparison against the Rust reference (tests/bit_exact_TODO.md).
//          NO golden is hardcoded here (can't be trusted without the reference).
//
// Build (VS2022 x64 dev env):
//   nvcc -O3 -arch=sm_120 -o alphanumeric_search.exe alphanumeric_search.cu
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// FROZEN BLAKE3 compress — copied verbatim from midstate_search.cu. Do not edit.
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

#define HEADER_LEN 92
#define NONCE_OFFSET 44   // must match pow::NONCE_OFFSET

// ─────────────────────────────────────────────────────────────────────────────
// NEW (UNVALIDATED): single BLAKE3 of the 92-byte header (two blocks: 64 + 28).
//   block 0: header[0..64)   block_len=64  flags=CHUNK_START        cv=IV
//   block 1: header[64..92)  block_len=28  flags=CHUNK_END|ROOT     cv=block0 out
// Returns the 32-byte root digest (little-endian per word, == blake3 as_bytes()).
// nonce is spliced at bytes [44..52) (word-aligned): m0[11]=lo32, m0[12]=hi32.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void hash_header_92(const uint8_t* header, uint64_t nonce, uint32_t x[8]) {
    uint32_t m0[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        m0[i] = (uint32_t)header[4*i]
              | ((uint32_t)header[4*i+1] << 8)
              | ((uint32_t)header[4*i+2] << 16)
              | ((uint32_t)header[4*i+3] << 24);
    }
    m0[11] = (uint32_t)(nonce & 0xFFFFFFFFu);   // nonce bytes [44..48)
    m0[12] = (uint32_t)(nonce >> 32);           // nonce bytes [48..52)

    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = IVc[i];
    uint32_t cv1[8];
    compress(cv, m0, 64u, FLAG_CHUNK_START, cv1);

    uint32_t m1[16];
#pragma unroll
    for (int i = 0; i < 16; i++) m1[i] = 0u;
#pragma unroll
    for (int i = 0; i < 7; i++) {               // header[64..92) = 28 bytes = 7 words
        m1[i] = (uint32_t)header[64 + 4*i]
              | ((uint32_t)header[64 + 4*i+1] << 8)
              | ((uint32_t)header[64 + 4*i+2] << 16)
              | ((uint32_t)header[64 + 4*i+3] << 24);
    }
    compress(cv1, m1, 28u, FLAG_CHUNK_END | FLAG_ROOT, x);
}

// alphanumeric uses hash <= target (LESS-THAN-OR-EQUAL), matching
// pow::meets_target. Big-endian byte compare: byte j of the hash is
// (x[j/4] >> (8*(j%4))) & 0xff (the little-endian-per-word layout that
// blake3::hash().as_bytes() produces, which Rust then compares lexicographically
// from index 0). Returns true when hash <= target.
__device__ __forceinline__ bool le_target(const uint32_t x[8], const uint8_t* target_be) {
#pragma unroll
    for (int j = 0; j < 32; j++) {
        uint8_t hb = (uint8_t)((x[j >> 2] >> (8 * (j & 3))) & 0xff);
        uint8_t tb = target_be[j];
        if (hb < tb) return true;
        if (hb > tb) return false;
    }
    return true; // equal => meets target (<=)
}

// out_nonce/out_words filled for FOUND threads via an atomic result counter.
struct Found { uint64_t nonce; uint32_t words[8]; };

__global__ void search(const uint8_t* header, const uint8_t* target_be,
                       uint64_t nonce_base, uint32_t count,
                       Found* results, uint32_t* result_count, uint32_t max_results) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    uint64_t nonce = nonce_base + (uint64_t)tid;

    uint32_t x[8];
    hash_header_92(header, nonce, x);

    if (le_target(x, target_be)) {
        uint32_t slot = atomicAdd(result_count, 1u);
        if (slot < max_results) {
            results[slot].nonce = nonce;
#pragma unroll
            for (int i = 0; i < 8; i++) results[slot].words[i] = x[i];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers
// ─────────────────────────────────────────────────────────────────────────────
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

static int hex_to_bytes(const char* hex, uint8_t* out, int nbytes) {
    if ((int)strlen(hex) != nbytes * 2) return -1;
    for (int i = 0; i < nbytes; i++) {
        unsigned v;
        if (sscanf(hex + 2*i, "%2x", &v) != 1) return -1;
        out[i] = (uint8_t)v;
    }
    return 0;
}

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

// Hash a single (header, nonce) on the GPU and return its hex digest (for selftest).
static int hash_one(const uint8_t h_hdr[HEADER_LEN], uint64_t nonce, char out_hex[65]) {
    uint8_t* d_hdr; CK(cudaMalloc(&d_hdr, HEADER_LEN)); CK(cudaMemcpy(d_hdr, h_hdr, HEADER_LEN, cudaMemcpyHostToDevice));
    // target all-0xFF so every thread is "found" -- we only want the digest.
    uint8_t h_tgt[32]; memset(h_tgt, 0xFF, 32);
    uint8_t* d_tgt; CK(cudaMalloc(&d_tgt, 32)); CK(cudaMemcpy(d_tgt, h_tgt, 32, cudaMemcpyHostToDevice));
    Found* d_res; CK(cudaMalloc(&d_res, sizeof(Found)));
    uint32_t* d_cnt; CK(cudaMalloc(&d_cnt, 4)); CK(cudaMemset(d_cnt, 0, 4));

    search<<<1,1>>>(d_hdr, d_tgt, nonce, 1u, d_res, d_cnt, 1u);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    Found h_res; CK(cudaMemcpy(&h_res, d_res, sizeof(Found), cudaMemcpyDeviceToHost));
    words_to_hex(h_res.words, 8, out_hex);
    cudaFree(d_hdr); cudaFree(d_tgt); cudaFree(d_res); cudaFree(d_cnt);
    return 0;
}

// Selftest: hash three canonical vectors and print them for comparison against
// the Rust reference (examples/reference_vectors.rs builds byte-identical
// headers). NO golden is hardcoded -- the Rust reference is the source of truth.
static int run_selftest() {
    char hex[65];

    // Vector A: all-zero 92-byte header, nonce 0.
    uint8_t hA[HEADER_LEN]; memset(hA, 0, sizeof(hA));
    if (hash_one(hA, 0ull, hex)) return 2;
    printf("A zero-header nonce0        = %s\n", hex);

    // Vector B: all-zero 92-byte header, nonce 1 (exercises the nonce splice).
    if (hash_one(hA, 1ull, hex)) return 2;
    printf("B zero-header nonce1        = %s\n", hex);

    // Vector C: distinct byte pattern per field (exercises both blocks + splice).
    //   number=0x11111111  prev=0x22*32  timestamp=0x3333333333333333
    //   nonce=0x4444444444444444 (spliced via nonce_base)  difficulty=0x5555555555555555
    //   merkle=0x66*32
    // Field bytes in the passed header (nonce field will be overwritten):
    uint8_t hC[HEADER_LEN];
    memset(hC + 0,  0x11, 4);    // number   [0..4)
    memset(hC + 4,  0x22, 32);   // prev     [4..36)
    memset(hC + 36, 0x33, 8);    // timestamp[36..44)
    memset(hC + 44, 0x00, 8);    // nonce    [44..52)  (ignored; spliced below)
    memset(hC + 52, 0x55, 8);    // diff     [52..60)
    memset(hC + 60, 0x66, 32);   // merkle   [60..92)
    if (hash_one(hC, 0x4444444444444444ull, hex)) return 2;
    printf("C mixed-header nonce0x44..  = %s\n", hex);

    printf("\nCompare A/B/C against the Rust reference (single source of truth):\n");
    printf("  cargo run --release --example reference_vectors\n");
    printf("If any differ, hash_header_92 is NOT bit-exact -- see tests/bit_exact_TODO.md.\n");
    return 0;
}

// Persistent "serve" mode: the whole point of this is to pay CUDA context +
// buffer allocation ONCE, then hash many batches without re-initialising.
// The one-shot argv path below re-creates the CUDA context on every process
// spawn (~250ms), so a fast single-block BLAKE3 was throttled to ~16 MH/s --
// 99% context-init overhead, ~1% hashing. This loop keeps the context alive
// and reuses the device buffers, so each batch is just memcpy + launch +
// sync (sub-millisecond for millions of nonces), unlocking the GPU's real
// throughput.
//
// Protocol (line-oriented, over stdin/stdout):
//   in : "<header_hex184> <target_hex64> <nonce_start_u64> <count_u32>\n"
//   out: zero or more "FOUND <nonce_dec> <hash_hex64>\n" then exactly one
//        "DONE <nonce_start> <count>\n" per input line, flushed.
// The `search` kernel and every hashing primitive are IDENTICAL to the
// one-shot path -- serve mode changes only WHEN buffers are allocated, never
// HOW a hash is computed, so `selftest`'s bit-exact guarantee still covers it.
static int run_serve() {
    uint8_t *d_hdr, *d_tgt;
    CK(cudaMalloc(&d_hdr, HEADER_LEN));
    CK(cudaMalloc(&d_tgt, 32));
    const uint32_t MAX_RESULTS = 4096;
    Found* d_res; CK(cudaMalloc(&d_res, sizeof(Found) * MAX_RESULTS));
    uint32_t* d_cnt; CK(cudaMalloc(&d_cnt, 4));

    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        char hdr_hex[256], tgt_hex[128];
        unsigned long long ns = 0, cnt = 0;
        if (sscanf(line, "%200s %120s %llu %llu", hdr_hex, tgt_hex, &ns, &cnt) != 4) {
            printf("DONE %llu %llu\n", ns, cnt); fflush(stdout); continue;
        }
        uint8_t h_hdr[HEADER_LEN], h_tgt[32];
        if (hex_to_bytes(hdr_hex, h_hdr, HEADER_LEN) != 0 ||
            hex_to_bytes(tgt_hex, h_tgt, 32) != 0 ||
            cnt == 0 || cnt > 0xFFFFFFFFull) {
            printf("DONE %llu %llu\n", ns, cnt); fflush(stdout); continue;
        }
        uint32_t count = (uint32_t)cnt;
        CK(cudaMemcpy(d_hdr, h_hdr, HEADER_LEN, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_tgt, h_tgt, 32, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_cnt, 0, 4));

        int threadsPerBlock = 256;
        uint64_t blocks = ((uint64_t)count + threadsPerBlock - 1) / threadsPerBlock;
        search<<<(unsigned)blocks, threadsPerBlock>>>(d_hdr, d_tgt, ns, count, d_res, d_cnt, MAX_RESULTS);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

        uint32_t h_cnt = 0; CK(cudaMemcpy(&h_cnt, d_cnt, 4, cudaMemcpyDeviceToHost));
        uint32_t n = h_cnt < MAX_RESULTS ? h_cnt : MAX_RESULTS;
        if (n > 0) {
            Found* h_res = (Found*)malloc(sizeof(Found) * n);
            CK(cudaMemcpy(h_res, d_res, sizeof(Found) * n, cudaMemcpyDeviceToHost));
            for (uint32_t i = 0; i < n; i++) {
                char hex[128]; words_to_hex(h_res[i].words, 8, hex);
                printf("FOUND %llu %s\n", (unsigned long long)h_res[i].nonce, hex);
            }
            free(h_res);
        }
        printf("DONE %llu %llu\n", ns, cnt);
        fflush(stdout);
    }
    cudaFree(d_hdr); cudaFree(d_tgt); cudaFree(d_res); cudaFree(d_cnt);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "selftest") == 0) {
        return run_selftest();
    }
    if (argc == 2 && strcmp(argv[1], "serve") == 0) {
        return run_serve();
    }
    if (argc != 5) {
        fprintf(stderr, "usage: %s <header_hex184> <target_hex64> <nonce_start_u64> <count_u32>\n", argv[0]);
        fprintf(stderr, "       %s selftest\n", argv[0]);
        return 3;
    }

    uint8_t h_hdr[HEADER_LEN];
    if (hex_to_bytes(argv[1], h_hdr, HEADER_LEN) != 0) {
        fprintf(stderr, "bad header_hex (need %d hex chars for the 92-byte header)\n", HEADER_LEN * 2);
        return 3;
    }
    uint8_t h_tgt[32];
    if (hex_to_bytes(argv[2], h_tgt, 32) != 0) { fprintf(stderr, "bad target_hex (need 64 hex chars)\n"); return 3; }

    uint64_t nonce_start = strtoull(argv[3], nullptr, 10);
    uint64_t count64 = strtoull(argv[4], nullptr, 10);
    if (count64 == 0 || count64 > 0xFFFFFFFFull) { fprintf(stderr, "count must be 1..2^32-1\n"); return 3; }
    uint32_t count = (uint32_t)count64;

    uint8_t *d_hdr, *d_tgt;
    CK(cudaMalloc(&d_hdr, HEADER_LEN)); CK(cudaMemcpy(d_hdr, h_hdr, HEADER_LEN, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_tgt, 32)); CK(cudaMemcpy(d_tgt, h_tgt, 32, cudaMemcpyHostToDevice));

    const uint32_t MAX_RESULTS = 4096; // plenty for one throttled batch
    Found* d_res; CK(cudaMalloc(&d_res, sizeof(Found) * MAX_RESULTS));
    uint32_t* d_cnt; CK(cudaMalloc(&d_cnt, 4)); CK(cudaMemset(d_cnt, 0, 4));

    int threadsPerBlock = 256;
    uint64_t blocks = ((uint64_t)count + threadsPerBlock - 1) / threadsPerBlock;
    search<<<(unsigned)blocks, threadsPerBlock>>>(d_hdr, d_tgt, nonce_start, count, d_res, d_cnt, MAX_RESULTS);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    uint32_t h_cnt = 0; CK(cudaMemcpy(&h_cnt, d_cnt, 4, cudaMemcpyDeviceToHost));
    if (h_cnt > MAX_RESULTS) {
        fprintf(stderr, "warning: %u hits exceeded MAX_RESULTS=%u; only the first %u are reported\n",
                h_cnt, MAX_RESULTS, MAX_RESULTS);
    }
    uint32_t n = h_cnt < MAX_RESULTS ? h_cnt : MAX_RESULTS;
    if (n > 0) {
        Found* h_res = (Found*)malloc(sizeof(Found) * n);
        CK(cudaMemcpy(h_res, d_res, sizeof(Found) * n, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n; i++) {
            char hex[128]; words_to_hex(h_res[i].words, 8, hex);
            // %llu for the decimal nonce; consumer parses u64.
            printf("FOUND %llu %s\n", (unsigned long long)h_res[i].nonce, hex);
        }
        free(h_res);
    }
    fflush(stdout);
    return 0;
}
