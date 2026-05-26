/*
 * Unified MGARD-X Huffman + ZSTD encoder/decoder.
 *
 * The two backends below produce a BIT-IDENTICAL compressed bitstream for
 * the same uint32 input + the same Config (verified by virtue of MGARD-X
 * `ComposedLosslessCompressor<uint32_t, uint64_t, DeviceType>` being a
 * template whose Huffman.Serialize / Zstd code path is device-agnostic).
 *
 * Both backends compile and link against MGARD; the SERIAL backend uses
 * `mgard_x::SERIAL` (host pointers) and the HIP backend uses
 * `mgard_x::HIP` (DEVICE pointers). One library may link both .o files
 * to dispatch at runtime via an env var (see operator code).
 *
 * Bitstream format produced here: raw `ComposedLosslessCompressor` payload
 * (no header, no quantum). Callers (the operator) are responsible for
 * framing (resMarker, quantum, payload length).
 *
 * Configuration knobs are fixed (hard-coded in each backend) because they
 * affect the bitstream. Tuning requires changing both files together.
 *   H = uint64_t                 (max codeword = 56 bits — safe for CFD)
 *   huff_dict_size       = 8192
 *   huff_block_size      = 20480
 *   zstd_compress_level  = 3
 *   estimate_outlier_ratio = 1.0
 *   lossless = Huffman_Zstd
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace mgardx_lossless {

/* ── SERIAL (CPU) backend ────────────────────────────────────────────
 * Pointers MUST be host (regular malloc / std::vector::data() etc.).
 * Implemented in MGARDXLossless_SERIAL.cpp (LANGUAGE CXX, links mgard).
 */
namespace serial {

/* Compress `n` uint32 values pointed to by `h_quant` (HOST pointer) into
 * `h_out` (HOST buffer of capacity `h_out_cap`). Returns the number of
 * bytes written, or 0 on failure (including overflow). */
std::size_t Compress(const std::uint32_t *h_quant, std::size_t n,
                     char *h_out, std::size_t h_out_cap);

/* Decompress `in_len` bytes from `h_in` (HOST pointer) into `h_quant_out`
 * (HOST buffer of length `n` uint32 values). Returns true on success. */
bool Decompress(const char *h_in, std::size_t in_len,
                std::size_t n, std::uint32_t *h_quant_out);

} // namespace serial


/* ── HIP (GPU) backend ───────────────────────────────────────────────
 * The quantized input pointer MUST be a HIP device pointer; the output
 * compressed bytes are copied back to a HOST buffer (`h_out`). Likewise
 * decompress reads from a HOST buffer and writes to a DEVICE pointer.
 * Implemented in MGARDXLossless_HIP.cpp (LANGUAGE HIP).
 */
namespace hip {

/* Compress `n` uint32 values pointed to by `d_quant` (DEVICE pointer)
 * into `h_out` (HOST buffer). Returns bytes written, or 0 on failure. */
std::size_t Compress(const std::uint32_t *d_quant, std::size_t n,
                     char *h_out, std::size_t h_out_cap);

/* Decompress `in_len` bytes from `h_in` (HOST pointer) into `d_quant_out`
 * (DEVICE pointer to `n` uint32 values). Returns true on success. */
bool Decompress(const char *h_in, std::size_t in_len,
                std::size_t n, std::uint32_t *d_quant_out);

} // namespace hip

} // namespace mgardx_lossless
