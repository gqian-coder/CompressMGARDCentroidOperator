/*
 * MGARDXLossless_SERIAL.cpp — SERIAL (CPU) backend.
 *
 * Compiled as plain C++ (no -x hip). Cray's CC does not define __HIPCC__
 * for CXX TUs, so DataTypes.h falls through to MGARDX_COMPILE_SERIAL and
 * RuntimeX.h only pulls in DeviceAdapterSerial.h.
 *
 * Links against MGARD (either mgard_serial install or HIP install — both
 * ship the SERIAL adapter; the bitstream is identical).
 */

#include "MGARDXLossless.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include <zstd.h>

/*
 * NOTE: The original implementation used MGARD-X's
 * `ComposedLosslessCompressor<uint32_t,uint64_t,SERIAL>` (Huffman+Zstd).
 * That code path is broken in the MGARD HIP-install build we link against
 * here on Frontier — every call (irrespective of n) triggers a corrupt
 * giant `Malloc1D` from inside `Huffman<SERIAL>::Serialize`. The top-level
 * `mgard_x::compress` works (used by the cell-average path), but the
 * lossless-only template path does not.
 *
 * Pragmatic fix: drop in libzstd directly on the raw uint32 byte buffer.
 * We lose the Huffman pass, so CR on the residual quantization stream is
 * typically 1.5-2x worse than the original spec. The bitstream remains
 * stable across SERIAL/HIP since we mirror the change in the HIP backend.
 *
 * Knobs:
 *   MGARDX_ZSTD_LEVEL  (default 3) — zstd compression level.
 * Removed knobs (no longer used by this backend):
 *   MGARDX_LOSSLESS, MGARDX_HUFF_DICT, MGARDX_HUFF_BLOCK, MGARDX_OUTLIER_RATIO.
 */

namespace mgardx_lossless {
namespace serial {

namespace {

int env_int(const char *name, int def) {
  const char *s = std::getenv(name);
  if (!s || !*s) return def;
  try { return std::stoi(s); } catch (...) { return def; }
}

} // namespace

std::size_t Compress(const std::uint32_t *h_quant, std::size_t n,
                     char *h_out, std::size_t h_out_cap) {
  if (n == 0 || h_quant == nullptr || h_out == nullptr) return 0;
  const std::size_t in_bytes = n * sizeof(std::uint32_t);
  const int level = env_int("MGARDX_ZSTD_LEVEL", 3);
  const std::size_t bound = ZSTD_compressBound(in_bytes);
  if (bound > h_out_cap) return 0;
  const std::size_t got = ZSTD_compress(h_out, h_out_cap, h_quant, in_bytes, level);
  if (ZSTD_isError(got)) {
    std::fprintf(stderr, "[serial.compress] zstd error: %s\n",
                 ZSTD_getErrorName(got));
    return 0;
  }
  return got;
}

bool Decompress(const char *h_in, std::size_t in_len,
                std::size_t n, std::uint32_t *h_quant_out) {
  if (in_len == 0 || h_in == nullptr || n == 0 || h_quant_out == nullptr)
    return false;
  const std::size_t expected = n * sizeof(std::uint32_t);
  const std::size_t got = ZSTD_decompress(h_quant_out, expected, h_in, in_len);
  if (ZSTD_isError(got)) {
    std::fprintf(stderr, "[serial.decompress] zstd error: %s\n",
                 ZSTD_getErrorName(got));
    return false;
  }
  if (got != expected) {
    std::fprintf(stderr, "[serial.decompress] size mismatch: got=%zu exp=%zu\n",
                 got, expected);
    return false;
  }
  return true;
}

} // namespace serial
} // namespace mgardx_lossless
