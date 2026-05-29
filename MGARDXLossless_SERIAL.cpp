/*
 * MGARDXLossless_SERIAL.cpp — SERIAL (CPU) lossless backend.
 *
 * Compiled as plain C++ (no -x hip). Cray's CC does not define __HIPCC__
 * for CXX TUs, so DataTypes.h falls through to MGARDX_COMPILE_SERIAL and
 * RuntimeX.h only pulls in DeviceAdapterSerial.h.
 *
 * Strategy:
 *   Use MGARD-X ComposedLosslessCompressor<uint32_t,uint64_t,SERIAL>
 *   (Huffman+Zstd) when every quantization code is < dict_size.  Fall back
 *   to raw Zstd when any code >= dict_size — those would cause an out-of-
 *   bounds atomic write in HistogramFunctor and an OOB codebook read in
 *   EncodeFixedLen with no guard in MGARD-X.
 *
 * Bug fix (without modifying MGARD-X):
 *   Huffman::outlier_count is a plain member with no default initializer.
 *   ComposedLosslessCompressor::Compress calls CompressPrimary (the unsigned
 *   path), which never populates the outlier buffer and never sets the host
 *   member.  Serialize then reads the garbage value → giant Malloc1D → crash.
 *   Fix: set clc.huffman.outlier_count = 0 immediately after construction.
 *   This is semantically correct: the unsigned/primary path has no outliers.
 *   Decompression is unaffected — Deserialize reads outlier_count from the
 *   bitstream header, overwriting the member before any outlier data is used.
 *
 * Bitstream layout: 1-byte tag + payload
 *   'H' (0x48)  Huffman+Zstd  (ComposedLosslessCompressor<SERIAL>)
 *   'Z' (0x5A)  raw Zstd      (fallback when max_code >= dict_size)
 * Backward compat: a first byte that is neither tag is assumed to be the
 * legacy raw-Zstd-no-tag format (ZSTD magic starts with 0x28).
 *
 * Knobs (env vars):
 *   MGARDX_HUFF_DICT   Huffman dict size        (default 8192)
 *   MGARDX_HUFF_BLOCK  Huffman chunk/block size  (default 20480)
 *   MGARDX_ZSTD_LEVEL  Zstd compression level   (default 3)
 */

#include "MGARDXLossless.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include <zstd.h>

#include <mgard/mgard-x/RuntimeX/RuntimeX.h>
#include <mgard/mgard-x/Lossless/Lossless.hpp>

namespace mgardx_lossless {
namespace serial {

namespace {

constexpr std::uint8_t kTagHuffman = 'H';   // 0x48
constexpr std::uint8_t kTagZstd    = 'Z';   // 0x5A

std::once_flag g_init;

void init_once() {
  std::call_once(g_init, []() {
    mgard_x::DeviceRuntime<mgard_x::SERIAL>::Initialize();
  });
}

int env_int(const char *name, int def) {
  const char *s = std::getenv(name);
  if (!s || !*s) return def;
  try { return std::stoi(s); } catch (...) { return def; }
}

mgard_x::Config make_config() {
  mgard_x::Config cfg;
  cfg.dev_type               = mgard_x::device_type::SERIAL;
  cfg.lossless               = mgard_x::lossless_type::Huffman_Zstd;
  cfg.huff_dict_size         = static_cast<mgard_x::SIZE>(env_int("MGARDX_HUFF_DICT",  8192));
  cfg.huff_block_size        = static_cast<mgard_x::SIZE>(env_int("MGARDX_HUFF_BLOCK", 1024 * 20));
  cfg.zstd_compress_level    = env_int("MGARDX_ZSTD_LEVEL", 3);
  cfg.estimate_outlier_ratio = 1.0;
  return cfg;
}

} // namespace

std::size_t Compress(const std::uint32_t *h_quant, std::size_t n,
                     char *h_out, std::size_t h_out_cap) {
  if (n == 0 || h_quant == nullptr || h_out == nullptr) return 0;
  if (h_out_cap < 2) return 0;   // need at least tag byte + 1 payload byte

  const auto cfg = make_config();

  /* Decide: scan codes to check whether all fit in [0, dict_size). */
  const std::uint32_t max_code =
      *std::max_element(h_quant, h_quant + n);
  const bool use_huffman =
      (max_code < static_cast<std::uint32_t>(cfg.huff_dict_size));

  char        *payload = h_out + 1;    // byte 0 is reserved for the tag
  std::size_t  pcap    = h_out_cap - 1;

  if (use_huffman) {
    init_once();

    mgard_x::Array<1, std::uint32_t, mgard_x::SERIAL> arr_quant(
        {static_cast<mgard_x::SIZE>(n)});
    arr_quant.load(h_quant);

    mgard_x::Array<1, mgard_x::Byte, mgard_x::SERIAL> arr_compressed;

    bool ok = true;
    try {
      mgard_x::ComposedLosslessCompressor<std::uint32_t, std::uint64_t,
                                          mgard_x::SERIAL>
          clc(static_cast<mgard_x::SIZE>(n), cfg);
      /* Fix: outlier_count is uninitialized in the CompressPrimary path.
       * CompressPrimary never writes to the outlier buffer, so 0 is correct.
       * Serialize reads this member to size the outlier section of the header. */
      clc.huffman.outlier_count = 0;
      clc.Compress(arr_quant, arr_compressed, 0);
    } catch (const std::exception &e) {
      std::fprintf(stderr,
                   "[serial.compress] Huffman+Zstd failed (%s), "
                   "falling back to raw Zstd\n", e.what());
      ok = false;
    }

    if (ok) {
      const std::size_t comp_bytes = arr_compressed.shape(0);
      if (comp_bytes > pcap) return 0;
      std::memcpy(payload, arr_compressed.data(), comp_bytes);
      h_out[0] = static_cast<char>(kTagHuffman);
      return 1 + comp_bytes;
    }
    /* fall through to raw Zstd */
  }

  /* Raw Zstd path: max_code >= dict_size, or Huffman threw. */
  const std::size_t in_bytes = n * sizeof(std::uint32_t);
  const int level = env_int("MGARDX_ZSTD_LEVEL", 3);
  const std::size_t got = ZSTD_compress(payload, pcap, h_quant, in_bytes, level);
  if (ZSTD_isError(got)) {
    std::fprintf(stderr, "[serial.compress] zstd error: %s\n",
                 ZSTD_getErrorName(got));
    return 0;
  }
  h_out[0] = static_cast<char>(kTagZstd);
  return 1 + got;
}

bool Decompress(const char *h_in, std::size_t in_len,
                std::size_t n, std::uint32_t *h_quant_out) {
  if (in_len == 0 || h_in == nullptr || n == 0 || h_quant_out == nullptr)
    return false;

  const std::uint8_t tag     = static_cast<std::uint8_t>(h_in[0]);
  const char        *payload = h_in + 1;
  const std::size_t  plen    = in_len - 1;

  /* Huffman+Zstd path. */
  if (tag == kTagHuffman) {
    init_once();
    const auto cfg = make_config();

    mgard_x::Array<1, mgard_x::Byte, mgard_x::SERIAL> arr_compressed(
        {static_cast<mgard_x::SIZE>(plen)});
    arr_compressed.load(reinterpret_cast<const mgard_x::Byte *>(payload));

    mgard_x::Array<1, std::uint32_t, mgard_x::SERIAL> arr_decompressed(
        {static_cast<mgard_x::SIZE>(n)});

    try {
      mgard_x::ComposedLosslessCompressor<std::uint32_t, std::uint64_t,
                                          mgard_x::SERIAL>
          clc(static_cast<mgard_x::SIZE>(n), cfg);
      /* Deserialize reads outlier_count from the bitstream header, so no
       * manual fix needed here — it is overwritten before any outlier data
       * is accessed. */
      clc.Decompress(arr_compressed, arr_decompressed, 0);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[serial.decompress] Huffman+Zstd failed: %s\n",
                   e.what());
      return false;
    }

    std::memcpy(h_quant_out, arr_decompressed.data(),
                n * sizeof(std::uint32_t));
    return true;
  }

  /* Raw Zstd path (tag == 'Z'), or legacy no-tag format (tag == 0x28,
   * the first byte of a ZSTD frame magic — decompress the full buffer). */
  const char        *zsrc = (tag == kTagZstd) ? payload  : h_in;
  const std::size_t  zlen = (tag == kTagZstd) ? plen     : in_len;

  const std::size_t expected = n * sizeof(std::uint32_t);
  const std::size_t got = ZSTD_decompress(h_quant_out, expected, zsrc, zlen);
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
