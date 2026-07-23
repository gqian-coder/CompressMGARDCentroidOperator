/*
 * MGARDXLossless_HIP.cpp — HIP (GPU) backend.
 *
 * Compiled with LANGUAGE HIP. Quantized input must be a HIP DEVICE
 * pointer; output compressed bytes are host. The bitstream produced
 * here is bit-identical to the SERIAL backend for the same uint32
 * input + Config (Huffman.Serialize and Zstd are device-agnostic).
 *
 * Links against MGARD HIP install (`mgard::mgard`) which provides
 * `mgard_x::HIP` device adapter.
 */

#include "MGARDXLossless.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <mgard/mgard-x/RuntimeX/RuntimeX.h>
#include <mgard/mgard-x/Lossless/Lossless.hpp>

extern "C" {
std::size_t ZSTD_compress(void *dst, std::size_t dstCap, const void *src,
                          std::size_t srcSize, int level);
unsigned    ZSTD_isError(std::size_t code);
const char *ZSTD_getErrorName(std::size_t code);
}

namespace mgardx_lossless {
namespace hip {

constexpr std::uint8_t kTagHuffman = 'H';
constexpr std::uint8_t kTagZstd    = 'Z';

namespace {

std::once_flag g_init;

void init_once() {
  std::call_once(g_init, []() {
    mgard_x::DeviceRuntime<mgard_x::HIP>::Initialize();
  });
}

long env_long(const char *name, long def) {
  const char *s = std::getenv(name);
  if (!s || !*s) return def;
  try { return std::stol(s); } catch (...) { return def; }
}
double env_double(const char *name, double def) {
  const char *s = std::getenv(name);
  if (!s || !*s) return def;
  try { return std::stod(s); } catch (...) { return def; }
}

mgard_x::Config make_config() {
  mgard_x::Config cfg;
  cfg.dev_type               = mgard_x::device_type::HIP;
  const long lt = env_long("MGARDX_LOSSLESS", 2);
  cfg.lossless               = (lt == 0) ? mgard_x::lossless_type::Huffman
                              : (lt == 1) ? mgard_x::lossless_type::Huffman_LZ4
                                          : mgard_x::lossless_type::Huffman_Zstd;
  cfg.huff_dict_size         = static_cast<mgard_x::SIZE>(env_long("MGARDX_HUFF_DICT", 8192));
  cfg.huff_block_size        = static_cast<mgard_x::SIZE>(env_long("MGARDX_HUFF_BLOCK", 1024 * 20));
  cfg.zstd_compress_level    = static_cast<int>(env_long("MGARDX_ZSTD_LEVEL", 3));
  cfg.estimate_outlier_ratio = env_double("MGARDX_OUTLIER_RATIO", 1.0);
  return cfg;
}

} // namespace

std::size_t Compress(const std::uint32_t *d_quant, std::size_t n,
                     char *h_out, std::size_t h_out_cap) {
  if (n == 0 || d_quant == nullptr || h_out == nullptr) return 0;
  init_once();

  const auto cfg = make_config();

  /* Wrap the external device pointer (zero-copy, no free on dtor). */
  mgard_x::Array<1, std::uint32_t, mgard_x::HIP> arr_quant(
      {static_cast<mgard_x::SIZE>(n)},
      const_cast<std::uint32_t *>(d_quant));

  mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> arr_compressed;
  try {
    mgard_x::ComposedLosslessCompressor<std::uint32_t, std::uint64_t,
                                        mgard_x::HIP>
        clc(static_cast<mgard_x::SIZE>(n), cfg);
    clc.Compress(arr_quant, arr_compressed, 0);
    mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
  } catch (const std::exception &e) {
    std::cerr << "[mgardx_lossless::hip] Compress failed: " << e.what()
              << std::endl;
    return 0;
  }

  const std::size_t comp_bytes = arr_compressed.shape(0);
  if (comp_bytes > h_out_cap) return 0;

  /* hostCopy() pulls compressed bytes back to host. */
  const mgard_x::Byte *p = arr_compressed.hostCopy(false, 0);
  mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
  std::memcpy(h_out, p, comp_bytes);
  return comp_bytes;
}

bool Decompress(const char *h_in, std::size_t in_len,
                std::size_t n, std::uint32_t *d_quant_out) {
  if (in_len == 0 || h_in == nullptr || n == 0 || d_quant_out == nullptr)
    return false;
  init_once();

  const auto cfg = make_config();

  /* Load host bytes into a HIP device buffer. */
  mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> arr_compressed(
      {static_cast<mgard_x::SIZE>(in_len)});
  arr_compressed.load(reinterpret_cast<const mgard_x::Byte *>(h_in));

  /* Let MGARD-X allocate its own decompressed buffer (safer than passing
   * an external pointer because DecompressPrimary may reshape). */
  mgard_x::Array<1, std::uint32_t, mgard_x::HIP> arr_decompressed(
      {static_cast<mgard_x::SIZE>(n)});

  try {
    mgard_x::ComposedLosslessCompressor<std::uint32_t, std::uint64_t,
                                        mgard_x::HIP>
        clc(static_cast<mgard_x::SIZE>(n), cfg);
    clc.Decompress(arr_compressed, arr_decompressed, 0);
    mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
  } catch (const std::exception &e) {
    std::cerr << "[mgardx_lossless::hip] Decompress failed: " << e.what()
              << std::endl;
    return false;
  }

  /* Device-to-device copy into caller's external buffer. */
  const std::uint32_t *src = arr_decompressed.data();
  hipError_t err = hipMemcpy(d_quant_out, src, n * sizeof(std::uint32_t),
                             hipMemcpyDeviceToDevice);
  if (err != hipSuccess) {
    std::cerr << "[mgardx_lossless::hip] hipMemcpy D2D failed: "
              << hipGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

std::size_t CompressFromHost(const std::uint32_t *h_quant, std::size_t n,
                             char *h_out, std::size_t h_out_cap) {
  if (n == 0 || h_quant == nullptr || h_out == nullptr) return 0;
  if (h_out_cap < 2) return 0;
  hipDeviceSynchronize();
  (void)hipGetLastError();

  // Match serial::Compress dict (8192) so the tagged stream is serial-decodable.
  const auto cfg = make_config();
  const std::uint32_t max_code = *std::max_element(h_quant, h_quant + n);
  const bool use_huffman = (max_code < static_cast<std::uint32_t>(cfg.huff_dict_size));
  char *payload = h_out + 1; std::size_t pcap = h_out_cap - 1;

  if (use_huffman) {
    init_once();
    mgard_x::Array<1, std::uint32_t, mgard_x::HIP> arr_quant({static_cast<mgard_x::SIZE>(n)});
    arr_quant.load(h_quant);
    mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> arr_compressed;
    bool ok = true;
    try {
      mgard_x::ComposedLosslessCompressor<std::uint32_t, std::uint64_t, mgard_x::HIP>
          clc(static_cast<mgard_x::SIZE>(n), cfg);
      clc.Compress(arr_quant, arr_compressed, 0);
      mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
    } catch (const std::exception &e) {
      std::cerr << "[mgardx_lossless::hip] Huffman failed (" << e.what()
                << "), raw Zstd fallback" << std::endl;
      ok = false;
    }
    if (ok) {
      const std::size_t cb = arr_compressed.shape(0);
      if (cb > pcap) return 0;
      const mgard_x::Byte *pp = arr_compressed.hostCopy(false, 0);
      mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
      std::memcpy(payload, pp, cb);
      h_out[0] = static_cast<char>(kTagHuffman);
      return 1 + cb;
    }
  }
  const std::size_t in_bytes = n * sizeof(std::uint32_t);
  const int level = static_cast<int>(env_long("MGARDX_ZSTD_LEVEL", 3));
  const std::size_t got = ZSTD_compress(payload, pcap, h_quant, in_bytes, level);
  if (ZSTD_isError(got)) { std::cerr << "[mgardx_lossless::hip] zstd err: "
      << ZSTD_getErrorName(got) << std::endl; return 0; }
  h_out[0] = static_cast<char>(kTagZstd);
  return 1 + got;
}

std::size_t CompressFromDevice(const std::uint32_t *d_quant, std::size_t n,
                               std::uint32_t max_code, char *h_out,
                               std::size_t h_out_cap) {
  // Device-resident variant of CompressFromHost: the quantized codes are ALREADY
  // on the GPU (from the fused pipeline), so the Huffman path consumes them
  // in place with NO H2D. Same tagged framing as serial::Compress. max_code is
  // supplied by the caller (device-side reduce) so the dict guard needs no D2H.
  if (n == 0 || d_quant == nullptr || h_out == nullptr) return 0;
  if (h_out_cap < 2) return 0;
  hipDeviceSynchronize();
  (void)hipGetLastError();

  const auto cfg = make_config();
  const bool use_huffman = (max_code < static_cast<std::uint32_t>(cfg.huff_dict_size));
  char *payload = h_out + 1; std::size_t pcap = h_out_cap - 1;

  if (use_huffman) {
    init_once();
    // zero-copy wrap of the external DEVICE pointer (no allocation, no H2D)
    mgard_x::Array<1, std::uint32_t, mgard_x::HIP> arr_quant(
        {static_cast<mgard_x::SIZE>(n)}, const_cast<std::uint32_t *>(d_quant));
    mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> arr_compressed;
    bool ok = true;
    try {
      mgard_x::ComposedLosslessCompressor<std::uint32_t, std::uint64_t, mgard_x::HIP>
          clc(static_cast<mgard_x::SIZE>(n), cfg);
      clc.Compress(arr_quant, arr_compressed, 0);
      mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
    } catch (const std::exception &e) {
      std::cerr << "[mgardx_lossless::hip] Huffman failed (" << e.what()
                << "), raw Zstd fallback" << std::endl;
      ok = false;
    }
    if (ok) {
      const std::size_t cb = arr_compressed.shape(0);
      if (cb > pcap) return 0;
      const mgard_x::Byte *pp = arr_compressed.hostCopy(false, 0);
      mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
      std::memcpy(payload, pp, cb);
      h_out[0] = static_cast<char>(kTagHuffman);
      return 1 + cb;
    }
  }
  // Zstd fallback (rare: max_code >= dict or Huffman overflow) -- needs host
  // data, so D2H the codes here.
  std::vector<std::uint32_t> h(n);
  hipMemcpy(h.data(), d_quant, n * sizeof(std::uint32_t), hipMemcpyDeviceToHost);
  const std::size_t in_bytes = n * sizeof(std::uint32_t);
  const int level = static_cast<int>(env_long("MGARDX_ZSTD_LEVEL", 3));
  const std::size_t got = ZSTD_compress(payload, pcap, h.data(), in_bytes, level);
  if (ZSTD_isError(got)) { std::cerr << "[mgardx_lossless::hip] zstd err: "
      << ZSTD_getErrorName(got) << std::endl; return 0; }
  h_out[0] = static_cast<char>(kTagZstd);
  return 1 + got;
}

} // namespace hip
} // namespace mgardx_lossless
