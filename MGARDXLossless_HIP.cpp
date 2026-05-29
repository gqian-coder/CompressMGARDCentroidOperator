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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include <mgard/mgard-x/RuntimeX/RuntimeX.h>
#include <mgard/mgard-x/Lossless/Lossless.hpp>

namespace mgardx_lossless {
namespace hip {

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

// for_ab=true: first check MGARDX_HUFF_AB_DICT (defaults to 64 — safe for
// residual distributions that trigger degenerate Huffman at larger dict sizes).
mgard_x::Config make_config(bool for_ab = false) {
  mgard_x::Config cfg;
  cfg.dev_type               = mgard_x::device_type::HIP;
  const long lt = env_long("MGARDX_LOSSLESS", 2);
  cfg.lossless               = (lt == 0) ? mgard_x::lossless_type::Huffman
                              : (lt == 1) ? mgard_x::lossless_type::Huffman_LZ4
                                          : mgard_x::lossless_type::Huffman_Zstd;
  // A/B calls use MGARDX_HUFF_AB_DICT (default 64) to avoid degenerate
  // GPU Huffman code lengths (max_CL > 56) for fine-quantization residuals.
  // Regular calls use MGARDX_HUFF_DICT (default 8192).
  if (for_ab)
    cfg.huff_dict_size = static_cast<mgard_x::SIZE>(
        env_long("MGARDX_HUFF_AB_DICT", env_long("MGARDX_HUFF_DICT", 64)));
  else
    cfg.huff_dict_size = static_cast<mgard_x::SIZE>(env_long("MGARDX_HUFF_DICT", 8192));
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
  init_once();

  const auto cfg = make_config(/*for_ab=*/true);

  /* Let MGARD-X own the device-side input buffer (avoids passing an
   * externally-hipMalloc'd pointer that some intermediate ops then try to
   * copy with hipMemcpyDefault — which fails on GPUs with
   * unifiedAddressing == 0, e.g. the Frontier login-node MI210). */
  mgard_x::Array<1, std::uint32_t, mgard_x::HIP> arr_quant(
      {static_cast<mgard_x::SIZE>(n)});
  arr_quant.load(h_quant);

  mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> arr_compressed;
  try {
    mgard_x::ComposedLosslessCompressor<std::uint32_t, std::uint64_t,
                                        mgard_x::HIP>
        clc(static_cast<mgard_x::SIZE>(n), cfg);
    clc.Compress(arr_quant, arr_compressed, 0);
    mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
  } catch (const std::exception &e) {
    std::cerr << "[mgardx_lossless::hip] CompressFromHost failed: "
              << e.what() << std::endl;
    return 0;
  }

  const std::size_t comp_bytes = arr_compressed.shape(0);
  if (comp_bytes > h_out_cap) return 0;

  const mgard_x::Byte *p = arr_compressed.hostCopy(false, 0);
  mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
  std::memcpy(h_out, p, comp_bytes);
  return comp_bytes;
}

} // namespace hip
} // namespace mgardx_lossless
