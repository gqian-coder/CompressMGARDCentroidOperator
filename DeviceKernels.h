/*
 * DeviceKernels.h — C ABI for centroid-operator device kernels.
 *
 * Two backends are available (CMG_CPU always; CMG_HIP only when the project
 * is built with ENABLE_HIP=ON). All entry points take *host* pointers; the
 * HIP backend transfers to device internally, runs the kernel, and copies
 * results back. Higher-level zero-copy pipelines (device-resident u → device
 * quantized residual → MGARD-X HIP Huffman) are exposed through the fused
 * helpers further down.
 *
 * Outputs are bit-identical across backends (centroid sum is per-cell, so
 * accumulation order is fixed; SFC codes use deterministic integer math).
 */

#ifndef CMG_DEVICE_KERNELS_H
#define CMG_DEVICE_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backend selector. */
typedef enum cmg_backend_e
{
    CMG_BACKEND_CPU = 0,
    CMG_BACKEND_HIP = 1
} cmg_backend_t;

/* SFC kind. Matches V5 bitstream sfcMode values. */
typedef enum cmg_sfc_e
{
    CMG_SFC_NONE    = 0,
    CMG_SFC_MORTON  = 1,
    CMG_SFC_MINNODE = 2,
    CMG_SFC_HILBERT = 3
} cmg_sfc_t;

/* Return non-zero if the HIP backend is built in and a device is present. */
int cmg_hip_available(void);

/* ------------------------------------------------------------------ */
/* Centroid split (forward)                                            */
/*   out_avg  : [ncells]                                               */
/*   out_resi : [nnodes]                                               */
/* ------------------------------------------------------------------ */
void cmg_centroid_split_f32(cmg_backend_t be,
    const float *u, const int64_t *conn,
    size_t ncells, size_t npc, size_t nnodes,
    float *out_avg, float *out_resi);

void cmg_centroid_split_f64(cmg_backend_t be,
    const double *u, const int64_t *conn,
    size_t ncells, size_t npc, size_t nnodes,
    double *out_avg, double *out_resi);

/* ------------------------------------------------------------------ */
/* Centroid recombine (inverse)                                        */
/*   inout starts as residual r(n); on exit holds u(n) = bar_u + r.    */
/* ------------------------------------------------------------------ */
void cmg_centroid_recombine_f32(cmg_backend_t be,
    const float *avg, const int64_t *conn,
    size_t ncells, size_t npc, size_t nnodes,
    float *inout);

void cmg_centroid_recombine_f64(cmg_backend_t be,
    const double *avg, const int64_t *conn,
    size_t ncells, size_t npc, size_t nnodes,
    double *inout);

/* ------------------------------------------------------------------ */
/* SFC permutation builder                                             */
/*   On input  : conn[ncells*npc], optional X/Y/Z[nnodes].             */
/*   On output : perm[ncells] s.t. perm[i] = original cell at slot i.  */
/*   Returns the SFC kind actually used (may fall back to MINNODE when */
/*   coords are NULL or the requested coord-based curve is rejected).  */
/* ------------------------------------------------------------------ */
cmg_sfc_t cmg_build_sfc_perm(cmg_backend_t be,
    const int64_t *conn, size_t ncells, size_t npc,
    const double *X, const double *Y, const double *Z, size_t nnodes,
    cmg_sfc_t requested,
    uint32_t *perm_out);

/* Forward gather: dst[i] = src[perm[i]]. */
void cmg_perm_forward_f32(const float  *src, float  *dst, const uint32_t *perm, size_t n);
void cmg_perm_forward_f64(const double *src, double *dst, const uint32_t *perm, size_t n);
/* Inverse scatter: dst[perm[i]] = src[i]. */
void cmg_perm_inverse_f32(const float  *src, float  *dst, const uint32_t *perm, size_t n);
void cmg_perm_inverse_f64(const double *src, double *dst, const uint32_t *perm, size_t n);

/* ------------------------------------------------------------------ */
/* Linear quantization (mid-tread, quantum = 2*tolerance) + zigzag.    */
/*   Output is uint32_t suitable for MGARD-X Huffman+ZSTD.             */
/* ------------------------------------------------------------------ */
void cmg_quantize_zigzag_f32(cmg_backend_t be,
    const float *in, size_t n, double tolerance, uint32_t *out_u32);
void cmg_quantize_zigzag_f64(cmg_backend_t be,
    const double *in, size_t n, double tolerance, uint32_t *out_u32);

void cmg_dequantize_zigzag_f32(cmg_backend_t be,
    const uint32_t *in_u32, size_t n, double tolerance, float *out);
void cmg_dequantize_zigzag_f64(cmg_backend_t be,
    const uint32_t *in_u32, size_t n, double tolerance, double *out);

#ifdef __cplusplus
}
#endif
#endif /* CMG_DEVICE_KERNELS_H */
