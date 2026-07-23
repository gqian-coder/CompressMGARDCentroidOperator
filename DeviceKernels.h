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
/* NODAL SFC permutation builder.                                      */
/*                                                                     */
/*   Orders the *nodes* (not the cells) along a space-filling curve    */
/*   built directly from the node coordinates. Used by the             */
/*   reorder+MGARD path to gather the nodal field into spatially       */
/*   local order before handing it to MGARD's 1-D multilevel           */
/*   transform (array-order neighbour correlation ~0.81 -> ~0.99).     */
/*                                                                     */
/*   On input  : X/Y/Z[nnodes] (required; no connectivity needed).     */
/*   On output : perm_out[nnodes] s.t. perm[i] = original node at      */
/*               slot i  (same convention as cmg_build_sfc_perm).      */
/*   Returns the SFC kind actually used; returns CMG_SFC_NONE and      */
/*   leaves perm as identity when coords are missing or `requested`    */
/*   is not a coordinate-based curve.                                  */
/*                                                                     */
/*   Pair with cmg_perm_forward_* (compress) / cmg_perm_inverse_*      */
/*   (decompress) for the gather/scatter.                              */
/* ------------------------------------------------------------------ */
cmg_sfc_t cmg_build_sfc_perm_nodes(cmg_backend_t be,
    const double *X, const double *Y, const double *Z, size_t nnodes,
    cmg_sfc_t requested,
    uint32_t *perm_out);

/* ------------------------------------------------------------------ */
/* Coordinate grouping (group-location decomposition).                 */
/*                                                                     */
/*   This mesh stores each distinct physical location many times over  */
/*   (measured 12.7x-23.3x per block, 15.2x dataset-wide), and         */
/*   co-located nodes carry near-identical values. Grouping nodes by   */
/*   exact (x,y,z) therefore yields a coarse representation ~6.6% the  */
/*   size of the nodal field that predicts it far better than the      */
/*   cell-average does.                                                */
/*                                                                     */
/*   Builds, for each node, the id of its coordinate group:            */
/*     gid_out[n] in [0, ngroups)                                      */
/*   Groups are numbered in ascending lexicographic (x,y,z) order and  */
/*   ties broken by original node index, so the mapping is bit-        */
/*   reproducible between compress and decompress from the same        */
/*   coordinate array (nothing about it is stored in the bitstream).   */
/*                                                                     */
/*   Optionally also emits one representative coordinate per group     */
/*   (gx/gy/gz, may be NULL) for building an SFC over the groups.      */
/*                                                                     */
/*   Returns the number of distinct groups.                            */
/* ------------------------------------------------------------------ */
/* Group nodes by coordinate.
 *
 * tol > 0 selects TOLERANT grouping: coordinates are snapped to a lattice of
 * step tol * (per-axis bbox extent) before comparison, so nodes that describe
 * the same physical point but differ by round-off are merged. tol <= 0 keeps
 * exact bitwise grouping.
 *
 * Snapping can in principle split a cluster that straddles a lattice boundary,
 * but that is harmless: the grouping is rebuilt identically on read, so a split
 * costs a handful of extra coarse values and never breaks reconstruction. The
 * usable tol range is wide -- on the p3 mesh the distinct count is flat at
 * 23.2% from 1e-9 to 1e-5, and on p1 tol up to 1e-6 reproduces the exact count
 * to the node -- so 1e-7 sits ~2 decades inside the plateau on both. */
/* Host-side bbox reduction feeding the tolerant-grouping lattice. Exposed so
 * the HIP translation unit uses the SAME bounds as the CPU one. */
void cmg_coord_quant_scale(const double *X, const double *Y, const double *Z,
                           size_t nnodes, double tol, double *omin, double *oinvh);

size_t cmg_build_coord_groups(cmg_backend_t be,
    const double *X, const double *Y, const double *Z, size_t nnodes,
    double tol, uint32_t *gid_out,
    double *gx, double *gy, double *gz);

/* Fused open-loop group_centroid per-field compress (device-resident on HIP).
 * Runs group mean -> coarse SFC -> open-loop residual -> stats -> residual SFC
 * -> quantize all on the GPU, uploading the field once and returning only the
 * small coarse stream, the quantized codes and the stats. Geometry (gid,gperm,
 * nperm) is device-cached per blockId. Returns 1 if it ran on the GPU, 0 if the
 * caller must fall back to the staged CPU path (non-HIP backend). gperm/nperm
 * may be null to skip the corresponding SFC reorder. quantC is written only
 * when *drop == 0 (adaptive rule kept the residual).
 *
 * Device-resident mode: pass non-null d_coarse_out / d_q_out to keep the coarse
 * stream and quantized codes on the GPU (returned as device pointers, freed by
 * the caller via cmg_hip_free) so MGARD-HIP and hip::CompressFromDevice consume
 * them with no D2H/H2D; maxcode receives the device-side max code for the
 * lossless dict guard. Pass all three null for the legacy host-copy behaviour. */
int cmg_group_compress_f32(cmg_backend_t be, size_t blockId, const float *field,
    const uint32_t *gid, const uint32_t *gperm, const uint32_t *nperm,
    size_t nnodes, size_t ngroups, double tol_resi, int adaptive,
    float *coarseSfc, double *rmin, double *rmax, double *rsumsq,
    int *drop, uint32_t *quantC,
    void **d_coarse_out, void **d_q_out, uint32_t *maxcode);
int cmg_group_compress_f64(cmg_backend_t be, size_t blockId, const double *field,
    const uint32_t *gid, const uint32_t *gperm, const uint32_t *nperm,
    size_t nnodes, size_t ngroups, double tol_resi, int adaptive,
    double *coarseSfc, double *rmin, double *rmax, double *rsumsq,
    int *drop, uint32_t *quantC,
    void **d_coarse_out, void **d_q_out, uint32_t *maxcode);

/* Device-resident decompress recombine: undo group SFC on the coarse stream,
 * dequantize the residual, undo the nodal SFC, and add the group-broadcast mean
 * -- all on device (one H2D of inputs, one D2H of the field). gperm==NULL: group
 * SFC was NONE; quant==NULL: residual was dropped; nperm==NULL: residual SFC was
 * NONE. Returns 1 when HIP handled it, 0 on the CPU backend (use the host tail). */
int cmg_group_recombine_f32(cmg_backend_t be, const float *coarseSfc, const uint32_t *gperm,
    const uint32_t *quant, const uint32_t *gid, const uint32_t *nperm,
    size_t nnodes, size_t ngroups, double tolR, float *out);
int cmg_group_recombine_f64(cmg_backend_t be, const double *coarseSfc, const uint32_t *gperm,
    const uint32_t *quant, const uint32_t *gid, const uint32_t *nperm,
    size_t nnodes, size_t ngroups, double tolR, double *out);

/* Free a device pointer returned by cmg_group_compress_* (device-resident mode).
 * No-op on non-HIP builds / null. */
void cmg_hip_free(void *p);

/* Device->host copy (bytes). Used to bring MGARD-X's device-resident compressed
 * output back to the host bitstream buffer. No-op on non-HIP builds. */
void cmg_hip_d2h(void *h_dst, const void *d_src, size_t bytes);

/* Group reduction / broadcast for the coarse stream.
 *   mean : out_mean[g] = average of f over the nodes of group g
 *   bcast: out[n]      = mean[gid[n]]                                  */
void cmg_group_mean_f32(cmg_backend_t be, const float  *f, const uint32_t *gid,
                        size_t nnodes, size_t ngroups, float  *out_mean);
void cmg_group_mean_f64(cmg_backend_t be, const double *f, const uint32_t *gid,
                        size_t nnodes, size_t ngroups, double *out_mean);
void cmg_group_bcast_sub_f32(const float  *mean, const uint32_t *gid, size_t nnodes,
                             const float  *f, float  *out_resid);
void cmg_group_bcast_sub_f64(const double *mean, const uint32_t *gid, size_t nnodes,
                             const double *f, double *out_resid);
/* inout[n] += mean[gid[n]]  (decompress recombine) */
void cmg_group_bcast_add_f32(const float  *mean, const uint32_t *gid, size_t nnodes, float  *inout);
void cmg_group_bcast_add_f64(const double *mean, const uint32_t *gid, size_t nnodes, double *inout);

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

/* ------------------------------------------------------------------ */
/* Block reduction for relative-tolerance normalization.               */
/*   Computes the minimum, maximum, and sum-of-squares of the input    */
/*   block in a single pass. The L2 norm is sqrt(out_sumsq); the value */
/*   range is (out_max - out_min). Outputs are double for precision.   */
/*   Runs on the GPU when be == CMG_BACKEND_HIP, else on the CPU.       */
/* ------------------------------------------------------------------ */
void cmg_reduce_stats_f32(cmg_backend_t be,
    const float *in, size_t n, double *out_min, double *out_max, double *out_sumsq);
void cmg_reduce_stats_f64(cmg_backend_t be,
    const double *in, size_t n, double *out_min, double *out_max, double *out_sumsq);

#ifdef __cplusplus
}
#endif
#endif /* CMG_DEVICE_KERNELS_H */
