/*
 * DeviceKernels_HIP.cpp — HIP kernels for centroid operator.
 *
 * Compiled with LANGUAGE HIP. Exports cmg_hip_* symbols that mirror the
 * cmg_* CPU ABI; the dispatcher in DeviceKernels_CPU.cpp forwards to them
 * whenever the caller passes CMG_BACKEND_HIP.
 *
 * All public entry points accept host pointers and H2D/D2H internally. This
 * keeps the operator code device-agnostic at the cost of two extra copies
 * on the GPU path; a fully device-resident pipeline can be layered on top
 * later by exposing variants that take device pointers.
 */

#include "DeviceKernels.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#define HIP_OK(call)                                                                       \
    do {                                                                                   \
        hipError_t _e = (call);                                                            \
        if (_e != hipSuccess) {                                                            \
            std::fprintf(stderr, "[cmg-hip] %s at %s:%d -> %s\n", #call, __FILE__,         \
                         __LINE__, hipGetErrorString(_e));                                 \
        }                                                                                  \
    } while (0)

namespace {

constexpr int BLOCK_X = 256;

inline size_t grid(size_t n) { return (n + BLOCK_X - 1) / BLOCK_X; }

/* ---- device helpers (Morton, Hilbert, zigzag quantize) ------------- */

__device__ __forceinline__ uint64_t dev_morton_expand3(uint32_t v)
{
    uint64_t x = v & 0x1fffffull;
    x = (x | (x << 32)) & 0x001f00000000ffffull;
    x = (x | (x << 16)) & 0x001f0000ff0000ffull;
    x = (x | (x << 8))  & 0x100f00f00f00f00full;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2))  & 0x1249249249249249ull;
    return x;
}
__device__ __forceinline__ uint64_t dev_morton3d(uint32_t x, uint32_t y, uint32_t z)
{ return dev_morton_expand3(x) | (dev_morton_expand3(y) << 1) | (dev_morton_expand3(z) << 2); }

__device__ __forceinline__ uint64_t dev_hilbert3d(uint32_t x, uint32_t y, uint32_t z)
{
    constexpr int b = 21;
    uint32_t X[3] = { x & 0x1fffffu, y & 0x1fffffu, z & 0x1fffffu };
    const uint32_t M = 1u << (b - 1);
    uint32_t P, Q, t;
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (X[i] & Q) { X[0] ^= P; }
            else { t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; }
        }
    }
    #pragma unroll
    for (int i = 1; i < 3; ++i) X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1) if (X[2] & Q) t ^= Q - 1;
    #pragma unroll
    for (int i = 0; i < 3; ++i) X[i] ^= t;
    uint64_t h = 0;
    for (int bit = b - 1; bit >= 0; --bit) {
        h = (h << 1) | ((X[0] >> bit) & 1u);
        h = (h << 1) | ((X[1] >> bit) & 1u);
        h = (h << 1) | ((X[2] >> bit) & 1u);
    }
    return h;
}

/* ---- SFC kernels --------------------------------------------------- */

__global__ void k_cell_centroid(const int64_t *conn, size_t ncells, size_t npc,
                                const double *X, const double *Y, const double *Z,
                                double *cx, double *cy, double *cz)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    const int64_t *cc = conn + c * npc;
    double ax = 0.0, ay = 0.0, az = 0.0; int u = 0;
    for (size_t k = 0; k < npc; ++k) {
        bool dup = false;
        for (size_t j = 0; j < k; ++j) if (cc[j] == cc[k]) { dup = true; break; }
        if (!dup) { const int64_t n = cc[k]; ax += X[n]; ay += Y[n]; az += Z[n]; ++u; }
    }
    const double inv = u ? 1.0 / (double)u : 0.0;
    cx[c] = ax * inv; cy[c] = ay * inv; cz[c] = az * inv;
}

__global__ void k_sfc_codes(const double *cx, const double *cy, const double *cz,
                            double xmin, double ymin, double zmin,
                            double sx, double sy, double sz,
                            uint32_t maxQ, int useHilbert, size_t ncells,
                            uint64_t *codes, uint32_t *idx)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    double qxd = (cx[c] - xmin) * sx;
    double qyd = (cy[c] - ymin) * sy;
    double qzd = (cz[c] - zmin) * sz;
    if (qxd < 0.0) qxd = 0.0; if (qxd > (double)maxQ) qxd = maxQ;
    if (qyd < 0.0) qyd = 0.0; if (qyd > (double)maxQ) qyd = maxQ;
    if (qzd < 0.0) qzd = 0.0; if (qzd > (double)maxQ) qzd = maxQ;
    const uint32_t qx = (uint32_t)qxd, qy = (uint32_t)qyd, qz = (uint32_t)qzd;
    codes[c] = useHilbert ? dev_hilbert3d(qx, qy, qz) : dev_morton3d(qx, qy, qz);
    idx[c]   = (uint32_t)c;
}


/* ---- GPU sort helpers (rocPRIM) ------------------------------------ */

/* Order-preserving map double -> uint64 so a radix sort reproduces numeric
 * order: flip the sign bit for positives, invert all bits for negatives. */
__device__ __forceinline__ uint64_t dev_ordkey(double d)
{
    uint64_t u = __double_as_longlong(d);
    return (u & 0x8000000000000000ull) ? ~u : (u | 0x8000000000000000ull);
}

/* Per-axis comparison key. With quantize != 0 the coordinate is snapped to a
 * lattice of step 1/invh so round-off-separated duplicates collide; otherwise
 * the raw bits are used and grouping stays exact. Either way the result is a
 * uint64 whose numeric order matches the double's, so one radix pass sorts it. */
__global__ void k_make_axis_key(const double *coord, double cmin, double invh,
                                int quantize, uint64_t *q, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = quantize ? dev_ordkey(floor((coord[i] - cmin) * invh + 0.5))
                    : dev_ordkey(coord[i]);
}

/* key[i] = q[idx[i]] -- gathers the next sort key through the permutation
 * produced by the previous (stable) pass. */
__global__ void k_gather_u64(const uint64_t *q, const uint32_t *idx,
                             uint64_t *key, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    key[i] = q[idx[i]];
}

__global__ void k_iota_u32(uint32_t *a, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (uint32_t)i;
}

/* flag[i] = 1 when sorted element i starts a new group, i.e. its quantized
 * (x,y,z) key differs from that of the element before it. */
__global__ void k_mark_group_start(const uint64_t *qx, const uint64_t *qy,
                                   const uint64_t *qz, const uint32_t *idx,
                                   uint32_t *flag, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0) { flag[0] = 1u; return; }
    const uint32_t a = idx[i], b = idx[i - 1];
    flag[i] = (qx[a] != qx[b] || qy[a] != qy[b] || qz[a] != qz[b]) ? 1u : 0u;
}

/* scan holds the 1-based group number of each sorted element. Scatter it back
 * to original node order, and emit one representative coordinate per group. */
__global__ void k_scatter_gid(const uint32_t *idx, const uint32_t *scan,
                              const uint32_t *flag, const double *X, const double *Y,
                              const double *Z, uint32_t *gid, double *gx, double *gy,
                              double *gz, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t node = idx[i];
    const uint32_t g = scan[i] - 1u;
    gid[node] = g;
    if (flag[i] && gx) { gx[g] = X[node]; gy[g] = Y[node]; gz[g] = Z[node]; }
}

namespace {
/* stable radix sort of (uint64 key, uint32 value) pairs, in place via buffers */
inline void sort_pairs_u64(uint64_t *&k_in, uint64_t *&k_out,
                           uint32_t *&v_in, uint32_t *&v_out, size_t n)
{
    void  *tmp = nullptr;
    size_t bytes = 0;
    rocprim::radix_sort_pairs(tmp, bytes, k_in, k_out, v_in, v_out, n);
    HIP_OK(hipMalloc(&tmp, bytes));
    rocprim::radix_sort_pairs(tmp, bytes, k_in, k_out, v_in, v_out, n);
    HIP_OK(hipDeviceSynchronize());
    hipFree(tmp);
    std::swap(k_in, k_out);
    std::swap(v_in, v_out);
}
} // namespace

/* ---- quantize / dequantize kernels --------------------------------- */

template <typename T>
__global__ void k_quantize_zigzag(const T *r, uint32_t *out, size_t n, double inv_q)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = (double)r[i] * inv_q;
    if (v >  2147483647.0) v =  2147483647.0;
    if (v < -2147483647.0) v = -2147483647.0;
    int32_t sq = (int32_t)llround(v);
    out[i] = (sq >= 0) ? (uint32_t)(sq * 2) : (uint32_t)((-sq - 1) * 2 + 1);
}

template <typename T>
__global__ void k_dequantize_zigzag(const uint32_t *in, T *out, size_t n, double q)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t z = in[i];
    const int32_t s = (z & 1u) ? -(int32_t)(z >> 1) - 1 : (int32_t)(z >> 1);
    out[i] = (T)((double)s * q);
}

/* ---- group mean / broadcast / gather (device-resident pipeline) ---- */

/* acc[gid[i]] += f[i]; cnt[gid[i]]++  -- atomic segmented sum. The order of
 * atomic adds is nondeterministic, so the group mean can differ from the CPU
 * (sequential) mean in the last ULP. That is harmless here: the coarse stream
 * is lossy-MGARD'd and the residual is formed against THIS mean, so any coarse
 * value round-trips to itself +/- the MGARD error regardless of how it was
 * summed. (double atomicAdd is native on gfx90a.) */
template <typename T>
__global__ void k_group_accum(const T *f, const uint32_t *gid, size_t n,
                              double *acc, uint32_t *cnt)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t g = gid[i];
    atomicAdd(&acc[g], (double)f[i]);
    atomicAdd(&cnt[g], 1u);
}

template <typename T>
__global__ void k_group_finalize(const double *acc, const uint32_t *cnt,
                                 size_t ngroups, T *mean)
{
    size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    mean[g] = (T)(cnt[g] ? acc[g] / (double)cnt[g] : 0.0);
}

/* resid[i] = f[i] - mean[gid[i]]   (open-loop group broadcast-subtract) */
template <typename T>
__global__ void k_bcast_sub(const T *mean, const uint32_t *gid, size_t n,
                           const T *f, T *resid)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    resid[i] = f[i] - mean[gid[i]];
}

/* d[i] = s[p[i]]  -- forward gather through a permutation (SFC reorder) */
template <typename T>
__global__ void k_gather_perm(const T *s, const uint32_t *p, T *d, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = s[p[i]];
}

/* d[p[i]] = s[i]  -- inverse scatter through a permutation (undo SFC reorder) */
template <typename T>
__global__ void k_scatter_perm(const T *s, const uint32_t *p, T *d, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[p[i]] = s[i];
}

/* out[i] = mean[gid[i]] + (f ? f[i] : 0)  -- group broadcast-add (recombine) */
template <typename T>
__global__ void k_group_bcast_add(const T *mean, const uint32_t *gid, const T *f,
                                  T *out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = mean[gid[i]] + (f ? f[i] : (T)0);
}

template <typename T>
void quantize_hip(const T *h_in, size_t n, double tol, uint32_t *h_out)
{
    T        *d_in  = nullptr;
    uint32_t *d_out = nullptr;
    HIP_OK(hipMalloc(&d_in,  n * sizeof(T)));
    HIP_OK(hipMalloc(&d_out, n * sizeof(uint32_t)));
    HIP_OK(hipMemcpy(d_in, h_in, n * sizeof(T), hipMemcpyHostToDevice));
    const double inv_q = 1.0 / (2.0 * tol);
    hipLaunchKernelGGL(k_quantize_zigzag<T>, dim3(grid(n)), dim3(BLOCK_X), 0, 0,
                       d_in, d_out, n, inv_q);
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(h_out, d_out, n * sizeof(uint32_t), hipMemcpyDeviceToHost));
    hipFree(d_in); hipFree(d_out);
}

template <typename T>
void dequantize_hip(const uint32_t *h_in, size_t n, double tol, T *h_out)
{
    uint32_t *d_in  = nullptr;
    T        *d_out = nullptr;
    HIP_OK(hipMalloc(&d_in,  n * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_out, n * sizeof(T)));
    HIP_OK(hipMemcpy(d_in, h_in, n * sizeof(uint32_t), hipMemcpyHostToDevice));
    const double q = 2.0 * tol;
    hipLaunchKernelGGL(k_dequantize_zigzag<T>, dim3(grid(n)), dim3(BLOCK_X), 0, 0,
                       d_in, d_out, n, q);
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(h_out, d_out, n * sizeof(T), hipMemcpyDeviceToHost));
    hipFree(d_in); hipFree(d_out);
}

/* ---- reduction: min / max / sum-of-squares ------------------------- */

template <typename T>
__global__ void k_reduce_stats(const T *in, size_t n,
                               double *blk_min, double *blk_max, double *blk_sumsq)
{
    __shared__ double s_min[BLOCK_X];
    __shared__ double s_max[BLOCK_X];
    __shared__ double s_sq[BLOCK_X];
    const unsigned tid = threadIdx.x;
    const size_t i0 = (size_t)blockIdx.x * blockDim.x + tid;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    /* Identity elements so threads with no work do not perturb the reduction. */
    double vmin = 1.0e300, vmax = -1.0e300, ssq = 0.0;
    for (size_t k = i0; k < n; k += stride) {
        const double v = (double)in[k];
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
        ssq += v * v;
    }
    s_min[tid] = vmin; s_max[tid] = vmax; s_sq[tid] = ssq;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = s_min[tid] < s_min[tid + s] ? s_min[tid] : s_min[tid + s];
            s_max[tid] = s_max[tid] > s_max[tid + s] ? s_max[tid] : s_max[tid + s];
            s_sq[tid] += s_sq[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_min[blockIdx.x]   = s_min[0];
        blk_max[blockIdx.x]   = s_max[0];
        blk_sumsq[blockIdx.x] = s_sq[0];
    }
}

template <typename T>
void reduce_stats_hip(const T *h_in, size_t n,
                      double *out_min, double *out_max, double *out_sumsq)
{
    if (n == 0) { *out_min = 0.0; *out_max = 0.0; *out_sumsq = 0.0; return; }
    size_t nblk = grid(n);
    if (nblk > 4096) nblk = 4096; /* grid-stride loop covers the remainder */

    T *d_in = nullptr;
    double *d_min = nullptr, *d_max = nullptr, *d_sq = nullptr;
    HIP_OK(hipMalloc(&d_in, n * sizeof(T)));
    HIP_OK(hipMalloc(&d_min, nblk * sizeof(double)));
    HIP_OK(hipMalloc(&d_max, nblk * sizeof(double)));
    HIP_OK(hipMalloc(&d_sq,  nblk * sizeof(double)));
    HIP_OK(hipMemcpy(d_in, h_in, n * sizeof(T), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(k_reduce_stats<T>, dim3(nblk), dim3(BLOCK_X), 0, 0,
                       d_in, n, d_min, d_max, d_sq);
    HIP_OK(hipDeviceSynchronize());

    std::vector<double> h_min(nblk), h_max(nblk), h_sq(nblk);
    HIP_OK(hipMemcpy(h_min.data(), d_min, nblk * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(h_max.data(), d_max, nblk * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(h_sq.data(),  d_sq,  nblk * sizeof(double), hipMemcpyDeviceToHost));

    double vmin = h_min[0], vmax = h_max[0], ssq = 0.0;
    for (size_t i = 0; i < nblk; ++i) {
        vmin = std::min(vmin, h_min[i]);
        vmax = std::max(vmax, h_max[i]);
        ssq += h_sq[i];
    }
    *out_min = vmin; *out_max = vmax; *out_sumsq = ssq;

    hipFree(d_in); hipFree(d_min); hipFree(d_max); hipFree(d_sq);
}

/* min/max/sum-of-squares over a DEVICE array (no H2D; used by the fused
 * pipeline where the residual is already on the GPU). */
template <typename T>
void reduce_stats_dev(const T *d_in, size_t n,
                      double *out_min, double *out_max, double *out_sumsq)
{
    if (n == 0) { *out_min = 0.0; *out_max = 0.0; *out_sumsq = 0.0; return; }
    size_t nblk = grid(n);
    if (nblk > 4096) nblk = 4096;
    double *d_min = nullptr, *d_max = nullptr, *d_sq = nullptr;
    HIP_OK(hipMalloc(&d_min, nblk * sizeof(double)));
    HIP_OK(hipMalloc(&d_max, nblk * sizeof(double)));
    HIP_OK(hipMalloc(&d_sq,  nblk * sizeof(double)));
    hipLaunchKernelGGL(k_reduce_stats<T>, dim3(nblk), dim3(BLOCK_X), 0, 0,
                       d_in, n, d_min, d_max, d_sq);
    HIP_OK(hipDeviceSynchronize());
    std::vector<double> h_min(nblk), h_max(nblk), h_sq(nblk);
    HIP_OK(hipMemcpy(h_min.data(), d_min, nblk * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(h_max.data(), d_max, nblk * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(h_sq.data(),  d_sq,  nblk * sizeof(double), hipMemcpyDeviceToHost));
    double vmin = h_min[0], vmax = h_max[0], ssq = 0.0;
    for (size_t i = 0; i < nblk; ++i) {
        vmin = std::min(vmin, h_min[i]); vmax = std::max(vmax, h_max[i]); ssq += h_sq[i];
    }
    *out_min = vmin; *out_max = vmax; *out_sumsq = ssq;
    hipFree(d_min); hipFree(d_max); hipFree(d_sq);
}

/* ---- per-block geometry, cached on the DEVICE ---------------------- */
/* gid / gperm / nperm are geometry-only (depend on the mesh, not the field), so
 * they are uploaded once per block and reused across all variables/timesteps of
 * that block -- the core of the device-resident pipeline. Keyed by blockId, held
 * for the process lifetime (freed implicitly at exit). */
struct GeomDev {
    uint32_t *gid   = nullptr;   /* nnodes  */
    uint32_t *gperm = nullptr;   /* ngroups, null if group SFC disabled  */
    uint32_t *nperm = nullptr;   /* nnodes,  null if residual SFC disabled */
    size_t    nnodes = 0, ngroups = 0;
};
std::mutex g_geomMutex;
std::unordered_map<size_t, GeomDev> g_geomCache;

const GeomDev &get_geom_dev(size_t blockId, const uint32_t *gid, const uint32_t *gperm,
                            const uint32_t *nperm, size_t nnodes, size_t ngroups)
{
    std::lock_guard<std::mutex> lk(g_geomMutex);
    auto it = g_geomCache.find(blockId);
    if (it != g_geomCache.end() &&
        it->second.nnodes == nnodes && it->second.ngroups == ngroups)
    {
        GeomDev &G = it->second;
        /* nperm may arrive on a later variable than the one that first built the
         * geometry (residual dropped on the first), so upload it lazily. */
        if (nperm && !G.nperm) {
            HIP_OK(hipMalloc(&G.nperm, nnodes * sizeof(uint32_t)));
            HIP_OK(hipMemcpy(G.nperm, nperm, nnodes * sizeof(uint32_t), hipMemcpyHostToDevice));
        }
        return G;
    }
    GeomDev G; G.nnodes = nnodes; G.ngroups = ngroups;
    HIP_OK(hipMalloc(&G.gid, nnodes * sizeof(uint32_t)));
    HIP_OK(hipMemcpy(G.gid, gid, nnodes * sizeof(uint32_t), hipMemcpyHostToDevice));
    if (gperm) {
        HIP_OK(hipMalloc(&G.gperm, ngroups * sizeof(uint32_t)));
        HIP_OK(hipMemcpy(G.gperm, gperm, ngroups * sizeof(uint32_t), hipMemcpyHostToDevice));
    }
    if (nperm) {
        HIP_OK(hipMalloc(&G.nperm, nnodes * sizeof(uint32_t)));
        HIP_OK(hipMemcpy(G.nperm, nperm, nnodes * sizeof(uint32_t), hipMemcpyHostToDevice));
    }
    return g_geomCache.emplace(blockId, G).first->second;
}

/* ---- fused open-loop group_centroid per-field pipeline ------------- */
/* One H2D of the field; group mean, coarse SFC, open-loop residual, stats,
 * residual SFC and quantize all run device-resident. Only the small coarse
 * stream (-> CPU MGARD), the quantized codes (-> CPU Huffman) and the stats
 * come back. Geometry is device-cached via get_geom_dev. */
/* If d_coarse_out / d_q_out are non-null the corresponding buffer is kept on
 * the DEVICE and returned (the caller frees it via cmg_hip_free) -- the coarse
 * goes straight to MGARD-HIP and the codes straight to hip::CompressFromDevice
 * with NO D2H/H2D. In that mode h_coarseSfc / h_quantC may be null and
 * out_maxcode (device reduce of the codes) is filled for the lossless guard.
 * Otherwise the legacy host-copy behaviour is used. */
template <typename T>
void group_compress_hip(size_t blockId, const T *h_field, const uint32_t *h_gid,
                        const uint32_t *h_gperm, const uint32_t *h_nperm,
                        size_t nnodes, size_t ngroups, double tol_resi, int adaptive,
                        T *h_coarseSfc, double *out_rmin, double *out_rmax,
                        double *out_rsumsq, int *out_drop, uint32_t *h_quantC,
                        void **d_coarse_out, void **d_q_out, uint32_t *out_maxcode)
{
    const GeomDev &G = get_geom_dev(blockId, h_gid, h_gperm, h_nperm, nnodes, ngroups);
    const bool devResident = (d_coarse_out != nullptr) || (d_q_out != nullptr);

    T *d_field = nullptr, *d_coarse = nullptr, *d_resid = nullptr;
    double *d_acc = nullptr; uint32_t *d_cnt = nullptr;
    HIP_OK(hipMalloc(&d_field,  nnodes  * sizeof(T)));
    HIP_OK(hipMalloc(&d_coarse, ngroups * sizeof(T)));
    HIP_OK(hipMalloc(&d_resid,  nnodes  * sizeof(T)));
    HIP_OK(hipMalloc(&d_acc,    ngroups * sizeof(double)));
    HIP_OK(hipMalloc(&d_cnt,    ngroups * sizeof(uint32_t)));
    HIP_OK(hipMemcpy(d_field, h_field, nnodes * sizeof(T), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_acc, 0, ngroups * sizeof(double)));
    HIP_OK(hipMemset(d_cnt, 0, ngroups * sizeof(uint32_t)));

    /* coarse[g] = mean of field over group g */
    hipLaunchKernelGGL(k_group_accum<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       d_field, G.gid, nnodes, d_acc, d_cnt);
    hipLaunchKernelGGL(k_group_finalize<T>, dim3(grid(ngroups)), dim3(BLOCK_X), 0, 0,
                       d_acc, d_cnt, ngroups, d_coarse);

    /* SFC-order the coarse stream; keep on device (device-resident) or D2H. */
    T *d_coarse_keep = nullptr;   /* separate from d_coarse in both branches */
    if (G.gperm) {
        T *d_csfc = nullptr;
        HIP_OK(hipMalloc(&d_csfc, ngroups * sizeof(T)));
        hipLaunchKernelGGL(k_gather_perm<T>, dim3(grid(ngroups)), dim3(BLOCK_X), 0, 0,
                           d_coarse, G.gperm, d_csfc, ngroups);
        HIP_OK(hipDeviceSynchronize());
        if (d_coarse_out) d_coarse_keep = d_csfc;   /* hand out, don't free */
        else { HIP_OK(hipMemcpy(h_coarseSfc, d_csfc, ngroups * sizeof(T),
                                hipMemcpyDeviceToHost)); hipFree(d_csfc); }
    } else {
        HIP_OK(hipDeviceSynchronize());
        if (d_coarse_out) {   /* no SFC: dedicated copy so d_coarse can be freed */
            HIP_OK(hipMalloc(&d_coarse_keep, ngroups * sizeof(T)));
            HIP_OK(hipMemcpy(d_coarse_keep, d_coarse, ngroups * sizeof(T),
                             hipMemcpyDeviceToDevice));
        } else {
            HIP_OK(hipMemcpy(h_coarseSfc, d_coarse, ngroups * sizeof(T),
                             hipMemcpyDeviceToHost));
        }
    }
    if (d_coarse_out) *d_coarse_out = d_coarse_keep;

    /* open-loop residual against the same (natural-order) coarse */
    hipLaunchKernelGGL(k_bcast_sub<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       d_coarse, G.gid, nnodes, d_field, d_resid);

    /* stats on the natural-order residual (Linf/rms are permutation-invariant) */
    double rmin = 0.0, rmax = 0.0, rsumsq = 0.0;
    reduce_stats_dev<T>(d_resid, nnodes, &rmin, &rmax, &rsumsq);
    *out_rmin = rmin; *out_rmax = rmax; *out_rsumsq = rsumsq;
    const double residLinf = std::max(std::fabs(rmin), std::fabs(rmax));
    const int drop = (adaptive && residLinf <= tol_resi) ? 1 : 0;
    *out_drop = drop;
    if (d_q_out) *d_q_out = nullptr;
    if (out_maxcode) *out_maxcode = 0;

    if (!drop) {
        /* residual SFC gather (if enabled) then zigzag-quantize, device-resident */
        const T *src = d_resid; T *d_tmp = nullptr;
        if (G.nperm) {
            HIP_OK(hipMalloc(&d_tmp, nnodes * sizeof(T)));
            hipLaunchKernelGGL(k_gather_perm<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                               d_resid, G.nperm, d_tmp, nnodes);
            src = d_tmp;
        }
        uint32_t *d_q = nullptr;
        HIP_OK(hipMalloc(&d_q, nnodes * sizeof(uint32_t)));
        const double inv_q = 1.0 / (2.0 * tol_resi);
        hipLaunchKernelGGL(k_quantize_zigzag<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                           src, d_q, nnodes, inv_q);
        HIP_OK(hipDeviceSynchronize());
        /* max code (device reduce) for the lossless dict guard */
        if (out_maxcode) {
            double qmn = 0.0, qmx = 0.0, qss = 0.0;
            reduce_stats_dev<uint32_t>(d_q, nnodes, &qmn, &qmx, &qss);
            *out_maxcode = (uint32_t)qmx;
        }
        if (d_q_out) { *d_q_out = d_q; }   /* hand out, don't free */
        else { HIP_OK(hipMemcpy(h_quantC, d_q, nnodes * sizeof(uint32_t),
                                hipMemcpyDeviceToHost)); hipFree(d_q); }
        if (d_tmp) hipFree(d_tmp);
    }

    hipFree(d_field); hipFree(d_coarse); hipFree(d_resid);
    hipFree(d_acc); hipFree(d_cnt);
    (void)devResident;
}

/* Device-resident decompress recombine (mirror of group_compress_hip). Does the
 * whole decode tail on the GPU with ONE H2D of the inputs and ONE D2H of the
 * field, instead of round-tripping the coarse + residual through the host for
 * the two perm-inverses and the broadcast-add:
 *   coarse[gperm[i]] = coarseSfc[i]          (undo group SFC, ngroups)
 *   resid            = dequant(quant) * 2*tol (nnodes)
 *   field[nperm[i]]  = resid[i]              (undo nodal SFC, nnodes)
 *   out[n]           = coarse[gid[n]] + field[n]
 * h_quant == nullptr => residual was dropped (out = coarse[gid]);
 * h_gperm == nullptr => group SFC was NONE; h_nperm == nullptr => residual SFC
 * was NONE. */
template <typename T>
void group_recombine_hip(const T *h_coarseSfc, const uint32_t *h_gperm,
                         const uint32_t *h_quant, const uint32_t *h_gid,
                         const uint32_t *h_nperm, size_t nnodes, size_t ngroups,
                         double tolR, T *h_out)
{
    T *d_coarse = nullptr, *d_out = nullptr;
    uint32_t *d_gid = nullptr;
    HIP_OK(hipMalloc(&d_coarse, ngroups * sizeof(T)));
    HIP_OK(hipMalloc(&d_gid,    nnodes  * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_out,    nnodes  * sizeof(T)));
    HIP_OK(hipMemcpy(d_gid, h_gid, nnodes * sizeof(uint32_t), hipMemcpyHostToDevice));

    /* undo the group SFC on the coarse stream (ngroups) */
    uint32_t *d_gperm = nullptr;
    if (h_gperm) {
        T *d_coarseSfc = nullptr;
        HIP_OK(hipMalloc(&d_coarseSfc, ngroups * sizeof(T)));
        HIP_OK(hipMemcpy(d_coarseSfc, h_coarseSfc, ngroups * sizeof(T), hipMemcpyHostToDevice));
        HIP_OK(hipMalloc(&d_gperm, ngroups * sizeof(uint32_t)));
        HIP_OK(hipMemcpy(d_gperm, h_gperm, ngroups * sizeof(uint32_t), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_scatter_perm<T>, dim3(grid(ngroups)), dim3(BLOCK_X), 0, 0,
                           d_coarseSfc, d_gperm, d_coarse, ngroups);
        hipFree(d_coarseSfc);
    } else {
        HIP_OK(hipMemcpy(d_coarse, h_coarseSfc, ngroups * sizeof(T), hipMemcpyHostToDevice));
    }

    /* residual: dequantize -> undo nodal SFC (scatter) */
    const T *field = nullptr;
    uint32_t *d_q = nullptr, *d_nperm = nullptr;
    T *d_resid = nullptr, *d_field = nullptr;
    if (h_quant) {
        HIP_OK(hipMalloc(&d_q, nnodes * sizeof(uint32_t)));
        HIP_OK(hipMemcpy(d_q, h_quant, nnodes * sizeof(uint32_t), hipMemcpyHostToDevice));
        HIP_OK(hipMalloc(&d_resid, nnodes * sizeof(T)));
        hipLaunchKernelGGL(k_dequantize_zigzag<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                           d_q, d_resid, nnodes, 2.0 * tolR);
        if (h_nperm) {
            HIP_OK(hipMalloc(&d_nperm, nnodes * sizeof(uint32_t)));
            HIP_OK(hipMemcpy(d_nperm, h_nperm, nnodes * sizeof(uint32_t), hipMemcpyHostToDevice));
            HIP_OK(hipMalloc(&d_field, nnodes * sizeof(T)));
            hipLaunchKernelGGL(k_scatter_perm<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                               d_resid, d_nperm, d_field, nnodes);
            field = d_field;
        } else {
            field = d_resid;
        }
    }

    hipLaunchKernelGGL(k_group_bcast_add<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       d_coarse, d_gid, field, d_out, nnodes);
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(h_out, d_out, nnodes * sizeof(T), hipMemcpyDeviceToHost));

    hipFree(d_coarse); hipFree(d_gid); hipFree(d_out);
    if (d_gperm) hipFree(d_gperm);
    if (d_q)     hipFree(d_q);
    if (d_resid) hipFree(d_resid);
    if (d_nperm) hipFree(d_nperm);
    if (d_field) hipFree(d_field);
}

} // namespace

/* ========================== exported C ABI ========================== */

extern "C" {

int cmg_hip_available(void)
{
    int n = 0;
    return (hipGetDeviceCount(&n) == hipSuccess && n > 0) ? 1 : 0;
}

/* These are the strong definitions called by the CPU-TU dispatcher when the
 * caller passes CMG_BACKEND_HIP. They take the *same* host pointers as the
 * primary ABI. Naming uses cmg_hip_* to coexist with the CPU symbols. */

void cmg_hip_quantize_zigzag_f32(const float *in, size_t n, double tol, uint32_t *out)
{ quantize_hip<float>(in, n, tol, out); }
void cmg_hip_quantize_zigzag_f64(const double *in, size_t n, double tol, uint32_t *out)
{ quantize_hip<double>(in, n, tol, out); }

void cmg_hip_dequantize_zigzag_f32(const uint32_t *in, size_t n, double tol, float *out)
{ dequantize_hip<float>(in, n, tol, out); }
void cmg_hip_dequantize_zigzag_f64(const uint32_t *in, size_t n, double tol, double *out)
{ dequantize_hip<double>(in, n, tol, out); }

void cmg_hip_free(void *p) { if (p) hipFree(p); }

/* D2H copy: MGARD-X returns the compressed output as a DEVICE pointer when its
 * input was on device, so the operator D2H's those (small) bytes to bufferOut. */
void cmg_hip_d2h(void *h_dst, const void *d_src, size_t bytes)
{ if (bytes) HIP_OK(hipMemcpy(h_dst, d_src, bytes, hipMemcpyDeviceToHost)); }

void cmg_hip_group_compress_f32(size_t blockId, const float *field, const uint32_t *gid,
                                const uint32_t *gperm, const uint32_t *nperm,
                                size_t nnodes, size_t ngroups, double tol_resi, int adaptive,
                                float *coarseSfc, double *rmin, double *rmax, double *rsumsq,
                                int *drop, uint32_t *quantC,
                                void **d_coarse_out, void **d_q_out, uint32_t *maxcode)
{ group_compress_hip<float>(blockId, field, gid, gperm, nperm, nnodes, ngroups, tol_resi,
                            adaptive, coarseSfc, rmin, rmax, rsumsq, drop, quantC,
                            d_coarse_out, d_q_out, maxcode); }

void cmg_hip_group_compress_f64(size_t blockId, const double *field, const uint32_t *gid,
                                const uint32_t *gperm, const uint32_t *nperm,
                                size_t nnodes, size_t ngroups, double tol_resi, int adaptive,
                                double *coarseSfc, double *rmin, double *rmax, double *rsumsq,
                                int *drop, uint32_t *quantC,
                                void **d_coarse_out, void **d_q_out, uint32_t *maxcode)
{ group_compress_hip<double>(blockId, field, gid, gperm, nperm, nnodes, ngroups, tol_resi,
                             adaptive, coarseSfc, rmin, rmax, rsumsq, drop, quantC,
                             d_coarse_out, d_q_out, maxcode); }

void cmg_hip_reduce_stats_f32(const float *in, size_t n,
                              double *omin, double *omax, double *osumsq)
{ reduce_stats_hip<float>(in, n, omin, omax, osumsq); }
void cmg_hip_reduce_stats_f64(const double *in, size_t n,
                              double *omin, double *omax, double *osumsq)
{ reduce_stats_hip<double>(in, n, omin, omax, osumsq); }

void cmg_hip_group_recombine_f32(const float *coarseSfc, const uint32_t *gperm,
                                 const uint32_t *quant, const uint32_t *gid,
                                 const uint32_t *nperm, size_t nnodes, size_t ngroups,
                                 double tolR, float *out)
{ group_recombine_hip<float>(coarseSfc, gperm, quant, gid, nperm, nnodes, ngroups, tolR, out); }
void cmg_hip_group_recombine_f64(const double *coarseSfc, const uint32_t *gperm,
                                 const uint32_t *quant, const uint32_t *gid,
                                 const uint32_t *nperm, size_t nnodes, size_t ngroups,
                                 double tolR, double *out)
{ group_recombine_hip<double>(coarseSfc, gperm, quant, gid, nperm, nnodes, ngroups, tolR, out); }

/* The SFC permutation builder runs centroid + code on device, then sorts on
 * host (sort overhead is small vs. centroid pass and avoids a thrust dep). */
cmg_sfc_t cmg_hip_build_sfc_perm(const int64_t *h_conn, size_t ncells, size_t npc,
                                 const double *X, const double *Y, const double *Z, size_t nnodes,
                                 cmg_sfc_t requested, uint32_t *perm_out);

} // extern "C"

/* Out-of-line because it allocates std::vector for the host sort. */
extern "C" cmg_sfc_t cmg_hip_build_sfc_perm(const int64_t *h_conn, size_t ncells, size_t npc,
                                            const double *X, const double *Y, const double *Z,
                                            size_t nnodes, cmg_sfc_t requested,
                                            uint32_t *perm_out)
{
    const bool wantCoord = (requested == CMG_SFC_MORTON || requested == CMG_SFC_HILBERT);
    if (!(wantCoord && X && Y && Z))
    {
        /* fall back to CPU MinNodeId path (cheap, no kernel needed) */
        return cmg_build_sfc_perm(CMG_BACKEND_CPU, h_conn, ncells, npc,
                                  nullptr, nullptr, nullptr, nnodes,
                                  CMG_SFC_MINNODE, perm_out);
    }

    int64_t *d_conn = nullptr;
    double *d_X = nullptr, *d_Y = nullptr, *d_Z = nullptr;
    double *d_cx = nullptr, *d_cy = nullptr, *d_cz = nullptr;
    uint64_t *d_codes = nullptr; uint32_t *d_idx = nullptr;
    HIP_OK(hipMalloc(&d_conn, ncells * npc * sizeof(int64_t)));
    HIP_OK(hipMalloc(&d_X, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_Y, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_Z, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_cx, ncells * sizeof(double)));
    HIP_OK(hipMalloc(&d_cy, ncells * sizeof(double)));
    HIP_OK(hipMalloc(&d_cz, ncells * sizeof(double)));
    HIP_OK(hipMalloc(&d_codes, ncells * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&d_idx,   ncells * sizeof(uint32_t)));

    HIP_OK(hipMemcpy(d_conn, h_conn, ncells * npc * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_X, X, nnodes * sizeof(double), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_Y, Y, nnodes * sizeof(double), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_Z, Z, nnodes * sizeof(double), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(k_cell_centroid, dim3(grid(ncells)), dim3(BLOCK_X), 0, 0,
                       d_conn, ncells, npc, d_X, d_Y, d_Z, d_cx, d_cy, d_cz);
    HIP_OK(hipDeviceSynchronize());

    /* host-side bbox reduction (small) */
    std::vector<double> cx(ncells), cy(ncells), cz(ncells);
    HIP_OK(hipMemcpy(cx.data(), d_cx, ncells * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(cy.data(), d_cy, ncells * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(cz.data(), d_cz, ncells * sizeof(double), hipMemcpyDeviceToHost));

    double xmin = cx[0], xmax = cx[0];
    double ymin = cy[0], ymax = cy[0];
    double zmin = cz[0], zmax = cz[0];
    for (size_t c = 1; c < ncells; ++c) {
        if (cx[c] < xmin) xmin = cx[c]; if (cx[c] > xmax) xmax = cx[c];
        if (cy[c] < ymin) ymin = cy[c]; if (cy[c] > ymax) ymax = cy[c];
        if (cz[c] < zmin) zmin = cz[c]; if (cz[c] > zmax) zmax = cz[c];
    }
    const uint32_t maxQ = (1u << 21) - 1u;
    const double sx = (xmax > xmin) ? double(maxQ) / (xmax - xmin) : 0.0;
    const double sy = (ymax > ymin) ? double(maxQ) / (ymax - ymin) : 0.0;
    const double sz = (zmax > zmin) ? double(maxQ) / (zmax - zmin) : 0.0;
    const int useHilbert = (requested == CMG_SFC_HILBERT) ? 1 : 0;

    hipLaunchKernelGGL(k_sfc_codes, dim3(grid(ncells)), dim3(BLOCK_X), 0, 0,
                       d_cx, d_cy, d_cz, xmin, ymin, zmin, sx, sy, sz,
                       maxQ, useHilbert, ncells, d_codes, d_idx);
    HIP_OK(hipDeviceSynchronize());

    /* device radix sort by code; rocPRIM's radix sort is stable, so ties keep
     * ascending original index -- byte-identical to the old host std::sort. */
    uint64_t *d_codes2 = nullptr; uint32_t *d_idx2 = nullptr;
    HIP_OK(hipMalloc(&d_codes2, ncells * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&d_idx2,   ncells * sizeof(uint32_t)));
    sort_pairs_u64(d_codes, d_codes2, d_idx, d_idx2, ncells);

    HIP_OK(hipMemcpy(perm_out, d_idx, ncells * sizeof(uint32_t), hipMemcpyDeviceToHost));

    hipFree(d_conn); hipFree(d_X); hipFree(d_Y); hipFree(d_Z);
    hipFree(d_cx); hipFree(d_cy); hipFree(d_cz);
    hipFree(d_codes); hipFree(d_idx); hipFree(d_codes2); hipFree(d_idx2);
    return requested;
}

/* ------------------------------------------------------------------ */
/* NODAL SFC permutation (GPU).                                        */
/* Reuses k_sfc_codes directly on the node coordinates -- no cell      */
/* centroid pass is needed since the points ARE the nodes.             */
/* ------------------------------------------------------------------ */
extern "C" cmg_sfc_t cmg_hip_build_sfc_perm_nodes(const double *X, const double *Y,
                                                  const double *Z, size_t nnodes,
                                                  cmg_sfc_t requested, uint32_t *perm_out)
{
    const bool wantCoord = (requested == CMG_SFC_MORTON || requested == CMG_SFC_HILBERT);
    if (!(wantCoord && X && Y && Z) || nnodes == 0)
    {
        for (size_t i = 0; i < nnodes; ++i) perm_out[i] = (uint32_t)i;
        return CMG_SFC_NONE;
    }

    double *d_X = nullptr, *d_Y = nullptr, *d_Z = nullptr;
    uint64_t *d_codes = nullptr; uint32_t *d_idx = nullptr;
    HIP_OK(hipMalloc(&d_X, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_Y, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_Z, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_codes, nnodes * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&d_idx,   nnodes * sizeof(uint32_t)));
    HIP_OK(hipMemcpy(d_X, X, nnodes * sizeof(double), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_Y, Y, nnodes * sizeof(double), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_Z, Z, nnodes * sizeof(double), hipMemcpyHostToDevice));

    /* host-side bbox over node coords (single pass, cheap vs the sort) */
    double xmin = X[0], xmax = X[0], ymin = Y[0], ymax = Y[0], zmin = Z[0], zmax = Z[0];
    for (size_t i = 1; i < nnodes; ++i) {
        if (X[i] < xmin) xmin = X[i]; if (X[i] > xmax) xmax = X[i];
        if (Y[i] < ymin) ymin = Y[i]; if (Y[i] > ymax) ymax = Y[i];
        if (Z[i] < zmin) zmin = Z[i]; if (Z[i] > zmax) zmax = Z[i];
    }
    const uint32_t maxQ = (1u << 21) - 1u;
    const double sx = (xmax > xmin) ? double(maxQ) / (xmax - xmin) : 0.0;
    const double sy = (ymax > ymin) ? double(maxQ) / (ymax - ymin) : 0.0;
    const double sz = (zmax > zmin) ? double(maxQ) / (zmax - zmin) : 0.0;
    const int useHilbert = (requested == CMG_SFC_HILBERT) ? 1 : 0;

    hipLaunchKernelGGL(k_sfc_codes, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       d_X, d_Y, d_Z, xmin, ymin, zmin, sx, sy, sz,
                       maxQ, useHilbert, nnodes, d_codes, d_idx);
    HIP_OK(hipDeviceSynchronize());

    /* device radix sort by code -- stable, so coincident nodes keep ascending
     * original index and compress/decompress agree. */
    uint64_t *d_codes2 = nullptr; uint32_t *d_idx2 = nullptr;
    HIP_OK(hipMalloc(&d_codes2, nnodes * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&d_idx2,   nnodes * sizeof(uint32_t)));
    sort_pairs_u64(d_codes, d_codes2, d_idx, d_idx2, nnodes);

    HIP_OK(hipMemcpy(perm_out, d_idx, nnodes * sizeof(uint32_t), hipMemcpyDeviceToHost));

    hipFree(d_X); hipFree(d_Y); hipFree(d_Z);
    hipFree(d_codes); hipFree(d_idx); hipFree(d_codes2); hipFree(d_idx2);
    return requested;
}


/* ------------------------------------------------------------------ */
/* Coordinate grouping (GPU).                                          */
/* Lexicographic (x,y,z) order with ascending-original-index tie-break */
/* is obtained by three *stable* LSD radix passes (z, then y, then x), */
/* re-gathering each pass's key through the running permutation. The   */
/* group ids then come from a boundary flag + inclusive scan.          */
/* ------------------------------------------------------------------ */
extern "C" size_t cmg_hip_build_coord_groups(const double *X, const double *Y,
                                             const double *Z, size_t nnodes,
                                             double tol, uint32_t *gid_out,
                                             double *gx, double *gy, double *gz)
{
    if (nnodes == 0) return 0;

    double *d_X = nullptr, *d_Y = nullptr, *d_Z = nullptr;
    HIP_OK(hipMalloc(&d_X, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_Y, nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_Z, nnodes * sizeof(double)));
    HIP_OK(hipMemcpy(d_X, X, nnodes * sizeof(double), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_Y, Y, nnodes * sizeof(double), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_Z, Z, nnodes * sizeof(double), hipMemcpyHostToDevice));

    /* Comparison keys, one uint64 per node per axis. The bbox reduction is a
     * single cheap host pass (cmg_coord_quant_scale) -- it is not the bottleneck
     * (the radix sort dominates groupbuild), and a device reduction here measured
     * SLOWER due to per-axis allocation overhead. Host min/max also guarantees
     * bit-identical bounds vs the CPU backend -- compress on one must group the
     * same as decompress on the other. */
    const int quant = (tol > 0.0) ? 1 : 0;
    double qmin[3] = {0,0,0}, qinvh[3] = {0,0,0};
    if (quant) cmg_coord_quant_scale(X, Y, Z, nnodes, tol, qmin, qinvh);

    uint64_t *d_q[3] = {nullptr, nullptr, nullptr};
    const double *d_axis[3] = { d_X, d_Y, d_Z };
    for (int d = 0; d < 3; ++d) {
        HIP_OK(hipMalloc(&d_q[d], nnodes * sizeof(uint64_t)));
        hipLaunchKernelGGL(k_make_axis_key, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                           d_axis[d], qmin[d], qinvh[d], quant, d_q[d], nnodes);
    }
    HIP_OK(hipDeviceSynchronize());

    uint64_t *k_a = nullptr, *k_b = nullptr;
    uint32_t *v_a = nullptr, *v_b = nullptr;
    HIP_OK(hipMalloc(&k_a, nnodes * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&k_b, nnodes * sizeof(uint64_t)));
    HIP_OK(hipMalloc(&v_a, nnodes * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&v_b, nnodes * sizeof(uint32_t)));

    hipLaunchKernelGGL(k_iota_u32, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0, v_a, nnodes);
    HIP_OK(hipDeviceSynchronize());

    const int order[3] = { 2, 1, 0 };            /* least- to most-significant */
    for (int p = 0; p < 3; ++p)
    {
        hipLaunchKernelGGL(k_gather_u64, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                           d_q[order[p]], v_a, k_a, nnodes);
        HIP_OK(hipDeviceSynchronize());
        sort_pairs_u64(k_a, k_b, v_a, v_b, nnodes);
    }
    /* v_a now holds node indices in lexicographic (x,y,z) key order */

    uint32_t *d_flag = nullptr, *d_scan = nullptr;
    HIP_OK(hipMalloc(&d_flag, nnodes * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_scan, nnodes * sizeof(uint32_t)));
    hipLaunchKernelGGL(k_mark_group_start, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       d_q[0], d_q[1], d_q[2], v_a, d_flag, nnodes);
    HIP_OK(hipDeviceSynchronize());

    {
        void *tmp = nullptr; size_t bytes = 0;
        rocprim::inclusive_scan(tmp, bytes, d_flag, d_scan, nnodes, rocprim::plus<uint32_t>());
        HIP_OK(hipMalloc(&tmp, bytes));
        rocprim::inclusive_scan(tmp, bytes, d_flag, d_scan, nnodes, rocprim::plus<uint32_t>());
        HIP_OK(hipDeviceSynchronize());
        hipFree(tmp);
    }

    uint32_t *d_gid = nullptr;
    double *d_gx = nullptr, *d_gy = nullptr, *d_gz = nullptr;
    HIP_OK(hipMalloc(&d_gid, nnodes * sizeof(uint32_t)));

    uint32_t ngroups = 0;
    HIP_OK(hipMemcpy(&ngroups, d_scan + (nnodes - 1), sizeof(uint32_t), hipMemcpyDeviceToHost));

    if (gx) {
        HIP_OK(hipMalloc(&d_gx, (size_t)ngroups * sizeof(double)));
        HIP_OK(hipMalloc(&d_gy, (size_t)ngroups * sizeof(double)));
        HIP_OK(hipMalloc(&d_gz, (size_t)ngroups * sizeof(double)));
    }
    hipLaunchKernelGGL(k_scatter_gid, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       v_a, d_scan, d_flag, d_X, d_Y, d_Z,
                       d_gid, d_gx, d_gy, d_gz, nnodes);
    HIP_OK(hipDeviceSynchronize());

    HIP_OK(hipMemcpy(gid_out, d_gid, nnodes * sizeof(uint32_t), hipMemcpyDeviceToHost));
    if (gx) {
        HIP_OK(hipMemcpy(gx, d_gx, (size_t)ngroups * sizeof(double), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(gy, d_gy, (size_t)ngroups * sizeof(double), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(gz, d_gz, (size_t)ngroups * sizeof(double), hipMemcpyDeviceToHost));
        hipFree(d_gx); hipFree(d_gy); hipFree(d_gz);
    }

    hipFree(d_X); hipFree(d_Y); hipFree(d_Z);
    hipFree(d_q[0]); hipFree(d_q[1]); hipFree(d_q[2]);
    hipFree(k_a); hipFree(k_b); hipFree(v_a); hipFree(v_b);
    hipFree(d_flag); hipFree(d_scan); hipFree(d_gid);
    return (size_t)ngroups;
}
