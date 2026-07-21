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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

/* ---- centroid-split kernels ---------------------------------------- */

template <typename T>
__global__ void k_cell_avg(const T *u, const int64_t *conn, size_t ncells, size_t npc,
                           T *cell_avg)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    const int64_t *cc = conn + c * npc;
    double acc = 0.0; int cnt = 0;
    for (size_t k = 0; k < npc; ++k) {
        bool dup = false;
        for (size_t j = 0; j < k; ++j) if (cc[j] == cc[k]) { dup = true; break; }
        if (!dup) { acc += (double)u[cc[k]]; ++cnt; }
    }
    cell_avg[c] = (T)(cnt ? acc / (double)cnt : 0.0);
}

template <typename T>
__global__ void k_node_accumulate(const T *cell_avg, const int64_t *conn, size_t ncells,
                                  size_t npc, double *sum, uint32_t *cnt)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncells) return;
    const int64_t *cc = conn + c * npc;
    const double a = (double)cell_avg[c];
    for (size_t k = 0; k < npc; ++k) {
        bool dup = false;
        for (size_t j = 0; j < k; ++j) if (cc[j] == cc[k]) { dup = true; break; }
        if (!dup) { atomicAdd(&sum[cc[k]], a); atomicAdd(&cnt[cc[k]], 1u); }
    }
}

template <typename T>
__global__ void k_residual(const T *u, const double *sum, const uint32_t *cnt,
                           size_t nnodes, T *r)
{
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nnodes) return;
    const uint32_t c = cnt[n];
    const double bar = c ? sum[n] / (double)c : 0.0;
    r[n] = (T)((double)u[n] - bar);
}

template <typename T>
__global__ void k_recombine(const double *sum, const uint32_t *cnt,
                            size_t nnodes, T *inout)
{
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nnodes) return;
    const uint32_t c = cnt[n];
    const double bar = c ? sum[n] / (double)c : 0.0;
    inout[n] = (T)(bar + (double)inout[n]);
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

/* ---- generic split/recombine drivers ------------------------------- */

template <typename T>
void split_hip(const T *h_u, const int64_t *h_conn,
               size_t ncells, size_t npc, size_t nnodes,
               T *h_avg, T *h_resi)
{
    T       *d_u = nullptr, *d_avg = nullptr, *d_resi = nullptr;
    int64_t *d_conn = nullptr;
    double  *d_sum = nullptr; uint32_t *d_cnt = nullptr;

    HIP_OK(hipMalloc(&d_u,    nnodes  * sizeof(T)));
    HIP_OK(hipMalloc(&d_avg,  ncells  * sizeof(T)));
    HIP_OK(hipMalloc(&d_resi, nnodes  * sizeof(T)));
    HIP_OK(hipMalloc(&d_conn, ncells * npc * sizeof(int64_t)));
    HIP_OK(hipMalloc(&d_sum,  nnodes  * sizeof(double)));
    HIP_OK(hipMalloc(&d_cnt,  nnodes  * sizeof(uint32_t)));

    HIP_OK(hipMemcpy(d_u,    h_u,    nnodes * sizeof(T),         hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_conn, h_conn, ncells * npc * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_sum, 0, nnodes * sizeof(double)));
    HIP_OK(hipMemset(d_cnt, 0, nnodes * sizeof(uint32_t)));

    hipLaunchKernelGGL(k_cell_avg<T>, dim3(grid(ncells)), dim3(BLOCK_X), 0, 0,
                       d_u, d_conn, ncells, npc, d_avg);
    hipLaunchKernelGGL(k_node_accumulate<T>, dim3(grid(ncells)), dim3(BLOCK_X), 0, 0,
                       d_avg, d_conn, ncells, npc, d_sum, d_cnt);
    hipLaunchKernelGGL(k_residual<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       d_u, d_sum, d_cnt, nnodes, d_resi);
    HIP_OK(hipDeviceSynchronize());

    HIP_OK(hipMemcpy(h_avg,  d_avg,  ncells * sizeof(T), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(h_resi, d_resi, nnodes * sizeof(T), hipMemcpyDeviceToHost));

    hipFree(d_u); hipFree(d_avg); hipFree(d_resi);
    hipFree(d_conn); hipFree(d_sum); hipFree(d_cnt);
}

template <typename T>
void recombine_hip(const T *h_avg, const int64_t *h_conn,
                   size_t ncells, size_t npc, size_t nnodes, T *h_inout)
{
    T       *d_avg = nullptr, *d_inout = nullptr;
    int64_t *d_conn = nullptr;
    double  *d_sum = nullptr; uint32_t *d_cnt = nullptr;
    HIP_OK(hipMalloc(&d_avg,   ncells * sizeof(T)));
    HIP_OK(hipMalloc(&d_inout, nnodes * sizeof(T)));
    HIP_OK(hipMalloc(&d_conn,  ncells * npc * sizeof(int64_t)));
    HIP_OK(hipMalloc(&d_sum,   nnodes * sizeof(double)));
    HIP_OK(hipMalloc(&d_cnt,   nnodes * sizeof(uint32_t)));

    HIP_OK(hipMemcpy(d_avg,   h_avg,   ncells * sizeof(T),         hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_inout, h_inout, nnodes * sizeof(T),         hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_conn,  h_conn,  ncells * npc * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_sum, 0, nnodes * sizeof(double)));
    HIP_OK(hipMemset(d_cnt, 0, nnodes * sizeof(uint32_t)));

    hipLaunchKernelGGL(k_node_accumulate<T>, dim3(grid(ncells)), dim3(BLOCK_X), 0, 0,
                       d_avg, d_conn, ncells, npc, d_sum, d_cnt);
    hipLaunchKernelGGL(k_recombine<T>, dim3(grid(nnodes)), dim3(BLOCK_X), 0, 0,
                       d_sum, d_cnt, nnodes, d_inout);
    HIP_OK(hipDeviceSynchronize());

    HIP_OK(hipMemcpy(h_inout, d_inout, nnodes * sizeof(T), hipMemcpyDeviceToHost));

    hipFree(d_avg); hipFree(d_inout); hipFree(d_conn); hipFree(d_sum); hipFree(d_cnt);
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

void cmg_hip_centroid_split_f32(const float *u, const int64_t *conn,
                                size_t ncells, size_t npc, size_t nnodes,
                                float *out_avg, float *out_resi)
{ split_hip<float>(u, conn, ncells, npc, nnodes, out_avg, out_resi); }

void cmg_hip_centroid_split_f64(const double *u, const int64_t *conn,
                                size_t ncells, size_t npc, size_t nnodes,
                                double *out_avg, double *out_resi)
{ split_hip<double>(u, conn, ncells, npc, nnodes, out_avg, out_resi); }

void cmg_hip_centroid_recombine_f32(const float *avg, const int64_t *conn,
                                    size_t ncells, size_t npc, size_t nnodes,
                                    float *inout)
{ recombine_hip<float>(avg, conn, ncells, npc, nnodes, inout); }

void cmg_hip_centroid_recombine_f64(const double *avg, const int64_t *conn,
                                    size_t ncells, size_t npc, size_t nnodes,
                                    double *inout)
{ recombine_hip<double>(avg, conn, ncells, npc, nnodes, inout); }

void cmg_hip_quantize_zigzag_f32(const float *in, size_t n, double tol, uint32_t *out)
{ quantize_hip<float>(in, n, tol, out); }
void cmg_hip_quantize_zigzag_f64(const double *in, size_t n, double tol, uint32_t *out)
{ quantize_hip<double>(in, n, tol, out); }

void cmg_hip_dequantize_zigzag_f32(const uint32_t *in, size_t n, double tol, float *out)
{ dequantize_hip<float>(in, n, tol, out); }
void cmg_hip_dequantize_zigzag_f64(const uint32_t *in, size_t n, double tol, double *out)
{ dequantize_hip<double>(in, n, tol, out); }

void cmg_hip_reduce_stats_f32(const float *in, size_t n,
                              double *omin, double *omax, double *osumsq)
{ reduce_stats_hip<float>(in, n, omin, omax, osumsq); }
void cmg_hip_reduce_stats_f64(const double *in, size_t n,
                              double *omin, double *omax, double *osumsq)
{ reduce_stats_hip<double>(in, n, omin, omax, osumsq); }

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

    std::vector<uint64_t> codes(ncells);
    std::vector<uint32_t> idx(ncells);
    HIP_OK(hipMemcpy(codes.data(), d_codes, ncells * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(idx.data(),   d_idx,   ncells * sizeof(uint32_t), hipMemcpyDeviceToHost));

    hipFree(d_conn); hipFree(d_X); hipFree(d_Y); hipFree(d_Z);
    hipFree(d_cx); hipFree(d_cy); hipFree(d_cz);
    hipFree(d_codes); hipFree(d_idx);

    /* stable sort by (code, original cell index) */
    std::vector<uint32_t> order(ncells);
    for (size_t i = 0; i < ncells; ++i) order[i] = (uint32_t)i;
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) {
                  return (codes[a] != codes[b]) ? (codes[a] < codes[b])
                                                : (idx[a]   < idx[b]);
              });
    for (size_t i = 0; i < ncells; ++i) perm_out[i] = idx[order[i]];
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

    std::vector<uint64_t> codes(nnodes);
    std::vector<uint32_t> idx(nnodes);
    HIP_OK(hipMemcpy(codes.data(), d_codes, nnodes * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(idx.data(),   d_idx,   nnodes * sizeof(uint32_t), hipMemcpyDeviceToHost));
    hipFree(d_X); hipFree(d_Y); hipFree(d_Z); hipFree(d_codes); hipFree(d_idx);

    /* stable sort by (code, original node index) -- ties (coincident nodes)
     * resolve deterministically so compress/decompress agree. */
    std::vector<uint32_t> order(nnodes);
    for (size_t i = 0; i < nnodes; ++i) order[i] = (uint32_t)i;
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) {
                  return (codes[a] != codes[b]) ? (codes[a] < codes[b])
                                                : (idx[a]   < idx[b]);
              });
    for (size_t i = 0; i < nnodes; ++i) perm_out[i] = idx[order[i]];
    return requested;
}

