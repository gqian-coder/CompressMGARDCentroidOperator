/*
 * DeviceKernels_CPU.cpp — single-threaded reference implementation.
 *
 * Per Plan.md decision #4 the CPU path stays single-threaded for
 * deterministic accumulation order. All kernels here are direct ports of
 * the host helpers that previously lived inside
 * backup/CompressMGARDCentroidOperator.cpp.
 */

#include "DeviceKernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace {

/* ------ SFC helpers (Morton + Hilbert, 21 bits per axis) ------------- */

inline uint64_t morton_expand3(uint32_t v)
{
    uint64_t x = v & 0x1fffffull;
    x = (x | (x << 32)) & 0x001f00000000ffffull;
    x = (x | (x << 16)) & 0x001f0000ff0000ffull;
    x = (x | (x << 8))  & 0x100f00f00f00f00full;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2))  & 0x1249249249249249ull;
    return x;
}
inline uint64_t morton3d(uint32_t x, uint32_t y, uint32_t z)
{
    return morton_expand3(x) | (morton_expand3(y) << 1) | (morton_expand3(z) << 2);
}
inline uint64_t hilbert3d(uint32_t x, uint32_t y, uint32_t z)
{
    constexpr int b = 21;
    uint32_t X[3] = { x & 0x1fffffu, y & 0x1fffffu, z & 0x1fffffu };
    const uint32_t M = 1u << (b - 1);
    uint32_t P, Q, t;
    for (Q = M; Q > 1; Q >>= 1)
    {
        P = Q - 1;
        for (int i = 0; i < 3; ++i)
        {
            if (X[i] & Q) { X[0] ^= P; }
            else { t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; }
        }
    }
    for (int i = 1; i < 3; ++i) X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1)
        if (X[2] & Q) t ^= Q - 1;
    for (int i = 0; i < 3; ++i) X[i] ^= t;
    uint64_t h = 0;
    for (int bit = b - 1; bit >= 0; --bit)
    {
        h = (h << 1) | ((X[0] >> bit) & 1u);
        h = (h << 1) | ((X[1] >> bit) & 1u);
        h = (h << 1) | ((X[2] >> bit) & 1u);
    }
    return h;
}

/* ------ quantize / dequantize (mid-tread + zigzag) ------------------- */

inline uint32_t zigzag_quantize_double(double x, double inv_q)
{
    double v = x * inv_q;
    if (v >  2147483647.0) v =  2147483647.0;
    if (v < -2147483647.0) v = -2147483647.0;
    const int32_t s = static_cast<int32_t>(std::llround(v));
    return (s >= 0) ? static_cast<uint32_t>( s * 2)
                    : static_cast<uint32_t>((-s - 1) * 2 + 1);
}
inline int32_t zigzag_decode(uint32_t z)
{
    return (z & 1u) ? -static_cast<int32_t>(z >> 1) - 1
                    :  static_cast<int32_t>(z >> 1);
}

} // namespace

/* ====================== exported C ABI ============================== */

extern "C" {

#ifdef CMG_HAVE_HIP
/* Strong definitions live in DeviceKernels_HIP.cpp; forward-declare here. */
void cmg_hip_quantize_zigzag_f32(const float *, size_t, double, uint32_t *);
void cmg_hip_quantize_zigzag_f64(const double *, size_t, double, uint32_t *);
void cmg_hip_dequantize_zigzag_f32(const uint32_t *, size_t, double, float *);
void cmg_hip_dequantize_zigzag_f64(const uint32_t *, size_t, double, double *);
cmg_sfc_t cmg_hip_build_sfc_perm(const int64_t *, size_t, size_t,
                                 const double *, const double *, const double *, size_t,
                                 cmg_sfc_t, uint32_t *);
cmg_sfc_t cmg_hip_build_sfc_perm_nodes(const double *, const double *, const double *,
                                       size_t, cmg_sfc_t, uint32_t *);
size_t cmg_hip_build_coord_groups(const double *, const double *, const double *, size_t,
                                  double, uint32_t *, double *, double *, double *);
void cmg_hip_group_compress_f32(size_t, const float *, const uint32_t *, const uint32_t *,
                                const uint32_t *, size_t, size_t, double, int,
                                float *, double *, double *, double *, int *, uint32_t *,
                                void **, void **, uint32_t *);
void cmg_hip_group_compress_f64(size_t, const double *, const uint32_t *, const uint32_t *,
                                const uint32_t *, size_t, size_t, double, int,
                                double *, double *, double *, double *, int *, uint32_t *,
                                void **, void **, uint32_t *);
void cmg_hip_reduce_stats_f32(const float *, size_t, double *, double *, double *);
void cmg_hip_reduce_stats_f64(const double *, size_t, double *, double *, double *);
void cmg_hip_group_recombine_f32(const float *, const uint32_t *, const uint32_t *,
                                 const uint32_t *, const uint32_t *, size_t, size_t,
                                 double, float *);
void cmg_hip_group_recombine_f64(const double *, const uint32_t *, const uint32_t *,
                                 const uint32_t *, const uint32_t *, size_t, size_t,
                                 double, double *);
#endif

cmg_sfc_t cmg_build_sfc_perm(cmg_backend_t be, const int64_t *conn,
                             size_t ncells, size_t npc,
                             const double *X, const double *Y, const double *Z,
                             size_t nnodes, cmg_sfc_t requested,
                             uint32_t *perm_out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        return cmg_hip_build_sfc_perm(conn, ncells, npc, X, Y, Z, nnodes, requested, perm_out);
    }
#else
    (void)be;
#endif
    const bool wantCoord = (requested == CMG_SFC_MORTON || requested == CMG_SFC_HILBERT);

    if (wantCoord && X && Y && Z)
    {
        std::vector<double> cx(ncells), cy(ncells), cz(ncells);
        double xmin =  std::numeric_limits<double>::infinity(), xmax = -xmin;
        double ymin =  xmin, ymax = -xmin;
        double zmin =  xmin, zmax = -xmin;
        for (size_t c = 0; c < ncells; ++c)
        {
            const int64_t *cc = conn + c * npc;
            double ax = 0.0, ay = 0.0, az = 0.0; size_t u = 0;
            for (size_t k = 0; k < npc; ++k)
            {
                bool dup = false;
                for (size_t j = 0; j < k; ++j)
                    if (cc[j] == cc[k]) { dup = true; break; }
                if (!dup)
                {
                    const int64_t n = cc[k];
                    ax += X[n]; ay += Y[n]; az += Z[n]; ++u;
                }
            }
            const double inv = u ? 1.0 / static_cast<double>(u) : 0.0;
            cx[c] = ax * inv; cy[c] = ay * inv; cz[c] = az * inv;
            if (cx[c] < xmin) xmin = cx[c]; if (cx[c] > xmax) xmax = cx[c];
            if (cy[c] < ymin) ymin = cy[c]; if (cy[c] > ymax) ymax = cy[c];
            if (cz[c] < zmin) zmin = cz[c]; if (cz[c] > zmax) zmax = cz[c];
        }
        const uint32_t maxQ = (1u << 21) - 1u;
        const double sx = (xmax > xmin) ? double(maxQ) / (xmax - xmin) : 0.0;
        const double sy = (ymax > ymin) ? double(maxQ) / (ymax - ymin) : 0.0;
        const double sz = (zmax > zmin) ? double(maxQ) / (zmax - zmin) : 0.0;
        std::vector<std::pair<uint64_t, uint32_t>> kv(ncells);
        for (size_t c = 0; c < ncells; ++c)
        {
            double qxd = (cx[c] - xmin) * sx;
            double qyd = (cy[c] - ymin) * sy;
            double qzd = (cz[c] - zmin) * sz;
            if (qxd < 0.0) qxd = 0.0; if (qxd > double(maxQ)) qxd = maxQ;
            if (qyd < 0.0) qyd = 0.0; if (qyd > double(maxQ)) qyd = maxQ;
            if (qzd < 0.0) qzd = 0.0; if (qzd > double(maxQ)) qzd = maxQ;
            const uint32_t qx = (uint32_t)qxd, qy = (uint32_t)qyd, qz = (uint32_t)qzd;
            kv[c] = { (requested == CMG_SFC_HILBERT)
                         ? hilbert3d(qx, qy, qz) : morton3d(qx, qy, qz),
                      (uint32_t)c };
        }
        std::sort(kv.begin(), kv.end(),
                  [](const auto &a, const auto &b) {
                      return (a.first != b.first) ? (a.first < b.first)
                                                  : (a.second < b.second);
                  });
        for (size_t i = 0; i < ncells; ++i) perm_out[i] = kv[i].second;
        return requested;
    }

    /* MinNodeId fallback */
    std::vector<std::tuple<int64_t, uint32_t>> kv(ncells);
    for (size_t c = 0; c < ncells; ++c)
    {
        const int64_t *cc = conn + c * npc;
        int64_t mn = std::numeric_limits<int64_t>::max();
        for (size_t k = 0; k < npc; ++k) if (cc[k] < mn) mn = cc[k];
        kv[c] = std::make_tuple(mn, (uint32_t)c);
    }
    std::sort(kv.begin(), kv.end());
    for (size_t i = 0; i < ncells; ++i) perm_out[i] = std::get<1>(kv[i]);
    return CMG_SFC_MINNODE;
}

/* ------------------------------------------------------------------ */
/* Coordinate grouping: nodes sharing an exact (x,y,z) form one group.  */
/* Deterministic: lexicographic (x,y,z) with the original node index as */
/* tie-break, so compress and decompress derive identical group ids.    */
/* ------------------------------------------------------------------ */
/* Per-axis lattice step for tolerant grouping. Returned as 1/h so the hot loop
 * multiplies; h == 0 (degenerate axis) yields invh == 0, collapsing that axis.
 * The bbox is always reduced on the host, on both backends, so compress and
 * decompress quantize against bit-identical bounds. */
void cmg_coord_quant_scale(const double *X, const double *Y, const double *Z,
                           size_t nnodes, double tol,
                           double *omin, double *oinvh)
{
    double lo[3] = {X[0], Y[0], Z[0]}, hi[3] = {X[0], Y[0], Z[0]};
    const double *ax[3] = {X, Y, Z};
    for (int d = 0; d < 3; ++d)
        for (size_t i = 1; i < nnodes; ++i) {
            const double v = ax[d][i];
            if (v < lo[d]) lo[d] = v;
            if (v > hi[d]) hi[d] = v;
        }
    for (int d = 0; d < 3; ++d) {
        const double h = (hi[d] - lo[d]) * tol;
        omin[d]  = lo[d];
        oinvh[d] = (h > 0.0) ? 1.0 / h : 0.0;
    }
}

size_t cmg_build_coord_groups(cmg_backend_t be,
                              const double *X, const double *Y, const double *Z,
                              size_t nnodes, double tol, uint32_t *gid_out,
                              double *gx, double *gy, double *gz)
{
    if (nnodes == 0) return 0;

    /* Quantized comparison keys. With tol <= 0 the key IS the coordinate, so
     * both modes run the identical sort/scan below. */
    const bool quant = (tol > 0.0);

#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP && cmg_hip_available())
        /* cmg_hip_build_coord_groups computes its own bbox; doing it here too
         * would be a wasted host pass over all coords on the GPU path. */
        return cmg_hip_build_coord_groups(X, Y, Z, nnodes, tol, gid_out, gx, gy, gz);
#endif
    (void)be;

    /* CPU path only: bbox-based quantization scale (host reduction). */
    double qmin[3] = {0,0,0}, qinvh[3] = {0,0,0};
    if (quant) cmg_coord_quant_scale(X, Y, Z, nnodes, tol, qmin, qinvh);

    auto key = [&](int d, const double *A, uint32_t i) -> double {
        return quant ? std::floor((A[i] - qmin[d]) * qinvh[d] + 0.5) : A[i];
    };

    std::vector<uint32_t> ord(nnodes);
    for (size_t i = 0; i < nnodes; ++i) ord[i] = (uint32_t)i;
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t c) {
        const double xa = key(0,X,a), xc = key(0,X,c); if (xa != xc) return xa < xc;
        const double ya = key(1,Y,a), yc = key(1,Y,c); if (ya != yc) return ya < yc;
        const double za = key(2,Z,a), zc = key(2,Z,c); if (za != zc) return za < zc;
        return a < c;                       /* deterministic tie-break */
    });

    size_t g = 0;
    for (size_t i = 0; i < nnodes; ++i)
    {
        const uint32_t n = ord[i];
        if (i > 0)
        {
            const uint32_t p = ord[i - 1];
            if (key(0,X,n) != key(0,X,p) || key(1,Y,n) != key(1,Y,p) ||
                key(2,Z,n) != key(2,Z,p)) ++g;
        }
        gid_out[n] = (uint32_t)g;
        /* representative = first node of the group in sort order. Under tolerant
         * grouping the members differ by round-off, so any member will do. */
        if (gx) { gx[g] = X[n]; gy[g] = Y[n]; gz[g] = Z[n]; }
    }
    return g + 1;
}

/* Fused device-resident group_centroid compress. Only implemented on HIP; on
 * the CPU backend it returns 0 so the operator uses the staged path. */
int cmg_group_compress_f32(cmg_backend_t be, size_t blockId, const float *field,
    const uint32_t *gid, const uint32_t *gperm, const uint32_t *nperm,
    size_t nnodes, size_t ngroups, double tol_resi, int adaptive,
    float *coarseSfc, double *rmin, double *rmax, double *rsumsq, int *drop, uint32_t *quantC,
    void **d_coarse_out, void **d_q_out, uint32_t *maxcode)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP && cmg_hip_available()) {
        cmg_hip_group_compress_f32(blockId, field, gid, gperm, nperm, nnodes, ngroups,
                                   tol_resi, adaptive, coarseSfc, rmin, rmax, rsumsq,
                                   drop, quantC, d_coarse_out, d_q_out, maxcode);
        return 1;
    }
#endif
    (void)be; (void)blockId; (void)field; (void)gid; (void)gperm; (void)nperm;
    (void)nnodes; (void)ngroups; (void)tol_resi; (void)adaptive; (void)coarseSfc;
    (void)rmin; (void)rmax; (void)rsumsq; (void)drop; (void)quantC;
    (void)d_coarse_out; (void)d_q_out; (void)maxcode;
    return 0;
}
int cmg_group_compress_f64(cmg_backend_t be, size_t blockId, const double *field,
    const uint32_t *gid, const uint32_t *gperm, const uint32_t *nperm,
    size_t nnodes, size_t ngroups, double tol_resi, int adaptive,
    double *coarseSfc, double *rmin, double *rmax, double *rsumsq, int *drop, uint32_t *quantC,
    void **d_coarse_out, void **d_q_out, uint32_t *maxcode)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP && cmg_hip_available()) {
        cmg_hip_group_compress_f64(blockId, field, gid, gperm, nperm, nnodes, ngroups,
                                   tol_resi, adaptive, coarseSfc, rmin, rmax, rsumsq,
                                   drop, quantC, d_coarse_out, d_q_out, maxcode);
        return 1;
    }
#endif
    (void)be; (void)blockId; (void)field; (void)gid; (void)gperm; (void)nperm;
    (void)nnodes; (void)ngroups; (void)tol_resi; (void)adaptive; (void)coarseSfc;
    (void)rmin; (void)rmax; (void)rsumsq; (void)drop; (void)quantC;
    (void)d_coarse_out; (void)d_q_out; (void)maxcode;
    return 0;
}
#ifndef CMG_HAVE_HIP
extern "C" void cmg_hip_free(void *) {}   /* CPU build: no device memory */
extern "C" void cmg_hip_d2h(void *, const void *, size_t) {}
#endif

/* Group means. Accumulated in double for both types so the coarse value is
 * independent of the storage precision (and of accumulation order, since each
 * group is summed sequentially). */
void cmg_group_mean_f32(cmg_backend_t be, const float *f, const uint32_t *gid,
                        size_t nnodes, size_t ngroups, float *out_mean)
{
    (void)be;
    std::vector<double>   acc(ngroups, 0.0);
    std::vector<uint32_t> cnt(ngroups, 0u);
    for (size_t n = 0; n < nnodes; ++n) { acc[gid[n]] += (double)f[n]; cnt[gid[n]] += 1u; }
    for (size_t g = 0; g < ngroups; ++g)
        out_mean[g] = (float)(cnt[g] ? acc[g] / (double)cnt[g] : 0.0);
}
void cmg_group_mean_f64(cmg_backend_t be, const double *f, const uint32_t *gid,
                        size_t nnodes, size_t ngroups, double *out_mean)
{
    (void)be;
    std::vector<double>   acc(ngroups, 0.0);
    std::vector<uint32_t> cnt(ngroups, 0u);
    for (size_t n = 0; n < nnodes; ++n) { acc[gid[n]] += f[n]; cnt[gid[n]] += 1u; }
    for (size_t g = 0; g < ngroups; ++g)
        out_mean[g] = cnt[g] ? acc[g] / (double)cnt[g] : 0.0;
}

void cmg_group_bcast_sub_f32(const float *mean, const uint32_t *gid, size_t nnodes,
                             const float *f, float *out_resid)
{ for (size_t n = 0; n < nnodes; ++n) out_resid[n] = f[n] - mean[gid[n]]; }
void cmg_group_bcast_sub_f64(const double *mean, const uint32_t *gid, size_t nnodes,
                             const double *f, double *out_resid)
{ for (size_t n = 0; n < nnodes; ++n) out_resid[n] = f[n] - mean[gid[n]]; }

void cmg_group_bcast_add_f32(const float *mean, const uint32_t *gid, size_t nnodes, float *inout)
{ for (size_t n = 0; n < nnodes; ++n) inout[n] += mean[gid[n]]; }
void cmg_group_bcast_add_f64(const double *mean, const uint32_t *gid, size_t nnodes, double *inout)
{ for (size_t n = 0; n < nnodes; ++n) inout[n] += mean[gid[n]]; }

/* Device-resident decompress recombine (undo group SFC + dequant + undo nodal
 * SFC + broadcast-add, all on device). Returns 1 when the HIP backend handled
 * it; 0 on the CPU backend so the caller runs the staged host tail. */
int cmg_group_recombine_f32(cmg_backend_t be, const float *coarseSfc, const uint32_t *gperm,
                            const uint32_t *quant, const uint32_t *gid, const uint32_t *nperm,
                            size_t nnodes, size_t ngroups, double tolR, float *out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP && cmg_hip_available()) {
        cmg_hip_group_recombine_f32(coarseSfc, gperm, quant, gid, nperm, nnodes, ngroups, tolR, out);
        return 1;
    }
#endif
    (void)be; (void)coarseSfc; (void)gperm; (void)quant; (void)gid; (void)nperm;
    (void)nnodes; (void)ngroups; (void)tolR; (void)out;
    return 0;
}
int cmg_group_recombine_f64(cmg_backend_t be, const double *coarseSfc, const uint32_t *gperm,
                            const uint32_t *quant, const uint32_t *gid, const uint32_t *nperm,
                            size_t nnodes, size_t ngroups, double tolR, double *out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP && cmg_hip_available()) {
        cmg_hip_group_recombine_f64(coarseSfc, gperm, quant, gid, nperm, nnodes, ngroups, tolR, out);
        return 1;
    }
#endif
    (void)be; (void)coarseSfc; (void)gperm; (void)quant; (void)gid; (void)nperm;
    (void)nnodes; (void)ngroups; (void)tolR; (void)out;
    return 0;
}

void cmg_perm_forward_f32(const float  *s, float  *d, const uint32_t *p, size_t n)
{ for (size_t i = 0; i < n; ++i) d[i] = s[p[i]]; }
void cmg_perm_forward_f64(const double *s, double *d, const uint32_t *p, size_t n)
{ for (size_t i = 0; i < n; ++i) d[i] = s[p[i]]; }
void cmg_perm_inverse_f32(const float  *s, float  *d, const uint32_t *p, size_t n)
{ for (size_t i = 0; i < n; ++i) d[p[i]] = s[i]; }
void cmg_perm_inverse_f64(const double *s, double *d, const uint32_t *p, size_t n)
{ for (size_t i = 0; i < n; ++i) d[p[i]] = s[i]; }

/* ------------------------------------------------------------------ */
/* NODAL SFC permutation: order the nodes themselves along the curve.  */
/* ------------------------------------------------------------------ */
cmg_sfc_t cmg_build_sfc_perm_nodes(cmg_backend_t be,
                                   const double *X, const double *Y, const double *Z,
                                   size_t nnodes, cmg_sfc_t requested, uint32_t *perm_out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        return cmg_hip_build_sfc_perm_nodes(X, Y, Z, nnodes, requested, perm_out);
    }
#else
    (void)be;
#endif
    const bool wantCoord = (requested == CMG_SFC_MORTON || requested == CMG_SFC_HILBERT);
    if (!(wantCoord && X && Y && Z) || nnodes == 0)
    {
        for (size_t i = 0; i < nnodes; ++i) perm_out[i] = (uint32_t)i;
        return CMG_SFC_NONE;
    }
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

    std::vector<std::pair<uint64_t, uint32_t>> kv(nnodes);
    for (size_t i = 0; i < nnodes; ++i)
    {
        double qxd = (X[i] - xmin) * sx;
        double qyd = (Y[i] - ymin) * sy;
        double qzd = (Z[i] - zmin) * sz;
        if (qxd < 0.0) qxd = 0.0; if (qxd > double(maxQ)) qxd = maxQ;
        if (qyd < 0.0) qyd = 0.0; if (qyd > double(maxQ)) qyd = maxQ;
        if (qzd < 0.0) qzd = 0.0; if (qzd > double(maxQ)) qzd = maxQ;
        const uint32_t qx = (uint32_t)qxd, qy = (uint32_t)qyd, qz = (uint32_t)qzd;
        kv[i] = { (requested == CMG_SFC_HILBERT) ? hilbert3d(qx, qy, qz)
                                                 : morton3d(qx, qy, qz),
                  (uint32_t)i };
    }
    /* ties (coincident nodes) break on original index -> deterministic */
    std::sort(kv.begin(), kv.end(),
              [](const auto &a, const auto &b) {
                  return (a.first != b.first) ? (a.first < b.first) : (a.second < b.second);
              });
    for (size_t i = 0; i < nnodes; ++i) perm_out[i] = kv[i].second;
    return requested;
}


void cmg_quantize_zigzag_f32(cmg_backend_t be, const float *in, size_t n,
                              double tol, uint32_t *out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) { cmg_hip_quantize_zigzag_f32(in, n, tol, out); return; }
#else
    (void)be;
#endif
    const double inv_q = 1.0 / (2.0 * tol);
    for (size_t i = 0; i < n; ++i) out[i] = zigzag_quantize_double(in[i], inv_q);
}
void cmg_quantize_zigzag_f64(cmg_backend_t be, const double *in, size_t n,
                              double tol, uint32_t *out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) { cmg_hip_quantize_zigzag_f64(in, n, tol, out); return; }
#else
    (void)be;
#endif
    const double inv_q = 1.0 / (2.0 * tol);
    for (size_t i = 0; i < n; ++i) out[i] = zigzag_quantize_double(in[i], inv_q);
}
void cmg_dequantize_zigzag_f32(cmg_backend_t be, const uint32_t *in, size_t n,
                                double tol, float *out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) { cmg_hip_dequantize_zigzag_f32(in, n, tol, out); return; }
#else
    (void)be;
#endif
    const double q = 2.0 * tol;
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(zigzag_decode(in[i]) * q);
}
void cmg_dequantize_zigzag_f64(cmg_backend_t be, const uint32_t *in, size_t n,
                                double tol, double *out)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) { cmg_hip_dequantize_zigzag_f64(in, n, tol, out); return; }
#else
    (void)be;
#endif
    const double q = 2.0 * tol;
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(zigzag_decode(in[i])) * q;
}

void cmg_reduce_stats_f32(cmg_backend_t be, const float *in, size_t n,
                          double *out_min, double *out_max, double *out_sumsq)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        cmg_hip_reduce_stats_f32(in, n, out_min, out_max, out_sumsq);
        return;
    }
#else
    (void)be;
#endif
    if (n == 0) { *out_min = 0.0; *out_max = 0.0; *out_sumsq = 0.0; return; }
    double vmin = static_cast<double>(in[0]), vmax = vmin, ssq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(in[i]);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        ssq += v * v;
    }
    *out_min = vmin; *out_max = vmax; *out_sumsq = ssq;
}
void cmg_reduce_stats_f64(cmg_backend_t be, const double *in, size_t n,
                          double *out_min, double *out_max, double *out_sumsq)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        cmg_hip_reduce_stats_f64(in, n, out_min, out_max, out_sumsq);
        return;
    }
#else
    (void)be;
#endif
    if (n == 0) { *out_min = 0.0; *out_max = 0.0; *out_sumsq = 0.0; return; }
    double vmin = in[0], vmax = in[0], ssq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = in[i];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        ssq += v * v;
    }
    *out_min = vmin; *out_max = vmax; *out_sumsq = ssq;
}

/* HIP availability: HIP TU provides the strong definition; otherwise stub. */
#ifndef CMG_HAVE_HIP
int cmg_hip_available(void) { return 0; }
#endif

} // extern "C"
