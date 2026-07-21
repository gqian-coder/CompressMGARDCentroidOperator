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

template <typename T>
void split_impl(const T *u, const int64_t *conn,
                size_t ncells, size_t npc, size_t nnodes,
                T *cell_avg, T *residual)
{
    /* 1) per-cell average over unique nodes only (handles padded conns). */
    for (size_t c = 0; c < ncells; ++c)
    {
        double acc = 0.0;
        size_t ucount = 0;
        const int64_t *cc = conn + c * npc;
        for (size_t k = 0; k < npc; ++k)
        {
            bool dup = false;
            for (size_t j = 0; j < k; ++j)
                if (cc[j] == cc[k]) { dup = true; break; }
            if (!dup) { acc += static_cast<double>(u[cc[k]]); ++ucount; }
        }
        cell_avg[c] = static_cast<T>(ucount ? acc / static_cast<double>(ucount) : 0.0);
    }

    /* 2) bar_u(n) = mean over incident cells. */
    std::vector<double>   sum(nnodes, 0.0);
    std::vector<uint32_t> cnt(nnodes, 0u);
    for (size_t c = 0; c < ncells; ++c)
    {
        const int64_t *cc = conn + c * npc;
        const double   a  = static_cast<double>(cell_avg[c]);
        for (size_t k = 0; k < npc; ++k)
        {
            bool dup = false;
            for (size_t j = 0; j < k; ++j)
                if (cc[j] == cc[k]) { dup = true; break; }
            if (!dup) { sum[cc[k]] += a; cnt[cc[k]] += 1u; }
        }
    }

    /* 3) residual. */
    for (size_t n = 0; n < nnodes; ++n)
    {
        const double bar = cnt[n] ? sum[n] / static_cast<double>(cnt[n]) : 0.0;
        residual[n] = static_cast<T>(static_cast<double>(u[n]) - bar);
    }
}

template <typename T>
void recombine_impl(const T *avg, const int64_t *conn,
                    size_t ncells, size_t npc, size_t nnodes,
                    T *inout)
{
    std::vector<double>   sum(nnodes, 0.0);
    std::vector<uint32_t> cnt(nnodes, 0u);
    for (size_t c = 0; c < ncells; ++c)
    {
        const int64_t *cc = conn + c * npc;
        const double   a  = static_cast<double>(avg[c]);
        for (size_t k = 0; k < npc; ++k)
        {
            bool dup = false;
            for (size_t j = 0; j < k; ++j)
                if (cc[j] == cc[k]) { dup = true; break; }
            if (!dup) { sum[cc[k]] += a; cnt[cc[k]] += 1u; }
        }
    }
    for (size_t n = 0; n < nnodes; ++n)
    {
        const double bar = cnt[n] ? sum[n] / static_cast<double>(cnt[n]) : 0.0;
        inout[n] = static_cast<T>(bar + static_cast<double>(inout[n]));
    }
}

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
void cmg_hip_centroid_split_f32(const float *, const int64_t *, size_t, size_t, size_t,
                                float *, float *);
void cmg_hip_centroid_split_f64(const double *, const int64_t *, size_t, size_t, size_t,
                                double *, double *);
void cmg_hip_centroid_recombine_f32(const float *, const int64_t *, size_t, size_t, size_t,
                                    float *);
void cmg_hip_centroid_recombine_f64(const double *, const int64_t *, size_t, size_t, size_t,
                                    double *);
void cmg_hip_quantize_zigzag_f32(const float *, size_t, double, uint32_t *);
void cmg_hip_quantize_zigzag_f64(const double *, size_t, double, uint32_t *);
void cmg_hip_dequantize_zigzag_f32(const uint32_t *, size_t, double, float *);
void cmg_hip_dequantize_zigzag_f64(const uint32_t *, size_t, double, double *);
cmg_sfc_t cmg_hip_build_sfc_perm(const int64_t *, size_t, size_t,
                                 const double *, const double *, const double *, size_t,
                                 cmg_sfc_t, uint32_t *);
cmg_sfc_t cmg_hip_build_sfc_perm_nodes(const double *, const double *, const double *,
                                       size_t, cmg_sfc_t, uint32_t *);
void cmg_hip_reduce_stats_f32(const float *, size_t, double *, double *, double *);
void cmg_hip_reduce_stats_f64(const double *, size_t, double *, double *, double *);
#endif

void cmg_centroid_split_f32(cmg_backend_t be, const float *u, const int64_t *conn,
                            size_t ncells, size_t npc, size_t nnodes,
                            float *out_avg, float *out_resi)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        cmg_hip_centroid_split_f32(u, conn, ncells, npc, nnodes, out_avg, out_resi);
        return;
    }
#else
    (void)be;
#endif
    split_impl<float>(u, conn, ncells, npc, nnodes, out_avg, out_resi);
}
void cmg_centroid_split_f64(cmg_backend_t be, const double *u, const int64_t *conn,
                            size_t ncells, size_t npc, size_t nnodes,
                            double *out_avg, double *out_resi)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        cmg_hip_centroid_split_f64(u, conn, ncells, npc, nnodes, out_avg, out_resi);
        return;
    }
#else
    (void)be;
#endif
    split_impl<double>(u, conn, ncells, npc, nnodes, out_avg, out_resi);
}

void cmg_centroid_recombine_f32(cmg_backend_t be, const float *avg, const int64_t *conn,
                                size_t ncells, size_t npc, size_t nnodes,
                                float *inout)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        cmg_hip_centroid_recombine_f32(avg, conn, ncells, npc, nnodes, inout);
        return;
    }
#else
    (void)be;
#endif
    recombine_impl<float>(avg, conn, ncells, npc, nnodes, inout);
}
void cmg_centroid_recombine_f64(cmg_backend_t be, const double *avg, const int64_t *conn,
                                size_t ncells, size_t npc, size_t nnodes,
                                double *inout)
{
#ifdef CMG_HAVE_HIP
    if (be == CMG_BACKEND_HIP) {
        cmg_hip_centroid_recombine_f64(avg, conn, ncells, npc, nnodes, inout);
        return;
    }
#else
    (void)be;
#endif
    recombine_impl<double>(avg, conn, ncells, npc, nnodes, inout);
}

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
