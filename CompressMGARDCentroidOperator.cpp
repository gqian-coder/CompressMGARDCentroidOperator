/*
 * CompressMGARDCentroidOperator.cpp — V5 unified implementation.
 *
 * See CompressMGARDCentroidOperator.h for the bitstream layout. This is the
 * only writer; the legacy V1..V4 read paths have been retired (drop after
 * V5 stable per Plan.md). The active path always emits V5 and InverseOperate
 * only accepts bufferVersion == 5.
 *
 * Backend selection:
 *   - CENTROID_DEVICE=cpu (or unset on CPU build)       → CMG_BACKEND_CPU
 *   - CENTROID_DEVICE=gpu / hip                         → CMG_BACKEND_HIP
 *   - falls back to MGARD_X_DEVICE_TYPE (HIP|SERIAL)    for back-compat.
 *
 * The MGARD-X Huffman+ZSTD payload is bit-identical across the SERIAL and
 * HIP backends for the same uint32 input (verified in tests/). The CPU/GPU
 * choice therefore only changes where the centroid split + SFC + quantize
 * kernels run; the lossless step uses the SERIAL backend on either path,
 * which avoids an extra H2D round-trip on the GPU path with no CR change.
 */

#include "CompressMGARDCentroidOperator.h"
#include "MGARDXLossless.h"

#include <mgard/MGARDConfig.hpp>
#include <mgard/compress_x.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace adios2
{
namespace plugin
{

namespace
{

constexpr uint8_t kBufferVersion = 5;
constexpr uint8_t kResMarker_MGARD = 0;
constexpr uint8_t kResMarker_MGARDX_Huff = 1;

/* First byte of the 4-byte per-operator header. Matches the value the
 * v1 (core::Operator-derived) interface wrote via MakeCommonHeader
 * (OperatorType::PLUGIN_INTERFACE), so existing V5 files stay readable. */
constexpr uint8_t kOperatorTypeByte = 53;

/* ------ env-var helpers --------------------------------------------- */

bool EnvIsOff(const std::string &s)
{
    auto ls = s;
    for (auto &c : ls) c = (char)std::tolower(c);
    return (ls == "off" || ls == "0" || ls == "false" || ls == "no");
}

cmg_backend_t ChooseBackend()
{
    const char *e = std::getenv("CENTROID_DEVICE");
    if (e)
    {
        std::string s(e);
        for (auto &c : s) c = (char)std::tolower(c);
        if (s == "gpu" || s == "hip" || s == "cuda") return CMG_BACKEND_HIP;
        return CMG_BACKEND_CPU;
    }
    if (const char *e2 = std::getenv("MGARD_X_DEVICE_TYPE"))
    {
        std::string s(e2);
        if (s == "HIP" || s == "CUDA") return CMG_BACKEND_HIP;
    }
#ifdef CMG_HAVE_HIP
    return CMG_BACKEND_HIP;
#else
    return CMG_BACKEND_CPU;
#endif
}

cmg_sfc_t ParseSFC(const Params &params)
{
    auto lc = [](std::string s) {
        for (auto &c : s) c = (char)std::tolower(c); return s;
    };
    auto pick = [&](const std::string &v) -> cmg_sfc_t {
        if (v == "morton" || v == "z")        return CMG_SFC_MORTON;
        if (v == "hilbert" || v == "h")       return CMG_SFC_HILBERT;
        if (v == "minnode" || v == "min_node") return CMG_SFC_MINNODE;
        return CMG_SFC_HILBERT;
    };
    auto it = params.find("sfc_type");
    if (it != params.end()) return pick(lc(it->second));
    if (const char *e = std::getenv("CENTROID_SFC_TYPE")) return pick(lc(std::string(e)));
    return CMG_SFC_HILBERT;
}

bool IsSFCEnabled(const Params &params)
{
    auto it = params.find("cell_sfc");
    if (it != params.end() && EnvIsOff(it->second)) return false;
    if (const char *e = std::getenv("CENTROID_SFC"))
        if (EnvIsOff(std::string(e))) return false;
    return true;
}

/* Compression path selector.
 *
 *   "centroid" (default) : cell-average + nodal-residual split; the cell
 *                          averages are SFC-reordered and MGARD-compressed,
 *                          the residual is quantized + entropy coded.
 *   "reorder_mgard"      : gather the nodal field along a nodal SFC (Hilbert)
 *                          and hand the whole field to MGARD's 1-D multilevel
 *                          transform. No split, so no cell-average stream and
 *                          no N_cell expansion -- MGARD's own critically
 *                          sampled hierarchy performs the coarse/detail
 *                          separation instead.
 *
 * Select with the "method" parameter or CENTROID_METHOD env var. */
std::string MethodName(const Params &params)
{
    std::string v;
    auto it = params.find("method");
    if (it != params.end()) v = it->second;
    else if (const char *e = std::getenv("CENTROID_METHOD")) v = e;
    for (auto &c : v) c = (char)std::tolower((unsigned char)c);
    return v;
}

bool UseReorderMgardPath(const Params &params)
{
    const std::string v = MethodName(params);
    return v == "reorder_mgard" || v == "sfc_mgard" || v == "reorder";
}

/* method=group_centroid : coarse stream is one value per DISTINCT COORDINATE
 * LOCATION (this mesh stores each location 12.7x-23.3x over, so the coarse
 * stream is ~6.6% of nnodes), SFC-reordered over the group locations and
 * MGARD-compressed; the per-node residual u - coarse[group(n)] is nodal-SFC
 * reordered, quantized and entropy coded. The grouping is derived from the
 * coordinates alone, so it is cached per block and stored in zero bytes. */
bool UseGroupCentroidPath(const Params &params)
{
    const std::string v = MethodName(params);
    return v == "group_centroid" || v == "group" || v == "location";
}

/* Nodal SFC reorder of the RESIDUAL before entropy coding.
 *
 * Note this can only ever help the ZSTD/LZ stage: a Huffman code's length is
 * fixed by the symbol frequencies, which a permutation cannot change. Measured
 * benefit in the centroid path was ~1.5% (72 blocks), against a per-field
 * gather on write plus a scatter on every read. Left ON by default to preserve
 * behaviour, but switchable so the trade can be measured per dataset:
 *   residual_sfc=off / CENTROID_RESID_SFC=0 */
bool IsResidualSFCEnabled(const Params &params)
{
    auto it = params.find("residual_sfc");
    if (it != params.end()) return !EnvIsOff(it->second);
    if (const char *e = std::getenv("CENTROID_RESID_SFC")) return !EnvIsOff(std::string(e));
    return true;
}

/* Adaptive no-residual rule (group_centroid).
 *
 * The residual is only worth storing if it carries information above the
 * quantization floor. When the coarse (group-mean) reconstruction is already
 * inside the residual's slice of the error budget, the quantizer maps every
 * residual to zero and the encoded stream is pure overhead -- measured at
 * eb=1e-2 for P_aver: 38,572 B (49% of the output) spent encoding an all-zero
 * array, because rms(residual)=13 against a quantum of 2350.
 *
 * So: compute rms(u - coarse[group]) and drop the residual when it is within
 * tol_resi. The rms comes from cmg_reduce_stats_*, which runs on the GPU when
 * the backend is HIP. Disable with adaptive_residual=off /
 * CENTROID_ADAPTIVE_RESID=0. */
/* Two criteria, both fed by the same cmg_reduce_stats_* call:
 *
 *   "linf" (default) : drop when max|resid| <= tol_resi. Every residual then
 *                      quantizes to zero, so the stored stream carries no
 *                      information and dropping it is PROVABLY output-identical
 *                      -- a free win, no accuracy change.
 *   "rms"            : drop when rms(resid) <= tol_resi. More aggressive: stays
 *                      inside the error budget but SPENDS the residual's whole
 *                      allocation, so delivered accuracy degrades toward the
 *                      bound (measured U_aver eb=1e-2: relL2 2.84e-3 -> 6.57e-3
 *                      for CR 73.6 -> 128.2).
 *   "off"            : never drop.
 *
 * adaptive_residual= linf | rms | off   /  CENTROID_ADAPTIVE_RESID=... */
enum class AdaptResid { Off, Linf, Rms };
AdaptResid AdaptiveResidualMode(const Params &params)
{
    std::string v;
    auto it = params.find("adaptive_residual");
    if (it != params.end()) v = it->second;
    else if (const char *e = std::getenv("CENTROID_ADAPTIVE_RESID")) v = e;
    for (auto &c : v) c = (char)std::tolower((unsigned char)c);
    if (v.empty()) return AdaptResid::Linf;          /* default: the free rule */
    if (v == "rms") return AdaptResid::Rms;
    if (EnvIsOff(v)) return AdaptResid::Off;
    return AdaptResid::Linf;                          /* "linf", "1", "on", ... */
}

/* Residual backend selector: "huffman" (default) = quantize -> MGARD-X
 * Huffman/ZSTD; "mgard" = MGARD-lossy directly on the raw residual. Settable
 * via the residual_method parameter or MGARD_RESIDUAL_METHOD env var.
 * MGARD-lossy exploits the residual's spatial smoothness (useful once the
 * nodal SFC reorder has made the residual spatially local), whereas the
 * entropy coder only sees the value distribution / sequential runs. */
bool UseMgardResidual(const Params &params)
{
    auto pick = [](std::string v) {
        for (auto &c : v) c = (char)std::tolower((unsigned char)c);
        return v == "mgard";
    };
    auto it = params.find("residual_method");
    if (it != params.end()) return pick(it->second);
    if (const char *e = std::getenv("MGARD_RESIDUAL_METHOD")) return pick(std::string(e));
    return false;
}

/* ------ per-process mesh cache (connectivity + coords) -------------- */

std::mutex                                                  g_meshMutex;
std::string                                                 g_meshFileName;
std::string                                                 g_connVarName;
adios2::ADIOS                                               g_ad;
adios2::IO                                                  g_io_mesh;
adios2::Engine                                              g_reader_mesh;
std::map<size_t, std::vector<int64_t>>                      g_connCache;
std::map<std::pair<int, size_t>, std::vector<double>>       g_coordsCache;
/* SFC permutation cache: keyed by block id. The mesh (connectivity + coords) is
 * time- and variable-invariant within a process, so the permutation is the same
 * for every variable/timestep of a block; caching it avoids rebuilding the
 * Hilbert sort on every (de)compress call. */
std::map<size_t, std::vector<uint32_t>>                     g_permCache;
std::map<size_t, cmg_sfc_t>                                 g_permMode;
/* Nodal SFC permutation + the connectivity remapped through it. Same caching
 * rationale as g_permCache: geometry-derived, so identical for every variable
 * and timestep of a block. */
std::map<size_t, std::vector<uint32_t>>                     g_nodePermCache;
std::map<size_t, cmg_sfc_t>                                 g_nodePermMode;
/* Coordinate-group cache (group_centroid path): gid per node, the group count,
 * and the SFC permutation over the group locations. All geometry-derived, so
 * identical for every variable/timestep of a block and never stored. */
struct CoordGroups
{
    std::vector<uint32_t> gid;        /* [nnodes] -> group id                 */
    size_t                ngroups = 0;
    std::vector<uint32_t> gperm;      /* [ngroups] SFC order over groups      */
    cmg_sfc_t             gmode = CMG_SFC_NONE;
};
std::map<size_t, CoordGroups>                               g_groupCache;

const std::vector<int64_t> &LoadConnectivity(const std::string &meshFile,
                                             const std::string &connVar, size_t blockId)
{
    std::lock_guard<std::mutex> lck(g_meshMutex);
    if (!g_reader_mesh)
    {
        g_meshFileName = meshFile;
        g_connVarName  = connVar;
        const std::string ioName = "CentroidOperator_MeshIn";
        try { g_io_mesh = g_ad.AtIO(ioName); }
        catch (...) {
            g_io_mesh = g_ad.DeclareIO(ioName);
        }
        g_reader_mesh = g_io_mesh.Open(meshFile, adios2::Mode::ReadRandomAccess);
    }
    else if (g_meshFileName != meshFile || g_connVarName != connVar)
    {
        throw std::invalid_argument(
            "[CompressMGARDCentroidOperator::LoadConnectivity] "
            "Mesh-file/connectivity-variable cannot change between calls in the same process");
    }
    auto it = g_connCache.find(blockId);
    if (it != g_connCache.end()) return it->second;
    auto vConn = g_io_mesh.InquireVariable<int64_t>(connVar);
    if (!vConn)
        throw std::invalid_argument(
            "[CompressMGARDCentroidOperator::LoadConnectivity] Connectivity variable '" + connVar +
            "' not found in " + meshFile);
    vConn.SetBlockSelection(blockId);
    std::vector<int64_t> conn;
    g_reader_mesh.Get<int64_t>(vConn, conn, adios2::Mode::Sync);
    auto &slot = g_connCache[blockId];
    slot = std::move(conn);
    return slot;
}

const std::vector<double> *TryLoadCoordsAxis(const std::string &meshFile,
                                             const std::string &coordVar,
                                             int axisIdx, size_t blockId)
{
    std::lock_guard<std::mutex> lck(g_meshMutex);
    auto key = std::make_pair(axisIdx, blockId);
    auto it = g_coordsCache.find(key);
    if (it != g_coordsCache.end()) return &it->second;
    if (!g_reader_mesh || g_meshFileName != meshFile) return nullptr;

    auto vDbl = g_io_mesh.InquireVariable<double>(coordVar);
    if (vDbl)
    {
        vDbl.SetBlockSelection(blockId);
        std::vector<double> buf;
        try { g_reader_mesh.Get<double>(vDbl, buf, adios2::Mode::Sync); }
        catch (...) { return nullptr; }
        auto &slot = g_coordsCache[key];
        slot = std::move(buf);
        return &slot;
    }
    auto vFlt = g_io_mesh.InquireVariable<float>(coordVar);
    if (vFlt)
    {
        vFlt.SetBlockSelection(blockId);
        std::vector<float> bufF;
        try { g_reader_mesh.Get<float>(vFlt, bufF, adios2::Mode::Sync); }
        catch (...) { return nullptr; }
        std::vector<double> buf(bufF.begin(), bufF.end());
        auto &slot = g_coordsCache[key];
        slot = std::move(buf);
        return &slot;
    }
    return nullptr;
}

bool ResolveAndLoadCoords(const Params &params, const std::string &meshFile,
                          const std::string &connVar, size_t blockId,
                          const double *&outX, const double *&outY, const double *&outZ)
{
    outX = outY = outZ = nullptr;
    std::string prefix;
    auto it = params.find("coordinates_prefix");
    if (it != params.end()) prefix = it->second;
    if (prefix.empty())
        if (const char *e = std::getenv("CENTROID_COORD_PREFIX")) prefix = e;
    if (prefix.empty())
    {
        const std::string token = "/Elem/";
        auto pos = connVar.rfind(token);
        if (pos != std::string::npos)
            prefix = connVar.substr(0, pos) + "/GridCoordinates/Coordinate";
    }
    if (prefix.empty()) return false;
    auto *vX = TryLoadCoordsAxis(meshFile, prefix + "X", 0, blockId);
    auto *vY = TryLoadCoordsAxis(meshFile, prefix + "Y", 1, blockId);
    auto *vZ = TryLoadCoordsAxis(meshFile, prefix + "Z", 2, blockId);
    if (!vX || !vY || !vZ) return false;
    outX = vX->data(); outY = vY->data(); outZ = vZ->data();
    return true;
}

/* Build, or fetch from the per-process cache, the SFC permutation for a block.
 * Cached by block id: connectivity + coordinates are invariant within a process,
 * so a block's permutation is identical across all variables and timesteps. This
 * turns O(vars * blocks) permutation builds into O(blocks). Used by both compress
 * (forward reorder) and decompress (inverse reorder). */
const std::vector<uint32_t> &GetOrBuildSFCPerm(cmg_backend_t be, size_t blockId,
                                               cmg_sfc_t requested, const int64_t *conn,
                                               size_t ncells, size_t npc, const double *cx,
                                               const double *cy, const double *cz, size_t nnodes,
                                               cmg_sfc_t &actualMode)
{
    std::lock_guard<std::mutex> lck(g_meshMutex);
    auto it = g_permCache.find(blockId);
    if (it != g_permCache.end())
    {
        actualMode = g_permMode[blockId];
        return it->second;
    }
    std::vector<uint32_t> perm(ncells);
    actualMode =
        cmg_build_sfc_perm(be, conn, ncells, npc, cx, cy, cz, nnodes, requested, perm.data());
    g_permMode[blockId] = actualMode;
    auto &slot = g_permCache[blockId];
    slot = std::move(perm);
    return slot;
}

/* ------ nodal SFC permutation (reorder BEFORE the centroid split) ---- */
/* Like the cell permutation, the nodal ordering depends only on the mesh
 * geometry, so it is identical for every variable and timestep of a block and
 * is built once per block. The remapped connectivity derived from it is cached
 * alongside. Neither is stored in the bitstream: both are recomputed from the
 * mesh coordinates at decompress time, so the reorder costs no extra bytes. */
const std::vector<uint32_t> &GetOrBuildNodeSFCPerm(cmg_backend_t be, size_t blockId,
                                                   cmg_sfc_t requested, const double *X,
                                                   const double *Y, const double *Z,
                                                   size_t nnodes, cmg_sfc_t &actualMode)
{
    std::lock_guard<std::mutex> lck(g_meshMutex);
    auto it = g_nodePermCache.find(blockId);
    if (it != g_nodePermCache.end())
    {
        actualMode = g_nodePermMode[blockId];
        return it->second;
    }
    std::vector<uint32_t> perm(nnodes);
    actualMode = cmg_build_sfc_perm_nodes(be, X, Y, Z, nnodes, requested, perm.data());
    g_nodePermMode[blockId] = actualMode;
    auto &slot = g_nodePermCache[blockId];
    slot = std::move(perm);
    return slot;
}


/* Build, or fetch from cache, the coordinate grouping for a block plus an SFC
 * permutation over the group locations. Geometry-only: identical for every
 * variable and timestep, and re-derived on read, so it costs no stored bytes. */
const CoordGroups &GetOrBuildCoordGroups(cmg_backend_t be, size_t blockId,
                                         cmg_sfc_t requested, const double *X,
                                         const double *Y, const double *Z, size_t nnodes)
{
    std::lock_guard<std::mutex> lck(g_meshMutex);
    auto it = g_groupCache.find(blockId);
    if (it != g_groupCache.end()) return it->second;

    CoordGroups cg;
    cg.gid.resize(nnodes);
    std::vector<double> gx(nnodes), gy(nnodes), gz(nnodes);   /* upper bound */
    cg.ngroups = cmg_build_coord_groups(be, X, Y, Z, nnodes, cg.gid.data(),
                                        gx.data(), gy.data(), gz.data());
    gx.resize(cg.ngroups); gy.resize(cg.ngroups); gz.resize(cg.ngroups);
    cg.gperm.resize(cg.ngroups);
    cg.gmode = cmg_build_sfc_perm_nodes(be, gx.data(), gy.data(), gz.data(), cg.ngroups,
                                        requested, cg.gperm.data());
    auto &slot = g_groupCache[blockId];
    slot = std::move(cg);
    return slot;
}

/* ------ MGARD compress helpers (cell averages) ---------------------- */

mgard_x::data_type ToMgardType(DataType t)
{
    if (t == DataType::Float)  return mgard_x::data_type::Float;
    if (t == DataType::Double) return mgard_x::data_type::Double;
    throw std::invalid_argument(
        "[CompressMGARDCentroidOperator::ToMgardType] Only float/double supported");
}
size_t TypeNBytes(DataType t)
{
    if (t == DataType::Float || t == DataType::Double) return GetDataTypeSize(t);
    throw std::invalid_argument(
        "[CompressMGARDCentroidOperator::TypeNBytes] Only float/double supported");
}

mgard_x::Config MakeMgardConfig(cmg_backend_t /*be*/)
{
    mgard_x::Config cfg;
    cfg.lossless = mgard_x::lossless_type::Huffman_Zstd;
    /* Default the internal MGARD to SERIAL even when the centroid device kernels
     * run on the GPU: MGARD here only compresses the small per-cell average
     * array, for which CPU MGARD (no H2D/D2H or GPU workspace-rebuild overhead)
     * is faster than HIP. Set MGARD_X_DEVICE_TYPE=HIP to override for a
     * fully-GPU pipeline. */
    cfg.dev_type = mgard_x::device_type::SERIAL;
    /* explicit override wins */
    if (const char *e = std::getenv("MGARD_X_DEVICE_TYPE"))
    {
        std::string s(e);
        if      (s == "SERIAL") cfg.dev_type = mgard_x::device_type::SERIAL;
        else if (s == "HIP")    cfg.dev_type = mgard_x::device_type::HIP;
        else if (s == "CUDA")   cfg.dev_type = mgard_x::device_type::CUDA;
        else if (s == "OPENMP") cfg.dev_type = mgard_x::device_type::OPENMP;
        else if (s == "AUTO")   cfg.dev_type = mgard_x::device_type::AUTO;
    }
    /* Huffman tuning: smaller dict_size concentrates the codebook on the
     * most-frequent symbols, keeping code lengths short. The GPU parallel
     * Huffman (GenerateCL kernel) can produce degenerate 60+ bit codewords
     * when the frequency histogram is nearly flat across 8192 bins (happens
     * at tight tolerances). A smaller dict forces more outlier-bypass and
     * avoids that degenerate case. */
    if (const char *e = std::getenv("MGARDX_HUFF_DICT"); e && *e)
    {
        try { cfg.huff_dict_size = static_cast<mgard_x::SIZE>(std::stol(e)); }
        catch (...) {}
    }
    if (const char *e = std::getenv("MGARDX_HUFF_BLOCK"); e && *e)
    {
        try { cfg.huff_block_size = static_cast<mgard_x::SIZE>(std::stol(e)); }
        catch (...) {}
    }
    return cfg;
}

} // anonymous namespace

/* ====================================================================== */

CompressMGARDCentroidOperator::CompressMGARDCentroidOperator(const Params &parameters)
: PluginOperatorInterface(parameters)
{
    // Single env var gating all debug output from the centroid operator.
    if (std::getenv("CENTROID_DEBUG")) m_Verbose = true;
    m_Backend = ChooseBackend();
    if (m_Backend == CMG_BACKEND_HIP)
    {
        if (!cmg_hip_available())
        {
            std::cerr << "[centroid] CENTROID_DEVICE=gpu requested but HIP backend not available; "
                    << "falling back to CPU.\n";
            m_Backend = CMG_BACKEND_CPU;
        }
        if (m_Verbose)
            std::cerr << "[centroid] CENTROID_DEVICE=gpu\n";
    }
    else
    {
        if (m_Verbose)
            std::cerr << "[centroid] CENTROID_DEVICE=cpu\n";
    }
}

CompressMGARDCentroidOperator::~CompressMGARDCentroidOperator()
{
    if (g_reader_mesh) g_reader_mesh.Close();
}

bool CompressMGARDCentroidOperator::IsDataTypeValid(const DataType type) const
{
    return (type == DataType::Float || type == DataType::Double);
}

size_t CompressMGARDCentroidOperator::GetEstimatedSize(const size_t ElemCount,
                                                       const size_t ElemSize,
                                                       const size_t, const size_t *) const
{
    /* MGARD cell-avg upper bound + uint32-quant residual upper bound + header */
    const size_t mgardWorst = ElemCount * ElemSize;
    const size_t lossWorst  = ElemCount * sizeof(uint32_t) * 4 + 4096;
    return mgardWorst + lossWorst + 1024;
}

void CompressMGARDCentroidOperator::AddExtraParameters(const Params &params)
{
    for (auto &it : params)
    {
        if      (it.first == "EngineName")   m_EngineName = it.second;
        else if (it.first == "VariableName") m_VariableName = it.second;
    }
}

/* ====================================================================== */

size_t CompressMGARDCentroidOperator::Operate(const char *dataIn, const Dims & /*blockStart*/,
                                              const Dims &blockCount, const DataType type,
                                              char *bufferOut)
{
    using namespace std::chrono;
    const auto t0 = high_resolution_clock::now();

    if (!IsDataTypeValid(type))
        throw std::invalid_argument(
            "[CompressMGARDCentroidOperator::Operate] Only float/double types are supported");

    /* ---- required params ---- */
    auto need = [&](const char *k) -> std::string {
        auto it = m_Parameters.find(k);
        if (it == m_Parameters.end())
            throw std::invalid_argument(
                std::string("[CompressMGARDCentroidOperator::Operate] missing parameter '") + k +
                "'");
        return it->second;
    };

    m_MeshFile         = need("meshfile");
    std::string connVar = need("connectivity_variable");
    m_NodesPerCell     = std::stoul(need("nodes_per_cell"));
    m_BlockId          = std::stoul(need("blockid"));

    double tolerance = 0.0;
    bool   hasTol = false;
    {
        auto it = m_Parameters.find("tolerance");
        if (it == m_Parameters.end()) it = m_Parameters.find("accuracy");
        if (it != m_Parameters.end()) { tolerance = std::stod(it->second); hasTol = true; }
    }
    if (!hasTol)
        throw std::invalid_argument("[CompressMGARDCentroidOperator::Operate] missing mandatory "
                                    "parameter 'tolerance' (relative)");

    double ebratio = 0.8;
    {
        auto it = m_Parameters.find("ebratio");
        if (it != m_Parameters.end()) ebratio = std::stod(it->second);
    }
    if (ebratio <= 0.0 || ebratio >= 1.0)
        throw std::invalid_argument(
            "[CompressMGARDCentroidOperator::Operate] 'ebratio' must be in (0, 1)");

    /* Relative-tolerance mode: the input 'tolerance' is relative and is
     * converted to an absolute tolerance by normalizing with a statistic of
     * the data block (see below).
     *   REL_L2  : normalize by the data L2 norm  = sqrt(sum x^2)
     *   REL_VAL : normalize by the value range   = (max - min)             */
    enum class RelMode { L2, VAL };
    RelMode relMode = RelMode::L2; /* default */
    {
        auto it = m_Parameters.find("mode");
        if (it != m_Parameters.end())
        {
            if (it->second == "REL_L2")
                relMode = RelMode::L2;
            else if (it->second == "REL_VAL")
                relMode = RelMode::VAL;
            else
                throw std::invalid_argument(
                    "[CompressMGARDCentroidOperator::Operate] Only 'REL_L2' and 'REL_VAL' "
                    "mode is supported");
        }
    }
    double s = 0.0;
    {
        auto it = m_Parameters.find("s");
        if (it != m_Parameters.end()) s = std::stod(it->second);
    }
    size_t thresholdSize = 100000;
    {
        auto it = m_Parameters.find("threshold");
        if (it != m_Parameters.end()) thresholdSize = std::stoul(it->second);
    }
    /* ---- shapes: treat the block as one flat 1D nodal array ---- */
    const size_t nnodes    = GetTotalSize(blockCount);
    const size_t typeSize  = TypeNBytes(type);
    const size_t inputBytes = nnodes * typeSize;

    /* ---- write V5 header ---- */
    size_t off = 0;
    PutParameter(bufferOut, off, kOperatorTypeByte);
    PutParameter(bufferOut, off, kBufferVersion);
    PutParameter(bufferOut, off, (uint16_t)0); /* reserved */
    PutParameter(bufferOut, off, m_BlockId);
    PutParameter(bufferOut, off, (size_t)1); /* ndims: flattened to 1D */
    PutParameter(bufferOut, off, nnodes);
    PutParameter(bufferOut, off, type);
    PutParameter(bufferOut, off, (uint8_t)MGARD_VERSION_MAJOR);
    PutParameter(bufferOut, off, (uint8_t)MGARD_VERSION_MINOR);
    PutParameter(bufferOut, off, (uint8_t)MGARD_VERSION_PATCH);

    /* ---- relative -> absolute tolerance ----
     * Normalize the input relative 'tolerance' with a statistic of this data
     * block; the min/max and sum-of-squares are computed on the GPU when the
     * backend is HIP (else on the CPU). Only needed when we actually compress. */
    double vmin = 0.0, vmax = 0.0, sumsq = 0.0;
    if (inputBytes >= thresholdSize)
    {
        if (type == DataType::Float)
            cmg_reduce_stats_f32(m_Backend, reinterpret_cast<const float *>(dataIn), nnodes,
                                 &vmin, &vmax, &sumsq);
        else
            cmg_reduce_stats_f64(m_Backend, reinterpret_cast<const double *>(dataIn), nnodes,
                                 &vmin, &vmax, &sumsq);
    }
    /* REL_L2 uses the RMS value sqrt((1/N) sum x^2) (the per-element "L2 norm"
     * convention used by the driver and calc_err), so absTol is a per-element
     * scale consistent with the L-inf residual quantizer. */
    const double normFactor = (relMode == RelMode::L2)
                                  ? std::sqrt(sumsq / static_cast<double>(nnodes))
                                  : (vmax - vmin);
    const double absTol = tolerance * normFactor;

    /* Raw storage for tiny blocks or a degenerate normalization: a constant /
     * all-zero block gives normFactor == 0 => absTol == 0, which would make the
     * residual quantum 1/(2*absTol) non-finite. */
    if (inputBytes < thresholdSize || !(absTol > 0.0))
    {
        /* Store the raw data inside our own payload. Returning 0 (the core
         * "operator not applied" contract) does not work through the plugin
         * wrapper: PluginOperator::Operate adds its own header and returns a
         * nonzero size, so BP5 never takes its raw-copy fallback. */
        PutParameter(bufferOut, off, false);
        std::memcpy(bufferOut + off, dataIn, inputBytes);
        return off + inputBytes;
    }
    PutParameter(bufferOut, off, true);

    /* Split the absolute error budget between the cell-average and the residual. */
    const double tol_avg  = (1.0 - ebratio) * absTol;
    const double tol_resi = ebratio * absTol;

    if (m_Verbose)
    {
        const char *modeStr = (relMode == RelMode::L2) ? "REL_L2" : "REL_VAL";
        std::cerr << std::scientific << std::setprecision(4) << "[centroid] block=" << m_BlockId
                  << " mode=" << modeStr << " relTol=" << tolerance
                  << (relMode == RelMode::L2 ? " rms=" : " range=") << normFactor
                  << " abs_tol=" << absTol << " tol_avg=" << tol_avg << " tol_resi=" << tol_resi
                  << std::endl;
    }

    /* =================================================================
     * PATH C: group-location decomposition (method=group_centroid).
     *
     *   coarse[g] = mean of u over the nodes sharing location g
     *               -> SFC reorder over group locations -> MGARD
     *   resid[n]  = u[n] - coarse[group(n)]
     *               -> nodal SFC reorder -> quantize -> Huffman/ZSTD
     *
     * This mesh stores each distinct (x,y,z) 12.7x-23.3x over (15.2x
     * dataset-wide), so the coarse stream is ~6.6% of nnodes -- far fewer
     * MGARD elements than the cell-average stream -- and co-located nodes
     * carry near-identical values, so it predicts much better than the
     * cell average (residual rms 1.5x-2.6x smaller, measured).
     *
     * Marked in the bitstream by ncells == SIZE_MAX (a real centroid stream
     * can never emit that), so the centroid layout stays untouched.
     * ================================================================= */
    if (UseGroupCentroidPath(m_Parameters))
    {
        PutParameter(bufferOut, off, (size_t)SIZE_MAX);  /* PATH C marker */
        PutParameter(bufferOut, off, (size_t)0);         /* npc unused    */

        /* --- coordinates + grouping (cached, geometry-only) --- */
        const auto t_grp0 = high_resolution_clock::now();
        LoadConnectivity(m_MeshFile, connVar, m_BlockId);   /* opens mesh reader */
        const double *nx = nullptr, *ny = nullptr, *nz = nullptr;
        if (!ResolveAndLoadCoords(m_Parameters, m_MeshFile, connVar, m_BlockId, nx, ny, nz))
            throw std::runtime_error("[CompressMGARDCentroidOperator::Operate] group_centroid "
                                     "requires coordinates; set CENTROID_COORD_PREFIX");
        {
            auto vec = g_coordsCache.find({0, m_BlockId});
            if (vec == g_coordsCache.end() || vec->second.size() != nnodes)
                throw std::runtime_error("[CompressMGARDCentroidOperator::Operate] group_centroid: "
                                         "coordinate length != nnodes for block " +
                                         std::to_string(m_BlockId));
        }
        const CoordGroups &cg = GetOrBuildCoordGroups(m_Backend, m_BlockId,
                                                      ParseSFC(m_Parameters), nx, ny, nz, nnodes);
        const size_t ngroups = cg.ngroups;
        const auto t_grp1 = high_resolution_clock::now();
        PutParameter(bufferOut, off, ngroups);
        PutParameter(bufferOut, off, (uint8_t)cg.gmode);

        /* --- coarse: group means, SFC-ordered --- */
        const auto t_co0 = high_resolution_clock::now();
        std::vector<char> coarse(ngroups * typeSize);
        if (type == DataType::Float)
            cmg_group_mean_f32(m_Backend, reinterpret_cast<const float *>(dataIn), cg.gid.data(),
                               nnodes, ngroups, reinterpret_cast<float *>(coarse.data()));
        else
            cmg_group_mean_f64(m_Backend, reinterpret_cast<const double *>(dataIn), cg.gid.data(),
                               nnodes, ngroups, reinterpret_cast<double *>(coarse.data()));
        std::vector<char> coarseSfc(ngroups * typeSize);
        if (cg.gmode != CMG_SFC_NONE)
        {
            if (type == DataType::Float)
                cmg_perm_forward_f32(reinterpret_cast<const float *>(coarse.data()),
                                     reinterpret_cast<float *>(coarseSfc.data()),
                                     cg.gperm.data(), ngroups);
            else
                cmg_perm_forward_f64(reinterpret_cast<const double *>(coarse.data()),
                                     reinterpret_cast<double *>(coarseSfc.data()),
                                     cg.gperm.data(), ngroups);
        }
        else
            coarseSfc = coarse;
        const auto t_co1 = high_resolution_clock::now();

        /* --- MGARD the coarse stream --- */
        const mgard_x::data_type mgTypeC = ToMgardType(type);
        mgard_x::Config cfgC = MakeMgardConfig(m_Backend);
        std::vector<mgard_x::SIZE> shapeC = { (mgard_x::SIZE)ngroups };
        void *mgOut = nullptr; size_t mgSize = 0;
        const auto t_mg0 = high_resolution_clock::now();
        mgard_x::compress((mgard_x::DIM)1, mgTypeC, shapeC, tol_avg, s,
                          mgard_x::error_bound_type::ABS, coarseSfc.data(),
                          mgOut, mgSize, cfgC, false);
        const auto t_mg1 = high_resolution_clock::now();
        if (mgOut == nullptr || mgSize == 0 || mgSize > inputBytes)
        {
            if (mgOut) std::free(mgOut);
            throw std::runtime_error("[CompressMGARDCentroidOperator::Operate] group_centroid: "
                                     "coarse MGARD failed for block " + std::to_string(m_BlockId));
        }
        PutParameter(bufferOut, off, mgSize);
        std::memcpy(bufferOut + off, mgOut, mgSize);
        std::free(mgOut);
        mgard_x::release_cache(cfgC);
        off += mgSize;

        /* --- residual against the DECODED coarse (closed loop) --- */
        const auto t_rs0 = high_resolution_clock::now();
        void *coarseDec = nullptr;
        mgard_x::decompress(bufferOut + off - mgSize, mgSize, coarseDec, cfgC, false);
        if (coarseDec == nullptr)
            throw std::runtime_error("[CompressMGARDCentroidOperator::Operate] group_centroid: "
                                     "coarse round-trip failed");
        /* undo the group SFC so coarseDecNat is indexed by group id */
        std::vector<char> coarseNat(ngroups * typeSize);
        if (cg.gmode != CMG_SFC_NONE)
        {
            if (type == DataType::Float)
                cmg_perm_inverse_f32(reinterpret_cast<const float *>(coarseDec),
                                     reinterpret_cast<float *>(coarseNat.data()),
                                     cg.gperm.data(), ngroups);
            else
                cmg_perm_inverse_f64(reinterpret_cast<const double *>(coarseDec),
                                     reinterpret_cast<double *>(coarseNat.data()),
                                     cg.gperm.data(), ngroups);
        }
        else
            std::memcpy(coarseNat.data(), coarseDec, ngroups * typeSize);
        std::free(coarseDec);

        std::vector<char> resid(nnodes * typeSize);
        if (type == DataType::Float)
            cmg_group_bcast_sub_f32(reinterpret_cast<const float *>(coarseNat.data()),
                                    cg.gid.data(), nnodes,
                                    reinterpret_cast<const float *>(dataIn),
                                    reinterpret_cast<float *>(resid.data()));
        else
            cmg_group_bcast_sub_f64(reinterpret_cast<const double *>(coarseNat.data()),
                                    cg.gid.data(), nnodes,
                                    reinterpret_cast<const double *>(dataIn),
                                    reinterpret_cast<double *>(resid.data()));

        /* nodal SFC reorder of the residual (helps the ZSTD stage).
         * The permutation build is per-block geometry and cached, so it is
         * timed separately from the per-field residual work. */
        /* --- adaptive rule: is the residual worth storing at all? ---
         * rms via cmg_reduce_stats_*, which runs on the GPU when m_Backend is
         * HIP. Evaluated BEFORE the SFC gather so a dropped residual also skips
         * that work. */
        const auto t_rms0 = high_resolution_clock::now();
        double rmin = 0.0, rmax = 0.0, rsumsq = 0.0;
        if (type == DataType::Float)
            cmg_reduce_stats_f32(m_Backend, reinterpret_cast<const float *>(resid.data()),
                                 nnodes, &rmin, &rmax, &rsumsq);
        else
            cmg_reduce_stats_f64(m_Backend, reinterpret_cast<const double *>(resid.data()),
                                 nnodes, &rmin, &rmax, &rsumsq);
        const double residRms  = (nnodes > 0) ? std::sqrt(rsumsq / (double)nnodes) : 0.0;
        const double residLinf = std::max(std::fabs(rmin), std::fabs(rmax));
        const AdaptResid arMode = AdaptiveResidualMode(m_Parameters);
        const bool dropResidual =
            (arMode == AdaptResid::Linf && residLinf <= tol_resi) ||
            (arMode == AdaptResid::Rms  && residRms  <= tol_resi);
        const auto t_rms1 = high_resolution_clock::now();

        cmg_sfc_t rSfc = CMG_SFC_NONE;
        auto t_np0 = high_resolution_clock::now();
        auto t_np1 = t_np0, t_gat0 = t_np0, t_gat1 = t_np0;
        if (!dropResidual && IsResidualSFCEnabled(m_Parameters))
        {
            t_np0 = high_resolution_clock::now();
            const std::vector<uint32_t> &nperm =
                GetOrBuildNodeSFCPerm(m_Backend, m_BlockId, ParseSFC(m_Parameters), nx, ny, nz,
                                      nnodes, rSfc);
            t_np1 = high_resolution_clock::now();
            if (rSfc != CMG_SFC_NONE)
            {
                std::vector<char> tmp(nnodes * typeSize);
                t_gat0 = high_resolution_clock::now();
                if (type == DataType::Float)
                    cmg_perm_forward_f32(reinterpret_cast<const float *>(resid.data()),
                                         reinterpret_cast<float *>(tmp.data()), nperm.data(), nnodes);
                else
                    cmg_perm_forward_f64(reinterpret_cast<const double *>(resid.data()),
                                         reinterpret_cast<double *>(tmp.data()), nperm.data(), nnodes);
                t_gat1 = high_resolution_clock::now();
                resid.swap(tmp);
            }
        }
        PutParameter(bufferOut, off, (uint8_t)rSfc);
        PutParameter(bufferOut, off, tol_resi);

        /* capR == 0 marks "no residual stored"; the decoder reconstructs from
         * the coarse stream alone. */
        size_t capR = 0;
        if (!dropResidual)
        {
            std::vector<uint32_t> quantC(nnodes);
            if (type == DataType::Float)
                cmg_quantize_zigzag_f32(m_Backend, reinterpret_cast<const float *>(resid.data()),
                                        nnodes, tol_resi, quantC.data());
            else
                cmg_quantize_zigzag_f64(m_Backend, reinterpret_cast<const double *>(resid.data()),
                                        nnodes, tol_resi, quantC.data());
            char *outR = bufferOut + off + sizeof(size_t);
            capR = mgardx_lossless::serial::Compress(quantC.data(), nnodes, outR,
                                                     nnodes * sizeof(uint32_t) * 4 + 4096);
            if (capR == 0)
                throw std::runtime_error("[CompressMGARDCentroidOperator::Operate] group_centroid: "
                                         "residual entropy coding failed");
        }
        PutParameter(bufferOut, off, capR);
        off += capR;
        const auto t_rs1 = high_resolution_clock::now();

        if (m_Verbose)
            std::cerr << std::fixed << std::setprecision(2)
                      << "[centroid] block=" << m_BlockId << " path=group_centroid"
                      << " nnodes=" << nnodes << " ngroups=" << ngroups
                      << " (" << 100.0 * (double)ngroups / (double)nnodes << "%)"
                      << " in=" << inputBytes << " out=" << off
                      << " CR=" << (double)inputBytes / (double)off << "x"
                      << " (coarse=" << mgSize << "B resid=" << capR << "B)"
                      << (dropResidual ? " NO-RESIDUAL" : "")
                      << " residRms=" << residRms << " residLinf=" << residLinf
                      << " tol_resi=" << tol_resi
                      << " ms[groupbuild=" << duration<double, std::milli>(t_grp1 - t_grp0).count()
                      << " nodepermbuild=" << duration<double, std::milli>(t_np1 - t_np0).count()
                      << " gather=" << duration<double, std::milli>(t_gat1 - t_gat0).count()
                      << " rms=" << duration<double, std::milli>(t_rms1 - t_rms0).count()
                      << " coarse=" << duration<double, std::milli>(t_co1 - t_co0).count()
                      << " mgard=" << duration<double, std::milli>(t_mg1 - t_mg0).count()
                      << " resid=" << (duration<double, std::milli>(t_rs1 - t_rs0).count() -
                                       duration<double, std::milli>(t_np1 - t_np0).count() -
                                       duration<double, std::milli>(t_gat1 - t_gat0).count() -
                                       duration<double, std::milli>(t_rms1 - t_rms0).count())
                      << "]\n";
        return off;
    }

    /* =================================================================
     * PATH B: nodal SFC reorder + plain MGARD (no centroid split).
     *
     * Gathers the nodal field along a Hilbert curve built from the node
     * coordinates, then compresses the whole field with MGARD's 1-D
     * multilevel transform. Reordering raises the array-order neighbour
     * correlation from ~0.81 to ~0.99, which is what MGARD's coefficient
     * decay depends on. Because MGARD is critically sampled there is no
     * cell-average stream and no N_cell expansion.
     *
     * The permutation is derived from the mesh geometry and rebuilt on read,
     * so it costs no bytes. Marked in the bitstream by ncells == 0, which a
     * centroid stream can never emit, keeping the centroid layout untouched.
     * ================================================================= */
    if (UseReorderMgardPath(m_Parameters))
    {
        PutParameter(bufferOut, off, (size_t)0);   /* ncells == 0 => PATH B */
        PutParameter(bufferOut, off, (size_t)0);   /* npc unused */

        cmg_sfc_t nodeSfcMode = CMG_SFC_NONE;
        const char *src = dataIn;
        std::vector<char> gathered;
        /* Timed separately: the SFC permutation build is per-block geometry and
         * is cached (amortizes across variables/timesteps in a process), while
         * the gather is unavoidably per-variable per-timestep. */
        auto t_perm0 = high_resolution_clock::now();
        auto t_perm1 = t_perm0, t_gat0 = t_perm0, t_gat1 = t_perm0;
        const auto t_g0 = t_perm0;
        {
            /* The coordinate loader reads through the shared mesh reader, which
             * is opened lazily by LoadConnectivity. This path does not otherwise
             * need the connectivity, but must open the reader before asking for
             * coords (result is cached, so this is a no-op after block 0). */
            LoadConnectivity(m_MeshFile, connVar, m_BlockId);

            const double *nx = nullptr, *ny = nullptr, *nz = nullptr;
            bool haveCoords =
                ResolveAndLoadCoords(m_Parameters, m_MeshFile, connVar, m_BlockId, nx, ny, nz);
            if (haveCoords)
            {
                auto vec = g_coordsCache.find({0, m_BlockId});
                if (vec == g_coordsCache.end() || vec->second.size() != nnodes) haveCoords = false;
            }
            if (haveCoords)
            {
                t_perm0 = high_resolution_clock::now();
                const std::vector<uint32_t> &nperm = GetOrBuildNodeSFCPerm(
                    m_Backend, m_BlockId, ParseSFC(m_Parameters), nx, ny, nz, nnodes, nodeSfcMode);
                t_perm1 = high_resolution_clock::now();
                if (nodeSfcMode != CMG_SFC_NONE)
                {
                    gathered.resize(nnodes * typeSize);
                    t_gat0 = high_resolution_clock::now();
                    if (type == DataType::Float)
                        cmg_perm_forward_f32(reinterpret_cast<const float *>(dataIn),
                                             reinterpret_cast<float *>(gathered.data()),
                                             nperm.data(), nnodes);
                    else
                        cmg_perm_forward_f64(reinterpret_cast<const double *>(dataIn),
                                             reinterpret_cast<double *>(gathered.data()),
                                             nperm.data(), nnodes);
                    t_gat1 = high_resolution_clock::now();
                    src = gathered.data();
                }
            }
            else if (m_Verbose)
                std::cerr << "[centroid] block=" << m_BlockId
                          << " reorder_mgard: coordinates unavailable; compressing unordered\n";
        }
        const auto t_g1 = high_resolution_clock::now();
        PutParameter(bufferOut, off, (uint8_t)nodeSfcMode);

        const mgard_x::data_type mgardTypeB = ToMgardType(type);
        mgard_x::Config cfgB = MakeMgardConfig(m_Backend);
        std::vector<mgard_x::SIZE> shapeB = { (mgard_x::SIZE)nnodes };
        void *mgardOut = nullptr; size_t mgardSize = 0;
        const auto t_m0 = high_resolution_clock::now();
        mgard_x::compress((mgard_x::DIM)1, mgardTypeB, shapeB, absTol, s,
                          mgard_x::error_bound_type::ABS, const_cast<char *>(src),
                          mgardOut, mgardSize, cfgB, false);
        const auto t_m1 = high_resolution_clock::now();
        /* GetEstimatedSize reserves >= inputBytes for the MGARD stream, so a
         * payload larger than the raw input cannot fit (and never should). */
        if (mgardOut == nullptr || mgardSize == 0 || mgardSize > inputBytes)
        {
            if (mgardOut) std::free(mgardOut);
            throw std::runtime_error(
                "[CompressMGARDCentroidOperator::Operate] reorder_mgard: MGARD compression "
                "failed or exceeded buffer for block " + std::to_string(m_BlockId));
        }
        PutParameter(bufferOut, off, mgardSize);
        std::memcpy(bufferOut + off, mgardOut, mgardSize);
        std::free(mgardOut);
        mgard_x::release_cache(cfgB);
        off += mgardSize;

        if (m_Verbose)
            std::cerr << std::fixed << std::setprecision(2)
                      << "[centroid] block=" << m_BlockId << " path=reorder_mgard"
                      << " sfc=" << (int)nodeSfcMode
                      << " in=" << inputBytes << " out=" << off
                      << " CR=" << (double)inputBytes / (double)off << "x"
                      << " ms[permbuild=" << duration<double, std::milli>(t_perm1 - t_perm0).count()
                      << " gather=" << duration<double, std::milli>(t_gat1 - t_gat0).count()
                      << " coordio=" << duration<double, std::milli>(t_g1 - t_g0).count()
                      << " mgard=" << duration<double, std::milli>(t_m1 - t_m0).count() << "]\n";
        return off;
    }

    /* ---- connectivity ---- */
    const auto t_conn0 = high_resolution_clock::now();
    const auto &conn = LoadConnectivity(m_MeshFile, connVar, m_BlockId);
    const auto t_conn1 = high_resolution_clock::now();
    if (conn.size() % m_NodesPerCell != 0)
        throw std::invalid_argument("[CompressMGARDCentroidOperator::Operate] Connectivity "
                                    "length not a multiple of nodes_per_cell");
    const size_t ncells = conn.size() / m_NodesPerCell;
    PutParameter(bufferOut, off, ncells);
    PutParameter(bufferOut, off, m_NodesPerCell);

    /* ---- centroid split ---- */
    std::vector<char> cellAvgBuf (ncells * typeSize);
    std::vector<char> residualBuf(nnodes * typeSize);
    const auto t_split0 = high_resolution_clock::now();
    if (type == DataType::Float)
    {
        cmg_centroid_split_f32(m_Backend,
            reinterpret_cast<const float *>(dataIn), conn.data(),
            ncells, m_NodesPerCell, nnodes,
            reinterpret_cast<float *>(cellAvgBuf.data()),
            reinterpret_cast<float *>(residualBuf.data()));
    }
    else
    {
        cmg_centroid_split_f64(m_Backend,
            reinterpret_cast<const double *>(dataIn), conn.data(),
            ncells, m_NodesPerCell, nnodes,
            reinterpret_cast<double *>(cellAvgBuf.data()),
            reinterpret_cast<double *>(residualBuf.data()));
    }
    const auto t_split1 = high_resolution_clock::now();

    /* ---- SFC reorder of cell averages ---- */
    cmg_sfc_t sfcMode = CMG_SFC_NONE;
    const auto t_sfc0 = high_resolution_clock::now();
    if (IsSFCEnabled(m_Parameters))
    {
        const double *cx = nullptr, *cy = nullptr, *cz = nullptr;
        const bool haveCoords =
            ResolveAndLoadCoords(m_Parameters, m_MeshFile, connVar, m_BlockId, cx, cy, cz);
        if (haveCoords)
        {
            auto vec = g_coordsCache.find({0, m_BlockId});
            if (vec == g_coordsCache.end() || vec->second.size() != nnodes)
                cx = cy = cz = nullptr;
        }
        const cmg_sfc_t want = ParseSFC(m_Parameters);
        const std::vector<uint32_t> &perm =
            GetOrBuildSFCPerm(m_Backend, m_BlockId, want, conn.data(), ncells, m_NodesPerCell, cx,
                              cy, cz, nnodes, sfcMode);

        std::vector<char> sortedAvg(ncells * typeSize);
        if (type == DataType::Float)
            cmg_perm_forward_f32(reinterpret_cast<const float *>(cellAvgBuf.data()),
                                 reinterpret_cast<float *>(sortedAvg.data()),
                                 perm.data(), ncells);
        else
            cmg_perm_forward_f64(reinterpret_cast<const double *>(cellAvgBuf.data()),
                                 reinterpret_cast<double *>(sortedAvg.data()),
                                 perm.data(), ncells);
        cellAvgBuf.swap(sortedAvg);
    }
    PutParameter(bufferOut, off, (uint8_t)sfcMode);
    const auto t_sfc1 = high_resolution_clock::now();

    /* ---- MGARD compress cell averages ---- */
    const mgard_x::data_type mgardType = ToMgardType(type);
    mgard_x::Config cfg = MakeMgardConfig(m_Backend);

    auto compressMgard1D = [&](void *src, size_t n, double tol, void *dst, size_t &dstCap) {
        std::vector<mgard_x::SIZE> shape = { (mgard_x::SIZE)n };
        /* Let MGARD-X allocate its own output buffer; we then copy into the
         * ADIOS2 buffer. Passing a pre-allocated buffer triggers a corrupted
         * internal allocation request (PB-scale) on the SERIAL adapter when
         * the MGARD-X library is loaded inside an ADIOS2 plugin context. */
        void *mgardOut = nullptr;
        size_t mgardSize = 0;
        mgard_x::compress((mgard_x::DIM)1, mgardType, shape, tol, s,
                          mgard_x::error_bound_type::ABS, src,
                          mgardOut, mgardSize, cfg, false);
        if (mgardOut == nullptr || mgardSize == 0)
        {
            if (m_Verbose)
                std::cerr << "[centroid] mgard_x::compress returned empty buffer "
                             "(n=" << n << " tol=" << tol << ")\n";
            dstCap = 0;
            return;
        }
        if (mgardSize > dstCap)
        {
            if (m_Verbose)
                std::cerr << "[centroid] mgard_x output (" << mgardSize
                          << " B) exceeds reserved capacity (" << dstCap
                          << " B); aborting block.\n";
            std::free(mgardOut);
            dstCap = 0;
            return;
        }
        std::memcpy(dst, mgardOut, mgardSize);
        std::free(mgardOut);
        dstCap = mgardSize;
        mgard_x::release_cache(cfg);
    };

    size_t cap_avg = inputBytes;
    void *outAvg = bufferOut + off + sizeof(size_t);
    const auto t_avg0 = high_resolution_clock::now();
    compressMgard1D(cellAvgBuf.data(), ncells, tol_avg, outAvg, cap_avg);
    const auto t_avg1 = high_resolution_clock::now();
    PutParameter(bufferOut, off, cap_avg);
    off += cap_avg;

    /* ---- quantize residual and MGARD-X Huffman+ZSTD ---- */
    const size_t markerOff = off;
    PutParameter(bufferOut, off, (uint8_t)kResMarker_MGARDX_Huff);

    /* tol_resi follows the marker so the decoder reproduces the quantum. */
    PutParameter(bufferOut, off, tol_resi);

    /* Residual backend: quantize->Huffman/ZSTD (default) or MGARD-lossy on the
     * raw residual (residual_method=mgard / MGARD_RESIDUAL_METHOD=mgard). */
    const bool useMgardResidual = UseMgardResidual(m_Parameters);

    char *outRes = bufferOut + off + sizeof(size_t);
    const size_t resCap = nnodes * sizeof(uint32_t) * 4 + 4096;
    std::vector<uint32_t> quant;
    size_t cap_res = 0;
    auto t_q0 = high_resolution_clock::now();
    auto t_q1 = t_q0, t_h0 = t_q0, t_h1 = t_q0;

    if (!useMgardResidual)
    {
        /* quantize → host uint32 buffer */
        quant.resize(nnodes);
        t_q0 = high_resolution_clock::now();
        if (type == DataType::Float)
            cmg_quantize_zigzag_f32(m_Backend, reinterpret_cast<const float *>(residualBuf.data()),
                                    nnodes, tol_resi, quant.data());
        else
            cmg_quantize_zigzag_f64(m_Backend, reinterpret_cast<const double *>(residualBuf.data()),
                                    nnodes, tol_resi, quant.data());
        t_q1 = high_resolution_clock::now();

        /* MGARD-X Huffman+ZSTD on host quant buffer (SERIAL backend; same bytes on either device) */
        t_h0 = high_resolution_clock::now();
        cap_res = mgardx_lossless::serial::Compress(quant.data(), nnodes, outRes, resCap);
        t_h1 = high_resolution_clock::now();
    }

#ifdef CMG_HAVE_HIP
    /* Optional A/B measurement: also run the HIP Huffman path on the same
     * uint32 quant buffer and log both sizes side-by-side. Enabled by
     * env var CMG_HUFF_AB=1 (HIP build only). */
    if (const char *ab = std::getenv("CMG_HUFF_AB");
        !useMgardResidual && ab && *ab && *ab != '0')
    {
        std::vector<char> tmp(resCap);
        const auto t_ab0 = high_resolution_clock::now();
        size_t hip_bytes = mgardx_lossless::hip::CompressFromHost(
            quant.data(), nnodes, tmp.data(), resCap);
        const auto t_ab1 = high_resolution_clock::now();
        const double ms_ser = duration<double, std::milli>(t_h1 - t_h0).count();
        const double ms_hip = duration<double, std::milli>(t_ab1 - t_ab0).count();
        if (m_Verbose)
            std::cerr << "[HUFF_AB] block=" << m_BlockId
                    << " nnodes=" << nnodes
                    << " tol_resi=" << tol_resi
                    << " serial_bytes=" << cap_res
                    << " hip_bytes=" << hip_bytes
                    << " serial_ms=" << ms_ser
                    << " hip_ms=" << ms_hip
                    << "\n";
        if (const char *logp = std::getenv("CMG_HUFF_AB_LOG"); logp && *logp)
        {
            FILE *f = std::fopen(logp, "a");
            if (f)
            {
                std::fprintf(f, "%zu,%zu,%g,%zu,%zu,%g,%g\n",
                             (size_t)m_BlockId, nnodes, tol_resi,
                             cap_res, hip_bytes, ms_ser, ms_hip);
                std::fclose(f);
            }
        }
    }
#endif

    uint8_t actualResMarker = kResMarker_MGARDX_Huff;
    if (useMgardResidual || cap_res == 0)
    {
        /* MGARD-lossy on the raw residual: either selected explicitly, or as a
         * fallback when the Huffman path returns 0 bytes. */
        if (m_Verbose && !useMgardResidual)
            std::cerr << "[centroid] block=" << m_BlockId
                      << " MGARD-X Huffman failed; falling back to MGARD on residual\n";
        actualResMarker = kResMarker_MGARD;
        /* The marker byte is patched below; tol_resi was already written and is
         * always read back by the decoder, so the field layout stays fixed
         * ([marker][tol_resi][cap_res][payload]) regardless of which backend ran. */
        cap_res = inputBytes;
        t_h0 = high_resolution_clock::now();
        compressMgard1D(residualBuf.data(), nnodes, tol_resi, outRes, cap_res);
        t_h1 = high_resolution_clock::now();
    }
    *(reinterpret_cast<uint8_t *>(bufferOut + markerOff)) = actualResMarker;
    PutParameter(bufferOut, off, cap_res);
    off += cap_res;

    const auto t1 = high_resolution_clock::now();
    auto ms = [](auto a, auto b) {
        return duration<double, std::milli>(b - a).count();
    };
    if (m_Verbose)
        std::cerr << std::fixed << std::setprecision(2)
                << "[centroid] block=" << m_BlockId
                << " be=" << (m_Backend == CMG_BACKEND_HIP ? "gpu" : "cpu")
                << " in=" << inputBytes << " out=" << off
                << " CR=" << (double)inputBytes / (double)off << "x"
                << " (avg=" << cap_avg << "B res=" << cap_res << "B)"
                << " ms[conn=" << ms(t_conn0, t_conn1)
                << " split=" << ms(t_split0, t_split1)
                << " sfc=" << ms(t_sfc0, t_sfc1)
                << " mgard_avg=" << ms(t_avg0, t_avg1)
                << " quant=" << ms(t_q0, t_q1)
                << " huff=" << ms(t_h0, t_h1)
                << " total=" << ms(t0, t1) << "]\n";
    return off;
}

/* ====================================================================== */

size_t CompressMGARDCentroidOperator::InverseOperate(const char *bufferIn, const size_t sizeIn,
                                                     char *dataOut)
{
    size_t off = 0;
    GetParameter<uint8_t>(bufferIn, off);                       /* operator type byte */
    const uint8_t bufferVersion = GetParameter<uint8_t>(bufferIn, off);
    off += 2;                                                   /* reserved */
    if (bufferVersion != kBufferVersion)
        throw std::runtime_error(
            "[CompressMGARDCentroidOperator::InverseOperate] Unsupported buffer version " +
            std::to_string((int)bufferVersion) + " (this build only reads V" +
            std::to_string((int)kBufferVersion) + ")");
    return DecompressV5(bufferIn + off, sizeIn - off, dataOut);
}

size_t CompressMGARDCentroidOperator::DecompressV5(const char *bufferIn, const size_t /*sizeIn*/,
                                                    char *dataOut)
{
    using namespace std::chrono;
    auto ms = [](auto a, auto b) { return duration<double, std::milli>(b - a).count(); };
    const auto t_start = high_resolution_clock::now();
    size_t off = 0;
    const size_t blockId = GetParameter<size_t>(bufferIn, off);
    const size_t ndims   = GetParameter<size_t, size_t>(bufferIn, off);
    Dims blockCount(ndims);
    for (size_t i = 0; i < ndims; ++i)
        blockCount[i] = GetParameter<size_t, size_t>(bufferIn, off);
    const DataType type = GetParameter<DataType>(bufferIn, off);
    m_VersionInfo =
        " Compressed with MGARD " + std::to_string(GetParameter<uint8_t>(bufferIn, off)) + "." +
        std::to_string(GetParameter<uint8_t>(bufferIn, off)) + "." +
        std::to_string(GetParameter<uint8_t>(bufferIn, off));

    const bool isCompressed = GetParameter<bool>(bufferIn, off);
    const size_t nnodes   = blockCount[0];
    const size_t typeSize = TypeNBytes(type);
    const size_t sizeOut  = nnodes * typeSize;
    if (!isCompressed)
    {
        /* below-threshold block: raw data follows the header */
        std::memcpy(dataOut, bufferIn + off, sizeOut);
        return sizeOut;
    }

    const size_t  ncells  = GetParameter<size_t>(bufferIn, off);
    const size_t  npc     = GetParameter<size_t>(bufferIn, off);

    /* ---- PATH C: group-location decomposition (ncells == SIZE_MAX) ---- */
    if (ncells == SIZE_MAX)
    {
        const size_t  ngroups = GetParameter<size_t>(bufferIn, off);
        const uint8_t gmodeRaw = GetParameter<uint8_t>(bufferIn, off);
        const cmg_sfc_t gmode = (cmg_sfc_t)gmodeRaw;
        const size_t  capCoarse = GetParameter<size_t>(bufferIn, off);
        mgard_x::Config cfgC = MakeMgardConfig(m_Backend);

        /* rebuild the grouping from the mesh coordinates */
        if (!m_Parameters.count("meshfile"))
            if (const char *e = std::getenv("CENTROID_MESHFILE")) m_Parameters["meshfile"] = e;
        if (!m_Parameters.count("connectivity_variable"))
            if (const char *e = std::getenv("CENTROID_CONN_VAR"))
                m_Parameters["connectivity_variable"] = e;
        auto itM = m_Parameters.find("meshfile");
        auto itC = m_Parameters.find("connectivity_variable");
        if (itM == m_Parameters.end() || itC == m_Parameters.end())
            throw std::invalid_argument("[CompressMGARDCentroidOperator::DecompressV5] "
                                        "group_centroid requires meshfile + connectivity_variable");
        LoadConnectivity(itM->second, itC->second, blockId);
        const double *nx = nullptr, *ny = nullptr, *nz = nullptr;
        if (!ResolveAndLoadCoords(m_Parameters, itM->second, itC->second, blockId, nx, ny, nz))
            throw std::runtime_error("[CompressMGARDCentroidOperator::DecompressV5] "
                                     "group_centroid requires coordinates");
        const CoordGroups &cg = GetOrBuildCoordGroups(m_Backend, blockId, gmode, nx, ny, nz, nnodes);
        if (cg.ngroups != ngroups)
            throw std::runtime_error("[CompressMGARDCentroidOperator::DecompressV5] rebuilt group "
                                     "count " + std::to_string(cg.ngroups) + " != bitstream " +
                                     std::to_string(ngroups));

        /* coarse: MGARD decode, then undo the group SFC */
        void *coarseDec = nullptr;
        mgard_x::decompress(bufferIn + off, capCoarse, coarseDec, cfgC, false);
        if (coarseDec == nullptr)
            throw std::runtime_error("[CompressMGARDCentroidOperator::DecompressV5] "
                                     "group_centroid coarse decode failed");
        off += capCoarse;
        std::vector<char> coarseNat(ngroups * typeSize);
        if (gmode != CMG_SFC_NONE)
        {
            if (type == DataType::Float)
                cmg_perm_inverse_f32(reinterpret_cast<const float *>(coarseDec),
                                     reinterpret_cast<float *>(coarseNat.data()),
                                     cg.gperm.data(), ngroups);
            else
                cmg_perm_inverse_f64(reinterpret_cast<const double *>(coarseDec),
                                     reinterpret_cast<double *>(coarseNat.data()),
                                     cg.gperm.data(), ngroups);
        }
        else
            std::memcpy(coarseNat.data(), coarseDec, ngroups * typeSize);
        std::free(coarseDec);

        /* residual: entropy decode -> dequantize -> undo nodal SFC */
        const uint8_t rSfcRaw = GetParameter<uint8_t>(bufferIn, off);
        const cmg_sfc_t rSfc = (cmg_sfc_t)rSfcRaw;
        const double tolR = GetParameter<double>(bufferIn, off);
        const size_t capR = GetParameter<size_t>(bufferIn, off);
        /* capR == 0: the encoder dropped the residual (adaptive rule) -- the
         * coarse stream alone was already inside the residual error budget. */
        if (capR == 0)
        {
            std::memset(dataOut, 0, nnodes * typeSize);
            if (type == DataType::Float)
                cmg_group_bcast_add_f32(reinterpret_cast<const float *>(coarseNat.data()),
                                        cg.gid.data(), nnodes, reinterpret_cast<float *>(dataOut));
            else
                cmg_group_bcast_add_f64(reinterpret_cast<const double *>(coarseNat.data()),
                                        cg.gid.data(), nnodes, reinterpret_cast<double *>(dataOut));
            return sizeOut;
        }

        std::vector<uint32_t> quantC(nnodes);
        if (!mgardx_lossless::serial::Decompress(bufferIn + off, capR, nnodes, quantC.data()))
            throw std::runtime_error("[CompressMGARDCentroidOperator::DecompressV5] "
                                     "group_centroid residual decode failed");
        std::vector<char> resid(nnodes * typeSize);
        if (type == DataType::Float)
            cmg_dequantize_zigzag_f32(m_Backend, quantC.data(), nnodes, tolR,
                                      reinterpret_cast<float *>(resid.data()));
        else
            cmg_dequantize_zigzag_f64(m_Backend, quantC.data(), nnodes, tolR,
                                      reinterpret_cast<double *>(resid.data()));
        if (rSfc != CMG_SFC_NONE)
        {
            cmg_sfc_t rebuilt;
            const std::vector<uint32_t> &nperm =
                GetOrBuildNodeSFCPerm(m_Backend, blockId, rSfc, nx, ny, nz, nnodes, rebuilt);
            if (type == DataType::Float)
                cmg_perm_inverse_f32(reinterpret_cast<const float *>(resid.data()),
                                     reinterpret_cast<float *>(dataOut), nperm.data(), nnodes);
            else
                cmg_perm_inverse_f64(reinterpret_cast<const double *>(resid.data()),
                                     reinterpret_cast<double *>(dataOut), nperm.data(), nnodes);
        }
        else
            std::memcpy(dataOut, resid.data(), nnodes * typeSize);

        /* u[n] = coarse[group(n)] + resid[n] */
        if (type == DataType::Float)
            cmg_group_bcast_add_f32(reinterpret_cast<const float *>(coarseNat.data()),
                                    cg.gid.data(), nnodes, reinterpret_cast<float *>(dataOut));
        else
            cmg_group_bcast_add_f64(reinterpret_cast<const double *>(coarseNat.data()),
                                    cg.gid.data(), nnodes, reinterpret_cast<double *>(dataOut));
        return sizeOut;
    }

    /* ---- PATH B: reorder + plain MGARD (marked by ncells == 0) ---- */
    if (ncells == 0)
    {
        const uint8_t nodeSfcRaw = GetParameter<uint8_t>(bufferIn, off);
        const cmg_sfc_t nodeSfcMode = (cmg_sfc_t)nodeSfcRaw;
        const size_t cap = GetParameter<size_t>(bufferIn, off);
        mgard_x::Config cfgB = MakeMgardConfig(m_Backend);

        if (nodeSfcMode == CMG_SFC_NONE)
        {
            /* stored unordered: MGARD writes straight into the output */
            void *out = dataOut;
            mgard_x::decompress(bufferIn + off, cap, out, cfgB, true);
            return sizeOut;
        }
        /* decompress into a scratch buffer, then scatter back to original
         * node order using the geometry-derived permutation. */
        std::vector<char> sfcOrder(nnodes * typeSize);
        void *out = sfcOrder.data();
        mgard_x::decompress(bufferIn + off, cap, out, cfgB, true);

        if (!m_Parameters.count("meshfile"))
            if (const char *e = std::getenv("CENTROID_MESHFILE")) m_Parameters["meshfile"] = e;
        if (!m_Parameters.count("connectivity_variable"))
            if (const char *e = std::getenv("CENTROID_CONN_VAR"))
                m_Parameters["connectivity_variable"] = e;
        auto itMeshB = m_Parameters.find("meshfile");
        auto itConnB = m_Parameters.find("connectivity_variable");
        if (itMeshB == m_Parameters.end() || itConnB == m_Parameters.end())
            throw std::invalid_argument(
                "[CompressMGARDCentroidOperator::DecompressV5] reorder_mgard requires meshfile + "
                "connectivity_variable params or CENTROID_MESHFILE / CENTROID_CONN_VAR env vars");
        /* opens the mesh reader (coords are read from the same file) */
        LoadConnectivity(itMeshB->second, itConnB->second, blockId);

        const double *nx = nullptr, *ny = nullptr, *nz = nullptr;
        if (!ResolveAndLoadCoords(m_Parameters, itMeshB->second, itConnB->second, blockId,
                                  nx, ny, nz))
            throw std::runtime_error(
                "[CompressMGARDCentroidOperator::DecompressV5] reorder_mgard requires coordinates; "
                "set CENTROID_COORD_PREFIX or coordinates_prefix");
        cmg_sfc_t rebuilt;
        const std::vector<uint32_t> &nperm =
            GetOrBuildNodeSFCPerm(m_Backend, blockId, nodeSfcMode, nx, ny, nz, nnodes, rebuilt);
        if (rebuilt != nodeSfcMode)
            throw std::runtime_error("[CompressMGARDCentroidOperator::DecompressV5] Rebuilt nodal "
                                     "SFC mode disagrees with bitstream marker");
        if (type == DataType::Float)
            cmg_perm_inverse_f32(reinterpret_cast<const float *>(sfcOrder.data()),
                                 reinterpret_cast<float *>(dataOut), nperm.data(), nnodes);
        else
            cmg_perm_inverse_f64(reinterpret_cast<const double *>(sfcOrder.data()),
                                 reinterpret_cast<double *>(dataOut), nperm.data(), nnodes);
        return sizeOut;
    }

    const uint8_t sfcRaw  = GetParameter<uint8_t>(bufferIn, off);
    const cmg_sfc_t sfcMode = (cmg_sfc_t)sfcRaw;

    /* resolve mesh params (env fallback) */
    if (!m_Parameters.count("meshfile"))
        if (const char *e = std::getenv("CENTROID_MESHFILE")) m_Parameters["meshfile"] = e;
    if (!m_Parameters.count("connectivity_variable"))
        if (const char *e = std::getenv("CENTROID_CONN_VAR")) m_Parameters["connectivity_variable"] = e;
    auto itMesh = m_Parameters.find("meshfile");
    auto itConn = m_Parameters.find("connectivity_variable");
    if (itMesh == m_Parameters.end() || itConn == m_Parameters.end())
        throw std::invalid_argument(
            "[CompressMGARDCentroidOperator::DecompressV5] Decompression requires meshfile + "
            "connectivity_variable params or CENTROID_MESHFILE / CENTROID_CONN_VAR env vars");
    const auto t_conn0 = high_resolution_clock::now();
    const auto &conn = LoadConnectivity(itMesh->second, itConn->second, blockId);
    const auto t_conn1 = high_resolution_clock::now();
    if (conn.size() != ncells * npc)
        throw std::runtime_error(
            "[CompressMGARDCentroidOperator::DecompressV5] Connectivity length mismatch on "
            "decompression for block " + std::to_string(blockId));

    /* cell averages: MGARD decompress (SERIAL by default, HIP via
     * MGARD_X_DEVICE_TYPE -- same policy as compression). */
    const mgard_x::Config avgCfg = MakeMgardConfig(m_Backend);
    const size_t cap_avg = GetParameter<size_t>(bufferIn, off);
    void *avgOut = nullptr;
    const auto t_avg0 = high_resolution_clock::now();
    mgard_x::decompress(bufferIn + off, cap_avg, avgOut, avgCfg, false);
    const auto t_avg1 = high_resolution_clock::now();
    off += cap_avg;

    /* residual: marker-dispatched */
    const auto t_res0 = high_resolution_clock::now();
    const uint8_t resMarker = GetParameter<uint8_t>(bufferIn, off);
    /* tol_resi is ALWAYS written by the encoder (before the backend is chosen),
     * so it must always be read here -- reading it conditionally misaligns the
     * stream by 8 bytes whenever the MGARD residual path is taken. */
    const double tol_resi = GetParameter<double>(bufferIn, off);
    const size_t cap_res = GetParameter<size_t>(bufferIn, off);

    if (resMarker == kResMarker_MGARD)
    {
        void *resiOut = dataOut;
        mgard_x::decompress(bufferIn + off, cap_res, resiOut, avgCfg, true);
    }
    else if (resMarker == kResMarker_MGARDX_Huff)
    {
        std::vector<uint32_t> quant(nnodes);
        if (!mgardx_lossless::serial::Decompress(bufferIn + off, cap_res, nnodes, quant.data()))
            throw std::runtime_error(
                "[CompressMGARDCentroidOperator::DecompressV5] MGARD-X Huffman decompression "
                "failed for block " + std::to_string(blockId));
        if (type == DataType::Float)
            cmg_dequantize_zigzag_f32(m_Backend, quant.data(), nnodes, tol_resi,
                                      reinterpret_cast<float *>(dataOut));
        else
            cmg_dequantize_zigzag_f64(m_Backend, quant.data(), nnodes, tol_resi,
                                      reinterpret_cast<double *>(dataOut));
    }
    else
    {
        throw std::runtime_error("[CompressMGARDCentroidOperator::DecompressV5] Unknown residual "
                                 "marker " + std::to_string((int)resMarker));
    }
    off += cap_res;
    const auto t_res1 = high_resolution_clock::now();

    /* undo SFC permutation on cell averages */
    const auto t_sfc0 = high_resolution_clock::now();
    if (sfcMode != CMG_SFC_NONE)
    {
        const double *cx = nullptr, *cy = nullptr, *cz = nullptr;
        if (sfcMode == CMG_SFC_MORTON || sfcMode == CMG_SFC_HILBERT)
        {
            const bool ok = ResolveAndLoadCoords(m_Parameters, itMesh->second, itConn->second,
                                                 blockId, cx, cy, cz);
            if (!ok)
                throw std::runtime_error(
                    "[CompressMGARDCentroidOperator::DecompressV5] SFC mode requires coordinates; "
                    "set CENTROID_COORD_PREFIX or coordinates_prefix");
            auto vec = g_coordsCache.find({0, blockId});
            if (vec != g_coordsCache.end() && vec->second.size() != nnodes)
                throw std::runtime_error(
                    "[CompressMGARDCentroidOperator::DecompressV5] Coordinate length mismatch "
                    "for block " + std::to_string(blockId));
        }
        cmg_sfc_t rebuilt;
        const std::vector<uint32_t> &perm = GetOrBuildSFCPerm(
            m_Backend, blockId, sfcMode, conn.data(), ncells, npc, cx, cy, cz, nnodes, rebuilt);
        if (rebuilt != sfcMode)
            throw std::runtime_error("[CompressMGARDCentroidOperator::DecompressV5] Rebuilt SFC "
                                     "mode disagrees with bitstream marker");
        std::vector<char> unsorted(ncells * typeSize);
        if (type == DataType::Float)
            cmg_perm_inverse_f32(reinterpret_cast<const float *>(avgOut),
                                 reinterpret_cast<float *>(unsorted.data()),
                                 perm.data(), ncells);
        else
            cmg_perm_inverse_f64(reinterpret_cast<const double *>(avgOut),
                                 reinterpret_cast<double *>(unsorted.data()),
                                 perm.data(), ncells);
        std::memcpy(avgOut, unsorted.data(), unsorted.size());
    }
    const auto t_sfc1 = high_resolution_clock::now();

    /* recombine bar_u(n) + r(n) → u(n) */
    const auto t_rec0 = high_resolution_clock::now();
    if (type == DataType::Float)
        cmg_centroid_recombine_f32(m_Backend,
            reinterpret_cast<const float *>(avgOut), conn.data(),
            ncells, npc, nnodes, reinterpret_cast<float *>(dataOut));
    else
        cmg_centroid_recombine_f64(m_Backend,
            reinterpret_cast<const double *>(avgOut), conn.data(),
            ncells, npc, nnodes, reinterpret_cast<double *>(dataOut));
    const auto t_rec1 = high_resolution_clock::now();


    if (m_Verbose)
        std::cerr << std::fixed << std::setprecision(2) << "[centroid] block=" << blockId
                  << " decompress be=" << (m_Backend == CMG_BACKEND_HIP ? "gpu" : "cpu")
                  << " out=" << sizeOut << " ms[conn=" << ms(t_conn0, t_conn1)
                  << " mgard_avg=" << ms(t_avg0, t_avg1) << " residual=" << ms(t_res0, t_res1)
                  << " sfc=" << ms(t_sfc0, t_sfc1) << " recombine=" << ms(t_rec0, t_rec1)
                  << " total=" << ms(t_start, high_resolution_clock::now()) << "]" << std::endl;

    /* Release the MGARD-X cache so its (device) workspace is returned after each
     * decompress, mirroring the release_cache on the compress side. */
    mgard_x::release_cache(avgCfg);
    std::free(avgOut);
    return sizeOut;
}

} // namespace plugin
} // namespace adios2

/* ------------------- plugin registration -------------------------------- */

extern "C" {
adios2::plugin::CompressMGARDCentroidOperator *
OperatorCreate(const adios2::Params &parameters)
{ return new adios2::plugin::CompressMGARDCentroidOperator(parameters); }

void OperatorDestroy(adios2::plugin::CompressMGARDCentroidOperator *obj) { delete obj; }
}
