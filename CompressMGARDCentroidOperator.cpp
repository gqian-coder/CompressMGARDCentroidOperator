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

#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"

#include <mgard/MGARDConfig.hpp>
#include <mgard/compress_x.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
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
    return CMG_BACKEND_CPU;
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

/* ------ per-process mesh cache (connectivity + coords) -------------- */

std::mutex                                                  g_meshMutex;
std::string                                                 g_meshFileName;
std::string                                                 g_connVarName;
adios2::ADIOS                                               g_ad;
adios2::IO                                                  g_io_mesh;
adios2::Engine                                              g_reader_mesh;
std::map<size_t, std::vector<int64_t>>                      g_connCache;
std::map<std::pair<int, size_t>, std::vector<double>>       g_coordsCache;

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
            g_io_mesh.SetParameter("SelectSteps", "0");
        }
        g_reader_mesh = g_io_mesh.Open(meshFile, adios2::Mode::ReadRandomAccess);
    }
    else if (g_meshFileName != meshFile || g_connVarName != connVar)
    {
        helper::Throw<std::invalid_argument>(
            "Operator", "CompressMGARDCentroidOperator", "LoadConnectivity",
            "Mesh-file/connectivity-variable cannot change between calls in the same process");
    }
    auto it = g_connCache.find(blockId);
    if (it != g_connCache.end()) return it->second;
    auto vConn = g_io_mesh.InquireVariable<int64_t>(connVar);
    if (!vConn)
        helper::Throw<std::invalid_argument>(
            "Operator", "CompressMGARDCentroidOperator", "LoadConnectivity",
            "Connectivity variable '" + connVar + "' not found in " + meshFile);
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

/* ------ MGARD compress helpers (cell averages) ---------------------- */

mgard_x::data_type ToMgardType(DataType t)
{
    if (t == helper::GetDataType<float>())  return mgard_x::data_type::Float;
    if (t == helper::GetDataType<double>()) return mgard_x::data_type::Double;
    helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                         "ToMgardType", "Only float/double supported");
    return mgard_x::data_type::Double;
}
size_t TypeNBytes(DataType t)
{
    if (t == helper::GetDataType<float>())  return sizeof(float);
    if (t == helper::GetDataType<double>()) return sizeof(double);
    helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                         "TypeNBytes", "Only float/double supported");
    return 0;
}

mgard_x::Config MakeMgardConfig(cmg_backend_t be)
{
    mgard_x::Config cfg;
    cfg.lossless = mgard_x::lossless_type::Huffman_Zstd;
    cfg.dev_type = (be == CMG_BACKEND_HIP) ? mgard_x::device_type::HIP
                                           : mgard_x::device_type::SERIAL;
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
    return cfg;
}

} // anonymous namespace

/* ====================================================================== */

CompressMGARDCentroidOperator::CompressMGARDCentroidOperator(const Params &parameters)
: PluginOperatorInterface(parameters)
{
    m_Backend = ChooseBackend();
    if (m_Backend == CMG_BACKEND_HIP && !cmg_hip_available())
    {
        std::cerr << "[centroid] CENTROID_DEVICE=gpu requested but HIP backend not available; "
                  << "falling back to CPU.\n";
        m_Backend = CMG_BACKEND_CPU;
    }
}

CompressMGARDCentroidOperator::~CompressMGARDCentroidOperator()
{
    if (g_reader_mesh) g_reader_mesh.Close();
}

bool CompressMGARDCentroidOperator::IsDataTypeValid(const DataType type) const
{
    return (type == helper::GetDataType<float>() || type == helper::GetDataType<double>());
}

size_t CompressMGARDCentroidOperator::GetHeaderSize() const { return headerSize; }

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
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                             "Operate", "Only float/double types are supported");

    /* ---- required params ---- */
    auto need = [&](const char *k) -> std::string {
        auto it = m_Parameters.find(k);
        if (it == m_Parameters.end())
            helper::Throw<std::invalid_argument>(
                "Operator", "CompressMGARDCentroidOperator", "Operate",
                std::string("missing parameter '") + k + "'");
        return it->second;
    };
    m_MeshFile         = need("meshfile");
    std::string connVar = need("connectivity_variable");
    m_NodesPerCell     = std::stoul(need("nodes_per_cell"));
    m_BlockId          = helper::StringToSizeT(need("blockid"), "blockid");

    double tolerance = 0.0;
    bool   hasTol = false;
    {
        auto it = m_Parameters.find("tolerance");
        if (it == m_Parameters.end()) it = m_Parameters.find("accuracy");
        if (it != m_Parameters.end()) { tolerance = std::stod(it->second); hasTol = true; }
    }
    if (!hasTol)
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                             "Operate", "missing mandatory parameter 'tolerance' (ABS)");

    double ebratio = 0.5;
    {
        auto it = m_Parameters.find("ebratio");
        if (it != m_Parameters.end()) ebratio = std::stod(it->second);
    }
    if (ebratio <= 0.0 || ebratio >= 1.0)
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                             "Operate", "'ebratio' must be in (0, 1)");

    {
        auto it = m_Parameters.find("mode");
        if (it != m_Parameters.end() && it->second != "ABS")
            helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                                 "Operate", "Only 'ABS' mode is supported");
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
    const double tol_avg  = (1.0 - ebratio) * tolerance;
    const double tol_resi = ebratio * tolerance;

    /* ---- shapes ---- */
    const Dims convertedDims = ConvertDims(blockCount, type, 1);
    if (convertedDims.size() != 1)
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                             "Operate", "expects a 1D nodal-array block");
    const size_t nnodes    = convertedDims[0];
    const size_t typeSize  = TypeNBytes(type);
    const size_t inputBytes = nnodes * typeSize;

    /* ---- write V5 header ---- */
    size_t off = 0;
    MakeCommonHeader(bufferOut, off, kBufferVersion);
    PutParameter(bufferOut, off, m_BlockId);
    PutParameter(bufferOut, off, (size_t)convertedDims.size());
    for (auto d : convertedDims) PutParameter(bufferOut, off, d);
    PutParameter(bufferOut, off, type);
    PutParameter(bufferOut, off, (uint8_t)MGARD_VERSION_MAJOR);
    PutParameter(bufferOut, off, (uint8_t)MGARD_VERSION_MINOR);
    PutParameter(bufferOut, off, (uint8_t)MGARD_VERSION_PATCH);

    if (inputBytes < thresholdSize)
    {
        PutParameter(bufferOut, off, false);
        headerSize = off;
        return 0;
    }
    PutParameter(bufferOut, off, true);

    /* ---- connectivity ---- */
    const auto t_conn0 = high_resolution_clock::now();
    const auto &conn = LoadConnectivity(m_MeshFile, connVar, m_BlockId);
    const auto t_conn1 = high_resolution_clock::now();
    if (conn.size() % m_NodesPerCell != 0)
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDCentroidOperator",
                                             "Operate", "Connectivity length not a multiple of nodes_per_cell");
    const size_t ncells = conn.size() / m_NodesPerCell;
    PutParameter(bufferOut, off, ncells);
    PutParameter(bufferOut, off, m_NodesPerCell);

    /* ---- centroid split ---- */
    std::vector<char> cellAvgBuf (ncells * typeSize);
    std::vector<char> residualBuf(nnodes * typeSize);
    const bool zeroAvg = (std::getenv("CENTROID_ZERO_AVG") != nullptr);
    const auto t_split0 = high_resolution_clock::now();
    if (zeroAvg)
    {
        std::memset(cellAvgBuf.data(), 0, cellAvgBuf.size());
        std::memcpy(residualBuf.data(), dataIn, inputBytes);
    }
    else if (type == helper::GetDataType<float>())
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
    if (!zeroAvg && IsSFCEnabled(m_Parameters))
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
        std::vector<uint32_t> perm(ncells);
        sfcMode = cmg_build_sfc_perm(m_Backend, conn.data(), ncells, m_NodesPerCell,
                                     cx, cy, cz, nnodes, want, perm.data());

        std::vector<char> sortedAvg(ncells * typeSize);
        if (type == helper::GetDataType<float>())
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
            std::cerr << "[centroid] mgard_x::compress returned empty buffer "
                         "(n=" << n << " tol=" << tol << ")\n";
            dstCap = 0;
            return;
        }
        if (mgardSize > dstCap)
        {
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

    /* quantize → host uint32 buffer */
    std::vector<uint32_t> quant(nnodes);
    const auto t_q0 = high_resolution_clock::now();
    if (type == helper::GetDataType<float>())
        cmg_quantize_zigzag_f32(m_Backend, reinterpret_cast<const float *>(residualBuf.data()),
                                nnodes, tol_resi, quant.data());
    else
        cmg_quantize_zigzag_f64(m_Backend, reinterpret_cast<const double *>(residualBuf.data()),
                                nnodes, tol_resi, quant.data());
    const auto t_q1 = high_resolution_clock::now();

    /* MGARD-X Huffman+ZSTD on host quant buffer (SERIAL backend; same bytes on either device) */
    char *outRes = bufferOut + off + sizeof(size_t);
    const size_t resCap = nnodes * sizeof(uint32_t) * 4 + 4096;
    const auto t_h0 = high_resolution_clock::now();
    size_t cap_res = mgardx_lossless::serial::Compress(quant.data(), nnodes, outRes, resCap);
    const auto t_h1 = high_resolution_clock::now();

    uint8_t actualResMarker = kResMarker_MGARDX_Huff;
    if (cap_res == 0)
    {
        /* fallback: MGARD-lossy on raw residual */
        std::cerr << "[centroid] block=" << m_BlockId
                  << " MGARD-X Huffman failed; falling back to MGARD on residual\n";
        actualResMarker = kResMarker_MGARD;
        /* rewind: marker byte stays, but our V5 reader trusts the byte we
         * write below. tol_resi field is unused for MGARD path; leave it.
         * cap_res slot still at off; outRes still valid. */
        cap_res = inputBytes;
        compressMgard1D(residualBuf.data(), nnodes, tol_resi, outRes, cap_res);
    }
    *(reinterpret_cast<uint8_t *>(bufferOut + markerOff)) = actualResMarker;
    PutParameter(bufferOut, off, cap_res);
    off += cap_res;

    const auto t1 = high_resolution_clock::now();
    auto ms = [](auto a, auto b) {
        return duration<double, std::milli>(b - a).count();
    };
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
        helper::Throw<std::runtime_error>("Operator", "CompressMGARDCentroidOperator",
            "InverseOperate", "Unsupported buffer version " + std::to_string((int)bufferVersion) +
            " (this build only reads V" + std::to_string((int)kBufferVersion) + ")");
    return DecompressV5(bufferIn + off, sizeIn - off, dataOut);
}

size_t CompressMGARDCentroidOperator::DecompressV5(const char *bufferIn, const size_t /*sizeIn*/,
                                                    char *dataOut)
{
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
    if (!isCompressed) { headerSize += off; return 0; }

    const size_t  ncells  = GetParameter<size_t>(bufferIn, off);
    const size_t  npc     = GetParameter<size_t>(bufferIn, off);
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
        helper::Throw<std::invalid_argument>(
            "Operator", "CompressMGARDCentroidOperator", "DecompressV5",
            "Decompression requires meshfile + connectivity_variable params or "
            "CENTROID_MESHFILE / CENTROID_CONN_VAR env vars");
    const auto &conn = LoadConnectivity(itMesh->second, itConn->second, blockId);
    if (conn.size() != ncells * npc)
        helper::Throw<std::runtime_error>(
            "Operator", "CompressMGARDCentroidOperator", "DecompressV5",
            "Connectivity length mismatch on decompression for block " + std::to_string(blockId));

    /* cell averages: MGARD decompress */
    const size_t cap_avg = GetParameter<size_t>(bufferIn, off);
    void *avgOut = nullptr;
    mgard_x::decompress(bufferIn + off, cap_avg, avgOut, false);
    off += cap_avg;

    /* residual: marker-dispatched */
    const uint8_t resMarker = GetParameter<uint8_t>(bufferIn, off);
    double tol_resi = 0.0;
    if (resMarker == kResMarker_MGARDX_Huff)
        tol_resi = GetParameter<double>(bufferIn, off);
    const size_t cap_res = GetParameter<size_t>(bufferIn, off);

    if (resMarker == kResMarker_MGARD)
    {
        void *resiOut = dataOut;
        mgard_x::decompress(bufferIn + off, cap_res, resiOut, true);
    }
    else if (resMarker == kResMarker_MGARDX_Huff)
    {
        std::vector<uint32_t> quant(nnodes);
        if (!mgardx_lossless::serial::Decompress(bufferIn + off, cap_res, nnodes, quant.data()))
            helper::Throw<std::runtime_error>(
                "Operator", "CompressMGARDCentroidOperator", "DecompressV5",
                "MGARD-X Huffman decompression failed for block " + std::to_string(blockId));
        if (type == helper::GetDataType<float>())
            cmg_dequantize_zigzag_f32(m_Backend, quant.data(), nnodes, tol_resi,
                                      reinterpret_cast<float *>(dataOut));
        else
            cmg_dequantize_zigzag_f64(m_Backend, quant.data(), nnodes, tol_resi,
                                      reinterpret_cast<double *>(dataOut));
    }
    else
    {
        helper::Throw<std::runtime_error>(
            "Operator", "CompressMGARDCentroidOperator", "DecompressV5",
            "Unknown residual marker " + std::to_string((int)resMarker));
    }
    off += cap_res;

    /* undo SFC permutation on cell averages */
    if (sfcMode != CMG_SFC_NONE)
    {
        const double *cx = nullptr, *cy = nullptr, *cz = nullptr;
        if (sfcMode == CMG_SFC_MORTON || sfcMode == CMG_SFC_HILBERT)
        {
            const bool ok = ResolveAndLoadCoords(m_Parameters, itMesh->second, itConn->second,
                                                 blockId, cx, cy, cz);
            if (!ok)
                helper::Throw<std::runtime_error>(
                    "Operator", "CompressMGARDCentroidOperator", "DecompressV5",
                    "SFC mode requires coordinates; set CENTROID_COORD_PREFIX or coordinates_prefix");
            auto vec = g_coordsCache.find({0, blockId});
            if (vec != g_coordsCache.end() && vec->second.size() != nnodes)
                helper::Throw<std::runtime_error>(
                    "Operator", "CompressMGARDCentroidOperator", "DecompressV5",
                    "Coordinate length mismatch for block " + std::to_string(blockId));
        }
        std::vector<uint32_t> perm(ncells);
        const cmg_sfc_t rebuilt = cmg_build_sfc_perm(m_Backend, conn.data(), ncells, npc,
                                                     cx, cy, cz, nnodes, sfcMode, perm.data());
        if (rebuilt != sfcMode)
            helper::Throw<std::runtime_error>(
                "Operator", "CompressMGARDCentroidOperator", "DecompressV5",
                "Rebuilt SFC mode disagrees with bitstream marker");
        std::vector<char> unsorted(ncells * typeSize);
        if (type == helper::GetDataType<float>())
            cmg_perm_inverse_f32(reinterpret_cast<const float *>(avgOut),
                                 reinterpret_cast<float *>(unsorted.data()),
                                 perm.data(), ncells);
        else
            cmg_perm_inverse_f64(reinterpret_cast<const double *>(avgOut),
                                 reinterpret_cast<double *>(unsorted.data()),
                                 perm.data(), ncells);
        std::memcpy(avgOut, unsorted.data(), unsorted.size());
    }

    /* recombine bar_u(n) + r(n) → u(n) */
    if (type == helper::GetDataType<float>())
        cmg_centroid_recombine_f32(m_Backend,
            reinterpret_cast<const float *>(avgOut), conn.data(),
            ncells, npc, nnodes, reinterpret_cast<float *>(dataOut));
    else
        cmg_centroid_recombine_f64(m_Backend,
            reinterpret_cast<const double *>(avgOut), conn.data(),
            ncells, npc, nnodes, reinterpret_cast<double *>(dataOut));

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
