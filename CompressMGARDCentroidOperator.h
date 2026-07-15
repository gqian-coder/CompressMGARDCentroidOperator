/*
 * CompressMGARDCentroidOperator.h
 *
 * ADIOS2 plugin operator implementing the centroid-split + MGARD compression
 * scheme for unstructured FE meshes. Single shared library; runtime CPU/GPU
 * dispatch via the CENTROID_DEVICE env var (or MGARD_X_DEVICE_TYPE).
 *
 * The input 'tolerance' is RELATIVE; it is converted to an absolute tolerance
 *   absTol = tolerance * norm(block)
 * where norm is the data RMS value sqrt((1/N) sum x^2) (mode=REL_L2, the
 * default; the per-element "L2 norm" convention used by the driver/calc_err) or
 * the value range max-min (mode=REL_VAL). The min/max and sum-of-squares are
 * computed on the GPU (HIP/CUDA) when the backend is a GPU. absTol is split below.
 *
 * Encoder pipeline:
 *   1. Split          u(n) -> cell_avg(c)  +  residual(n)        [device kernel]
 *   2. SFC reorder    cell_avg sorted along a 3D Hilbert curve   [device kernel]
 *   3. MGARD compress cell_avg with tol_avg = (1-eb)*absTol      [mgard-x]
 *   4. Quantize+zigzag residual with quantum = 2*tol_resi, tol_resi = eb*absTol
 *   5. MGARD-X Huffman+ZSTD on the uint32 residual stream         [mgard-x]
 *
 * Bitstream version 5 (this file is the only writer):
 *   <ADIOS common header>
 *   size_t   blockId
 *   size_t   ndims
 *   size_t[] blockCount
 *   DataType type
 *   uint8    MGARD_VERSION_MAJOR
 *   uint8    MGARD_VERSION_MINOR
 *   uint8    MGARD_VERSION_PATCH
 *   bool     isCompressed
 *   ── if !isCompressed: raw input bytes follow (below-threshold path) ─
 *   ── if isCompressed: ────────────────────────────────────────────
 *   size_t   ncells
 *   size_t   nodes_per_cell
 *   uint8    sfcMode               (0=None, 1=Morton, 2=MinNode, 3=Hilbert)
 *   size_t   cap_avg
 *   bytes    cap_avg               (mgard_x::compress payload for cell averages)
 *   uint8    resMarker             (0=MGARD-lossy fallback,
 *                                   1=MGARD-X Huffman+ZSTD on uint32 quant)
 *   ── if resMarker == 1: ────────────────────────────────────────
 *   double   tol_resi              (quantum = 2 * tol_resi)
 *   ────────────────────────────────────────────────────────────────
 *   size_t   cap_res
 *   bytes    cap_res
 *
 *   Author: Qian Gong
 */

#ifndef COMPRESSMGARDCENTROIDOPERATOR_H_
#define COMPRESSMGARDCENTROIDOPERATOR_H_

#include "adios2.h"
#include "adios2/plugin/PluginOperatorInterface.h"

#include "DeviceKernels.h"

namespace adios2
{
namespace plugin
{

class CompressMGARDCentroidOperator : public PluginOperatorInterface
{
public:
    CompressMGARDCentroidOperator(const Params &parameters);
    ~CompressMGARDCentroidOperator();

    void AddExtraParameters(const Params &params) override;

    size_t Operate(const char *dataIn, const Dims &blockStart, const Dims &blockCount,
                   const DataType type, char *bufferOut) override;

    size_t InverseOperate(const char *bufferIn, const size_t sizeIn, char *dataOut) override;

    bool IsDataTypeValid(const DataType type) const override;

    size_t GetEstimatedSize(const size_t ElemCount, const size_t ElemSize,
                            const size_t ndims, const size_t *dims) const override;

private:
    size_t DecompressV5(const char *bufferIn, const size_t sizeIn, char *dataOut);

    std::string m_VersionInfo;
    std::string m_MeshFile;
    std::string m_ConnVarName;
    size_t      m_NodesPerCell = 0;
    size_t      m_BlockId = 0;
    cmg_backend_t m_Backend = CMG_BACKEND_CPU;
    bool        m_Verbose = false;

    /* extra parameters injected by the engine */
    std::string m_EngineName;
    std::string m_VariableName;
};

} // namespace plugin
} // namespace adios2

extern "C" {
adios2::plugin::CompressMGARDCentroidOperator *
OperatorCreate(const adios2::Params &parameters);
void OperatorDestroy(adios2::plugin::CompressMGARDCentroidOperator *obj);
}

#endif /* COMPRESSMGARDCENTROIDOPERATOR_H_ */
