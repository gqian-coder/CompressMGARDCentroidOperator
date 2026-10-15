#!/usr/bin/env python3
"""Run actual cache helpers with a checked fake HIP allocator (no GPU needed).

This supplements the HIP build/round-trip tests; it does not measure device
memory use or replace a real GPU run. The extracted helpers are compiled from
the current operator sources, so changes to eviction/reuse execute here.
"""

import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
operator = (ROOT / "CompressMGARDCentroidOperator.cpp").read_text()
hip = (ROOT / "DeviceKernels_HIP.cpp").read_text()


def section(source, start, end):
    return source[source.index(start):source.index(end, source.index(start))]


host_helpers = section(operator, "std::mutex                                                  g_operationMutex;",
                       "/* Reader initialization")
device_helpers = section(hip, "struct GeomDev {", "/* ---- fused open-loop")
device_evict = section(hip, "void cmg_hip_evict_geometry_cache(void)",
                       "/* These are the strong definitions")
range_parser = section(operator, "    bool hasNormalizationRange = false;",
                       "    double s = std::numeric_limits<double>::infinity();")
normalization = section(operator, "    const double normFactor = (relMode == RelMode::L2)",
                        "    const double absTol = tolerance * normFactor;")

# Returned cache references must remain protected for the complete operation.
for signature, end in [
    ("size_t CompressMGARDCentroidOperator::Operate(",
     "size_t CompressMGARDCentroidOperator::InverseOperate("),
    ("size_t CompressMGARDCentroidOperator::InverseOperate(",
     "void CompressMGARDCentroidOperator::ResolveMeshSource("),
]:
    assert "std::lock_guard<std::mutex> operation(g_operationMutex);" in section(operator, signature, end)
assert "SelectActiveBlock(m_BlockId);" in operator
assert "SelectActiveBlock(blockId);" in operator
assert "std::lock_guard<std::mutex> geometryUse(g_geomMutex);" in section(
    hip, "void group_compress_hip(", "/* Device-resident decompress recombine")

# Coordinate-only paths initialize the mesh reader without loading connectivity.
assert operator.count("LoadConnectivity(") == 1  # definition only
assert operator.count("EnsureMeshReader(") == 6  # definition, loader, four paths

preamble = r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace adios2 {
struct ADIOS {}; struct IO {};
struct Engine {
    bool open = false;
    explicit operator bool() const { return open; }
    void Close() { open = false; }
};
}
enum cmg_sfc_t { CMG_SFC_NONE, CMG_SFC_HILBERT };
thread_local int fakeDevice = 0;
std::unordered_map<void *, int> allocated;
std::atomic<size_t> allocationCount{0}, freeCount{0};
constexpr int hipMemcpyHostToDevice = 0;
int hipGetDevice(int *d) { *d = fakeDevice; return 0; }
int hipSetDevice(int d) { fakeDevice = d; return 0; }
template <typename T> int hipMalloc(T **p, size_t bytes) {
    *p = static_cast<T *>(std::malloc(bytes));
    assert(*p && allocated.emplace(*p, fakeDevice).second);
    ++allocationCount;
    return 0;
}
int hipMemcpy(void *dst, const void *src, size_t bytes, int) {
    assert(allocated.at(dst) == fakeDevice);
    std::memcpy(dst, src, bytes);
    return 0;
}
int hipFree(void *p) {
    assert(allocated.at(p) == fakeDevice); // wrong-device frees fail here
    allocated.erase(p);
    std::free(p);
    ++freeCount;
    return 0;
}
#define HIP_OK(call) do { assert((call) == 0); } while (0)
'''

main = r'''
void populateHost(size_t block) {
    g_connCache[block] = std::vector<int64_t>(16, 7);
    for (int axis = 0; axis < 3; ++axis)
        g_coordsCache[{axis, block}] = std::vector<double>(16, 2.0);
    g_nodePermCache[block] = std::vector<uint32_t>(16, 3);
    g_nodePermMode[block] = CMG_SFC_HILBERT;
    auto &groups = g_groupCache[block];
    groups.gid.resize(16); groups.gperm.resize(8); groups.ngroups = 8;
}
void assertHostEmpty() {
    assert(g_connCache.empty() && g_coordsCache.empty() && g_nodePermCache.empty());
    assert(g_nodePermMode.empty() && g_groupCache.empty());
}
int main() {
    RegisterOperatorInstance();
    RegisterOperatorInstance();
    g_reader_mesh.open = true;
    // The same supplied timestep-wide range yields the same budget for blocks
    // with different local extrema; absent range preserves the local behavior.
    assert(normalizedBudget({}, RelMode::VAL, 1e-3, 10, 20, 0, 4) == 0.01);
    const std::map<std::string, std::string> globalRange{{"normalization_range", "100"}};
    assert(normalizedBudget(globalRange, RelMode::VAL, 1e-3, 10, 20, 0, 4) == 0.1);
    assert(normalizedBudget(globalRange, RelMode::VAL, 1e-3, -1, 1, 0, 4) == 0.1);
    assert(normalizedBudget({{"normalization_range", "0"}}, RelMode::VAL,
                            1e-3, 10, 20, 0, 4) == 0.0);
    assert(normalizedBudget({}, RelMode::L2, 1e-3, 0, 0, 100, 4) == 0.005);
    for (const auto *invalid : {"nan", "inf", "-1"}) {
        bool rejected = false;
        try { normalizedBudget({{"normalization_range", invalid}}, RelMode::VAL,
                               1e-3, 0, 1, 0, 4); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    bool invalidModeRejected = false;
    try { normalizedBudget(globalRange, RelMode::L2, 1e-3, 0, 1, 0, 4); }
    catch (const std::invalid_argument &) { invalidModeRejected = true; }
    assert(invalidModeRejected);

    const uint32_t gid[] = {0, 0, 1, 1}, gperm[] = {1, 0}, nperm[] = {3, 2, 1, 0};
    SelectActiveBlock(0);
    populateHost(0);
    fakeDevice = 2;
    uint32_t *firstGid = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_geomMutex);
        const auto &g = get_geom_dev_locked(0, gid, gperm, nperm, 4, 2);
        firstGid = g.gid;
        assert(g.device == 2 && std::equal(g.gid, g.gid + 4, gid));
    }
    for (int variable = 0; variable < 13; ++variable) {
        SelectActiveBlock(0);
        std::lock_guard<std::mutex> lock(g_geomMutex);
        assert(get_geom_dev_locked(0, gid, gperm, nperm, 4, 2).gid == firstGid);
        assert(g_coordsCache.size() == 3 && g_connCache.size() == 1);
        assert(g_nodePermCache.size() == 1 && g_groupCache.size() == 1);
    }
    assert(allocationCount == 3 && freeCount == 0);
    // Removing an older variable's operator during the active block preserves
    // the geometry and shared reader still needed by another live instance.
    UnregisterOperatorInstance();
    assert(g_operatorInstances == 1 && g_reader_mesh.open);
    assert(allocated.size() == 3 && g_coordsCache.size() == 3 && freeCount == 0);
    RegisterOperatorInstance();
    fakeDevice = 0; // frees must switch to owning device 2 and restore 0
    SelectActiveBlock(1);
    assertHostEmpty();
    assert(g_geomCache.empty() && allocated.empty() && freeCount == 3 && fakeDevice == 0);

    // Lazy residual permutation and replacement of incompatible same-block data.
    populateHost(1);
    {
        std::lock_guard<std::mutex> lock(g_geomMutex);
        get_geom_dev_locked(1, gid, gperm, nullptr, 4, 2);
        assert(allocated.size() == 2);
        get_geom_dev_locked(1, gid, gperm, nperm, 4, 2);
        assert(allocated.size() == 3 && allocationCount == 6);
        get_geom_dev_locked(1, gid, gperm, nperm, 3, 2);
        assert(allocated.size() == 3 && freeCount == 6);
        get_geom_dev_locked(2, gid, gperm, nperm, 4, 2);
        assert(g_geomCache.size() == 1 && allocated.size() == 3 && freeCount == 9);
        fakeDevice = 5;
        get_geom_dev_locked(2, gid, gperm, nperm, 4, 2);
        assert(g_geomCache.at(2).device == 5 && allocated.size() == 3 && freeCount == 12);
    }
    fakeDevice = 0;

    // Simulate kernels holding the actual geometry-use mutex: eviction waits.
    std::unique_lock<std::mutex> geometryUse(g_geomMutex);
    std::promise<void> started;
    auto begun = started.get_future();
    auto eviction = std::async(std::launch::async, [&] {
        started.set_value();
        SelectActiveBlock(3);
    });
    begun.wait();
    assert(eviction.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    assert(allocated.size() == 3);
    geometryUse.unlock();
    eviction.get();
    assertHostEmpty();
    assert(allocated.empty() && g_geomCache.empty() && freeCount == allocationCount);

    // Revisiting an evicted block rebuilds it; raw-block transitions clear it too.
    SelectActiveBlock(0);
    populateHost(0);
    SelectActiveBlock(4);
    assertHostEmpty();
    assert(g_hasActiveBlock && g_activeBlock == 4);
    populateHost(4);
    UnregisterOperatorInstance();
    assert(g_reader_mesh.open && g_groupCache.size() == 1);
    UnregisterOperatorInstance();
    assertHostEmpty();
    assert(g_operatorInstances == 0 && !g_hasActiveBlock && !g_reader_mesh.open);
    std::cout << "PASS: same-block reuse; bounded host/device caches; device-owner frees; "
                 "lazy permutation; replacement; eviction waits for geometry use; "
                 "global range normalization and validation; shared operator lifetime\n";
}
'''

with tempfile.TemporaryDirectory(prefix="centroid-cache-regression-") as tmp:
    src = pathlib.Path(tmp) / "cache_regression.cpp"
    exe = pathlib.Path(tmp) / "cache_regression"
    src.write_text(preamble + device_helpers + device_evict +
                   "void cmg_evict_geometry_cache() { cmg_hip_evict_geometry_cache(); }\n" +
                   host_helpers + "enum class RelMode { L2, VAL };\n" +
                   "double normalizedBudget(const std::map<std::string, std::string> &m_Parameters, "
                   "RelMode relMode, double tolerance, double vmin, double vmax, "
                   "double sumsq, size_t nnodes) {\n" + range_parser + normalization +
                   "return tolerance * normFactor;\n}\n" + main)
    subprocess.run(["c++", "-std=c++17", "-pthread", "-Wall", "-Wextra",
                    str(src), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
