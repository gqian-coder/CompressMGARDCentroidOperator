# CompressMGARDCentroidOperator

An ADIOS2 plugin operator that compresses nodal fields defined on unstructured
finite-element meshes by

1. splitting each block of the nodal field into
   - a **cell-average** component  `u_ave[c] = mean over the vertices of cell c`
   - a **nodal residual**          `r[n] = u[n] - bar_u[n]`,
     where `bar_u[n] = mean over the cell-averages of all cells incident on n`,
2. compressing both components independently with MGARD (ABS error mode).

The connectivity needed for both encoder and decoder is read from a separate
ADIOS2 BP file given by the `meshfile` parameter — typically the same input
BP that already carries `/<zone>/Elem/ElementConnectivity` uncompressed.

See `../Unstructured-Centroid/Plan.md` (now in this directory) for the design
rationale and a comparison against the per-cell-duplicate (Option A) and
single-owner (Option B) alternatives.

## Build configurations

Two `runconf` scripts are provided. Run either from a fresh build directory
inside the repo root.

### `runconf_cpu` — CPU-only (SERIAL backend)

Use when:
- No GPU is available or needed.
- Debugging the centroid split/recombine logic on a login node.
- Linking against a CPU-only MGARD install (`MGARD-serial/install-serial`).

```bash
mkdir build && cd build
source ../runconf_cpu
make -j$(nproc)
export ADIOS2_PLUGIN_PATH=$(pwd)
```

Runtime defaults with this build:
```bash
export CENTROID_DEVICE=cpu        # only option; HIP backend not compiled
export MGARD_X_DEVICE_TYPE=SERIAL
export CENTROID_SFC=on
export CENTROID_SFC_TYPE=hilbert
```

### `runconf_hip` — CPU+HIP (GPU) unified build

Use when:
- Running on Frontier (AMD Instinct MI210/MI300, `gfx90a`).
- You want GPU-accelerated cell-average compression via `mgard_x::compress`
  (HIP backend) while the residual quantization/lossless path stays on CPU.
- A/B profiling of serial vs HIP Huffman paths (`CMG_HUFF_AB=1`).

```bash
mkdir build_hip && cd build_hip
source ../runconf_hip
make -j$(nproc)
export ADIOS2_PLUGIN_PATH=$(pwd)
```

The same `.so` supports both GPU and CPU paths at runtime:
```bash
# GPU path (cell averages compressed on GPU):
export CENTROID_DEVICE=gpu
export MGARD_X_DEVICE_TYPE=HIP
export CENTROID_SFC=on
export CENTROID_SFC_TYPE=hilbert

# CPU fallback from the same HIP-built .so:
export CENTROID_DEVICE=cpu
export MGARD_X_DEVICE_TYPE=SERIAL
```

> **Note (temporary):** `runconf_hip` currently unloads `cray-hdf5` and
> `cray-netcdf` and points `ADIOS2_DIR` at the PrgEnv-gnu ADIOS2 install
> (`install-adios`). This workaround (`LINKER:--as-needed` in CMakeLists) is
> needed until ADIOS2 is rebuilt with PrgEnv-cray (`install-adios-cray`).
> Once that rebuild is done, remove the `module unload` lines and update
> `ADIOS2_DIR` to `install-adios-cray`.

## Operator parameters

| Name | Required | Description |
|------|----------|-------------|
| `meshfile`              | yes | Path to a BP file containing the connectivity variable. |
| `connectivity_variable` | yes | Full name of the int64 connectivity variable, e.g. `/volume/Elem/ElementConnectivity`. |
| `nodes_per_cell`        | yes | `8` for `HEXA_8`, `4` for `QUAD_4`/`TETRA_4`, `3` for `TRI_3`, ... |
| `blockid`               | yes | Index of the block being compressed (set per `Put`). |
| `tolerance`             | yes | Absolute error bound for the reconstructed field. |
| `mode`                  | no  | Only `"ABS"` is supported (caller must convert REL→ABS). |
| `ebratio`               | no  | Fraction of the budget allocated to the residual component. Default `0.8`. `tol_resi = ebratio * tolerance`, `tol_avg = (1 - ebratio) * tolerance`. |
| `s`                     | no  | MGARD `s` (smoothness) parameter. Default `0` (L∞-type bound). |
| `threshold`             | no  | Skip compression for blocks smaller than this many bytes. Default `100000`. |

Reconstruction error bound (informal):
`|u_recon - u| <= tol_avg + tol_resi = tolerance`. A formal proof (using the
fact that averaging a set of values each within `tol_avg` is again within
`tol_avg`) is pending and will be added to `Plan.md`.

## Wrapper application

A reference wrapper that reads `/<zone>/FlowSolution/*` from the example
3-D rotor case and drives this operator block-by-block lives at
`/lustre/orion/cfd164/proj-shared/gongq/Unstructured-ReMesh/mgardCentroid_adios_ge.cpp`.

## GPU support

HIP support is fully implemented (`ENABLE_HIP=ON`). The cell-average
compression uses `mgard_x::compress` with the HIP device type; the nodal
residual quantization and lossless stages run on CPU (SERIAL path) in both
build variants.

The split/recombine kernels (`CentroidSplit`, `CentroidRecombine`) dispatch
to either a CPU or GPU implementation at runtime via `CENTROID_DEVICE`.
MGARD itself is CPU/GPU portable and the same operator `.so` handles both
paths when built with `ENABLE_HIP=ON`.
