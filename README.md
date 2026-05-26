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

## Building (CPU-only)

```bash
mkdir build && cd build
source ../runconf_cpu
make
export ADIOS2_PLUGIN_PATH=$(pwd)
```

## Operator parameters

| Name | Required | Description |
|------|----------|-------------|
| `meshfile`              | yes | Path to a BP file containing the connectivity variable. |
| `connectivity_variable` | yes | Full name of the int64 connectivity variable, e.g. `/volume/Elem/ElementConnectivity`. |
| `nodes_per_cell`        | yes | `8` for `HEXA_8`, `4` for `QUAD_4`/`TETRA_4`, `3` for `TRI_3`, ... |
| `blockid`               | yes | Index of the block being compressed (set per `Put`). |
| `tolerance`             | yes | Absolute error bound for the reconstructed field. |
| `mode`                  | no  | Only `"ABS"` is supported (caller must convert REL→ABS). |
| `ebratio`               | no  | Fraction of the budget allocated to the residual component. Default `0.5`. `tol_resi = ebratio * tolerance`, `tol_avg = (1 - ebratio) * tolerance`. |
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

## GPU portability

The split / recombine kernels are isolated in template functions
(`CentroidSplit_CPU`, `CentroidRecombine_CPU`) so that a HIP/CUDA backend can
be added later behind a runtime flag, mirroring the structure of
`CompressMGARDMeshToGridOperator`. MGARD itself is already CPU/GPU portable.
