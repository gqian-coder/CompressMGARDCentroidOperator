# CompressMGARDCentroidOperator

An ADIOS2 plugin operator that compresses nodal fields defined on unstructured
finite-element meshes by splitting the field into a **coarse** stream and a
**fine residual**, compressing each with the method best suited to it.

## Compression paths (`method` / `CENTROID_METHOD`)

### `group_centroid` (default)

Exploits the fact that the nodal field stores each *distinct* coordinate
location many times over (12.7x-23.3x on the rotor case, ~15x typical), because
every cell carries its own copy of shared nodes.

1. **Group by coordinate.** Nodes are bucketed by physical location (within
   `group_tol`), giving `ngroups` distinct locations -- only ~6.6% of `nnodes`.
2. **Coarse stream** = one value per distinct location, reordered along a
   space-filling curve over the *group locations* and compressed with MGARD.
3. **Residual** `u[n] - coarse[group(n)]` = per node, reordered along a
   space-filling curve over *all nodes*, then quantized and entropy coded.

The grouping and both permutations are derived from the mesh coordinates alone,
so they are recomputed at decompress time and **cost zero stored bytes**.

> **Note.** Despite the historical name, the coarse stream of `group_centroid`
> is *not* a per-cell average (p0). It is one value per distinct **vertex
> location** -- i.e. deduplication of replicated nodes, at p1 vertex
> granularity. The older per-cell-average formulation
> (`u_ave[c] = mean over the vertices of cell c`) is not the active default.

### `reorder_mgard`

Gathers the nodal field along a nodal space-filling curve and hands the whole
1-D sequence to MGARD's multilevel transform. No split, so no coarse stream and
no `N_cell` expansion -- MGARD's own critically sampled hierarchy performs the
coarse/detail separation instead.

## Space-filling curves

Two *independent* permutations are built per block, both via
`cmg_build_sfc_perm_nodes()` (Hilbert by default, Morton and min-node also
available). They are cached separately and serve different consumers:

| permutation | point set | size | used for |
|---|---|---|---|
| nodal   | all node coordinates          | `nnodes`  | residual ordering |
| group   | distinct-location representatives | `ngroups` | coarse-stream ordering |

A third builder, `cmg_build_sfc_perm()`, orders **cells** by their centroid
(p0). It exists in the device layer but is **not used** by the current operator
paths, both of which order points rather than cells.

The connectivity needed by encoder and decoder is read from a separate ADIOS2
BP file given by the `meshfile` parameter -- typically the same input BP that
already carries `/<zone>/Elem/ElementConnectivity` uncompressed.

See `Plan.md` for the design rationale and a comparison against the
per-cell-duplicate (Option A) and single-owner (Option B) alternatives.

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

## To print information
```bash
export CENTROID_VERBOSE=1
```

By default the operator does not print anything. Logs go to stderr.

## Operator parameters

| Name | Env override | Required | Description |
|------|--------------|----------|-------------|
| `meshfile`              | `CENTROID_MESHFILE`     | compress | BP file containing the connectivity variable. On decompress this is optional — see *Finding the mesh at decompress time* below. |
| `connectivity_variable` | `CENTROID_CONN_VAR`     | compress | Full name of the int64 connectivity variable, e.g. `/volume/Elem/ElementConnectivity`. On decompress this is optional — see below. |
| `nodes_per_cell`        | —                       | yes | `8` for `HEXA_8`, `4` for `QUAD_4`/`TETRA_4`, `3` for `TRI_3`, ... |
| `blockid`               | —                       | yes | Index of the block being compressed (set per `Put`). |
| `tolerance`             | —                       | yes | Absolute error bound for the reconstructed field. |
| `mode`                  | —                       | no  | Only `"ABS"` is supported (caller must convert REL→ABS). |
| `method`                | `CENTROID_METHOD`       | no  | `group_centroid` (default) or `reorder_mgard`. |
| `sfc_type`              | `CENTROID_SFC_TYPE`     | no  | `hilbert` (default), `morton`/`z`, or `minnode`. |
| `residual_sfc`          | `CENTROID_RESID_SFC`    | no  | Nodal SFC reorder of the residual before entropy coding. Default on. |
| `group_tol`             | `CENTROID_GROUP_TOL`    | no  | Coordinate-matching tolerance for the grouping. Too loose merges distinct locations; too tight inflates the coarse stream. |
| `coordinates_prefix`    | `CENTROID_COORD_PREFIX` | no  | Prefix of the coordinate variables in `meshfile`. Resolved at compress time and recorded in the bitstream; if unset it is derived from the connectivity name (`<zone>/Elem/...` → `<zone>/GridCoordinates/Coordinate`). |
| `adaptive_residual`     | `CENTROID_ADAPTIVE_RESID` | no | Adaptive residual quantization. |
| `fused`                 | `CENTROID_FUSED`        | no  | Fused split/quantize kernels. |
| `device_resident`       | `CENTROID_DEVRESIDENT`  | no  | Keep block geometry resident on device across a block's variables. |
| `hip_lossless`          | `CENTROID_HIP_LOSSLESS` | no  | Use the HIP lossless (Huffman) path. |
| `ebratio`               | —                       | no  | Fraction of the budget for the residual. Default `0.8`; `tol_resi = ebratio*tolerance`, `tol_avg = (1-ebratio)*tolerance`. |
| `s`                     | —                       | no  | MGARD `s` (smoothness). Default `0`. |
| `threshold`             | —                       | no  | Skip compression for blocks smaller than this many bytes. Default `100000`. |

## Finding the mesh at decompress time

Decoding needs the mesh: both `group_centroid` and `reorder_mgard` rebuild the
coordinate grouping / SFC permutation from the connectivity and coordinates
rather than storing them, which is what makes the reordering cost zero bytes.
Nothing in the payload itself identifies that mesh, so it has to be named.

From bitstream **V6** the compressor writes the connectivity variable name and
the resolved coordinate prefix into each block header (~70 bytes/block for
`/volume/Elem/ElementConnectivity`). A reader then has two modes:

**Self-contained** — set neither env var. The operator uses the file it is
decompressing as the mesh source, with the names from the header. This is the
normal case when the mesh was passed through into the compressed file, and it
needs no environment at all.

**External mesh** — set **both** `CENTROID_MESHFILE` and `CENTROID_CONN_VAR`.
Use this when the mesh is not stored alongside the compressed field.

Setting only one of the pair is rejected with a warning rather than being
half-honoured: an external file and a variable name inside it are only
meaningful together, and silently combining a user-supplied file with an
embedded name is how a decode ends up reading the wrong mesh. A user-supplied
`CENTROID_COORD_PREFIX` still overrides the embedded prefix.

V5 streams (written before this change) carry no names and can only be decoded
in external-mesh mode; the operator says so explicitly if both env vars are
missing. This build reads V5 and V6 and writes V6.

Additional runtime knobs: `CENTROID_DEVICE` (`cpu`/`hip`), `CENTROID_VERBOSE`,
`CENTROID_DEBUG`, `MGARD_X_DEVICE_TYPE`, `MGARDX_HUFF_BLOCK`, `MGARDX_HUFF_DICT`.

Reconstruction error bound (informal):
`|u_recon - u| <= tol_avg + tol_resi = tolerance`. A formal proof (using the
fact that averaging a set of values each within `tol_avg` is again within
`tol_avg`) is pending and will be added to `Plan.md`.

## Wrapper application

A reference wrapper that reads `/<zone>/FlowSolution/*` from the example
3-D rotor case and drives this operator block-by-block lives at
`/lustre/orion/cfd164/proj-shared/gongq/Unstructured-ReMesh/mgardCentroid_adios_ge.cpp`.

## GPU support

HIP support is fully implemented (`ENABLE_HIP=ON`). The coarse-stream
compression uses `mgard_x::compress` with the HIP device type; the nodal
residual quantization and lossless stages run on CPU (SERIAL path) in both
build variants.

The split/recombine kernels (`CentroidSplit`, `CentroidRecombine`) dispatch
to either a CPU or GPU implementation at runtime via `CENTROID_DEVICE`.
MGARD itself is CPU/GPU portable and the same operator `.so` handles both
paths when built with `ENABLE_HIP=ON`.
