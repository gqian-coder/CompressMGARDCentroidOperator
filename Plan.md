# Plan — Cell-Centroid (Average + Residual) Compression for Unstructured BP Data

## 0. Dependencies 
Run: source ~/frontier_model_to_load.sh 

## 1. Goal

For each mesh cell, split the nodal field `u` into two components:

- `u_ave(c)` — one scalar per cell (mean of the cell's vertex values)
- `u_resi(n)` — per-node residual capturing the deviation from the cell average

Compress both components independently with MGARD (or another error-bounded
codec). Reconstruction is the inverse: read both components, recombine, and
recover `u` within a guaranteed error bound.

This mimics the cell-average / spectral-point-residual split used in
`/lustre/orion/cfd164/proj-shared/gongq/refactorMesh/p_multilevel_delta/src`
(`nek_gpu_cellavg_split` / `nek_gpu_cellavg_unsplit`), but adapted to a CGNS-style
unstructured mesh stored in ADIOS2 BP, where nodal values are shared across
multiple cells.

---

## 2. Reference data format

File examined: `/lustre/orion/cfd164/proj-shared/gongq/sol_p1/sol_4114800_aver.bp`

Top-level groups (one per CGNS zone):

| Zone group | Element type | # blocks |
|---|---|---|
| `/volume`            | `HEXA_8` | 72 |
| `/quad_EXIT_Q2`      | `QUAD_4` | 8  |
| `/quad_INLET_Q2`     | `QUAD_4` | 2  |
| `/quad_PER_LOW_Q2`   | `QUAD_4` | 2  |
| `/quad_PER_UP_Q2`    | `QUAD_4` | 2  |
| `/tri_*_Q2` (8 zones)| `QUAD_4` (mislabeled `tri_*`, still QUAD_4) | varies |

Per zone we have:

- `<zone>/Elem/ElementType` — string attribute (`HEXA_8`, `QUAD_4`, …)
- `<zone>/Elem/ElementConnectivity` — `int64`, blocked, size `nodes_per_cell × ncells`
- `<zone>/GridCoordinates/CoordinateX|Y|Z` — `double`, blocked, size `nnodes`
- `<zone>/FlowSolution/<var>` — `double`, **2 timesteps**, blocked, size `nnodes`
  (vars: `P_aver`, `P_ms`, `Rho_aver`, `Rho_ms`, `U_aver`, `V_aver`, `W_aver`, `uu`, `uv`, `uw`, `vv`, `vw`, `ww`)

Verified for `/volume`:

- 72 blocks, each block has the SAME row count for ElementConnectivity, Coordinate*, and FlowSolution.
- For `HEXA_8`: connectivity length per block ≈ `6.40e6` = `8 × 800 037` cells.
- FlowSolution block size ≈ 3.2–4.5 M nodes; **`max(ElementConnectivity) < nnodes_in_block`**, so the connectivity uses **0-based, block-local node indices** — each block is a self-contained partition (shared boundary nodes are duplicated across blocks).

Other zones follow the same layout with `QUAD_4` (4 nodes per cell).

### Information already present in BP — nothing missing for the new method
- Element type (per zone, via attribute)
- Connectivity (per-block)
- Nodal field (per-block)
- Implicit `nodes_per_cell` from element type
- `ncells_per_block = connectivity_size / nodes_per_cell`

### Information that we may want to confirm / add
- The previous code (`mgard_adios_ge.cpp`) only auto-detected `FlowSolution/*` of type `double` and used **value-range relative tolerance per variable** (`tol*(max-min)`). The same strategy will be reused here.
- The BP file contains globals, time, and boundary-zone solutions. For the
  prototype we will start with `/volume` only (the bulk of the data) and add
  surface zones as a flag-controlled option later.
- The `Min()` / `Max()` calls used in `mgard_adios_ge.cpp` work on the full variable. We will keep the same logic for the split components (range computed from a first-pass scan of one timestep).

---

## 3. Key design question — shared vertices

Each mesh node is referenced by multiple cells (≈ 8 hex cells per interior
node). When we split, we must decide **which cell's average a given node's
residual is taken against**, and how reconstruction inverts that choice. Three
viable options:

### Option A — Per-cell duplicated representation (mirrors previous Nek5000 code)
For every (cell, local-vertex) pair, store
`r_{c,k} = u_{conn[c,k]} − u_ave(c)`.

- Per-cell averages: length `ncells`
- Residuals: length `nodes_per_cell × ncells`  (= same length as connectivity)
- Reconstruction: for each node, average the per-cell reconstructed values
  from every incident cell (or pick one).
- **Pros:** simplest, matches the previous code's data shapes 1-to-1.
  No mesh-topology metadata is consumed at decode time beyond connectivity.
- **Cons:** residual array is ≈ 8× larger than the original nodal array for
  HEX (4× for QUAD). MGARD compresses the residuals on this larger, redundant
  array. Useful when the residuals are very smooth/correlated cell-by-cell.

### Option B — Single residual per node, with node→cell assignment
Pick one cell per node (e.g., the lowest cell-id incident on it); store
`r_n = u_n − u_ave(owner(n))`.

- Per-cell averages: length `ncells`
- Residuals: length `nnodes`
- Owner map: length `nnodes` (small int) — must be stored or recomputed
  deterministically from connectivity at decode time.
- **Pros:** no storage expansion.
- **Cons:** residual magnitudes can be large for nodes whose owner-cell mean
  differs strongly from theirs (boundary nodes between two very different
  cells). Owner map adds metadata.

### Option C — Node-level "smoothed base" = mean of incident cell averages
Encoder:
1. Compute `u_ave(c)` for every cell.
2. For every node `n`, compute `bar_u(n) = mean_{c ∋ n} u_ave(c)` (average of
   incident cell averages, easily derived from connectivity).
3. Store `r_n = u_n − bar_u(n)`.

Decoder:
1. Decompress cell averages.
2. Recompute `bar_u(n)` from decompressed cell averages + connectivity
   (connectivity is already in the BP file, lossless).
3. `u_n = bar_u(n) + r_n`.

- **Pros:** no storage expansion; deterministic; symmetric encoder/decoder;
  small residual magnitudes (each node sees the local mean of its
  neighborhood).
- **Cons:** error in cell averages propagates through `bar_u`, so the effective
  error bound for `u` requires both components to be tightened — but this is
  the standard "operator + residual" pattern already used in
  `CompressMGARDMeshToGridOperator`.

**Recommendation: implement Option C as default, expose Option A behind a
CLI flag** for direct comparison with the previous Nek5000-style approach.
Option B can be added later if needed.

---

## 4. Error-bound accounting

Let the user-requested relative tolerance be `eps` and the variable range be
`R = max(u) − min(u)`. Target: `|u_recon − u| ≤ eps * R`.

### Option C
`u_recon = bar_u_recon + r_recon = (bar_u + e_b) + (r + e_r)`
`u_recon − u = e_b + e_r`
where:
- `e_b` is at most the average reconstruction error of the cell averages used
  by `bar_u(n)`. If MGARD compresses cell averages with absolute tolerance
  `tol_avg`, then `|e_b| ≤ tol_avg` (averaging reduces but cannot exceed the
  per-cell bound).
- `e_r` is bounded by MGARD's residual tolerance `tol_res`.

Choose `tol_avg + tol_res ≤ eps * R`. A safe default is to split evenly:
`tol_avg = tol_res = 0.5 * eps * R`. Make the split tunable via CLI.

### Option A
Per-(cell,vertex) residual `r_{c,k}` reconstructed with abs tol `tol_res` →
`|u_{conn[c,k]}_recon − u_{conn[c,k]}| ≤ tol_avg + tol_res` for any single
cell; averaging duplicates only decreases that bound.

---

## 5. Source-code reuse

We will pattern the new tool on `mgard_adios_ge.cpp`:

- BP reader / writer setup with MPI block distribution
- Variable auto-detection (`FlowSolution/*` of type `double`)
- `tolerance = tol * (max − min)` using `var.Min() / var.Max()`
- One MGARD ADIOS2 operator, two compressed variables per source variable
  (`<name>__cellavg`, `<name>__resi`)
- Block-by-block processing, contiguous block distribution per rank

GPU acceleration (HIP) for the split/unsplit kernels is **not in scope** for
the first prototype — CPU OpenMP is sufficient. The kernel signatures will
mirror `nek_gpu_cellavg_split` / `nek_gpu_cellavg_unsplit` so a HIP port
later is mechanical.

---

## 6. Repository layout to create

```
Unstructured-Centroid/
├── Plan.md                    (this file)
├── CMakeLists.txt
├── src/
│   ├── main_compress.cpp      # compression driver (Option C default)
│   ├── main_decompress.cpp    # reconstruction + L2/L_inf error check
│   ├── mesh_io.{h,cpp}        # BP zone enumeration + block iterator
│   ├── cell_avg_split.{h,cpp} # CPU/OpenMP implementations of Options A & C
│   └── element_topology.{h,cpp}  # nodes_per_cell from ElementType string
├── scripts/
│   ├── run_compress.sh
│   └── run_decompress.sh
└── README.md
```

---

## 7. Implementation phases

### Phase 1 — Scaffolding
1. Create `CMakeLists.txt` linking against ADIOS2, MGARD, MPI, OpenMP
   (mimic `Unstructured-ReMesh/mgard_adios_ge.cpp` build flags).
2. `element_topology.{h,cpp}`: map `"HEXA_8" → 8`, `"QUAD_4" → 4`,
   `"TRI_3" → 3`, `"TETRA_4" → 4`, etc.
3. `mesh_io.{h,cpp}`: discover zones, list `FlowSolution/*` doubles, read
   `ElementConnectivity` + nodal arrays block by block.

### Phase 2 — Split & Unsplit (CPU/OpenMP, Option C)
4. `cell_avg_split::split(connectivity, npc, ncells, nnodes, u, out_avg, out_resi)`
   - Pass 1 (parallel reduction over cells): compute `u_ave(c)`.
   - Pass 2 (atomic / per-thread accumulator over cells): for each node,
     accumulate `sum_u_ave(n) += u_ave(c)` and `incident_count(n) += 1`.
   - Pass 3 (parallel over nodes): `bar_u(n) = sum_u_ave(n) / incident_count(n)`,
     `r(n) = u(n) − bar_u(n)`.
5. `cell_avg_split::unsplit(connectivity, npc, ncells, nnodes, avg, resi, out_u)`
   - Recompute `bar_u(n)` from `avg` exactly as in step 4 (passes 2 & 3),
     then `u(n) = bar_u(n) + r(n)`.
6. (Optional) Implement Option A `split_per_cell` / `unsplit_per_cell` with
   length-`npc*ncells` residual buffer.

### Phase 3 — Compression driver
7. `main_compress.cpp`:
   - Parse `input.bp output.bp eps [--option A|C] [--avg-frac 0.5] [--zones volume,...]`.
   - For each zone, each timestep, each block:
     a. Read connectivity (once per file — connectivity is time-invariant).
     b. For each variable:
        - First pass (timestep 0 only): compute `min/max` via `Min()/Max()`,
          derive `abs_tol = eps * (max − min)`, split into
          `tol_avg = avg_frac * abs_tol`, `tol_res = (1 − avg_frac) * abs_tol`.
        - Run split → `(avg, resi)`.
        - Write `avg` and `resi` to two ADIOS2 variables, each with its own
          MGARD operator (ABS mode, distinct tolerances).
   - Also passthrough-copy non-FlowSolution variables (coords, connectivity,
     globals) so the output BP is self-describing for decompression.

### Phase 4 — Decompression / verification driver
8. `main_decompress.cpp`:
   - Read compressed BP, reconstruct `u` per block via `unsplit`.
   - Compare against original BP (path passed as CLI), report per-variable
     relative L2 and L_inf errors. Confirm `≤ eps`.

### Phase 5 — Reporting & tuning
9. Print compression ratios per variable, total bytes in/out, wall-clock time.
10. Sweep `avg_frac` ∈ {0.1, 0.3, 0.5, 0.7, 0.9} to find the optimum.
11. Compare against the 1D-block baseline (`mgard_adios_ge.cpp`) at matching
    `eps`.

### Phase 6 — Extensions (later)
- HIP port of `split` / `unsplit` (kernel signature already chosen).
- Option A and Option B variants.
- Surface zones (`quad_*`, `tri_*`).
- Bit-plane / ZFP backends to swap for MGARD.

---

## 8. Open questions for the user (please confirm)

1. **Shared-vertex strategy default**: confirm Option C (node-level smoothed
   base from incident cell averages) — that is the closest no-overhead
   analogue of the Nek5000 cellavg-split pattern. The previous code used
   Option-A-style duplication only because Nek5000 SP layout makes each
   solution point uniquely owned by one element; here, the same shape would
   imply ~8× residual storage on hex meshes.

2. **Variable scope**: process all `FlowSolution/*` doubles in `/volume`
   first, ignore surface zones for the prototype — OK?

3. **Tolerance split (`avg_frac`)**: start at 0.5 and tune empirically.

4. **GPU**: CPU/OpenMP for the first implementation, HIP port deferred.

If you reply with any overrides, I'll start at Phase 1.

---

# Refactor Plan — Cross-Device Portability (May 22, 2026)

## Goals (from user request)

1. **Bitstream portability**: data compressed on GPU must be decompressible on
   CPU and vice versa. This requires Huffman encoding, SFC ordering, and
   centroid split/residual to be byte-for-byte identical across devices.
2. **Single Huffman codec**: MGARD-X's `ComposedLosslessCompressor<T, H,
   DeviceType>` for both CPU and GPU operators (replaces the CPU-only
   `mgard::compress_memory_huffman` path in `LosslessCompression.hpp`).
3. **Source layout**: consolidate the CPU and GPU operators into a single
   translation unit (one `.cpp` plus one header), so the device-agnostic
   pipeline (parameter parsing, bitstream framing, Huffman call, ZSTD, ADIOS2
   plugin registration) cannot drift between CPU and GPU builds.
4. **Device-specific kernels** (SFC code generation, cell-centroid split,
   residual gather) remain isolated in a small device-only file with a stable
   host-callable C interface. Both operator targets link to **MGARD-serial**
   for the Huffman+ZSTD codec.

## Confirmation: MGARD-X Huffman is portable

Verified in `mgard/mgard-x/Lossless/Lossless.hpp` (serial install):

```cpp
template <typename T, typename H, typename DeviceType>
class ComposedLosslessCompressor : public LosslessCompressorInterface<T, DeviceType> {
  void Compress(Array<1,T,DeviceType>&, Array<1,Byte,DeviceType>&, int);
  void Decompress(Array<1,Byte,DeviceType>&, Array<1,T,DeviceType>&, int);
  Huffman<Q,S,H,DeviceType> huffman;
  Zstd<DeviceType>          zstd;
};
```

- The Huffman codebook serialization (`huffman.Serialize` / `Deserialize`) is
  written by a single device-agnostic implementation; the per-device adapter
  only changes *how* the tree is built/applied (parallel scan on GPU, serial
  loop on CPU), not the resulting bitstream.
- `Zstd<DeviceType>` always shells out to the same `libzstd`.
- Therefore: a payload `Compress`ed on `mgard_x::HIP` is bit-identical to one
  `Compress`ed on `mgard_x::SERIAL` for the same input + `Config`, and
  `Decompress` works on either device.
- We will pin **`H = uint64_t`** (max codeword 56 bits) for both devices to
  avoid the `exit(1)` in `GetCodebook.hpp` for heavy-tail residuals.

## Current state (before refactor)

| File | Lines | Role |
|---|---|---|
| `CompressMGARDCentroidOperator.cpp/.h` | 1421 + 109 | CPU operator. Uses `mgard::compress_memory_huffman` (CPU-only). Has full SFC (Morton, Hilbert, MinNodeId). |
| `CompressMGARDCentroidOperator_GPU.cpp/.h/.hpp` | 1757 + 80 + 423 | GPU operator. Has duplicated SFC (incl. CUDA/HIP kernel), centroid kernel, plus GPU+CPU Huffman dry-run paths. |
| `LosslessCompression.hpp` | 444 | CPU `mgard::compress_memory_huffman` wrapper (`lossless::CompressHuffmanZstd`). |
| `CompressResidualMGARDX_CPU_impl.{cpp,h}` | 127 + 29 | MGARD-X SERIAL Huffman wrapper compiled as pure CXX in the HIP target. |

Bitstream incompatibility points to eliminate:
- CPU `resMarker = 1` is `mgard::compress_memory_huffman` (CPU-only codec).
- GPU `resMarker = 2` is MGARD-X HIP Huffman (incompatible with CPU build).
- Hilbert/Morton routines duplicated across `.cpp` files → drift risk.
- `CompressMGARDCentroidOperator_GPU.hpp` (423 lines) holds HIP kernels +
  host helpers mixed together.

## Target layout

```
CompressMGARDCentroidOperator/
├── CompressMGARDCentroidOperator.hpp     # SINGLE header: operator class
│                                          # declaration, bitstream layout
│                                          # constants, SFC enums.
├── CompressMGARDCentroidOperator.cpp      # SINGLE implementation: ADIOS2
│                                          # plugin glue, Operate / Inverse,
│                                          # SFC (host), centroid split (host
│                                          # ref), MGARD-X Huffman call,
│                                          # bitstream framing. Compiled once
│                                          # per target as plain CXX.
├── DeviceKernels.h                        # Pure-C interface for device work:
│                                          #   compute_centroid_split_<T>()
│                                          #   gather_residual_<T>()
│                                          #   sfc_code_compute()
│                                          #   permutation_apply_<T>()
│                                          # All take host pointers; the impl
│                                          # decides H2D/D2H internally.
├── DeviceKernels_CPU.cpp                  # CXX impl: plain loops + OpenMP.
├── DeviceKernels_HIP.cpp                  # HIP impl: launches kernels.
└── CMakeLists.txt                         # Builds:
                                            #  - CompressMGARDCentroidOperator
                                            #    (CPU: links DeviceKernels_CPU
                                            #     + mgard_serial)
                                            #  - CompressMGARDCentroidOperator_GPU
                                            #    (HIP: links DeviceKernels_HIP
                                            #     + DeviceKernels_CPU (fallback)
                                            #     + mgard_serial. The .so
                                            #     re-exports the same operator
                                            #     class; ADIOS2 picks via
                                            #     ADIOS2_PLUGIN_PATH.)
```

### Why a single `.cpp` is feasible

The portable, device-agnostic code is:
- ADIOS2 plugin boilerplate (`OperatorCreate`, `MakeCommonHeader`,
  `PutParameter`, `GetParameter`)
- Parameter parsing (`tol`, `ebratio`, SFC mode, residual method)
- Block iteration / Array partitioning
- SFC permutation **application** (gather/scatter — trivially fast on host)
- Quantization → uint32 (1 pass over `nnodes`)
- MGARD-X Huffman+ZSTD compress/decompress call (one template
  instantiation, dispatches by `mgard_x::device_type` at runtime)
- Bitstream framing

The only device-specific code is:
- Centroid pass (sum vertex values into per-cell average using connectivity)
- Residual computation (`u_resi[n] = u[n] - u_ave[cell_of(n)]`)
- SFC code computation (per-cell uint64 Morton/Hilbert from centroid coords)
- Optionally: parallel quantize (the host loop is `O(nnodes)` and runs at
  memory bandwidth → keeping it on host is fine for the prototype).

These four operations have a stable C interface (one function each, host
pointers in/out). The HIP version internally allocates device buffers,
launches kernels, copies back. The CPU version uses OpenMP or plain loops.
Both produce **identical** results (Hilbert3D is integer math, identical on
CPU and GPU; centroid sum order is fixed by cell index, not parallel
reduction order).

### Single-file alternative considered

A single `.cu`/`.hip` file with `#ifdef __HIPCC__` blocks was rejected:
- Adds `LANGUAGE HIP` build constraint to the CPU target → MGARD-serial
  headers see `__HIPCC__` and switch to `MGARDX_COMPILE_HIP` → defeats the
  portability goal.
- Mixing `__global__` kernels and ADIOS2 boilerplate harms readability.

The chosen layout (one common `.cpp` + small device backends) keeps the
99% portable code in one place, isolates the 1% device code, and allows
both targets to compile cleanly.

## Bitstream format (V5, unified)

| Offset | Field | Notes |
|---|---|---|
| 0 | ADIOS2 common header | `MakeCommonHeader` |
| +1 | `uint8 version = 5` | bump from V4 |
| ... | `uint8 sfcMode` | 0/1/2/3 (None / Morton / MinNodeId / Hilbert) |
| ... | `uint64 nnodes`, `uint64 ncells`, `uint8 npc` | mesh shape |
| ... | `double tol`, `double avg_frac` | error bound config |
| ... | `uint8 resMarker` | 0 = MGARD (avg only); 1 = MGARD-X Huffman+ZSTD (both CPU and GPU) |
| ... | `uint64 avg_payload_len` + bytes | MGARD-compressed cell averages |
| ... | `uint64 resi_payload_len` + bytes | MGARD-X Huffman+ZSTD residuals (or MGARD residuals if `resMarker=0`) |
| ... | permutation (optional) | when `sfcMode != 0`, indices of cells in SFC order |

- **One single `resMarker = 1` for both devices** — produced by
  `ComposedLosslessCompressor<uint32_t, uint64_t, DeviceType>` where
  `DeviceType` is chosen at runtime via `cfg.dev_type`. The serialized bytes
  are device-independent.
- Old V4 files remain readable via a compatibility branch keyed on
  `version == 4`.

## Step-by-step plan

### Step 1 — Header consolidation
- Create `CompressMGARDCentroidOperator.hpp` (the single header).
- Move all enums (`SFCMode`, `ResidualMethod`), bitstream constants, and the
  operator class declaration into it.
- Mark the GPU-specific class as a thin alias / subclass of the CPU one,
  selecting different `mgard_x::device_type` at construction.

### Step 2 — Device-kernel split
- Create `DeviceKernels.h` with a C ABI for the four operations:
  ```c
  void cmg_centroid_split_f32(const float *u, const int64_t *conn,
                              size_t nnodes, size_t ncells, size_t npc,
                              float *u_ave, float *u_resi);
  void cmg_centroid_split_f64(const double *...);
  void cmg_sfc_codes(const double *cx, const double *cy, const double *cz,
                     size_t ncells, int useHilbert, uint64_t *codes_out);
  void cmg_gather_perm_f32 / f64(...);  /* if needed */
  ```
- `DeviceKernels_CPU.cpp` — extract reference impls from current
  `CompressMGARDCentroidOperator.cpp`. OpenMP optional.
- `DeviceKernels_HIP.cpp` — extract HIP kernels from
  `CompressMGARDCentroidOperator_GPU.hpp`/`.cpp`. Compiled with
  `LANGUAGE HIP`.

### Step 3 — Bitcompatibility test before MGARD-X switch
- Before deleting the legacy paths, add a unit-style test that:
  1. Compresses a block with the CPU operator.
  2. Decompresses with the GPU operator (and vice versa).
  3. Diffs reconstructed values element-wise with tolerance check.
- Once the test passes for the legacy V4 format, proceed with V5 migration.

### Step 4 — MGARD-X Huffman as the sole codec
- Replace all calls to `lossless::CompressHuffmanZstd` / `DecompressHuffmanZstd`
  (in both `.cpp` files) with a single helper:
  ```cpp
  template <typename T>
  size_t MGARDX_HuffmanZstd_Compress(const T *quant_u32, size_t n,
                                     mgard_x::device_type dev,
                                     char *outBuf, size_t cap);
  template <typename T>
  bool   MGARDX_HuffmanZstd_Decompress(const char *inBuf, size_t inLen,
                                       size_t n, mgard_x::device_type dev,
                                       T *quant_u32_out);
  ```
- This helper lives in the common `.cpp` and uses the same
  `ComposedLosslessCompressor<uint32_t, uint64_t, DeviceType>` template,
  selecting `DeviceType` at compile time via a thin `if constexpr` dispatch
  (or by templating the helper on `DeviceType` and instantiating both for
  the GPU build).

### Step 5 — Link both targets to MGARD-serial
- Already done for CPU operator and `mgardx_serial_huffman` static lib.
- For the GPU target, link `mgard_serial` as well so its translation units
  that use SERIAL can find headers; the HIP kernels remain in
  `DeviceKernels_HIP.cpp` and link only to `hip::device`.
- Remove `mgard::mgard` (HIP install) from `target_link_libraries` for both
  operators; HIP runtime needed only by `DeviceKernels_HIP.cpp`.
- Update `runconf_hip` to set `MGARD_SERIAL_DIR` if not already in cache.

### Step 6 — Delete legacy code
- Remove `LosslessCompression.hpp` (mgard::compress_memory_huffman path).
- Remove `CompressResidualMGARDX_CPU_impl.{cpp,h}` (merged into common cpp).
- Remove `CompressMGARDCentroidOperator_GPU.hpp` (HIP kernels move to
  `DeviceKernels_HIP.cpp`; the host-side helpers move to the common cpp).
- `CompressMGARDCentroidOperator_GPU.cpp` becomes a tiny stub that defines
  the GPU plugin entry point and constructs the operator with
  `dev_type = HIP`.

### Step 7 — End-to-end portability test
- Run the same input file through CPU and GPU operators with identical
  parameters (`tol`, `ebratio`, SFC mode).
- Verify:
  - Per-block CR matches within ~0.1% (Huffman+ZSTD output should be
    bit-identical, so CR matches exactly).
  - Reconstructed values match within `tol`.
  - Cross-device decompress (CPU-produced BP read by GPU operator and vice
    versa) succeeds and matches.

## Open decisions for user

1. **Quantization location**: keep on host (simplest, ~5 ms for 2.25M doubles)
   or push to device (saves the H2D for `u_resi` in GPU path)?
   Reply: given the residual-avg splitting is implemented on GPU in GPU path, the quantization should also stay in a kernel function; in addition, please check how does MGARD-x huffman function handles the memory internally: if it accepts GPU memory pointer, please pass the device pointer to mgard-x huffman function; if it does not accept GPU memory point, then please move the data back to host before calling huffman encoding. 
2. **Drop V4 read support** after V5 is stable, or keep forever for
   back-compat?
   Reply: drop all unused code after V5 is stable, but keep a version that calls MGARD's CPU Huffman encoding --> I want to compare the timing and compression ratio
3. **Single shared `.so`** with runtime device selection (one library,
   `dev_type` from env var) vs **two separate `.so`s** (current ADIOS2 plugin
   model). The user request prefers one source file; a single .so is the
   logical extension. Recommend: single `.so` named
   `libCompressMGARDCentroidOperator.so` that auto-selects device based on
   `cfg.dev_type` (env var `CENTROID_DEVICE=cpu|gpu`).
   Reply: single `.so`, but different runconfig to set up env var
4. **OpenMP for CPU centroid pass**: enable, or stay single-threaded for
   determinism?
   Reply: stay single-threaded for CPU path
