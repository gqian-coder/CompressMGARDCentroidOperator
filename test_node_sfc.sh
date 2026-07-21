#!/bin/bash
# A/B test the nodal SFC reorder (CENTROID_NODE_SFC) in the centroid operator.
#
# For a given variable it compresses twice -- node_sfc=0 (baseline centroid) and
# node_sfc=1 (nodal Hilbert reorder before the centroid split) -- decompresses
# both, and reports round-trip relative-L2 error and compressed size/CR. Equal
# error with a smaller size means the reorder is a pure win.
#
# Usage:  bash test_node_sfc.sh [var] [n_blocks] [eb]
#   var       default uu       (short name under /volume/FlowSolution/)
#   n_blocks  default 2        (use 72 for the full head-to-head)
#   eb        default 1e-2
#
# NOTE: uses mgard_adios_decompress, not mgardCentroid_adios_decompress -- the
# latter hardcodes the /hpMusic_base/hpMusic_Zone/ prefix and cannot read this
# /volume/ dataset. ADIOS2 applies the plugin's inverse operator on read either way.

set -o pipefail
source ~/frontier_model_to_load.sh >/dev/null 2>&1

VAR=${1:-uu}
NBLK=${2:-2}
EB=${3:-1e-2}
EBRATIO=0.8

WRAP=/lustre/orion/cfd164/proj-shared/gongq/Unstructured-ReMesh/build_cfd164
PLUGIN=/lustre/orion/cfd164/proj-shared/gongq/CompressMGARDCentroidOperator/build_hip
ADIOS2=/lustre/orion/cfd164/proj-shared/gongq/Software/ADIOS2/install-adios-cray
export LD_LIBRARY_PATH=$PLUGIN:$ADIOS2/lib64:/ccs/proj/cfd164/mgard/lib64:/ccs/proj/cfd164/mgard/lib:$LD_LIBRARY_PATH
export ADIOS2_PLUGIN_PATH=$PLUGIN
# SERIAL MGARD backend: HIP mode SIGSEGVs on Frontier login nodes. On a GPU
# compute node, export MGARD_X_DEVICE_TYPE=HIP to exercise the GPU SFC path.
unset MGARD_X_DEVICE_TYPE

INPUT=/lustre/orion/proj-shared/cfd164/3d_case/p1/perf-centroid/sol-uncompressed/sol_2895940_aver.bp
export CENTROID_MESHFILE=$INPUT
export CENTROID_CONN_VAR=/volume/Elem/ElementConnectivity
OUT=/lustre/orion/cfd164/proj-shared/gongq/CompressMGARDCentroidOperator/nsfc_test
export TMPDIR=$OUT/_tmp; mkdir -p $OUT $TMPDIR

export FLOW_VAR_LIST=$VAR
export NO_PASSTHROUGH=1

echo "=== node_sfc A/B : var=$VAR blocks=$NBLK eb=$EB ebratio=$EBRATIO ==="
for mode in 0 1; do
    export CENTROID_NODE_SFC=$mode
    c=$OUT/c_${VAR}_ns${mode}.bp; d=$OUT/d_${VAR}_ns${mode}.bp
    rm -rf "$c" "$d"
    MAX_STEPS=1 MAX_BLOCKS=$NBLK \
        $WRAP/mgardCentroid_adios_ge "$INPUT" "$c" $EB $EBRATIO --norm=l2 \
        > $OUT/c_${VAR}_ns${mode}.log 2>&1
    ce=$?
    $WRAP/mgard_adios_decompress "$c" "$d" > $OUT/d_${VAR}_ns${mode}.log 2>&1
    de=$?
    echo "  node_sfc=$mode compress_exit=$ce decompress_exit=$de data.0=$(stat -c%s $c/data.0 2>/dev/null)"
done

echo "--- round-trip accuracy + CR ---"
VAR=$VAR NBLK=$NBLK OUT=$OUT python3 - <<'PY' 2>&1 | grep -vE "Warning|perfstubs|libfabric"
import numpy as np, adios2.bindings as b, os
VAR=os.environ["VAR"]; NBLK=int(os.environ["NBLK"]); OUT=os.environ["OUT"]
ORIG="/lustre/orion/proj-shared/cfd164/3d_case/p1/perf-centroid/sol-uncompressed/sol_2895940_aver.bp"
V=f"/volume/FlowSolution/{VAR}"
def rd(path, nb):
    ad=b.ADIOS(); io=ad.DeclareIO("r"+os.path.basename(path))
    e=io.Open(path,b.Mode.Read); e.BeginStep()
    v=io.InquireVariable(V); out=[]
    for k in range(nb):
        v.SetBlockSelection(k); a=np.zeros(v.SelectionSize()); e.Get(v,a,b.Mode.Sync); out.append(a.copy())
    e.PerformGets(); e.Close(); return np.concatenate(out)
f=rd(ORIG,NBLK); unc=f.size*8
print(f"  uncompressed ({NBLK} blocks) = {unc} bytes")
res={}
for m in (0,1):
    try:
        g=rd(f"{OUT}/d_{VAR}_ns{m}.bp",NBLK)
        sz=os.path.getsize(f"{OUT}/c_{VAR}_ns{m}.bp/data.0")
        rel=np.linalg.norm(g-f)/np.linalg.norm(f); res[m]=(rel,sz)
        print(f"  node_sfc={m}: relL2={rel:.4e}  size={sz}  CR={unc/sz:.2f}")
    except Exception as ex:
        print(f"  node_sfc={m}: FAILED {ex}")
if 0 in res and 1 in res:
    print(f"  --> size change {100*(res[1][1]-res[0][1])/res[0][1]:+.1f}%  "
          f"CR gain {res[0][1]/res[1][1]:.3f}x  "
          f"(accuracy {'MATCHED' if abs(res[1][0]-res[0][0])<1e-12 else 'DIFFERS'})")
PY
echo "NODE_SFC TEST DONE"
