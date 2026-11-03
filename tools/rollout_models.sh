#!/bin/sh
# On a machine that has the target GPU, walk the whole CPU-packaging chain for the given models:
#   GPU reference artifact -> CPU recomputed artifact -> replay both on fresh inputs and compare outputs
#
# usage: TREE=<source tree> MODELS=<onnx dir> WORK=<output dir> tools/rollout_models.sh <model> [...]
#   <model> resolves to $MODELS/<model>.onnx, at least one is required
#   TREE must contain tinygrad_repo/ and tools/model_codegen.py; default model names are in the list below
set -eu

tree=${TREE:?TREE=<source tree> is required}
models=${MODELS:?MODELS=<onnx dir> is required}
work=${WORK:?WORK=<output dir> is required}
py=${PY:-python3}
codegen=$tree/tools/model_codegen.py
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
[ -f "$codegen" ] || { echo "cannot find $codegen" >&2; exit 1; }
mkdir -p "$work"
export PYTHONPATH="$tree/tinygrad_repo:$here"

for name in "$@"; do
  onnx=$models/$name.onnx
  [ -f "$onnx" ] || { echo "skipping: no $onnx" >&2; continue; }
  echo "== $name: GPU reference"
  "$py" -u "$codegen" "$onnx" "$work/gpu_$name.pkl" --keep-exec > "$work/gpu_$name.log" 2>&1
  echo "== $name: CPU recompute"
  "$py" -u "$codegen" "$onnx" "$work/cpu_$name.pkl" --cpu-exec > "$work/cpu_$name.log" 2>&1
  echo "== $name: output comparison + buffer classification"
  "$py" -u "$here/run_pkl_pair.py" "$work/gpu_$name.pkl" "$work/cpu_$name.pkl"
  "$py" -u "$here/check_data_seed.py" "$work/gpu_$name.pkl" "$work/cpu_$name.pkl"
done
