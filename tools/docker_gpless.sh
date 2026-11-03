#!/bin/sh
# Produce a complete pkl (code half + data half) on a machine with no Qualcomm GPU.
# The host only needs Docker: the shim, the closed-source offline OpenCL compiler and the CPU
# recompute all run inside the container.
#
# usage: tools/docker_gpless.sh <source tree> <model.onnx> [out.pkl]
#   <source tree>  the tree carrying tinygrad_repo/ and tools/model_codegen.py
#   [out.pkl]      defaults to <model name>.gpless.pkl in the current directory
#
# environment:
#   IMAGE      required arm64 image: vendor libllvm-qcom (the offline compiler) + clang + a python with numpy
#   CONTAINER  persistent container name, default kgsl_gpless (created on first run, reused after that)
#   PYTHON     interpreter inside the container, default /usr/local/venv/bin/python3
#   STAGE      staging directory for the shim, default the image-agnostic ~/.cache/libkgsl_fake/gpless
#   KEEP_EXEC  set to 0 to remove the container afterwards (kept by default, handy for a second run)
set -eu

[ $# -ge 2 ] || { echo "usage: $0 <source tree> <model.onnx> [out.pkl]" >&2; exit 2; }

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(dirname "$here")
tree=$(CDPATH= cd -- "$1" && pwd) || { echo "source tree not found: $1" >&2; exit 2; }
model_dir=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd) || { echo "model not found: $2" >&2; exit 2; }
model_name=$(basename -- "$2")
out=${3:-$PWD/$(basename "${model_name%.onnx}").gpless.pkl}

image=${IMAGE:?IMAGE=<arm64 image carrying the vendor libllvm-qcom + clang> is required}
name=${CONTAINER:-kgsl_gpless}
python=${PYTHON:-/usr/local/venv/bin/python3}
stage=${STAGE:-$HOME/.cache/libkgsl_fake/gpless}

[ -d "$tree/tinygrad_repo" ] || { echo "no tinygrad_repo under $tree" >&2; exit 1; }
[ -f "$tree/tools/model_codegen.py" ] || { echo "no tools/model_codegen.py under $tree" >&2; exit 1; }
case "$out" in /*) ;; *) out=$PWD/$out ;; esac

# copy the shim sources into the staging area: make inside the container writes build/*.so (Linux ELF),
# and those must not land in a repository mounted from macOS
mkdir -p "$stage/out"
for d in src include tools tests props Makefile; do cp -R "$repo/$d" "$stage/"; done
if ! docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
  echo "creating container ${name} (mounts $tree / $model_dir / ${stage})"
  docker run -d --name "$name" --platform linux/arm64 \
    -v "$stage:/fake" -v "$tree:/tree" -v "$model_dir:/models" -v "$stage/out:/out" \
    --entrypoint bash "$image" -c 'sleep infinity' >/dev/null
fi

echo "== build the shim inside the container"
docker exec "$name" sh -c 'cd /fake && make' 2>&1 | tail -3

echo "== GPU-less packaging of $model_name"
docker exec "$name" sh -c "cd /fake && PYTHONPATH=/tree/tinygrad_repo:/tree/tools DEFAULT_DEVICE=qcom \
  tools/fkgsl --log /tmp/fk.log -- $python -u /tree/tools/model_codegen.py \
  /models/$model_name /out/$(basename "$out") --cpu-exec"

echo "== red-line self-check (all must be 0)"
docker exec "$name" sh -c "for k in GPU_COMMAND UNHANDLED TRIPWIRE; do \
  printf '   %-12s %s\n' \"\$k\" \$(grep -c \"\$k\" /tmp/fk.log || true); done"

cp "$stage/out/$(basename "$out")" "$out"
echo "== artifact $out"
echo "   acceptance: on a machine with the target GPU run tools/run_pkl_pair.py <device-made.pkl> $(basename "$out")"
[ "${KEEP_EXEC:-1}" = 0 ] && docker rm -f "$name" >/dev/null || true
