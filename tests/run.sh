#!/bin/sh
# Runs on any GPU-less aarch64 Linux: proves the fake device makes the closed-source compiler emit
# ISA that is bit-identical to the real hardware's, and holds the three red lines: 0 GPU_COMMAND,
# 0 unrecorded ioctl, 0 real-device leaks.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fkgsl="$root/tools/fkgsl"
cd "$root"
build=${BUILD:-build}
log=${FK_LOG:-/tmp/libkgsl_fake_test.log}
isa=${FK_ISA_OUT:-/tmp/libkgsl_fake_test_isa.bin}
golden=tests/golden/a6xx_probe.sha256
say() { printf '%s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
wipe() { for f in "$@"; do if [ -e "$f" ]; then python3 -c "import sys,os; os.unlink(sys.argv[1])" "$f"; fi; done; }

[ "$(uname -s)" = Linux ] || fail "self-test only runs on Linux (on macOS use a container, see README)"
make -s all >/dev/null
[ -x "$build/probe" ] || fail "probe did not build: this environment lacks the OpenCL runtime/headers"
# guard against a stale binary passing for a fresh one: make can fail and leave the old product behind
python3 - "$build/probe" tests/probe.c <<'PYCHK' || fail "probe is not newer than tests/probe.c: a previous build failed and the test used the stale artifact"
import os, sys
raise SystemExit(0 if os.path.getmtime(sys.argv[1]) >= os.path.getmtime(sys.argv[2]) else 1)
PYCHK

wipe "$log" "$isa"

say "== 0) both compile paths must produce the same ISA"
compile_pass() {                                 # $1=destination $2=label
  if ! FK_LOG="$log" FK_ISA_OUT="$1" "$fkgsl" --abort-exec -- "$build"/probe >/tmp/libkgsl_fake_test.out 2>&1; then
    cat /tmp/libkgsl_fake_test.out
    fail "the compile stage failed ($2)"
  fi
  grep -q '^== DEVICE QUALCOMM Adreno' /tmp/libkgsl_fake_test.out || fail "no Adreno device enumerated ($2)"
  grep -q '^== MARK DONE' /tmp/libkgsl_fake_test.out || fail "never reached DONE ($2)"
  printf '   %-24s %s bytes  %s\n' "$2" \
    "$(python3 -c "import os,sys;print(os.path.getsize(sys.argv[1]))" "$1")" \
    "$(sha256sum "$1" | cut -d" " -f1)"
}
compile_pass "$isa" "embedded property table"
base_sha=$(sha256sum "$isa" | cut -d" " -f1)

wipe "$log" /tmp/libkgsl_fake_props.bin
export FK_PROPS="$root/props/a6xx.properties"
compile_pass /tmp/libkgsl_fake_props.bin "FK_PROPS runtime override"
unset FK_PROPS
[ "$(sha256sum /tmp/libkgsl_fake_props.bin | cut -d" " -f1)" = "$base_sha" ] \
  || fail "the ISA changed after the FK_PROPS override, so property parsing is not equivalent"

say "== 1) compile-stage red lines: no execution, no leaked ioctl, no touch of the real device"
log_absent() {
  if grep -q "$1" "$log"; then
    grep -n "$1" "$log" | head -5
    fail "$2"
  fi
}
log_absent 'GPU_COMMAND' "GPU_COMMAND appeared during compilation: the closed runtime's compile path needs execution, so the fake-device approach does not hold"
log_absent 'UNHANDLED' "an unrecorded KGSL ioctl showed up, which the fake device cannot answer - re-record on real hardware with tools/fkgsl --rec"
log_absent 'TRIPWIRE' "a real device fd leaked into this run, so the \"no GPU needed\" conclusion of this round is void"

sha=$(sha256sum "$isa" | cut -d' ' -f1)
if [ -f "$golden" ]; then
  want=$(awk '{print $1}' "$golden")
  [ "$sha" = "$want" ] || fail "does not match the baseline: $sha != ${want} (baseline in ${golden})"
  say "   bit-identical to the real-device baseline"
else
  say "   no baseline file; create one on a real device with sh tests/golden.sh update"
fi

say "== 2) execute stage: correct numbers are NOT expected (tier 3 is not implemented)"
wipe "$log" "$isa"
if timeout 30 env FK_LOG="$log" "$fkgsl" --allow-exec -- "$build"/probe full >/tmp/libkgsl_fake_test_full.out 2>&1; then
  if grep -q '^== VERIFY OK' /tmp/libkgsl_fake_test_full.out; then
    fail "the fake device computed the right answer, which means it got a real GPU: strict leak detection failed"
  fi
  say "   ran but the numbers are wrong ($(grep '^== VERIFY' /tmp/libkgsl_fake_test_full.out || echo none)) - as expected"
else
  say "   the execute stage crashed or hung (no execution simulation, so the runtime never gets a GPU write-back) - as expected, see /tmp/libkgsl_fake_test_full.out"
fi

say "== 3) leak guard: a real device fd that bypassed symbol interception must still be caught"
if [ -x "$build/tripwire" ]; then
  if "$fkgsl" -- "$build"/tripwire >/tmp/libkgsl_fake_test_tw.out 2>&1; then
    if grep -q 'TRIPWIRE skip' /tmp/libkgsl_fake_test_tw.out; then
      say "   skipped: no usable /dev/kgsl-3d0 on this machine (the guard itself still builds here)"
    else
      cat /tmp/libkgsl_fake_test_tw.out
      fail "a KGSL ioctl on a real device fd was not intercepted, so the fake device may have been running alongside real hardware"
    fi
  else
    twrc=$?
    if grep -q 'is not a fake fd issued by this shim' /tmp/libkgsl_fake_test_tw.out; then
      say "   guard works: strict mode aborted on the spot (rc=${twrc})"
    else
      cat /tmp/libkgsl_fake_test_tw.out
      fail "tripwire exited abnormally, but not because the guard fired"
    fi
  fi
else
  say "   skipped: no tripwire binary"
fi

say "== 4) a real kernel (with loops/memory access) also only needs the compile path"
if [ -f tests/kernels/e2x.cl ]; then
  wipe "$log" /tmp/libkgsl_fake_e2x.bin
  FK_SRC_FILE=tests/kernels/e2x.cl compile_pass /tmp/libkgsl_fake_e2x.bin "e2x.cl compile"
  e2x_sha=$(sha256sum /tmp/libkgsl_fake_e2x.bin | cut -d" " -f1)
  [ "$e2x_sha" != "$base_sha" ] || fail "e2x hashed the same as the fixed probe, so FK_SRC_FILE did not take effect"
  if [ -f tests/golden/a6xx_e2x.sha256 ]; then
    want_e2x=$(awk '{print $1}' tests/golden/a6xx_e2x.sha256)
    [ "$e2x_sha" = "$want_e2x" ] || fail "e2x does not match the real-device baseline: $e2x_sha != $want_e2x"
    say "   e2x is bit-identical to the real-device baseline"
  fi
  log_absent 'GPU_COMMAND' "GPU_COMMAND appeared while compiling e2x"
  log_absent 'UNHANDLED' "an unrecorded ioctl appeared while compiling e2x"
  log_absent 'TRIPWIRE' "compiling e2x leaked to the real device"
else
  say "   skipped: no tests/kernels/e2x.cl"
fi

say "== 5) symbols taken from a libc handle via ctypes must also land on the shim"
if command -v python3 >/dev/null 2>&1; then
  "$fkgsl" -- ${PYTHON:-python3} "$root/tests/ctypes_lookup.py" "$root/$build/libkgsl_fake.so" \
    || fail "the ctypes path bypassed symbol interception, see tests/ctypes_lookup.py"
else
  say "   skipped: no python3 on this machine"
fi

say "== PASS"
