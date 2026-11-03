#!/bin/sh
# Baseline management: run once on a real device (fkgsl --real), or accept the current fake-device
# result, and write it into tests/golden/
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
build=${BUILD:-build}
isa=/tmp/libkgsl_fake_golden.bin
note=${1:-record}
if [ "$note" = update ]; then
  FK_ISA_OUT="$isa" "$build"/probe >/dev/null 2>&1 || true
  sha=$(sha256sum "$isa" | cut -d' ' -f1)
  size=$(python3 -c "import os;print(os.path.getsize('$isa'))")
  printf '%s  bytes=%s  kernel=probe(source fixed in tests/probe.c)  recorded=%s\n' \
    "$sha" "$size" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > tests/golden/a6xx_probe.sha256
  cat tests/golden/a6xx_probe.sha256
else
  cat tests/golden/a6xx_probe.sha256
fi
