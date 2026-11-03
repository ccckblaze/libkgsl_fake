#!/usr/bin/env python3
"""Drive the real tinygrad QCOM compiler onto the fake device, to show codegen needs no GPU at all.

Prerequisites: a source tree that carries tinygrad_repo, and libOpenCL on this machine.
usage:
  FK_TINYGRAD=../<your-tree>/tinygrad_repo tools/fkgsl --abort-exec -- \\
      python3 tests/qcom_compiler.py /tmp/qcom_isa.bin
Difference from tests/probe.c: this feeds genuine tinygrad-generated OpenCL C, not a hand-written kernel.
"""
import os
import pathlib
import sys

root = pathlib.Path(os.environ.get("FK_TINYGRAD") or "../tinygrad_repo")
if not (root / "tinygrad").is_dir():
    print(f"tinygrad source tree not found at {root}; point FK_TINYGRAD at it", file=sys.stderr)
    raise SystemExit(2)
sys.path.insert(0, str(root))
sys.path.insert(0, str(root.parent))

from tinygrad.runtime.ops_qcom import QCOMCompiler  # noqa: E402

src = pathlib.Path(os.environ.get("FK_SRC") or "tests/kernels/e2x.cl").read_text()
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qcom_isa.bin"

lib = QCOMCompiler("cl").compile(src)
blob = bytes(lib)
pathlib.Path(out).write_bytes(blob)
print(f"QCOMCompiler -> {len(blob)} bytes, sha256={__import__('hashlib').sha256(blob).hexdigest()}")
