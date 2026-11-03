"""Diagnosis: recompute image kernels with CPU+shim and compare them against the real-device result
half by half, to tell rounding noise apart from a genuine misalignment.

  DIAG_KERNEL=0 FP_CONTRACT=off PYTHONPATH=... python3 /tmp/diag_cpu_image.py <model.onnx>
"""
import os, sys, numpy as np
os.environ.setdefault("FLOAT16", "1"); os.environ.setdefault("IMAGE", "2")
os.environ.setdefault("NOLOCALS", "1"); os.environ.setdefault("JIT_BATCH_SIZE", "0")
os.environ.setdefault("TCACHE", "0")
sys.path.insert(0, "/tmp")
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.dtype import ImageDType
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.runtime import ops_cpu
import model_codegen as cg
from cpu_image_shim import make_renderer

TARGET = int(os.environ.get("DIAG_KERNEL", "0"))
FP = os.environ.get("FP_CONTRACT")
if FP is not None:
  orig = ops_cpu.ClangJITCompiler.compile
  def compile(self, src):
    import subprocess, platform, sys as _s
    args = ['-march=native', f'--target=aarch64-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding',
            '-fno-math-errno', '-nostdlib', '-fno-ident', f'-ffp-contract={FP}']
    from tinygrad.helpers import getenv
    obj = subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', *args, '-', '-o', '-'], input=src.encode())
    from tinygrad.runtime.support.elf import jit_loader
    return jit_loader(obj)
  ops_cpu.ClangJITCompiler.compile = compile

cg.install_numpy_weights()
jit = cg.build_graph(sys.argv[1])
items = [ei for ei in jit.captured._jit_cache if isinstance(ei.prg, CompiledRunner)
         and None not in ei.bufs and any(isinstance(b.dtype, ImageDType) for b in ei.bufs)]
ei = items[TARGET]
shim = make_renderer(Device["CPU"].renderer)
gb = ei.bufs
print(f"kernel#{TARGET} bufs={[str(b.dtype) for b in gb]} contract={FP}")
nb, saved = [], []
for b in gb:
  cb = Buffer("CPU", b.size, b.dtype).ensure_allocated()
  data = bytes(b.as_buffer()[:]) if b.is_allocated() else bytes(b.nbytes)
  saved.append(data); cb.copyin(memoryview(bytearray(data))); nb.append(cb)
runner = CompiledRunner(get_program(ei.prg.p.ast, shim))
runner(nb, ei.fixedvars, wait=True)
Device["CPU"].synchronize()

for i, (b, cb) in enumerate(zip(gb, nb)):
  if not isinstance(b.dtype, ImageDType): continue
  H, W = b.dtype.shape[0], b.dtype.shape[1]
  it = np.dtype(np.float16) if b.dtype.itemsize == 2 else np.dtype(np.float32)
  exp = np.frombuffer(saved[i], dtype=it).reshape(H, W, 4)
  got = np.frombuffer(bytes(cb.as_buffer()[:]), dtype=it).reshape(H, W, 4)
  if i not in set(getattr(runner.p, "outs", ()) or [0]): 
    print(f"  buf{i} input: {'identical' if (exp == got).all() else 'REWRITTEN!'}")
    continue
  ne = (exp != got)
  n = int(ne.sum())
  ulp = np.abs(got.astype(np.float32) - exp.astype(np.float32))
  eps = np.maximum(np.abs(exp.astype(np.float32)), 1e-30)
  if n == 0:
    print(f"  buf{i} output {H}x{W}: element-for-element identical")
    continue
  rows, cols = np.where(ne.any(axis=2))
  ch = np.zeros(4, dtype=int)
  for c in range(4): ch[c] = int(ne[:, :, c].sum())
  print(f"  buf{i} output {H}x{W}: differing elements {n}/{exp.size} ({100.0*n/exp.size:.2f}%) abs diff max={ulp.max():.6g} "
        f"rel max={(ulp/eps).max():.4g}  per channel={list(ch)}")
  print(f"        rows {rows.min()}..{rows.max()} (differences span {len(set(rows.tolist()))} rows) "
        f"cols {cols.min()}..{cols.max()}")
  print(f"        border only? first row {int(ne[0].sum())} last row {int(ne[-1].sum())} "
        f"first col {int(ne[:,0].sum())} last col {int(ne[:,-1].sum())} interior {int(ne[1:-1,1:-1].sum())}")
