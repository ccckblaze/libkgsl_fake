import os, sys
os.environ.setdefault("FLOAT16", "1"); os.environ.setdefault("IMAGE", "2")
os.environ.setdefault("NOLOCALS", "1"); os.environ.setdefault("JIT_BATCH_SIZE", "0")
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.dtype import _to_np_dtype
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv
sys.path.insert(0, "/tmp")
import model_codegen as cg

MODE, LIMIT = getenv("MODE", "noexec"), int(getenv("ONNXLIMIT", "15"))
SHIM = getenv("SHIM", "1")
onnx_file = sys.argv[1]
if MODE == "noexec":
  if SHIM == "1":
    cg.install_numpy_weights()
  cg.no_execution_mode("runner")

from extra.onnx import OnnxRunner
run = OnnxRunner(onnx_file)
import extra.onnx as onnx_mod
onnx_mod.limit = LIMIT
np.random.seed(100)
types = {n: (dtypes.float32 if s.dtype is dtypes.float16 else s.dtype) for n, s in run.graph_inputs.items()}
vals = {k: (np.random.randn(*s.shape).astype(_to_np_dtype(types[k])) * np.float32(8.0)).astype(_to_np_dtype(types[k]))
        for k, s in sorted(run.graph_inputs.items())}
inputs = {k: Tensor(v, device=Device.DEFAULT).realize() for k, v in vals.items()}
ret = run(inputs)
for name, t in ret.items():
  sched = Tensor.schedule(t)
  chain = [str(u.op) for u in t.uop.toposort()][:8]
  print(f"limit={LIMIT} mode={MODE} shim={SHIM} {name}: sched items={len(sched)} op={t.uop.op} arg={getattr(t.uop,'arg',None)}")
  print("   upstream:", chain)
