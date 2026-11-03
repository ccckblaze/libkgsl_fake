"""Probe: take the image kernels a real GPU already ran, recompute them with CPU+shim, compare byte for byte.

runs on an aarch64 machine with clang:
  PYTHONPATH=<tree>/tinygrad_repo python3 tools/probe_cpu_image.py <model.onnx>
"""
import os, sys, traceback, faulthandler
faulthandler.enable()
os.environ.setdefault("FLOAT16", "1"); os.environ.setdefault("IMAGE", "2")
os.environ.setdefault("NOLOCALS", "1"); os.environ.setdefault("JIT_BATCH_SIZE", "0")
sys.path.insert(0, "/tmp")
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.dtype import ImageDType
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.uop.ops import Variable
import model_codegen as cg
from cpu_image_shim import make_renderer

KEEP = int(os.environ.get("KEEP", "4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
stat = {"image": 0, "exact": 0, "diff": 0, "fail": 0}
bad = []

cg.install_numpy_weights()
jit = cg.build_graph(sys.argv[1])                      # executed on the real device, so the buffers hold ground truth
all_items = [ei for ei in jit.captured._jit_cache if isinstance(ei.prg, CompiledRunner)]
image_items = [ei for ei in all_items
               if None not in ei.bufs and any(isinstance(b.dtype, ImageDType) for b in ei.bufs)]
print(f"captured kernels={len(all_items)}  with image={len(image_items)}", flush=True)

shim = make_renderer(Device["CPU"].renderer)
print(f"shim={type(shim).__name__} base={type(Device['CPU'].renderer).__name__} "
      f"device={shim.device} has_local={shim.has_local} gep_arr_threshold={shim.gep_arr_threshold}", flush=True)

for n, ei in enumerate(image_items[:LIMIT or len(image_items)]):
  gb = ei.bufs
  print(f"[{n}] {ei.prg.p.name} bufs={[str(b.dtype) for b in gb]}", flush=True)
  try:
    nb, saved_data = [], []
    for b in gb:
      cb = Buffer("CPU", b.size, b.dtype).ensure_allocated()
      data = bytes(b.as_buffer()[:]) if b.is_allocated() else bytes(b.nbytes)
      saved_data.append(data)
      cb.copyin(memoryview(bytearray(data)))
      nb.append(cb)
    prg = get_program(ei.prg.p.ast, shim)
    assert prg.device == "CPU", prg.device
    if n < KEEP: open(f"/tmp/shim_src_{n}.c", "w").write(prg.src)
    runner = CompiledRunner(prg)
    var_vals = ei.fixedvars
    runner(nb, var_vals, wait=True)
    Device["CPU"].synchronize()
    got = [bytes(cb.as_buffer()[:]) for cb in nb]
    stat["image"] += 1
    diff_idx = [i for i, (g, s) in enumerate(zip(got, saved_data)) if g != s]
    outs = set(list(getattr(prg, "outs", ()) or ()))
    hit_outs = [i for i in diff_idx if (i in outs) or not outs]
    if not diff_idx:
      stat["exact"] += 1
    else:
      stat["diff"] += 1
      detail = [(i, sum(a != b for a, b in zip(got[i], saved_data[i])), len(saved_data[i])) for i in diff_idx[:4]]
      bad.append({"idx": n, "name": prg.name, "outs": sorted(outs), "diff_bufs": diff_idx[:8],
                  "byte_diff": detail, "strict": bool(hit_outs)})
  except Exception:
    stat["fail"] += 1
    if len(bad) < KEEP + 4: bad.append({"idx": n, "trace": traceback.format_exc()})
    if os.environ.get("RAISE"): raise
  if (n + 1) % 20 == 0: print(f"  {n+1}/{len(image_items)} {stat}", flush=True)

print("stats:", stat)
strict_diff = [b for b in bad if "trace" not in b and b.get("strict")]
print(f"output buffers genuinely differing: {len(strict_diff)}")
for b in bad[:KEEP]:
  print("---", {k: v for k, v in b.items() if k != "trace"})
  if "trace" in b: print(b["trace"][-1200:])
for b in strict_diff[:3]:
  print("source kept at /tmp/shim_src_%d.c" % b["idx"] if b["idx"] < KEEP else "", b)
sys.exit(0 if stat["fail"] == 0 and stat["diff"] == 0 else 2)
