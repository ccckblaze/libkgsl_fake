"""The second half of GPU-less packaging: replay QCOM kernels on the CPU backend to restore the data that
pruned one-shot kernels would otherwise leave empty.

  usage (a build or dev box with clang):
  PYTHONPATH=<tree>/tinygrad_repo python3 offdevice_cpuexec.py <model.onnx> <out.pkl> [ref.pkl]

Read this together with model_codegen.py in no-exec mode:
  - model_codegen only proves the "code half" reproduces bit for bit (kernel source + ISA match the device);
  - the other half of the pkl is buffer contents, copyout-baked at save time by Buffer.__reduce__.
    One-shot kernels pruned away (weight transpose / padding / width conversion) are not in jit_cache,
    so without execution they have no value: here every QCOM kernel call is redirected to run the same
    AST on the CPU, and the result is written back into the original QCOM buffer. That is what the pkl
    then saves: bytes recomputed here instead of bytes copied off a card.

Image buffers need an extra layer of OpenCL image2d semantics, see cpu_image_shim.py.

Note: the CPU only computes data; the kernel source/ISA stored in the pkl stays the QCOM rendering,
since the shim is attached to the CPU device alone and cannot touch the code half.
"""
import os, sys, pickle, collections
sys.path.insert(0, "/tmp")
os.environ.setdefault("FLOAT16", "1"); os.environ.setdefault("IMAGE", "2")
os.environ.setdefault("NOLOCALS", "1"); os.environ.setdefault("JIT_BATCH_SIZE", "0")
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.dtype import ImageDType
from tinygrad.engine import jit as jitmod
from tinygrad.engine import realize as rz
from tinygrad.engine.realize import CompiledRunner, ScheduleItem, lower_schedule_item
import model_codegen as cg
from cpu_image_shim import make_renderer

SKIP = "QCOM"
KEY = "_cpu_shadow"                # host-side bytes, hung off the buffer object itself
ENTRY = "_cpu_entry"               # (ast, metadata, fixedvars), hung off the runner object itself
stat = {"cpu_ok": 0, "cpu_fail": 0, "skipped": 0, "alias": 0, "image": 0}
errors = []

Device["CPU"].renderer = make_renderer(Device["CPU"].renderer)

orig_copyin = Buffer.copyin
orig_as_buffer = Buffer.as_buffer

def copyin(self, mv):
  ret = orig_copyin(self, mv)
  try: setattr(self, KEY, bytes(memoryview(mv).cast("B", shape=(memoryview(mv).nbytes,))))
  except Exception: pass
  return ret
Buffer.copyin = copyin

def host_bytes(b):
  data = getattr(b, KEY, None)
  if data is not None: return data
  if b.is_allocated():
    try: data = bytes(orig_as_buffer(b)[:])
    except Exception: data = bytes(b.nbytes)
  else: data = bytes(b.nbytes)
  setattr(b, KEY, data)
  return data

def cpu_run(ast, bufs, metadata, fixedvars, var_vals):
  if len({id(b.base) for b in bufs}) != len(bufs): stat["alias"] += 1
  if any(isinstance(b.dtype, ImageDType) for b in bufs): stat["image"] += 1
  nb = []
  for b in bufs:
    cb = Buffer("CPU", b.size, b.dtype).ensure_allocated()
    orig_copyin(cb, memoryview(bytearray(host_bytes(b))))
    nb.append(cb)
  ei = lower_schedule_item(ScheduleItem(ast, tuple(nb), metadata, fixedvars))
  ei.run(var_vals if var_vals is not None else fixedvars, wait=True, do_update_stats=False)
  for b, cb in zip(bufs, nb):
    data = bytes(orig_as_buffer(cb)[:])
    setattr(b, KEY, data)
    if b.is_allocated(): orig_copyin(b, memoryview(bytearray(data)))
  for cb in nb: cb.deallocate()

original_call = CompiledRunner.__call__
def call(self, rawbufs, var_vals, wait=False):
  if self.device != SKIP: return original_call(self, rawbufs, var_vals, wait)
  self.first_run = False
  entry = getattr(self, ENTRY, None)
  if entry is None:
    stat["skipped"] += 1
    return None
  ast, metadata, fixedvars = entry
  try:
    cpu_run(ast, list(rawbufs), metadata, dict(fixedvars), var_vals)
    stat["cpu_ok"] += 1
  except Exception:
    import traceback
    stat["cpu_fail"] += 1
    if len(errors) < 2: errors.append(traceback.format_exc())
    if os.environ.get("RAISE"): raise
  return None
CompiledRunner.__call__ = call

orig_run_schedule = rz.run_schedule
def run_schedule(schedule, var_vals=None, do_update_stats=True):
  schedule = list(schedule)
  for si in schedule:
    ei = rz.lower_schedule_item(si)
    if isinstance(ei.prg, CompiledRunner): setattr(ei.prg, ENTRY, (si.ast, si.metadata, si.fixedvars))
  return orig_run_schedule(schedule, var_vals, do_update_stats=do_update_stats)
rz.run_schedule = run_schedule
import tinygrad.tensor as tz
tz.run_schedule = run_schedule

orig_tj = jitmod.TinyJit.__call__
def tj(self, *a, **k):
  ret = orig_tj(self, *a, **k)
  if self.cnt > 1:                      # stop recomputing after the first pass, halving the runtime
    CompiledRunner.__call__ = original_call
  return ret
jitmod.TinyJit.__call__ = tj

cg.install_numpy_weights()
jit = cg.build_graph(sys.argv[1])
print("stats:", stat, "failures:", len(errors))
for e in errors: print(e[-1500:])
with open(sys.argv[2], "wb") as f: pickle.dump(jit, f)
print("pkl written", os.path.getsize(sys.argv[2]) / 1e6, "M")
def root_buffers(j):
  seen, out = set(), collections.defaultdict(list)
  for ei in j.captured._jit_cache:
    for b in ei.bufs:
      if b is None or b._base is not None or id(b) in seen: continue
      seen.add(id(b))
      out[(b.device, b.size, str(b.dtype))].append(bytes(b.as_buffer()[:]) if b.is_allocated() else b"")
  return out

def analyze(ref):
  np_dtype = {"dtypes.half": np.float16, "dtypes.float": np.float32, "dtypes.float16": np.float16,
              "dtypes.float32": np.float32}
  def elem(dt):
    if "imageh" in dt: return np.float16
    if "imagef" in dt: return np.float32
    return np_dtype.get(dt)
  a, b = root_buffers(ref), root_buffers(jit)
  cats = collections.Counter()
  for key in sorted(set(a) & set(b), key=lambda k: -len(b[k][0])):
    it = elem(key[2])
    for n, (x, y) in enumerate(zip(a[key], b[key])):
      if x == y: cats["identical"] += 1; continue
      if it is None or not x: cats["non-float or differs(unknown)"] += 1; continue
      ex, go = np.frombuffer(x, it).astype(np.float32), np.frombuffer(y, it).astype(np.float32)
      d = np.abs(go - ex)
      scale = np.maximum(np.abs(ex), np.finfo(np.float16).tiny)
      ulp = d / np.ldexp(scale, -10 if it == np.float16 else -23)
      cats[f"max {int(np.ceil(ulp.max())) if ulp.max() < 8 else '>8'} ULP"] += 1
      print(f"  {key[2]:26s} n={key[1]:>8d} copy {n} differing elements {int((d > 0).sum()):>8d}/{len(ex)} "
            f"abs diff max={d.max():.6g} in ULP max={ulp.max():.3f}")
  print("buffer classes:", dict(cats))

if len(sys.argv) > 3:
  with open(sys.argv[3], "rb") as f: ref = pickle.load(f)
  ok = cg.compare(sys.argv[3], jit)
  analyze(ref)
  sys.exit(0 if ok else 2)
