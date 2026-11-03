"""Work out which buffers of the "data half" must be bit-identical to a real device, and hard-check only those.

Saving a pkl bakes the contents of every allocated buffer in through Buffer.__reduce__. Once the
device loads it, the kernels that run are the ones kept in jit_cache, which means:
  - a buffer those kernels write in full -> overwritten on the first run, the baked-in value is irrelevant;
  - a buffer that is read but never written by them -> its value can only come from a pruned one-shot
    kernel (weight transpose / zero padding / width conversion), so it has to be bit-exact or the model changed;
  - a buffer written only in part (a view shows up in outs) -> the untouched remainder matters just as much.

This reads the ExecItem cache layout (captured.jit_cache); artifacts built on a linear-graph backend
have no such field and are out of scope here.

usage: python3 check_data_seed.py <ref.pkl> <ours.pkl>
"""
import os, sys, pickle
os.environ.setdefault("FLOAT16", "1"); os.environ.setdefault("IMAGE", "2")
os.environ.setdefault("NOLOCALS", "1"); os.environ.setdefault("JIT_BATCH_SIZE", "0")
import numpy as np
from tinygrad.engine.realize import CompiledRunner


def input_specs(cap):
  """(element count, dtype) of each declared model input -> how many buffers may still claim it.

  Loading a pkl runs CapturedJit.__post_init__, whose _clear_inputs() blanks the slots named by
  input_replace, so slots no longer identify the inputs; and the declared device is the *source*
  device (NPY/CPU) while the graph holds the buffer after the copy. So match on (size, dtype),
  letting each declared input claim one buffer only - anything left over stays in the critical class.
  """
  specs = {}
  for st, _vs, dt, _dev in getattr(cap, "expected_st_vars_dtype_device", []) or []:
    dims = getattr(st, "shape", None) or ()
    if dims and all(isinstance(d, int) for d in dims):
      key = (int(np.prod(dims)), str(dt))
      specs[key] = specs.get(key, 0) + 1
  return specs

def seed_order(jit):
  """Collect root buffers in first-seen order, tagged full-write / partial-write / read-only / input slot."""
  in_slots = {tuple(k) for k in (getattr(jit.captured, "input_replace", None) or {})}
  order, index = [], {}
  for n, ei in enumerate(jit.captured._jit_cache):
    outs = set(getattr(ei.prg.p, "outs", ()) or ()) if isinstance(ei.prg, CompiledRunner) else set()
    for i, b in enumerate(ei.bufs):
      if b is None: continue
      root, key = b.base, id(b.base)
      if key not in index:
        index[key] = len(order)
        order.append({"root": root, "full_write": False, "partial_write": False, "read": False, "input": False})
      ent = order[index[key]]
      if (n, i) in in_slots: ent["input"] = True
      if i in outs:
        if b is root: ent["full_write"] = True
        else: ent["partial_write"] = True
      else:
        ent["read"] = True
  return order

def elem_np(dt):
  if "imageh" in dt or dt in ("dtypes.half", "dtypes.float16"): return np.float16
  if "imagef" in dt or dt in ("dtypes.float", "dtypes.float32"): return np.float32
  return None

def report(dt, a, b):
  it = elem_np(dt)
  if it is None or not len(a) or len(a) != len(b): return "non-float or different length"
  x = np.frombuffer(a, it).astype(np.float32); y = np.frombuffer(b, it).astype(np.float32)
  d = np.abs(x - y)
  ulp = d / np.ldexp(np.maximum(np.abs(x), 1e-30), -10 if it == np.float16 else -23)
  return f"abs diff max={d.max():.6g} in ULP max={ulp.max():.4g} differing elements {int((d>0).sum())}/{len(x)}"

def main(ref_path, our_path):
  with open(ref_path, "rb") as f: ref = pickle.load(f)
  with open(our_path, "rb") as f: ours = pickle.load(f)
  ro, oo = seed_order(ref), seed_order(ours)
  specs = input_specs(ref.captured)
  if not specs:
    print("  note: this pkl declares Variable input shapes, so input slots fall back to position matching "
          "and may still land in the critical class")
  print(f"root buffers: reference={len(ro)} ours={len(oo)}")
  cats = {}
  bad, rest = [], []
  for n, (r, o) in enumerate(zip(ro, oo)):
    rb, ob = r["root"], o["root"]
    if (rb.device, rb.size, str(rb.dtype)) != (ob.device, ob.size, str(ob.dtype)):
      print(f"  #{n} structure mismatch: reference={rb} ours={ob}"); break
    a = bytes(rb.as_buffer()[:]) if rb.is_allocated() else b""
    b = bytes(ob.as_buffer()[:]) if ob.is_allocated() else b""
    spec = (int(rb.size), str(rb.dtype))
    placeholder = r["input"]
    if not placeholder and spec in specs and not r["full_write"] and specs[spec] > 0:
      specs[spec] -= 1
      placeholder = True
    kind = ("input-slot" if placeholder else
            "critical/read-only" if (r["read"] and not r["full_write"]) else
            "critical/partial-write" if (r["read"] and r["partial_write"]) else
            "overwritten" if r["full_write"] else "unreferenced")
    cats.setdefault(kind, [0, 0])
    cats[kind][0] += 1
    if a != b:
      cats[kind][1] += 1
      dt = str(rb.dtype)
      if kind.startswith("critical"):
        bad.append((n, kind, dt, int(rb.size), report(dt, a, b)))
      else:
        rest.append((n, kind, dt, int(rb.size), report(dt, a, b)))
  for kind, (total, diff) in sorted(cats.items()):
    print(f"  {kind:22s} {total:4d} buffer(s), differing from the reference: {diff}")
  for row in bad[:15]: print("   !critical", row)
  print(f"  the rest (overwritten / unreferenced) differ in {len(rest)} cases, sample:")
  for row in rest[:6]: print("     -", row)
  critical_bad = sum(1 for _, kind, *_ in bad if kind.startswith("critical"))
  print(f"verdict: {critical_bad} critical buffer(s) differ")
  return 0 if critical_bad == 0 else 2

if __name__ == "__main__":
  sys.exit(main(sys.argv[1], sys.argv[2]))
