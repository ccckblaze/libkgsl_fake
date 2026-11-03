#!/usr/bin/env python3
"""Output-level acceptance: replay two artifacts on the same fresh inputs and compare what the
model produces. This is the final judge of whether the *data* half is equivalent.

It must run on a machine with the target GPU: the artifact holds QCOM kernels, and only the
card can execute them. There is no offline fallback here -- by design, a run without hardware
would either hang or produce numbers that prove nothing. For cardless inspection use
`tools/pkl_dims.py` (launch dimensions) and `tools/check_data_seed.py` (buffer classification).

usage: python3 run_pkl_pair.py <a.pkl> [<b.pkl> ...]        (SEEDS=7,3,... to widen the check)
"""
import io
import os
import pickle
import sys

KGSL_NODE = os.environ.get("FK_KGSL_PATH", "/dev/kgsl-3d0")
if not os.path.exists(KGSL_NODE):
  print(f"run_pkl_pair: {KGSL_NODE} not found -- this is the GPU-side acceptance step, it replays the "
        f"artifact on the real card and there is nothing to compare against without it.\n"
        f"  on a cardless box use: tools/pkl_dims.py report <pkl>  /  tools/check_data_seed.py", file=sys.stderr)
  sys.exit(2)
try:
  _fd = os.open(KGSL_NODE, os.O_RDONLY)
  os.close(_fd)
except OSError as exc:
  print(f"run_pkl_pair: {KGSL_NODE} exists but cannot be opened ({exc}); run this as a user with GPU access",
        file=sys.stderr)
  sys.exit(2)
os.environ.setdefault("DEFAULT_DEVICE", "qcom")
os.environ.setdefault("FLOAT16", "1"); os.environ.setdefault("IMAGE", "2")
os.environ.setdefault("NOLOCALS", "1"); os.environ.setdefault("JIT_BATCH_SIZE", "0")
import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import _to_np_dtype


def read_any(path):
  if os.path.isfile(path):
    return open(path, "rb").read()
  manifest = path + ".chunkmanifest"
  if not os.path.isfile(manifest):
    raise FileNotFoundError(path)
  count = int(open(manifest).read().strip())
  return b"".join(open(f"{path}.chunk{i + 1:02d}of{count:02d}", "rb").read() for i in range(count))


def load_any(path):
  data = read_any(path)
  if data[0] == 0x80:
    return pickle.loads(data)
  from openpilot.selfdrive.modeld.helpers import load_oob
  return load_oob(io.BytesIO(data))


def input_info(cap):
  """(shape, dtype, device) per model input; the two backends name this field differently."""
  entries = getattr(cap, "expected_st_vars_dtype_device", None) or getattr(cap, "expected_input_info", None)
  if entries is None:
    raise SystemExit(f"run_pkl_pair: cannot find the input declaration on {type(cap).__name__}: "
                     f"expected expected_st_vars_dtype_device (ExecItem cache) or expected_input_info (linear graph)")
  return [(tuple(int(x) for x in spec.shape), dtype, device) for spec, _vars, dtype, device in entries]


def _input(rng, shape, it):
  if it == np.uint8: return rng.integers(0, 256, size=shape, dtype=np.uint8)
  if it.kind == "f": return (rng.standard_normal(shape) * 8.0).astype(it)
  return rng.integers(0, 4, size=shape).astype(it)


paths = sys.argv[1:]
if not paths:
  print(__doc__, file=sys.stderr)
  sys.exit(2)
seeds = [int(x) for x in os.environ.get("SEEDS", "7").split(",")]
results = []
for path in paths:
  jit = load_any(path)
  cap = jit.captured
  names = cap.expected_names
  info = input_info(cap)
  outs = []
  for seed in seeds:            # consecutive frames, the way the inference process uses it: load once
    rng = np.random.default_rng(seed)
    kw = {name: Tensor(_input(rng, shape, np.dtype(_to_np_dtype(dtype))), device=device).realize()
          for name, (shape, dtype, device) in zip(names, info)}
    outs.append(jit(**kw).contiguous().realize().uop.base.buffer.numpy())
  out = np.concatenate([o.astype(np.float32).ravel() for o in outs])
  print(f"{os.path.basename(path)}: {len(seeds)} frame(s) concatenated to {out.shape}, "
        f"last-frame argmax={outs[-1].argmax()} last-frame sum={outs[-1].astype(np.float32).sum():.4f}")
  np.save(path + ".seqout.npy", out)
  results.append((path, out))

base_path, base = results[0]
for path, out in results[1:]:
  a, b = base.astype(np.float32), out.astype(np.float32)
  d = np.abs(a - b)
  ulp = d / np.maximum(np.abs(a), 1e-6)
  verdict = "bit-identical" if np.array_equal(a, b) else "DIFFERS"
  print(f"  {os.path.basename(base_path)} vs {os.path.basename(path)}: {verdict}  "
        f"max abs diff={d.max():.6g} max relative diff={ulp.max():.4g} "
        f"differing elements={int((d > 0).sum())}/{a.size}")
