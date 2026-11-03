#!/usr/bin/env python3
"""Inspect the kernel launch dimensions baked into a compiled graph, and diff them against
a table harvested from a real device.

Why this exists: when the vendor compiler runs on a card, tinygrad times every kernel over
hundreds of thread-count candidates and freezes the winner into the artifact. On a machine
without the card the timing is always zero, so the pick degenerates (usually `local=1`), and
the same graph runs an order of magnitude slower on the real device. The dimension pair is
part of the program, not a performance preference: candidates that were never measured on
the card may compute wrong outputs, so this tool only *reports* -- it never invents a value.

Two artifact layouts are read here, since the trees that produce them pin different backends:
  - ExecItem cache  (`captured.jit_cache`, `CompiledRunner`)
  - linear graph    (`captured.linear`, `Ops.PROGRAM` args)

Loading an artifact instantiates its target device, so run it either on a machine with the card, or
on a cardless box through the fake device (which is exactly what that is for):

usage:
  tools/fkgsl -- python3 tools/pkl_dims.py report <model.pkl> [--json out.json]
  tools/fkgsl -- python3 tools/pkl_dims.py diff   <model.pkl> <table.json>

`<model.pkl>` may be a whole file or the base name of a chunked one (`*.chunk01ofNN` +
`*.chunkmanifest`). A table is `{"kernels": {"<src-hash>:<pre-global>": {...}}}` as written by
the `harvest` mode of the model toolchain's own dims helper; kernels sharing a source and a
pre-dimension share one entry, exactly as the apply step does.

`diff` exits 1 when the artifact and the table disagree, or when the table holds an entry no kernel
claimed (wrong table, or drifted sources). Kernels found outside the table are a warning only: a
symbolic dimension, or a kernel the card never gave a local group, is legitimate.
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import re
import sys


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text):
  """tinygrad colors kernel names; the report wants them readable and diffable."""
  return ANSI.sub("", str(text))


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


def jits(obj):
  if hasattr(obj, "captured"):
    return [obj]
  if isinstance(obj, dict):
    return [v for x in obj.values() for v in jits(x)]
  return []


def pre_global(global_size, local_size):
  """Whole launch shape before the thread count was divided out (a card-frozen global may be fractional)."""
  glob = [float(x) for x in global_size]
  if not local_size:
    return [int(x) for x in glob] if all(x.is_integer() for x in glob) else None
  return [int(round(g * l)) for g, l in zip(glob, [float(x) for x in local_size])]


def exec_item_kernels(cap):
  """ExecItem-cache artifacts: yield (name, src, global_size, local_size) per compiled kernel."""
  cache = getattr(cap, "_jit_cache", None)
  if cache is None:
    cache = getattr(cap, "jit_cache", None)
  if cache is None:
    return None                     # a linear-graph backend has neither field, and no CompiledRunner either
  from tinygrad.engine.realize import CompiledRunner
  out = []
  for ei in cache:
    prg = getattr(ei, "prg", None)
    if isinstance(prg, CompiledRunner):
      out.append((plain(prg.p.name), str(prg.p.src), list(prg.p.global_size or []), list(prg.p.local_size or [])))
  return out


def linear_kernels(cap):
  """linear-graph artifacts: the same triple, read off the PROGRAM nodes of the captured graph."""
  from tinygrad.uop.ops import Ops
  out = []
  for call in cap.linear.toposort(gate=lambda x: x.op is not Ops.PROGRAM):
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM:
      continue
    prg = call.src[0]
    arg = prg.arg
    src = arg.src if getattr(arg, "src", None) else (prg.src[3].arg if len(prg.src) > 3 else "")
    out.append((plain(getattr(arg, "name", "") or name_of(prg)), str(src),
                list(arg.global_size or []), list(arg.local_size or [])))
  return out


def name_of(prg):
  for u in prg.src:
    if u.op.__name__ == "NAME":
      return str(u.arg)
  return ""


def programs(obj):
  """(name, src, global_size, local_size) for every compiled kernel, in execution order."""
  out = []
  for jit in jits(obj):
    cap = jit.captured
    kernels = exec_item_kernels(cap)
    if kernels is None:
      kernels = linear_kernels(cap)
    out.extend(kernels)
  return out


def key_of(src, pre):
  return hashlib.sha256(src.encode("utf8", "surrogateescape")).hexdigest()[:32] + ":" + str(tuple(pre))


def product(dims):
  n = 1
  for x in dims:
    n *= float(x)
  return n


def row_stats(rows):
  total = len(rows)
  no_local = sum(1 for _, _, _, loc in rows if not loc)
  local_one = sum(1 for _, _, _, loc in rows if loc and product(loc) == 1)
  fractional = sum(1 for _, _, glob, _ in rows if any(float(x) % 1 for x in glob))
  return {"kernels": total, "no_local_group": no_local, "local_is_1": local_one,
          "fractional_global": fractional}


def report(path, as_json=None):
  obj = load_any(path)
  rows = programs(obj)
  stats = row_stats(rows)
  print(f"{os.path.basename(path)}: {stats['kernels']} kernels "
        f"(no local group {stats['no_local_group']}, local=1 {stats['local_is_1']}, "
        f"fractional global {stats['fractional_global']})")
  print(f"  {'#':>4}  {'global':28}  {'local':18}  name")
  for n, (name, src, glob, loc) in enumerate(rows):
    print(f"  {n:>4}  {str([int(x) if float(x).is_integer() else x for x in glob]):28}  "
          f"{str([int(x) if float(x).is_integer() else x for x in loc]):18}  {name[:52]}")
  if stats["kernels"] and stats["local_is_1"] + stats["no_local_group"] > stats["kernels"] // 2:
    print("  note: most kernels carry no real thread count -- this artifact was compiled where no card "
          "could time the candidates. Fine to run, slow on hardware; fix with `diff` against a harvested table.",
          file=sys.stderr)
  if as_json:
    json.dump({"source": os.path.basename(path), "stats": stats,
               "kernels": [{"key": key_of(src, pre_global(g, l)), "name": name,
                            "global": [float(x) for x in g], "local": [float(x) for x in l]}
                           for name, src, g, l in rows]},
              open(as_json, "w"), indent=1)
    print(f"  wrote {as_json}")
  return rows


def diff(path, table_path):
  table = json.load(open(table_path))["kernels"]
  rows = programs(load_any(path))
  used, same, changed, outside = set(), 0, [], []
  for name, src, glob, loc in rows:
    pre = pre_global(glob, loc)
    if pre is None:
      outside.append((name, "global is symbolic, resolved at run time"))
      continue
    key = key_of(src, pre)
    entry = table.get(key)
    if entry is None:
      outside.append((name, "picked a local group the table never saw" if loc else "no local group at all"))
      continue
    used.add(key)
    want_local = [int(x) for x in entry["local"]]
    have_local = [int(float(x)) for x in (loc or [])]
    if have_local != want_local:
      changed.append((name, have_local, want_local))
    else:
      same += 1
  unused = sorted(set(table) - used)
  print(f"{os.path.basename(path)}: agrees with table on {same}, differs on {len(changed)}, "
        f"outside the table on {len(outside)}, table entries unused {len(unused)}")
  for name, have, want in changed[:10]:
    print(f"    differs: {name[:44]:44s} here={have} table={want}")
  for name, why in outside[:10]:
    print(f"    outside: {name[:44]:44s} {why}")
  for key in unused[:5]:
    print(f"    unused table entry: {key[:32]}... local={table[key].get('local')}")
  if unused:
    print("  the table is not for this artifact: either a different model, or the kernel sources drifted "
          "(compiler options / backend version)", file=sys.stderr)
  if outside:
    print(f"  {len(outside)} kernels keep whatever this box picked; an unmeasured candidate can compute "
          "wrong output, so nothing here substitutes a value", file=sys.stderr)
  return 1 if (unused or changed) else 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=("report", "diff"))
  ap.add_argument("paths", nargs="+")
  ap.add_argument("--json", dest="as_json", help="also dump the dimensions as json (report only)")
  args = ap.parse_args()
  if args.cmd == "report":
    report(args.paths[0], args.as_json)
    return 0
  if len(args.paths) != 2:
    ap.error("diff needs <model.pkl> <table.json>")
  return diff(args.paths[0], args.paths[1])


if __name__ == "__main__":
  sys.exit(main())
