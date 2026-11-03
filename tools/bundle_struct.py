#!/usr/bin/env python3
"""Say *where* two serialized kernel bundles differ, because byte equality is not the bar.

Measured on a real Adreno a5xx: three runs of the vendor compiler over the same model file produced
three different bundles. So "the card changed the generated code" and "only the autotuned numbers
moved" have to be told apart, and a file hash cannot tell them.

A bundle here is a 4-byte length, a JSON description, then the payloads: buffer contents in the order
the description lists the ones that carry data, then one blob per compiled program.

  bundle_struct.py A B [C ...]     every one after the first is compared with the first

The verdict is hard when generated code, buffer layout or buffer contents differ, soft when only the
recorded launch dimensions moved, and dimensions are known not to be deterministic even on a card
(docs/findings.md, section 7), so a soft verdict is a pass.
"""
import hashlib
import json
import struct
import sys


def parse(path):
  with open(path, "rb") as f:
    blob = f.read()
  json_len = struct.unpack("I", blob[:4])[0]
  jdat = json.loads(blob[4:4 + json_len].decode("latin_1"))
  payloads = blob[4 + json_len:]
  at = 0
  for o in jdat["objects"]:
    if o["needs_load"]:
      o["data"] = payloads[at:at + o["size"]]
      at += o["size"]
  for b in jdat["binaries"]:
    b["data"] = payloads[at:at + b["length"]]
    at += b["length"]
  return jdat, at, len(blob)


def sha(data):
  return hashlib.sha256(data).hexdigest()[:16] if data else "-"


def skeleton(jdat):
  return {k: v for k, v in jdat.items() if k not in ("objects", "binaries", "kernels")}


def compare(a, b, la, lb):
  print("== %s  vs  %s" % (la, lb))
  for key in ("objects", "binaries", "kernels"):
    print("   %-9s %5d vs %5d  %s" % (key, len(a[key]), len(b[key]),
                                      "same" if len(a[key]) == len(b[key]) else "DIFFERENT"))

  kinds = {}
  for i, (x, y) in enumerate(zip(a["objects"], b["objects"])):
    if x.get("size") != y.get("size"):
      kinds.setdefault("size", []).append(i)
    elif x.get("data") != y.get("data"):
      kinds.setdefault("payload", []).append(i)
  flat = [i for v in kinds.values() for i in v]
  print("   object mismatches: %d  (%s)" %
        (len(flat), ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(kinds.items())) or "none"))
  for i in flat[:4]:
    x, y = a["objects"][i], b["objects"][i]
    print("      [%d] %s size %s/%s sha %s vs %s" % (i, x.get("arg_type"), hex(x.get("size", 0)),
          hex(y.get("size", 0)), sha(x.get("data")), sha(y.get("data"))))

  pairs = [(x["name"], x.get("data"), y.get("data")) for x, y in zip(a["binaries"], b["binaries"])]
  diff = [t for t in pairs if t[1] != t[2]]
  print("   ISA binaries mismatch: %d of %d" % (len(diff), len(pairs)))
  for name, x, y in diff[:5]:
    print("      %s %s/%s B sha %s vs %s" % (name, len(x), len(y), sha(x), sha(y)))

  names, gws, lws = [], [], []
  for x, y in zip(a["kernels"], b["kernels"]):
    if x.get("name") != y.get("name"):
      names.append((x.get("name"), y.get("name")))
    else:
      if x.get("global_work_size") != y.get("global_work_size"):
        gws.append(x.get("name"))
      if x.get("local_work_size") != y.get("local_work_size"):
        lws.append(x.get("name"))
  print("   kernels: name mismatch %d, global_work_size mismatch %d, local_work_size mismatch %d"
        % (len(names), len(gws), len(lws)))
  shown = 0
  for x, y in zip(a["kernels"], b["kernels"]):
    if x.get("local_work_size") != y.get("local_work_size") or \
       x.get("global_work_size") != y.get("global_work_size"):
      print("      %s\n         A gws=%s lws=%s\n         B gws=%s lws=%s" %
            (x.get("name"), x.get("global_work_size"), x.get("local_work_size"),
             y.get("global_work_size"), y.get("local_work_size")))
      shown += 1
      if shown >= 3:
        break

  other = [k for k in skeleton(a) if skeleton(a)[k] != skeleton(b).get(k)]
  print("   other json keys differ: %s" % (other if other else "none"))
  if not (flat or diff or names or gws or lws or other):
    print("   verdict: IDENTICAL")
  elif diff or names or kinds.get("size") or kinds.get("payload"):
    print("   verdict: HARD DIFF -- generated code, buffer layout or buffer contents changed")
  else:
    print("   verdict: soft only -- autotuned work sizes differ")


def main(argv):
  if len(argv) < 3:
    print(__doc__.strip(), file=sys.stderr)
    return 2
  loaded = []
  for path in argv[1:]:
    jdat, used, total = parse(path)
    print("%s: file %d B, json %d B, payloads %d B" % (path, total, total - used, used))
    loaded.append((path, jdat))
  print()
  base = loaded[0]
  for other in loaded[1:]:
    compare(base[1], other[1], base[0], other[0])
    print()
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
