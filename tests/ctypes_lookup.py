"""Regression: symbols taken from a libc handle through ctypes must land on the shim.

LD_PRELOAD only closes the PLT. `ctypes.CDLL("libc.so.6").mmap` dlopens a specific library and
dlsyms against that handle, which by default skips interception entirely (tinygrad's
FileIOInterface.mmap is written exactly that way, and measured it SIGBUSes). The shim interposes
dlsym, so both paths here have to resolve to the same address; if interception is not in place
they differ and this test fails.

usage: fkgsl -- python3 tests/ctypes_lookup.py <shim.so>
"""
import ctypes, ctypes.util, os, sys

NAMES = ("mmap", "mmap64", "ioctl", "open", "open64", "openat", "openat64",
         "stat", "lstat", "fstat", "access")

def addr(fn):
  return ctypes.cast(fn, ctypes.c_void_p).value

shim_path = sys.argv[1] if len(sys.argv) > 1 else None
if not shim_path or not os.path.exists(shim_path):
  print(f"SKIP: shim not found at {shim_path}")
  raise SystemExit(0)

libc_name = ctypes.util.find_library("c")
libc, shim = ctypes.CDLL(libc_name), ctypes.CDLL(shim_path)
bad = []
for name in NAMES:
  got, want = addr(getattr(libc, name)), addr(getattr(shim, name))
  if got != want:
    bad.append(name)
  print(f"  {name:8s} libc-handle={got:#x} shim={want:#x} {'same' if got == want else 'DIFFERENT'}")
if bad:
  print(f"FAIL: {len(bad)} symbols resolved from the libc handle are not the shim: {' '.join(bad)}")
  raise SystemExit(1)
print(f"OK: all {len(NAMES)} intercepted symbols hold on the ctypes path too")
