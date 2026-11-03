# libkgsl_fake

**README** · [简体中文](README.zh-CN.md)

An **LD_PRELOAD shim** for Linux/glibc and Android/bionic that replaces Qualcomm Adreno's KGSL device layer
(`/dev/kgsl-3d0`) with software answers, so that the **vendor's closed-source OpenCL compiler can run
offline on an aarch64 machine with no GPU — and emits ISA that is bit-for-bit identical to the real
device's output**. Measured on two generations: a6xx under glibc, a5xx under an Android device's own bionic.

Three things it is *not*, stated up front:

- **Not a replacement for `libgsl`/`libkgsl`.** It implements no user-mode driver API and exports no
  `kgsl_*` symbols. It intercepts libc entry points — `open`/`ioctl`/`mmap`/`dlsym`. The vendor runtime
  stays the original one.
- **Not a GPU emulator.** No instruction simulation of any kind: kernel submissions are acknowledged,
  never executed. Any value you expect the GPU to write back is untrustworthy.
- **Not a way to escape the vendor toolchain.** Without `libOpenCL`/`libgsl`/`libllvm-qcom` present in
  the image, no ISA comes out at all. What this project removes is only the *hardware precondition*.

In one line: **it separates "compiling" from "executing", and turns "does this step need the hardware?"
into something you can assert with one command.**

## Why this works
Adreno ISA generation happens in user space, inside `libllvm-qcom`. Along that path the kernel driver
only does four things: enumerate the device, answer its property queries, hand out memory handles, hand
out sync objects. Recorded from a real device, **one full OpenCL compile
lifetime touches just 12 ioctl numbers** (list and field meanings: `docs/findings.md`, section 1) — and
emits **zero** `GPU_COMMAND` packets. An interface surface that small makes "replay it" one to two
orders of magnitude cheaper than "simulate the hardware". That is the entire bet, and the reason the
project is viable.

## Capability boundary (all of it measured)
| Tier | Scope | Status |
|------|-------|--------|
| 1 | Capture and decode every ioctl and command batch on `/dev/kgsl-3d0` | Done (`src/rec.c`, incl. a6xx PM4 type-4/type-7 decoding) |
| 2 | `clCreateContext` + `clBuildProgram` with no GPU; ISA bit-identical to real hardware | Done -- a6xx/glibc and a5xx/bionic, through a whole bundle's save and reload, and through a whole model build in a container with neither a card nor a board (findings 8-9) |
| 3 | Simulate a6xx PM4 + ISA6 and actually **execute** kernels | Not done, and not planned (rationale: `docs/findings.md`, section 3) |

For numbers, use a public CPU backend (POCL / llvmpipe / `CPU=1`) as the reference. This shim only ever
guarantees the *code* half.

## What it does not do, measured
Two limits that look like bugs and are not:

- **The layer reproduces the code half, it does not run it.** Feeding a compiled artifact through the
  shim and expecting output is wrong by construction: on the default (`stub`) setting the artifact never
  produces output and the run has to be killed at 300 s; with `FK_EXEC=abort` the runtime dies at 30 s
  with `Wait timeout: 30000 ms! (the signal is not set to 2, but 0)`. Acceptance of a packaged artifact
  therefore happens on real hardware: `tools/run_pkl_pair.py`.
- **A build that serializes its buffers back records nothing for the ones the GPU was asked to copy.** On the
  a5xx some host→device uploads are CPU copies inside the runtime and arrive fine with no card; the rest are
  GPU commands, and with nothing executing they read back as zeros — measured on one bundle: 101 of its 190
  weight buffers correct, 89 empty. The numbers are not lost: they are in the model file, and the layout a
  kernel wants follows from shapes the bundle already records, so the honest fix is to recompute them rather
  than to ask a card for them. What this layer guarantees stays the code half.
- **Launch dimensions come from the card, not from a rule.** When the compiler can time candidates it
  freezes a thread count per kernel; without a card the timing is always zero and the pick degenerates.
  Measured on a 131-kernel monitoring model: a cardless box leaves 117 kernels without a card-chosen
  count (87 carry no local group at all, 30 land on `local=1`); the card-made artifact has 35 at
  `local=1` and 30 with a fractional `global`. That cardless artifact runs 277 ms on hardware against
  12 ms for the card-made one - same graph. Worse, inventing a dimension that the card never measured is not merely slow: with
  a hand-picked set 590 of 596 output elements differ. So the only legal values are harvested from a
  card-made artifact; `tools/pkl_dims.py report|diff` inspects them and compares against such a table.
  For scale, the card's own reading is not even deterministic: two compiles on the same card agreed on
  only 73 of 131 dimension pairs (and 75 with the deployed artifact), yet all three are bit-identical
  in output. Check it yourself: `tools/pkl_dims.py diff` reports 131/131 in agreement against a
  card-made table, and 44 agreed / 87 differed for the cardless artifact.

Details and the rest of the numbers: `docs/findings.md`, sections 5-7.

## Quick start
```sh
make          # build/libkgsl_fake.so  build/libkgsl_rec.so  build/probe  build/tripwire
make test     # six assertions; runs on any GPU-less aarch64 Linux

tools/fkgsl -- <cmd>                # inject the fake device (strict by default; GPU_COMMAND ack-only)
tools/fkgsl --abort-exec -- <cmd>    # abort on any GPU_COMMAND: proves the flow needs no GPU
tools/fkgsl --real -- <cmd>          # pass everything through, for a real-GPU comparison run
tools/fkgsl --rec -- <cmd>           # record only, no deception (needs a real GPU)
```
It cannot be built on macOS (it relies on glibc's `open64`/`mmap64`/`RTLD_NEXT`, or on bionic's libc). Use a container:
```sh
docker run --rm --platform linux/arm64 -v "$PWD":/work -w /work <your arm64 image> make test
```
For the Android/bionic build, run `make bionic` inside an image carrying that platform's own clang (the
device's user mode will do): it produces `build/bionic/libkgsl_fake.so`, defaulted to the a5xx property table
through `BIONIC_PROPS`. `docs/record.md`, section 5 covers recording against such a device.

Want the Android/bionic box assembled rather than put together by hand? The recipe is short: `make bionic`
here, plus a recorded device's property table and a recorded board's `/sys` identity sitting in one
directory. Point this layer's own three names (`FK_PROPS`, `FK_SYSFS`, `FK_TS_COVERAGE`) at them and
preload nothing -- what a build calls those paths, and whether it wraps a command with the layer at all,
belongs to the build, not to this repository.

Nothing with a GPU in sight? One command produces a usable model artifact (code half + data half):
```sh
tools/docker_gpless.sh <tree> <model.onnx> [out.pkl]
# builds the shim in the container, runs model_codegen.py --cpu-exec, then self-checks the red lines
# prints GPU_COMMAND / UNHANDLED / TRIPWIRE counts — all three must be 0
```
Accept the artifact on a machine that has the target GPU: `tools/run_pkl_pair.py <device-made.pkl> <yours.pkl>`
(replay both artifacts on the same fresh inputs and compare the model outputs; it refuses to start where
there is no card, on purpose), `tools/check_data_seed.py` for buffer-level classification, and
`tools/pkl_dims.py report|diff` to see the frozen launch dimensions and check them against a card-made
table. Measured results: `docs/findings.md`, sections 6-7.
The minimal reproducible sample is `build/probe`: enumerate → create context → build program → write the
ISA out (path via `FK_ISA_OUT`). Only `probe full` goes on to submit work, and that step is *expected*
to crash or hang.

## How it works
### 1. Intercept every libc entry point — none may be missed
`open` / `open64` / `openat` / `openat64` / `ioctl` / `mmap` / `mmap64` / `close` / `stat` / `lstat` /
`fstat` / `access` / `syscall()`. Miss one and the caller ends up holding the **real** device fd, then
produces a pile of "looks correct" results with a real GPU — the worst possible failure mode. So the
direct-`syscall(SYS_ioctl)` path is intercepted too, and `tests/tripwire.c` is its regression test.

Which nodes get faked is set by `FK_KGSL_PATH` / `FK_ION_PATH` / `FK_PMSG_PATH` (default
`/dev/kgsl-3d0`, …). The returned "fd" is a memfd; the fd → category table is what all later
dispatch keys off.

### 2. Device-property replay: what the compiler believes the card is
`DEVICE_GETPROPERTY` answers from a compile-time embedded property table generated from
`props/a6xx.properties` — **raw little-endian bytes recorded from a real card**, one entry per
`KGSL_PROP_*` (`DEVICE_INFO`, `DEVICE_SHADOW`, `UCHE_GMEM_VADDR`, `UCODE_VERSION`,
`HIGHEST_BANK_BIT`, `MIN_ACCESS_LENGTH`, …). `tools/genprops.py` turns that file into `props_table.h`
during `make`, and `FK_PROPS=<file>` overrides the whole table at runtime without recompiling. Test
tier 0 cross-checks those two paths and demands identical ISA.

Codegen decisions — register allocation, instruction selection, bank count, feature flags — are driven
by exactly these fields. To support a different GPU, record a new table; don't guess field values.
Two generations are recorded here: `props/a6xx.properties` (10 reads) and `props/a5xx.properties` (5 reads,
three of them different values — including `DEVICE_SHADOW`, which the mmap convention is built on).

### 3. `GPUOBJ` = one memfd, sliced, following KGSL's own mmap convention
`GPUOBJ_ALLOC` returns an object id, and `mmap(fd, len, …, offset)` dispatches on the kernel's convention
**`offset = id << 12`** to find the backing object. The backing is **one** memfd handed out in page-aligned
slices that are never reused, so two objects can never share a page. It used to be one memfd per object,
which costs the process a descriptor every allocation — something the real driver does not do — and on a
bionic device that is soft `RLIMIT_NOFILE` 1024 with a build peaking at 1005 live objects, the descriptor
table ran out, `GPUOBJ_ALLOC` failed, and the runtime's only complaint was a null buffer from
`clCreateBuffer` when it reloaded what it had just saved.
`GPUOBJ_INFO` deliberately does not invent a separate GPU page-table address: on a6xx the CPU and GPU share
an address space, and the runtime uses the host pointer returned by `mmap` as its `gpuaddr`, so returning
the host pointer is the self-consistent answer. The page named by `DEVICE_SHADOW` is special-cased — see next.

### 4. Timestamps: giving "wait for the GPU" an end
A real runtime submits work and then spins on `soptimestamp`/`eoptimestamp`. With no hardware write-back
it hangs forever, which looks like a broken shim. The shim advances the two fixed slots in the shadow
page (default offset `0x204`, stride `FK_TS_STRIDE=8`) each time a `GPU_COMMAND` is acknowledged.
**This exists so waiting terminates, not so results are correct.** The `FK_TS_COVERAGE=all` fallback —
painting every mapped region with the timestamp — was a debugging hammer on a6xx and is a **requirement** on
a5xx: the a5xx runtime spins on a field of the shadow page this layer has no name for, and without it a whole
build stops after 12 acknowledged commands inside `cl_a5x_wait_for_timestamp_thread`.

### 5. ION / dma-buf stand-in
`libgsl` asks `/dev/ion` for dma-bufs. `src/ion.c` serves those from the same memfd backend; without it,
enumeration dies before compilation ever starts.

### 6. Rewriting `dlsym`: the complement of `LD_PRELOAD`
The most overlooked and most load-bearing piece. **`LD_PRELOAD` only intercepts the PLT.** A call like
`ctypes.CDLL("libc.so.6").mmap` `dlopen`s a specific library and then `dlsym`s *that handle*, which
returns libc's own implementation — invisible to symbol interposition. This pattern is everywhere:
script FFI layers, JNI, and generally any loader that resolves symbols against a specific handle. So the shim also interposes `dlsym`, and
substitutes its own function **only when the address resolved is exactly libc's implementation**.
`RTLD_NEXT`/`RTLD_DEFAULT` lookups are left untouched — the shim itself resolves real implementations via
`RTLD_NEXT`, and rewriting those would recurse into itself.

Covered: `mmap/mmap64/ioctl/open*/stat*/fstat/access`; `tests/ctypes_lookup.py` asserts that both paths
land on the same address for all 11 symbols.

`dlvsym` is intentionally *not* hooked: taking `mmap` through it yields libc's version, which then faults
loudly on an out-of-range access rather than silently sliding onto real hardware and faking a pass.

### 7. Strict mode and the tripwire: abort rather than lie
`FK_STRICT=1` is the default, and it aborts in two cases: an ioctl number that was never recorded, and a
KGSL ioctl arriving on an fd the shim did not issue (i.e. the caller bypassed interception). Silent use of
real hardware is the worst outcome. `FK_EXEC=abort` (`--abort-exec`) turns "does this step need the GPU?"
into a single command's worth of evidence.

### 8. The recorder: measure the interface before deciding what to replay
`libkgsl_rec.so` (`src/rec.c`) copies without deceiving: it logs every ioctl with its fields and decodes
a6xx PM4 batches (type-4/type-7, with optional `LOAD_STATE6` dumps). Don't argue about whether a flow
needs hardware — run it once under `--rec` on a real device and read the answer.
Re-recording procedure: `docs/record.md`.

### 9. A board's identity is not a GPU question
Two things a cardless box has to be handed that this layer does not own, and either of them stops the
vendor runtime before it ever reaches the device node:

- **Which SoC this is.** The closed runtime reads `/sys/devices/soc0/soc_id` (older builds ask
  `/sys/devices/system/soc/soc0/id`) before it will look for a GPU at all. With no such node the target
  name comes back `UNKNOWN_SDM` and model validation refuses the GPU backend outright -- error 1006,
  `No GPU device available for target UNKNOWN_SDM` -- substitute device layer or not. `FK_SYSFS` names a
  directory standing in for `/sys`, so a recorded node answers in place of the missing one.
- **What the system says about itself.** Android/bionic below API 24 keeps every system property in one
  file, `/dev/__properties__`, and a container starts with no property area at all, so every property
  reads back empty. bionic maps that file `PROT_READ|PROT_WRITE` from an `O_RDONLY` open, a mapping
  tmpfs grants and other filesystems refuse: the box must *copy* it into its own `/dev` at start, and a
  bind-mounted file fails the mapping silently, which looks exactly like a board that has no properties.

Both come off a working board with one command, and nothing is invented:
`tools/record_board.sh <host> <dir>` writes the mirror, takes the property area, and prints the two
lines the box needs to use them.

One caveat that costs a confused hour if it is not said: a statically linked reader (`toybox cat`) is
invisible to `LD_PRELOAD`, so `cat /sys/devices/soc0/soc_id` goes on saying *no such file* while every
dynamically linked reader sees the mirror. Check a mirror with a dynamic reader, not with `cat`.

## What `make test` asserts
| Tier | Assertion |
|------|-----------|
| 0 | Embedded property table vs `FK_PROPS` runtime override must produce bit-identical ISA |
| 1 | Compile-phase red lines: 0 `GPU_COMMAND`, 0 unrecorded ioctl, 0 real-device leaks, and the result matches `tests/golden/` |
| 2 | Execute phase: passing means it *crashed or hung* (if it computed correctly, it secretly used a real GPU) |
| 3 | `tripwire`: a real device fd reaching a KGSL ioctl must abort under strict (skipped when there is no real card) |
| 4 | A real kernel (`tests/kernels/e2x.cl`) compiles with no GPU: 4220 bytes, bit-identical to the golden baseline |
| 5 | `ctypes_lookup.py`: all 11 intercepted symbols land on the shim through the ctypes/`dlsym` path too |

Also `tests/qcom_compiler.py`: a genuine codegen pipeline (tinygrad's QCOM compiler) running entirely on
the fake device, cross-checked against the fixed probe and a real card. Baselines are created on real
hardware with `tests/golden.sh update`; any change that drifts the ISA turns this suite red.

## Environment variables
| Variable | Default | Effect |
|----------|---------|--------|
| `FK_VERBOSE` | 0 | write a log |
| `FK_LOG` | `/tmp/libkgsl_fake.log` | log destination |
| `FK_STRICT` | 1 | abort on unrecorded ioctl / real-device fd leak |
| `FK_EXEC` | `stub` | `stub` = ack with timestamp only; `abort` = die on `GPU_COMMAND` |
| `FK_DISABLE` | unset | non-empty passes everything through (real-GPU control run) |
| `FK_PROPS` | unset | override the embedded device properties with another table |
| `FK_SYSFS` | unset | directory standing in for `/sys`: the identity nodes a box with no board lacks |
| `FK_KGSL_PATH` / `FK_ION_PATH` / `FK_PMSG_PATH` | `/dev/kgsl-3d0` etc. | which nodes to intercept |
| `FK_SHADOW_OFFSET` / `FK_TS_OFFSET` / `FK_SHADOW_SIZE` | from `DEVICE_SHADOW` | override the mapping convention |
| `FK_TS_STRIDE` | 8 | distance between the two timestamp slots in the shadow page |
| `FK_TS_COVERAGE` | `shadow` | `all` = paint every mapped region; required on a5xx, debugging hammer on a6xx |
| `FK_REC_OUT` / `FK_DUMP_DIR` | — | recorder log / PM4 dump directory |
| `FK_ISA_OUT` | `/tmp/fkgsl_probe_isa.bin` | where `build/probe` writes the ISA |
| `FK_FAKE_GPUADDR` | 0 | override the `gpuaddr` returned by `GPUOBJ_INFO` |

> Note: `tools/cpu_image_shim.py` is a second copy of a shim that also travels with a model toolchain.
> The rule is the same one this repository runs by: measure, do not guess -- when the two diverge, the
> one a build actually uses is the one to fix, and this copy exists so the experiments here run alone.

## Layout
```
src/interpose.c  interception: open*/ioctl/mmap*/close/stat*/access/syscall/dlsym
src/device.c     KGSL ioctl semantics (properties, GPUOBJ, contexts, sync, GPU_COMMAND)
src/props.c      device-property replay + FK_PROPS override
src/raw.c        raw syscalls: the libc entry points this layer needs, without asking the loader
src/mem.c        sliced memfd backend + shadow-page timestamps
src/ion.c        ION stand-in
src/rec.c        recorder (incl. a6xx typ4/typ7 PM4 decoding)
include/         trimmed KGSL UAPI header (only the structs actually used)
props/           device properties recorded from real hardware (a6xx, a5xx) -- the single source of
                 defaults -- and `a5xx-sys/`, the SoC identity mirror of that same board (section 9)
tests/           staged probes, leak assertions, ctypes regression, golden hashes
docs/           findings.md = measured conclusions; record.md = re-recording procedure
tools/           fkgsl wrapper, property/PM4 generators, docker_gpless.sh (offline packaging), the
               offline-codegen experiments, and the acceptance scripts (run_pkl_pair, check_data_seed,
               pkl_dims, bundle_struct)
```

## Requirements
- Linux/glibc or Android/bionic on aarch64, with the **vendor runtime for the target GPU generation** installed
  (`libOpenCL`, `libgsl`, `libllvm-qcom`). The shim does not substitute for any of them.
- An x86 host needs QEMU or a native arm runner; cross-architecture is out of scope here.
- New GPU generation or upgraded closed-source runtime → re-record the property table and ioctl set
  (`docs/record.md`). Don't guess fields.
- A box with no board needs the board's identity as much as it needs the device layer: a `/sys` mirror
  through `FK_SYSFS`, and below API 24 the property area copied onto its tmpfs `/dev`. One command
  records both -- `tools/record_board.sh`.
  The a5xx mirror ships at `props/a5xx-sys`; the property area does not, because that file holds this
  board's serial number and radio addresses -- record your own off a board you are allowed to copy.
- Only the compile phase is trustworthy. "Let's just run it and look at the numbers" is not covered.

## On the name
`libkgsl_fake` is too narrow for what this became: `libkgsl` reads as "drop-in vendor UMD replacement"
(it is not), and `fake` only describes half of what's here — recording and offline codegen are not fake
at all. The directory name stays, since renaming would ripple through docs, scripts and container
paths; the accurate description is **a KGSL device shim**: it takes over the device node and the
ioctl/mmap/dlsym surface, and that removes the vendor compiler's hardware precondition. The command you
actually type, `fkgsl`, has no such ambiguity.

## License / caveat worth knowing
No secrets and no project data live in this repository: `props/` holds only raw property bytes read off
a public device node. But note what this demonstrates — device presence checks and ioctl answers can be
replaced wholesale in software, including the property table itself. If you anchor trust on "I can read an
answer from the device", this codebase is a ready-made template for attacking exactly that: what you can
verify is a signature, never *who is answering*.
