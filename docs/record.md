# Re-recording a device: property table and command batches

New machine, new GPU generation, or an upgraded closed-source runtime, and both the property table and the
ioctl surface may change. Re-record in the four steps below.

## 1. Record
On any box carrying the same GPU (a dev machine or the target device itself):
```sh
make
tools/fkgsl --rec -- build/probe           # or --rec -- <any program that uses OpenCL>
# the log goes to /tmp/libkgsl_rec.log by default; override with FK_REC_OUT
```
`--rec` copies without deceiving, so first make sure this machine really runs OpenCL.

## 2. Extract the property table
```sh
python3 tools/genprops.py --emit-properties /tmp/libkgsl_rec.log > /tmp/rec.properties
diff <(grep -v '^#' props/a6xx.properties) <(grep -v '^#' /tmp/rec.properties)
```
A difference means this machine self-describes differently. Once confirmed, keep it as a second file, for
instance `props/a6xx_<machine codename>.properties`, and either point `PROPS` at it in the Makefile or set
`FK_PROPS=props/a6xx_xxx.properties` at runtime.

## 3. Dump command batches (optional, for analysis)
```sh
FK_DUMP_DIR=/tmp/dumps tools/fkgsl --rec -- build/probe
```
`/tmp/dumps/load_*` holds the raw bytes `CP_LOAD_STATE6` points at: shader ISA, constants, texture
descriptors. Feed them straight to tinygrad's `extra/disassemblers/adreno` disassembler.

## 4. Lock in a new baseline
```sh
tools/fkgsl --real -- build/probe && sha256sum /tmp/fkgsl_probe_isa.bin   # real card
```
Take the same hash with `build/probe` running on the fake device; the two must match. Then:
```sh
sh tests/golden.sh update
```

## 5. Recording on the device itself, when it runs Android/bionic
`build/probe` needs OpenCL headers and a glibc link line, and a shipping device has neither. Record against
**the workload the table has to serve** instead, which is also the more honest sample: a property a small probe
never asks for is a property the fake will be caught not knowing.

```sh
make bionic                     # inside the device's own image: its clang targets aarch64-linux-android
# then on the device, with the recorder only (it copies, it does not deceive -- the card must be working):
LD_PRELOAD=/data/local/tmp/build/bionic/libkgsl_rec.so FK_REC_OUT=/data/local/tmp/rec.log <the real workload>
python3 tools/genprops.py --emit-properties /data/local/tmp/rec.log > props/<generation>.properties
```
Check the list of ioctl numbers the log covers against `docs/findings.md` section 1 before believing it: the
a5xx sample (29,051 records over one full bundle build) asked for the same numbers as the a6xx one and read
only 5 properties, three of which carry different values -- including `DEVICE_SHADOW`, the one the mmap
convention is built on. `FK_STRICT=1` on the fake run is what proves the new table covers the flow: an unread
property is a fatal error there, not a shrug.

## 6. Recording what a box has no board for
The device layer answers questions about the GPU. A container also lacks two things that are not about the
GPU, and a closed runtime asks both before it reaches the device node (`docs/findings.md`, section 9): which
SoC this is, and what the system says about itself. Take them off a board that works -- one command:

```sh
tools/record_board.sh <host> <out-dir> [ssh-port]
# writes <out-dir>/devices/soc0/*   (the /sys mirror, passed to the build as FK_SYSFS)
# writes <out-dir>/__properties__   (the whole property area, Android below API 24)
# prints the two lines the box needs to use them
```

Then in the box, before anything compiles:

```sh
cp <out-dir>/__properties__ /dev/__properties__ && chmod 644 /dev/__properties__
LD_PRELOAD=<layer> FK_PROPS=<generation>.properties FK_SYSFS=<out-dir> FK_TS_COVERAGE=all <the build>
```

Only the second line is a precondition of the model build (measured: `docs/findings.md`, section 9). The
first is there for whatever else runs in the box and asks the platform a question, and it is *not*
committed here -- so a box assembled from this repository takes the mirror and leaves the area to whoever
owns a board: copy it onto the box's own `/dev` before the first process starts, because a bind mount
there reads as a file and changes nothing.

The a5xx mirror recorded this way ships in the repository as `props/a5xx-sys/`, for the same reason the
device property tables ship: it is a measured copy of public hardware identity, and a reader should be able
to reproduce section 9 without owning the board. **The property area is not committed** -- that one file
holds a serial number and radio addresses, which belong to whoever recorded them.

Two rules this procedure exists to keep honest:

- **Nothing is typed in.** A node the board does not have stays missing from the mirror. `soc_id` decides
  which codegen path the runtime takes, so a plausible-looking hand-written value produces an artifact for
  the wrong chip, and nothing downstream looks different.
- **The property area must be copied, never mounted.** bionic below API 24 maps it read/write from a
  read-only open; tmpfs allows that, a bind mount from another filesystem does not, and both failures look
  like a board with no properties. It is mapped when the process starts, so it belongs to the image or its
  entry point -- a `docker run -v .../properties:/dev/__properties__` line will sit there, work as a file,
  and change nothing.

Verify the mirror with a dynamically linked reader, not with `cat`: `toybox` is static, and a static binary
is invisible to `LD_PRELOAD`, so `cat /sys/devices/soc0/soc_id` keeps reporting *no such file* while the
runtime reads the mirror perfectly well.

What the recorded identity is worth is measured in `docs/findings.md`, section 9: with the mirror and the
property area in place, a container with no card and no board built a whole model graph -- 1045 buffers and
19 programs identical to the device's, code half bit-for-bit, numbers half within one half-precision ULP --
and the artifact replayed 368 of 368 kernels on the real card.
