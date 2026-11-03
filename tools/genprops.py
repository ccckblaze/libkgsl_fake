#!/usr/bin/env python3
"""Convert props/*.properties into the C table (build/props_table.h).

The reverse also works: --emit-properties extracts GETPROPERTY answers from a libkgsl_rec log and
writes a properties file (use it after re-recording on real hardware; see docs/record.md).
"""
import argparse
import re
import sys

HEX = re.compile(r"^[0-9a-fA-F]+$")
PROP_NAMES = {
    0x01: "DEVICE_INFO", 0x02: "DEVICE_SHADOW", 0x03: "DEVICE_POWER", 0x04: "SHMEM",
    0x05: "SHMEM_APERTURES", 0x06: "MMU_ENABLE", 0x07: "INTERRUPT_WAITS", 0x08: "VERSION",
    0x09: "GPU_RESET_STAT", 0x0E: "PWRCTRL", 0x12: "PWR_CONSTRAINT", 0x13: "UCHE_GMEM_VADDR",
    0x14: "SP_GENERIC_MEM", 0x15: "UCODE_VERSION", 0x16: "GPMU_VERSION", 0x17: "HIGHEST_BANK_BIT",
    0x18: "DEVICE_BITNESS", 0x19: "DEVICE_QDSS_STM", 0x1A: "MIN_ACCESS_LENGTH",
    0x1B: "UBWC_MODE", 0x1C: "GENERIC_POWER", 0x20: "QTIMER",
}




def parse_properties(text):
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise SystemExit(f"bad line: {raw!r}")
        key = parts[0]
        blob = parts[-1] if len(parts) > 2 else parts[1]
        if not HEX.match(blob) or len(blob) % 2:
            raise SystemExit(f"bad hex value: {raw!r}")
        entries.append((int(key, 0), bytes.fromhex(blob)))
    return entries


def emit_header(entries):
    out = [
        "/* generated from the properties file by tools/genprops.py -- do not edit by hand. */",
        "#ifndef FK_PROPS_TABLE_H",
        "#define FK_PROPS_TABLE_H",
        "static const struct fk_prop fk_prop_builtin[] = {",
    ]
    for typ, val in entries:
        body = ", ".join(f"0x{b:02x}" for b in val)
        out.append(f"  {{ 0x{typ:02x}, {{ {body}, }}, {len(val)} }},")
    out += [
        "};",
        f"#define FK_PROP_BUILTIN_COUNT {len(entries)}",
        "#endif",
    ]
    return "\n".join(out) + "\n"


def emit_properties(text):
    """Rebuild a properties file from the PROPSAVE lines in a recorder log."""
    found = {}
    for line in text.splitlines():
        m = re.search(r"PROPSAVE type=0x([0-9a-fA-F]+) size=(\d+) value=([0-9a-fA-F]+)", line)
        if m:
            found[int(m.group(1), 16)] = bytes.fromhex(m.group(3))[: int(m.group(2))]
    out = [
        "# generated from a recorder log by tools/genprops.py --emit-properties",
        "# one line per KGSL property read off real hardware: type is a KGSL_PROP_* number, value is raw little-endian bytes.",
    ]
    for typ in sorted(found):
        name = PROP_NAMES.get(typ, f"PROP_{typ:#04x}")
        out.append(f"0x{typ:02x} {name:<22} {found[typ].hex()}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="properties file")
    ap.add_argument("--emit-properties", metavar="REC_LOG", help="reverse-generate from a recorder log")
    ap.add_argument("-o", "--out", help="output path, stdout by default")
    args = ap.parse_args()

    if args.emit_properties:
        body = emit_properties(open(args.emit_properties, errors="replace").read())
    else:
        if not args.src:
            ap.error("need a properties file, or --emit-properties")
        body = emit_header(parse_properties(open(args.src).read()))

    if args.out:
        open(args.out, "w").write(body)
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
