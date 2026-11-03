#!/usr/bin/env python3
"""Generate the PM4 opcode name table (src/pm4_names.h) from adreno_pm4.xml.

usage: python3 tools/genpm4.py <adreno_pm4.xml> -o src/pm4_names.h
XML source: tinygrad upstream extra/qcom_gpu_driver/adreno_pm4.xml
"""
import argparse
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser()
ap.add_argument("xml")
ap.add_argument("-o", "--out", required=True)
args = ap.parse_args()

names = {}
root = ET.parse(args.xml).getroot()
for node in root:
    if node.attrib.get("name") != "adreno_pm4_type3_packets":
        continue
    for pkt in node:
        if "name" not in pkt.attrib or "value" not in pkt.attrib:
            continue
        variants = pkt.attrib.get("variants", "")
        if variants == "A2XX":
            continue
        names[int(pkt.attrib["value"], 0)] = pkt.attrib["name"]

lines = [
    "/* generated from adreno_pm4.xml by tools/genpm4.py -- do not edit by hand. */",
    "#ifndef FK_PM4_NAMES_H",
    "#define FK_PM4_NAMES_H",
    "static const struct fk_pm4_op fk_pm4_ops[] = {",
]
for op in sorted(names):
    lines.append(f'  {{ 0x{op:02x}, "{names[op]}" }},')
lines += ["};", f"#define FK_PM4_OP_COUNT {len(names)}", "#endif"]
open(args.out, "w").write("\n".join(lines) + "\n")
print(f"{len(names)} PM4 opcodes -> {args.out}")
