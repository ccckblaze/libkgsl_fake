#!/bin/sh
# Copy the platform identity a box has no board for, off a board that works.
#
#   tools/record_board.sh <host> <out-dir> [ssh-port]
#
# Two things sit outside the GPU question and stop a cardless box anyway:
#
#   /sys/devices/soc0/*      which SoC this is. A vendor runtime reads soc_id from there before it
#                            will look for a GPU at all; with no such node it names the target
#                            UNKNOWN and refuses the GPU backend -- no device layer can talk it
#                            into trying. FK_SYSFS names the mirror of /sys this script writes.
#   /dev/__properties__      Android/bionic below API 24 keeps every system property in this one
#                            file. A container starts with no property area, so every property
#                            reads back empty. bionic maps it read/write, so the box has to copy it
#                            onto its own tmpfs /dev at start; a bind-mounted file fails that
#                            mapping silently. The script prints the line that does it.
#
# What lands in <out-dir> is what was read off the board. Nothing here is invented or filled in:
# a node the board does not have is missing from the mirror, and the runtime will not read it.
#
set -eu

[ $# -ge 2 ] || { sed -n '2,21p' "$0"; exit 2; }
HOST=$1
OUT=$2
PORT=${3:-22}

mkdir -p "$OUT"

# the nodes a vendor runtime is known to read; extend the list for another generation
tmp=$(mktemp) || exit 1
ssh -p "$PORT" "$HOST" 'while IFS= read -r p; do
  [ -r "$p" ] || continue
  printf "== %s\n" "$p"
  cat "$p"
done' <<'NODES' > "$tmp"
/sys/devices/soc0/soc_id
/sys/devices/soc0/machine
/sys/devices/soc0/family
/sys/devices/soc0/hw_platform
/sys/devices/soc0/revision
/sys/devices/system/soc/soc0/id
NODES

node=
while IFS= read -r line; do
  case $line in
    "== "*)
      p=${line#'== '}
      node=$OUT/${p#/sys/}
      mkdir -p "$(dirname "$node")"
      : > "$node"
      continue ;;
  esac
  [ -n "$node" ] || continue
  printf '%s\n' "$line" >> "$node"
done < "$tmp"
rm -f "$tmp"

props=$OUT/__properties__
if ssh -p "$PORT" "$HOST" 'cat /dev/__properties__' > "$props" 2>/dev/null && [ -s "$props" ]; then
  chmod 644 "$props"
  echo "property area  $(wc -c < "$props" | tr -d ' ') bytes -> $props"
  echo "  put it where bionic looks, on the box's own tmpfs, before anything is compiled:"
  echo "  cp $props /dev/__properties__ && chmod 644 /dev/__properties__"
else
  rm -f "$props"
  echo "property area  none: this board keeps its properties elsewhere (newer bionic splits them by"
  echo "               context into a directory, which a container image can simply carry)"
fi

echo
echo "sys mirror     FK_SYSFS=$OUT"
for f in $(cd "$OUT" && find devices -type f 2>/dev/null | sort); do
  printf '  %-46s %s\n' "$f" "$(cat "$OUT/$f" | head -1)"
done
echo
echo "the mirror is a copy of one board's identity. Re-record it per generation, the same rule as"
echo "for the device property table (docs/record.md) -- a runtime that reads a plausible soc_id will"
echo "pick a codegen path for the wrong chip, and no amount of replaying afterwards looks like that."
