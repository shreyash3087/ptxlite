#!/usr/bin/env bash
# check_ptx.sh: compile every example kernel with ptxlite, then validate
# the resulting PTX with ptxas.  Requires ptxas on PATH (or extracted from
# the nvidia-cuda-nvcc-cu12 wheel as described in the README).
#
# Usage:
#   ./scripts/check_ptx.sh [path/to/ptxas]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PTXLITE="${REPO}/build/ptxlite"
PTXAS="${1:-ptxas}"
ARCH="sm_75"
EXAMPLES_DIR="${REPO}/examples"
TMP="$(mktemp -d)"

if [[ ! -x "$PTXLITE" ]]; then
  echo "error: '$PTXLITE' not found. Run 'make' first." >&2
  exit 1
fi

pass=0; fail=0

for kernel in "$EXAMPLES_DIR"/*.k; do
  name="$(basename "$kernel" .k)"
  ptx_out="$TMP/${name}.ptx"

  if "$PTXLITE" -O2 --arch="$ARCH" -o "$ptx_out" "$kernel"; then
    if "$PTXAS" -arch="$ARCH" "$ptx_out" -o /dev/null 2>/dev/null; then
      echo "PASS: $name"
      ((pass++))
    else
      echo "FAIL: $name (ptxas rejected)"
      "$PTXAS" -arch="$ARCH" "$ptx_out" -o /dev/null || true
      ((fail++))
    fi
  else
    echo "FAIL: $name (ptxlite failed)"
    ((fail++))
  fi
done

rm -rf "$TMP"
echo ""
echo "Results: ${pass} passed, ${fail} failed"
[[ $fail -eq 0 ]]
