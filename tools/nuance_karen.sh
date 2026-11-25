#!/bin/bash
# Superseded by the live character mode: tools/character_live.py karen [--test]  (kept as a shortcut)
cd "$(dirname "$0")/.."
ARGS=(karen); [ "${TEST:-0}" = 1 ] && ARGS+=(--test)
exec cloud/facefix/.venv/bin/python tools/character_live.py "${ARGS[@]}" "$@"
