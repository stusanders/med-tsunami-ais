#!/usr/bin/env bash
# T27 demo shortcuts (works from any directory, any shell):
#   scripts/demo/demo.sh 8 --still-only [--at 12]   one frame, ~1 s (scenes: 1..15, 9b, 9c)
#   scripts/demo/demo.sh 8 --sheet                  12 frames in one PNG, ~2 s
#   scripts/demo/demo.sh 8 --draft [--from 10 --to 15]   half-res 15 fps preview, ~3-10 s
#   scripts/demo/demo.sh 8                          final scene video (add --cut both for 2, 6, 13)
#   scripts/demo/demo.sh all [--draft] [--only 8,9] whole build (final ~5 min, draft ~1.5 min)
#   scripts/demo/demo.sh join                       re-join demo_full.mp4 and demo_3min.mp4
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/.venv/bin/python"
export PYTHONPATH=src TSUNAMI_BACKEND=numpy
what="${1:?usage: demo.sh <scene number|all|join> [options]}"; shift
case "$what" in
  all)  exec "$PY" scripts/demo/build_all.py "$@" ;;
  join) exec "$PY" -c "import sys; sys.path.insert(0, 'scripts/demo'); import build_all, common
for c in ('full', '3min'): build_all.join(c, common.ORDER)" ;;
  *)    stem="$("$PY" -c "import sys; sys.path.insert(0, 'scripts/demo'); import common; print(common.SCENES[common.scene_id('$what')].stem)")"
        exec "$PY" "scripts/demo/$stem.py" "$@" ;;
esac
