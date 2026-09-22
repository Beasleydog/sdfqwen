#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

"$PYTHON" -m pip uninstall -y torchao || true
"$PYTHON" -m pip install -r requirements.txt
exec "$PYTHON" train.py
