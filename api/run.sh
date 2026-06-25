#!/usr/bin/env bash
# Start the SOH API. Run from the project root.
set -e
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/opt/anaconda3/envs/voltup_ml/bin/python}"
exec "$PYTHON" -m uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
