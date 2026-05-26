#!/usr/bin/env bash
# Activate the venv and start the cache server.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo "venv not found — run ./setup.sh first"
    exit 1
fi

exec ./venv/bin/python cache_server.py "$@"
