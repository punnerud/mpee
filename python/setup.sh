#!/usr/bin/env bash
# One-time venv setup for the mpee cache server.
# Uses Python 3.12; the server itself depends only on the stdlib so
# `pip install` is a no-op for now (kept for future dependencies).

set -euo pipefail
cd "$(dirname "$0")"

if [ -d venv ]; then
    echo "venv already exists — delete python/venv to recreate"
    exit 0
fi

python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip >/dev/null
# No third-party requirements yet. If we add Flask or FastAPI later,
# pin them in requirements.txt and uncomment:
# pip install -r requirements.txt

echo
echo "venv created at python/venv"
echo "Start the server with:  ./run.sh"
