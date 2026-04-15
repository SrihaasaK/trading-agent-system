#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

exec "$REPO_DIR/venv/bin/python" "$REPO_DIR/main.py"
