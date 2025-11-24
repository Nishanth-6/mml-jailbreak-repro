#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m src.mml.cli --config config.yaml --limit 3
