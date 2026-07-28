#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1 && ! brew list --versions libomp >/dev/null 2>&1; then
  brew install libomp
fi
uv sync --extra dev --no-editable
