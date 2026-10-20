#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo "A CUDA toolkit with nvcc is required." >&2; exit 1; }
python -m pip install packaging ninja wheel setuptools
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-build-isolation ./third_party/causal-conv1d
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation -e ./third_party/mamba
python tools/check_environment.py
