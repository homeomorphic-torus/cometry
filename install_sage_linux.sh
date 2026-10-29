#!/usr/bin/env bash
# Install SageMath 10.9 into a user-owned, recreatable cache on Linux x86_64.
set -euo pipefail
CACHE="${SAGE_CACHE_DIR:-$HOME/.cache/bergman-sage}"
mkdir -p "$CACHE" "$CACHE/tmp"
if [ ! -x "$CACHE/bin/micromamba" ]; then
  curl --fail --location --silent --show-error https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$CACHE" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$CACHE/mamba"
export TMPDIR="$CACHE/tmp"
"$CACHE/bin/micromamba" create -y -p "$CACHE/env" -c conda-forge 'sage=10.9'
printf '\nRun tests: %q run -p %q python -m unittest -v test_bergman\n' "$CACHE/bin/micromamba" "$CACHE/env"
