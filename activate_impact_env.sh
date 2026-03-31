#!/usr/bin/env bash

IMPACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="$IMPACT_ROOT/.conda/envs/impact_as"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH." >&2
  return 1 2>/dev/null || exit 1
fi

if [ ! -d "$ENV_PATH" ]; then
  echo "IMPACT environment not found: $ENV_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$IMPACT_ROOT/.runtime/matplotlib" "$IMPACT_ROOT/.runtime/xdg"
export MPLCONFIGDIR="$IMPACT_ROOT/.runtime/matplotlib"
export XDG_CONFIG_HOME="$IMPACT_ROOT/.runtime/xdg"

eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

# Drop Qt plugin paths inherited from another Conda/base environment.
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH
unset QT_QPA_FONTDIR
