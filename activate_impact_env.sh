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

if [ -z "${FONTCONFIG_FILE:-}" ]; then
  if [ -f "$CONDA_PREFIX/etc/fonts/fonts.conf" ]; then
    export FONTCONFIG_FILE="$CONDA_PREFIX/etc/fonts/fonts.conf"
  elif [ -f "/etc/fonts/fonts.conf" ]; then
    export FONTCONFIG_FILE="/etc/fonts/fonts.conf"
  fi
fi

if [ -z "${FONTCONFIG_PATH:-}" ]; then
  if [ -n "${FONTCONFIG_FILE:-}" ] && [ -d "$(dirname "$FONTCONFIG_FILE")" ]; then
    export FONTCONFIG_PATH="$(dirname "$FONTCONFIG_FILE")"
  elif [ -d "$CONDA_PREFIX/etc/fonts" ]; then
    export FONTCONFIG_PATH="$CONDA_PREFIX/etc/fonts"
  elif [ -d "/etc/fonts" ]; then
    export FONTCONFIG_PATH="/etc/fonts"
  fi
fi

# Drop Qt plugin paths inherited from another Conda/base environment.
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH
unset QT_QPA_FONTDIR
