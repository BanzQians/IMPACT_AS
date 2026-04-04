#!/usr/bin/env bash
# Launch IMPACT AS without needing to source anything.
# Usage:  ./run.sh  [--oplog]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.conda/envs/impact_as/bin/python"
CONDA_PREFIX="$SCRIPT_DIR/.conda/envs/impact_as"

if [ ! -x "$PYTHON" ]; then
  echo "Python not found: $PYTHON" >&2
  exit 1
fi

# Runtime dirs (matplotlib cache, Qt XDG)
mkdir -p "$SCRIPT_DIR/.runtime/matplotlib" "$SCRIPT_DIR/.runtime/xdg"
export MPLCONFIGDIR="$SCRIPT_DIR/.runtime/matplotlib"
export XDG_CONFIG_HOME="$SCRIPT_DIR/.runtime/xdg"

# Fontconfig — prefer conda env's config, fall back to system
if [ -f "$CONDA_PREFIX/etc/fonts/fonts.conf" ]; then
  export FONTCONFIG_FILE="$CONDA_PREFIX/etc/fonts/fonts.conf"
  export FONTCONFIG_PATH="$CONDA_PREFIX/etc/fonts"
elif [ -f "/etc/fonts/fonts.conf" ]; then
  export FONTCONFIG_FILE="/etc/fonts/fonts.conf"
  export FONTCONFIG_PATH="/etc/fonts"
fi

# Clean Qt plugin paths from other environments
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH
unset QT_QPA_FONTDIR

# Preflight the GUI display unless the caller explicitly wants offscreen mode.
if [ "${QT_QPA_PLATFORM:-}" != "offscreen" ]; then
  if [ -z "${DISPLAY:-}" ]; then
    echo "No DISPLAY is set. This launcher starts a Qt GUI and needs an X11 desktop session." >&2
    echo "If you only want a headless smoke test, run: QT_QPA_PLATFORM=offscreen ./run.sh" >&2
    exit 2
  fi
  if command -v xdpyinfo >/dev/null 2>&1; then
    if ! xdpyinfo >/dev/null 2>&1; then
      echo "Cannot open X11 display ${DISPLAY} from this shell." >&2
      echo "The earlier 'qt.xkb.compose' line is only a warning; the real failure is missing GUI access." >&2
      echo "Open a terminal inside the active desktop session, or fix DISPLAY/XAUTHORITY, then rerun." >&2
      exit 2
    fi
  fi
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" app.py "$@"
