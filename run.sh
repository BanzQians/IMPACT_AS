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

# Normalize locale for Qt/XKB compose handling. Validation and other text
# input dialogs create compose tables lazily; inherited LC_ALL=C.UTF-8 can
# trigger noisy "qt.xkb.compose" warnings even when LANG already points to a
# real UTF-8 locale available on the system.
if command -v locale >/dev/null 2>&1; then
  _locale_norm() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/utf-8/utf8/g'
  }
  _available_locales="$(locale -a 2>/dev/null | tr '[:upper:]' '[:lower:]' | sed 's/utf-8/utf8/g')"
  _gui_locale=""
  if [ -n "${LANG:-}" ]; then
    _want_locale="$(_locale_norm "${LANG}")"
    if printf '%s\n' "$_available_locales" | grep -Fxq "$_want_locale"; then
      _gui_locale="${LANG}"
    fi
  fi
  if [ -z "$_gui_locale" ]; then
    if printf '%s\n' "$_available_locales" | grep -Fxq "zh_cn.utf8"; then
      _gui_locale="zh_CN.UTF-8"
    elif printf '%s\n' "$_available_locales" | grep -Fxq "c.utf8"; then
      _gui_locale="C.UTF-8"
    fi
  fi
  if [ -n "$_gui_locale" ]; then
    export LANG="$_gui_locale"
    export LC_CTYPE="$_gui_locale"
    export LC_ALL="$_gui_locale"
  fi
  _compose_file=""
  if [ -n "${LC_CTYPE:-}" ]; then
    _compose_file="/usr/share/X11/locale/${LC_CTYPE}/Compose"
  fi
  if [ ! -f "$_compose_file" ] && [ -n "${LANG:-}" ]; then
    _compose_file="/usr/share/X11/locale/${LANG}/Compose"
  fi
  if [ ! -f "$_compose_file" ] && [ -f "/usr/share/X11/locale/C/Compose" ]; then
    _compose_file="/usr/share/X11/locale/C/Compose"
  fi
  if [ -f "$_compose_file" ]; then
    export XCOMPOSEFILE="$_compose_file"
  fi
  unset _available_locales _gui_locale _want_locale
  unset _compose_file
  unset -f _locale_norm 2>/dev/null || true
fi

# Silence the known harmless Qt/XKB compose-table warning without muting other
# Qt diagnostics.
if [ -n "${QT_LOGGING_RULES:-}" ]; then
  case ";${QT_LOGGING_RULES};" in
    *";qt.xkb.compose.warning=false;"*) ;;
    *) export QT_LOGGING_RULES="${QT_LOGGING_RULES};qt.xkb.compose.warning=false" ;;
  esac
else
  export QT_LOGGING_RULES="qt.xkb.compose.warning=false"
fi

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
