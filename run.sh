#!/usr/bin/env bash
# Launch VoiceClone Studio. Safe to run repeatedly.
set -u
APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PORT="${VOICE_CLONE_PORT:-7860}"

# Free the port if a previous instance is still holding it.
if curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$PORT" 2>/dev/null; then
  echo "VoiceClone Studio is already running at http://127.0.0.1:$PORT"
  exit 0
fi
stale=$(pgrep -f "app\.py" 2>/dev/null | while read -r p; do
  # only kill processes whose cwd is this app dir
  [ "$(readlink -f "/proc/$p/cwd" 2>/dev/null)" = "$APP_DIR" ] && echo "$p"
done)
if [ -n "$stale" ]; then
  echo "Cleaning up a stale instance (pid $stale)..."
  kill $stale 2>/dev/null
  for _ in $(seq 1 20); do
    still=""
    for p in $stale; do [ -d "/proc/$p" ] && still="$still $p"; done
    [ -z "$still" ] && break
    sleep 0.5
  done
  for p in $stale; do [ -d "/proc/$p" ] && kill -9 "$p" 2>/dev/null; done
  sleep 1
fi

cd "$APP_DIR" || exit 1
exec "$APP_DIR/.venv/bin/python" app.py
