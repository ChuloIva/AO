#!/bin/bash
# Launch TUI app in browser via ttyd

PORT=7681
URL="http://localhost:$PORT"

# Kill any existing ttyd on this port
pkill -f "ttyd.*$PORT" 2>/dev/null

# Activate venv and start ttyd in background
cd "$(dirname "$0")"
ttyd -p $PORT -W -t fontSize=14 -t theme='{"background": "#1a1a2e"}' -t enableSixel=true -t enableTrzsz=false script -q /dev/null .venv/bin/python tui_app.py &
TTYD_PID=$!

# Wait for ttyd to start
sleep 1

# Open browser (macOS)
open "$URL"

echo "TUI running at $URL (PID: $TTYD_PID)"
echo "Press Ctrl+C to stop"

# Wait and cleanup on exit
trap "kill $TTYD_PID 2>/dev/null; echo 'Stopped'" EXIT
wait $TTYD_PID
