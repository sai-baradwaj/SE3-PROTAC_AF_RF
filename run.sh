#!/usr/bin/env bash
# ============================================================
# SE3AF v3.9.0 — PROTACPred  ·  One-shot startup script
# ============================================================
# Usage:
#   chmod +x run.sh
#   ./run.sh                  # install + start on port 5000
#   ./run.sh --port 8080      # custom port
#   ./run.sh --skip-install   # skip pip install (already done)
#   ./run.sh --stop           # stop the server
# ============================================================

set -e

PORT=5000
SKIP_INSTALL=0
STOP=0

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)    PORT="$2"; shift ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --stop)    STOP=1 ;;
        *) echo "Unknown: $1" ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Stop mode ────────────────────────────────────────────────
if [ "$STOP" -eq 1 ]; then
    echo "Stopping PROTACPred..."
    fuser -k ${PORT}/tcp 2>/dev/null && echo "Killed process on port $PORT" || echo "No process on port $PORT"
    exit 0
fi

echo "============================================"
echo "  SE3AF v3.9.0 — PROTACPred  "
echo "  Starting on http://localhost:$PORT"
echo "============================================"

# ── Python check ─────────────────────────────────────────────
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi
PYVER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "✅ Python $PYVER found at $PYTHON"

# ── Install dependencies ──────────────────────────────────────
if [ "$SKIP_INSTALL" -eq 0 ]; then
    echo ""
    echo "📦 Installing dependencies..."
    $PYTHON -m pip install -q -r requirements.txt
    echo "✅ Dependencies installed"
fi

# ── Create required directories ──────────────────────────────
mkdir -p history reports uploads data/alphafold checkpoints
echo "✅ Directories verified"

# ── Kill any existing process on port ────────────────────────
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

# ── Start Flask server ───────────────────────────────────────
echo ""
echo "🚀 Starting Flask server on port $PORT..."
echo "   Press Ctrl+C to stop"
echo ""
export PORT=$PORT
export FLASK_APP=app.py
export FLASK_ENV=production
$PYTHON app.py
