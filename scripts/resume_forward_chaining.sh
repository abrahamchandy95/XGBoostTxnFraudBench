#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src PYTHONUNBUFFERED=1
LOG="logs/stage1/forward-chaining-resume-$(date +%Y%m%d-%H%M%S).log"
{
    echo "=== resume: assemble + three arms ==="; date; echo
    .venv/bin/python -m tfgnn.tigergraph.assemble || exit 1
    echo; .venv/bin/python -m baseline.train_xgboost --variant all || exit 1
    echo; echo "=== complete ==="; date
} >"$LOG" 2>&1
echo "exit=$? log=$LOG"; tail -20 "$LOG"
