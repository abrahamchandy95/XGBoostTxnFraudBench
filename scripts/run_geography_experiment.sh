#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

mkdir -p logs/stage1
LOG="logs/stage1/geography-$(date +%Y%m%d-%H%M%S).log"

{
    echo "=== geography + te_merchant_id ==="
    date
    echo
    echo "--- [0/4] static + smoke ---"
    .venv/bin/python scripts/validate_stage1_static.py || exit 1
    .venv/bin/python scripts/smoke_stage1_offline.py 2>&1 | tail -4 || exit 1
    echo
    echo "--- [1/4] reinstall the two changed export queries + geography query ---"
    .venv/bin/python -m tfgnn.tigergraph.install_gsql \
        export_card_and_merchant_features card_home_distance || exit 1
    echo
    echo "--- [2/4] feature build, both snapshots (re-exports dimensions) ---"
    .venv/bin/python -m tfgnn.tigergraph.pipeline || exit 1
    echo
    echo "--- [3/4] assemble; fact table REUSED, dimensions fresh ---"
    .venv/bin/python -m tfgnn.tigergraph.assemble || exit 1
    echo
    echo "--- [4/4] three arms ---"
    .venv/bin/python -m baseline.train_xgboost --variant all || exit 1
    echo
    echo "=== complete ==="
    date
} >"$LOG" 2>&1

STATUS=$?
echo "exit=${STATUS} log=${LOG}"
tail -30 "$LOG"
exit "$STATUS"
