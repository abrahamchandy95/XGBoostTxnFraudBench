#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

mkdir -p logs/stage1
LOG="logs/stage1/forward-chaining-$(date +%Y%m%d-%H%M%S).log"

{
    echo "=== forward chaining ==="
    date
    .venv/bin/python -c "
import yaml
c = yaml.safe_load(open('config.yaml'))
print('train_folds       =', c['stage1_pipeline']['train_folds'])
print('entity_rank_bins  =', c['assemble']['entity_rank_bins'])
print('target_encode     =', c['features']['target_encode']['columns'])
"
    echo
    echo "--- [1/3] feature build: 4 training folds + val + test ---"
    .venv/bin/python -m tfgnn.tigergraph.pipeline || exit 1
    echo
    echo "--- [2/3] assemble; fact REUSED, warm-up fold withheld ---"
    .venv/bin/python -m tfgnn.tigergraph.assemble || exit 1
    echo
    echo "--- [3/3] three arms ---"
    .venv/bin/python -m baseline.train_xgboost --variant all || exit 1
    echo
    echo "=== complete ==="
    date
} >"$LOG" 2>&1

STATUS=$?
echo "exit=${STATUS} log=${LOG}"
tail -30 "$LOG"
exit "$STATUS"
