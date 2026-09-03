#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src

mkdir -p logs/stage1
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/stage1/rankbin-${STAMP}.log"

{
    echo "=== rank+bin experiment ==="
    date
    .venv/bin/python -c "
import yaml
c = yaml.safe_load(open('config.yaml'))
print('entity_rank_bins =', c['assemble']['entity_rank_bins'])
print('min_weight       =', c['stage1_pipeline']['structure']['min_weight'])
print('louvain          =', c['stage1_pipeline']['optional']['louvain'])
print('max_depth        =', c['model']['params']['max_depth'])
"
    echo
    echo "--- [1/2] re-assemble ---"
    .venv/bin/python -m tfgnn.tigergraph.assemble || exit 1
    echo
    echo "--- [2/2] three arms ---"
    .venv/bin/python -m baseline.train_xgboost --variant all || exit 1
    echo
    echo "=== complete ==="
    date
} >"$LOG" 2>&1

STATUS=$?
echo "exit=${STATUS} log=${LOG}"
tail -25 "$LOG"
exit "$STATUS"
