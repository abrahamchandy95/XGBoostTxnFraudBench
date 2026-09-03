#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

NO_GNN=0; GNN_ONLY=0; LIFT_ONLY=0; PAIR_GRAPH=0; FROM_EXPORT=0; DRY_RUN=0
for argument in "$@"; do
  case "${argument}" in
    --no-gnn)      NO_GNN=1 ;;
    --gnn-only)    GNN_ONLY=1 ;;
    --lift-only)   LIFT_ONLY=1 ;;
    --pair-graph)  PAIR_GRAPH=1 ;;
    --from-export) FROM_EXPORT=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    *) echo "unknown option: ${argument}" >&2; exit 2 ;;
  esac
done

mkdir -p logs artifacts/stage1 artifacts/stage2 data/stage1 data/stage2
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${ROOT}/logs/run_all-${STAMP}.log"
exec > >(tee -a "${LOG}") 2>&1

PYTHON="${ROOT}/.venv/bin/python"
step() { echo; echo "=============== $* ==============="; date '+%H:%M:%S'; }
fail() { echo; echo "FAILED at: $*"; echo "Log: ${LOG}"; exit 1; }
trap 'fail "line ${LINENO}"' ERR

echo "=================================================================="
echo "TF_GNN full run"
echo "started ${STAMP}   log ${LOG}"
echo "=================================================================="

[[ -f .env ]]        || fail ".env missing (HOST, GRAPHNAME, SECRET)"
[[ -f config.yaml ]] || fail "config.yaml missing"
if [[ ! -x "${PYTHON}" ]]; then
  step "creating .venv"
  python3 -m venv .venv
fi

step "0/8  dependencies"
# ONE requirements file, both halves. torch and torch-geometric used to be
# installed ad hoc on the line below this one, from a separate
# requirements-stage2.txt that claimed to be Linux-only and was not.
"${PYTHON}" -m pip install --quiet --upgrade pip
"${PYTHON}" -m pip install --quiet -r requirements.txt

# The four-arm comparison reads the matrix and the embeddings off DISK, so it
# needs neither TigerGraph nor a GPU. Exiting here is what makes the
# two-machine split's last step a normal run rather than a remembered
# incantation.
if [[ "${LIFT_ONLY}" -eq 1 ]]; then
  step "LIFT  four arms + charts (reuses the matrix and the embeddings)"
  "${PYTHON}" -u -m baseline.train_xgboost --variant all ${XGB_ARGS:-}
  "${PYTHON}" -m tfgnn.report
  echo; echo "--lift-only: done. Nothing was read from TigerGraph."; exit 0
fi

step "1/8  static validation (no TigerGraph)"
"${PYTHON}" scripts/validate_stage1_static.py

step "2/8  offline end-to-end smoke test (no TigerGraph)"
"${PYTHON}" scripts/smoke_stage1_offline.py | tail -12

# STRICT, not --report-only. This is the only entry point anyone uses, and the
# next phase runs reset_build, which DESTROYS the previous build's derived state.
# A schema mismatch found after that costs a rebuild; found here it costs
# nothing. Warnings still only warn -- preflight fails on `problems`, and the
# expected Merchant.distinct_location_count note is a warning.
step "3/8  auth / schema / installed-query preflight"
"${PYTHON}" -m tfgnn.tigergraph.preflight

if [[ "${DRY_RUN}" -eq 1 ]]; then
  # INSTALL FIRST, and yes, that is still a dry run. The plan is resolved by
  # CALLING derive_build_plan, and a query that is not installed has no REST
  # endpoint -- so on a FRESH LOAD, which is exactly when this command is
  # recommended, the listing used to die on a bare 404. Installing writes QUERY
  # DEFINITIONS, never data: no reset_build, no build, nothing that touches a
  # vertex or an edge. It is also idempotent, so paying for it here costs the
  # real run nothing.
  step "dry-run 1/2  CREATE + INSTALL (query definitions only; no data written)"
  "${PYTHON}" -m tfgnn.tigergraph.install_gsql --include-optional --include-stage2

  # CONNECTED --list, not --list --offline. Preflight just proved the connection,
  # and offline mode prints config TOKENS rather than resolved ranks: it cannot
  # expand train_folds into its 6 builds, and it cannot surface a fold_source
  # disagreement. Those are exactly what a dry run before a 40-minute build is
  # for -- on regenerated data `fold_source: auto` may find real causal_folds
  # and refuse a count that disagrees with train_folds, and this is where that
  # should be discovered.
  step "dry-run 2/2  plan (read-only, resolved against the live graph)"
  "${PYTHON}" -m tfgnn.tigergraph.pipeline --list
  echo; echo "--dry-run: queries installed; NO data was written to the graph."
  exit 0
fi

if [[ "${GNN_ONLY}" -eq 0 ]]; then
  if [[ "${FROM_EXPORT}" -eq 0 ]]; then
    step "4/8  CREATE + INSTALL all queries (core, optional, stage2 exporters)"
    # ALWAYS, AND THIS IS THE ONLY PLACE IT NEEDS TO HAPPEN. There is no
    # separate install command to remember: a normal run reinstalls every query
    # from the files on disk before it builds anything.
    #
    # It has to be unconditional rather than "only when something changed",
    # because TIGERGRAPH SILENTLY IGNORES UNKNOWN QUERY PARAMETERS. A query
    # whose signature grew -- card_home_distance gained cutoff_event_seq when
    # cardholder relocation landed -- would keep serving its OLD compiled body,
    # accept the new argument, discard it, and compute the old answer. No error
    # anywhere. Reinstalling costs minutes; that failure costs a run and looks
    # like a result.
    #
    # CREATE is not INSTALL: a created query has no REST endpoint until it
    # compiles. install_gsql does all the CREATEs then one batched INSTALL.
    "${PYTHON}" -m tfgnn.tigergraph.install_gsql --include-optional --include-stage2

    step "5/8  feature build: identity -> communities -> features -> dimensions"
    # This is the whole startup chain in its required order:
    #   reset -> measure_pii_degrees -> match_parties ->
    #   stamp_same_as_provenance -> unify_parties -> assert_resolved_entity ->
    #   resolved_entity_stats -> stamp_resolved_entity_keys ->
    #   build_interaction_edges -> card_merchant_degree_stats ->
    #   card_home_distance -> density_check -> projections -> wcc/louvain ->
    #   community_stats -> pagerank -> aggregates -> dimension export
    # repeated once per build (train_folds=5 -> 6 builds).
    "${PYTHON}" -m tfgnn.tigergraph.pipeline

    step "6/8  fact table export"
    "${PYTHON}" -m tfgnn.tigergraph.export

    # STAGE 3. Runs ONCE, not per build: every window is anchored at the row's
    # own unix_time, so one table serves every snapshot. Sits AFTER the fact
    # export because it reconciles against the transaction count, and BEFORE
    # assemble because assemble joins it on event_seq.
    step "6b/8  Stage 3: per-row temporal features"
    "${PYTHON}" -m tfgnn.tigergraph.temporal
  else
    step "4-6b/8  skipped (--from-export): reusing the build, fact table and temporal table"
    echo "  NOTE this also skips the INSTALL. Only safe when no .gsql file has"
    echo "  changed since the build being reused: a query whose signature grew"
    echo "  would silently serve its old body, because TigerGraph ignores"
    echo "  unknown parameters. Drop --from-export if in any doubt."
  fi

  step "7/8  assemble the matrix"
  "${PYTHON}" -m tfgnn.tigergraph.assemble

  step "8/8  Stage 1: three XGBoost arms"
  "${PYTHON}" -u -m baseline.train_xgboost --variant all

  # Charts, not a table of numbers. The lift plot is drawn on an axis centred
  # on ZERO because that is the decision: an interval touching the line has not
  # shown an effect, however large the point estimate reads. Pure stdlib, so it
  # adds no dependency to the run.
  echo; echo "Stage 1 complete."
  "${PYTHON}" -m tfgnn.report
fi

if [[ "${NO_GNN}" -eq 1 ]]; then
  echo; echo "--no-gnn: stopping after the XGBoost arms."; exit 0
fi

RELATIONS_FLAG=""
if [[ "${PAIR_GRAPH}" -eq 1 ]]; then
  RELATIONS_FLAG="--relations pair"
  echo "  --pair-graph: ~400k edges (incl. derived co-use pairs) instead of 81M"
fi

# Card_Merchant_Transaction is Stage 2's link-prediction TARGET and it is BUILT,
# not loaded: tf_gnn_loader_v2 declares the edge and deliberately skips the
# loading job (2 x 33.7M edges nothing read until now). The query that
# materialises it was installed by step 4 and, until 2026-07-30, called by
# NOTHING -- so the target relation sat at 0 and Stage 2 could not train.
#
# It runs ONCE here rather than per build: it carries no build_id and no cutoff
# (Stage 2 needs full history and filters at sample time), and reset_build does
# not clear it, so rebuilding it per build would rewrite the same edges six
# times. Count-gated and idempotent, so a completed build makes this a no-op
# and a killed one resumes.
if [[ "${PAIR_GRAPH}" -eq 0 ]]; then
  step "STAGE 2  materialise Card_Merchant_Transaction (once; no-op if present)"
  "${PYTHON}" -m tfgnn.tigergraph.target_edges
else
  echo
  echo "  --pair-graph: skipping the target-edge build, because pair mode does"
  echo "  not export Card_Merchant_Transaction. Pair mode proves the EXPORT"
  echo "  path only -- training needs the target relation for its seeds."
fi

step "STAGE 2  export the graph for cuGraph (cached after the first run)"
# shellcheck disable=SC2086
"${PYTHON}" -m tfgnn.stage2.run --export-only --graph-dir data/stage2/graph ${RELATIONS_FLAG}

step "STAGE 2  train the R-GCN and export embeddings"
# shellcheck disable=SC2086
"${PYTHON}" -m tfgnn.stage2.run \
    --graph-dir data/stage2/graph \
    --out-dir artifacts/stage2 \
    --epochs 3 ${RELATIONS_FLAG}

step "STAGE 2  fourth arm + THE GNN LIFT"
# `--variant all`, NOT `--variant raw_plus_embeddings`, and the difference is
# the whole point of the step. The single-variant path trains and saves that
# arm and computes NO lift: run_variant returns, and only run_all builds the
# comparison. The GNN lift is a PAIRED bootstrap of raw_plus_graph against
# raw_plus_embeddings, so both arms' predictions must exist in ONE process on
# identical rows -- which is also why all four are retrained rather than
# reloaded. This shipped as the single-variant form, so a full `make` run
# produced the embedding arm and never the number that justifies it.
# NOT `|| echo` (changed 2026-08-02). It used to swallow a non-zero exit and
# carry on to the report, so a FAILED fourth arm produced a three-arm chart
# that reads exactly like "the GNN added nothing" -- the most expensive
# possible way to be wrong about this project's headline number. The absent-
# tables case it was guarding is already handled inside the trainer, which
# skips the arm with a NOTE and exits 0; anything that reaches this line as a
# failure is a real one and must stop the run.
"${PYTHON}" -u -m baseline.train_xgboost --variant all || \
  fail "Stage 2 fourth arm (baseline.train_xgboost --variant all)"

# The full picture: all four arms, every lift with its interval, and the R-GCN
# loss curve. Re-run any time with `make report` -- it only reads artifacts.
step "REPORT  charts for every arm, every lift, and Stage 2 training"
"${PYTHON}" -m tfgnn.report

echo
echo "=================================================================="
echo "run complete   $(date '+%H:%M:%S')"
echo "log ${LOG}"
echo "charts        artifacts/report.png"
echo "=================================================================="
