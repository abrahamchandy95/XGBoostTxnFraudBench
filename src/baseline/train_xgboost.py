import argparse
import gc
import json
import time
from pathlib import Path
from typing import (
    Literal,
    NotRequired,
    Protocol,
    SupportsFloat,
    SupportsInt,
    TypedDict,
    cast,
    final,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from pydantic import TypeAdapter

from baseline import dataset, features, metrics, split
from baseline.types import PARTS, VARIANTS, Part, Partitions, Variant
from common.config import Config, load_raw_config, project_root
from tfgnn.features_meta import (
    AGGREGATE_FEATURES,
    FACT_ID,
    GRAPH_FEATURES,
    LABEL,
)


type ParameterValue = str | int | float | bool
type PredictionArray = npt.NDArray[np.float32]


class _Params(TypedDict):
    n_estimators: int
    max_depth: int
    learning_rate: float
    tree_method: str
    eval_metric: str
    early_stopping_rounds: int
    min_child_weight: NotRequired[float]
    subsample: NotRequired[float]
    colsample_bytree: NotRequired[float]
    objective: NotRequired[str]
    nthread: NotRequired[int]
    max_bin: NotRequired[int]


class _Model(TypedDict):
    name: Literal["xgboost"]
    params: _Params


class _Evaluation(TypedDict):
    bootstrap_resamples: int
    bootstrap_level: float


class _Stage1(TypedDict):
    output_dir: str
    evaluation: _Evaluation


class _Scores(TypedDict):
    train: float
    val: float
    test: float


class _Counts(TypedDict):
    train: int
    val: int
    test: int


class _Result(TypedDict):
    variant: Variant
    aucpr: _Scores
    roc_auc: _Scores
    test_report: metrics.Report
    val_report: metrics.Report
    threshold_from_validation: float
    best_iteration: int
    n_features: int
    feature_names: list[str]
    dropped_columns: dict[str, list[str]]
    rows: _Counts
    positives: _Counts
    prevalence: _Scores
    model_path: str
    test_predictions_path: str
    feature_manifest_path: str
    model_roundtrip_max_abs_diff: float
    evaluation_model: Literal["reloaded_saved_json"]
    inference_rows_per_second: float


class _Lift(TypedDict):
    baseline: str
    candidate: str
    baseline_test_aucpr: float
    candidate_test_aucpr: float
    absolute: float
    relative: float | None
    paired_interval: metrics.Interval | None
    probability_candidate_better: float | None
    claim: str


class _Comparison(TypedDict):
    results: dict[str, _Result]
    lifts: list[_Lift]
    notes: list[str]


class _BoosterIO(Protocol):
    def load_model(self, fname: str) -> None: ...
    def save_model(self, fname: str) -> None: ...


_DATA_ADAPTER = TypeAdapter(dataset.DataSource)
_SPLIT_ADAPTER = TypeAdapter(split.SplitPlan)
_FEATURE_ADAPTER = TypeAdapter(features.FeaturePlan)
_MODEL_ADAPTER = TypeAdapter(_Model)
_STAGE1_ADAPTER = TypeAdapter(_Stage1)
_SEED_ADAPTER = TypeAdapter(int)

_CLI_VARIANTS: tuple[str, ...] = (
    "all",
    "raw",
    "raw_plus_aggregates",
    "raw_plus_graph",
    "raw_plus_temporal",
    "raw_plus_embeddings",
    "raw_plus_temporal_embeddings",
    "raw_plus_aggregates_embeddings",
    "raw_plus_pair_scores",
    "raw_plus_temporal_pair_scores",
)


def _config_value(config: Config, key: str) -> object:
    typed = cast(dict[str, object], config)
    if key not in typed:
        raise KeyError(f"configuration section {key!r} is missing")
    return typed[key]


def _int(series: pd.Series) -> int:
    return int(cast(SupportsInt, series.sum()))


def _float(value: object) -> float:
    return float(cast(SupportsFloat, value))


@final
class XGBoostTrainer:
    _params: dict[str, ParameterValue]
    _num_boost_round: int
    _early_stopping_rounds: int
    _booster: xgb.Booster | None
    _best_iteration: int | None
    _train_matrix: xgb.DMatrix | None

    def __init__(self, model: _Model, seed: int) -> None:
        configured = model["params"]
        params: dict[str, ParameterValue] = {
            "max_depth": configured["max_depth"],
            "learning_rate": configured["learning_rate"],
            "tree_method": configured["tree_method"],
            "eval_metric": configured["eval_metric"],
            "seed": seed,
            "objective": configured.get("objective", "binary:logistic"),
        }
        # Written out rather than looped, so a key that is not in _Params is a
        # type error here instead of a parameter XGBoost silently ignores.
        if "min_child_weight" in configured:
            params["min_child_weight"] = configured["min_child_weight"]
        if "subsample" in configured:
            params["subsample"] = configured["subsample"]
        if "colsample_bytree" in configured:
            params["colsample_bytree"] = configured["colsample_bytree"]
        if "nthread" in configured:
            params["nthread"] = configured["nthread"]
        if "max_bin" in configured:
            params["max_bin"] = configured["max_bin"]

        self._params = params
        self._num_boost_round = configured["n_estimators"]
        self._early_stopping_rounds = configured["early_stopping_rounds"]
        self._booster = None
        self._best_iteration = None
        self._train_matrix = None

    @property
    def best_iteration(self) -> int:
        if self._best_iteration is None:
            raise RuntimeError("model has not been fitted or loaded")
        return self._best_iteration

    def fit(self, data: features.FeatureDataset) -> None:
        positives = _int(data.y.train)
        negatives = len(data.y.train) - positives
        if positives == 0 or negatives == 0:
            raise ValueError("training partition must contain both classes")

        params = self._params.copy()
        params["scale_pos_weight"] = negatives / positives

        # missing=NaN is the default, and it is the point: an entity this build
        # did not see reaches the tree as NaN and the tree learns a default
        # direction from the training data.
        train_matrix = xgb.DMatrix(
            data.X.train, label=data.y.train, feature_names=data.feature_names
        )
        val_matrix = xgb.DMatrix(
            data.X.val, label=data.y.val, feature_names=data.feature_names
        )
        # NO `(train_matrix, "train")` IN evals. Early stopping reads only the
        # LAST eval entry, so the train evaluation never influenced it -- verified
        # by comparing best_iteration and the serialised model with and without
        # it: byte-identical. It was one extra full-train prediction per boosting
        # round, and the train AUC-PR that is actually reported is recomputed
        # later from the saved model anyway.
        booster = xgb.train(
            params=params,
            dtrain=train_matrix,
            num_boost_round=self._num_boost_round,
            evals=[(val_matrix, "val")],
            early_stopping_rounds=self._early_stopping_rounds,
            verbose_eval=False,
        )
        self._booster = booster
        self._best_iteration = int(cast(SupportsInt, booster.best_iteration))
        # RETAINED so the train prediction can reuse it. Rebuilding a DMatrix over
        # the train frame later costs ~39 GiB on the widest arm, and this one
        # already exists. The caller frees it via `release_train_matrix()` once the
        # train prediction is taken -- and, in exchange, may drop its own reference
        # to X.train as soon as fit returns.
        self._train_matrix = train_matrix

        # TOP FEATURES BY TOTAL GAIN. Added 2026-08-03 because nothing in this
        # repo reported them, and the first time an arm moved by a suspicious
        # amount -- raw_plus_temporal at +0.3115 against +0.0098 for the whole
        # 37-column graph family -- there was no way to ask WHICH feature did
        # it. A large lift from a velocity count is a credible fraud result; a
        # large lift from a tenure or first-seen quantity is a CALENDAR PROXY
        # on a temporally split dataset, and systemprompt already disqualified
        # first_seen_unix_time as a node feature for exactly that. Those two
        # look identical in an AUCPR table and completely different here.
        gains = cast("dict[str, float]", booster.get_score(importance_type="gain"))
        if gains:
            total = sum(gains.values()) or 1.0
            ranked = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)
            print("  top features by gain:")
            for name, gain in ranked[:8]:
                print(f"    {name:<40s} {gain / total:6.1%}")
        if self._best_iteration + 1 >= self._num_boost_round:
            print(
                f"  ! early stopping never fired: best_iteration="
                f"{self._best_iteration} of n_estimators={self._num_boost_round}. "
                "The model may still be improving; raise n_estimators."
            )

    def load(self, path: Path) -> None:
        booster = xgb.Booster()
        cast(_BoosterIO, booster).load_model(str(path))
        best = booster.attr("best_iteration")
        if best is None:
            raise RuntimeError(f"saved model has no best_iteration: {path}")
        self._booster = booster
        self._best_iteration = int(best)

    def predict(self, frame: pd.DataFrame, names: list[str]) -> PredictionArray:
        if self._booster is None:
            raise RuntimeError("model has not been fitted or loaded")
        matrix = xgb.DMatrix(frame, feature_names=names)
        return cast(
            PredictionArray,
            self._booster.predict(matrix, iteration_range=(0, self.best_iteration + 1)),
        )

    def predict_matrix(self, matrix: xgb.DMatrix) -> PredictionArray:
        """Predict from an already-built DMatrix.

        Exists so the train prediction can reuse the matrix `fit` already built
        instead of constructing a second one over the same 19.6M x 260 frame.
        Verified 0.0 max abs difference against the freshly-built path.
        """
        if self._booster is None:
            raise RuntimeError("model has not been fitted or loaded")
        return cast(
            PredictionArray,
            self._booster.predict(matrix, iteration_range=(0, self.best_iteration + 1)),
        )

    @property
    def train_matrix(self) -> xgb.DMatrix:
        if self._train_matrix is None:
            raise RuntimeError("fit has not run, so no train matrix was retained")
        return self._train_matrix

    def release_train_matrix(self) -> None:
        """Drop the retained train DMatrix. Call once the train prediction is in."""
        self._train_matrix = None
        _ = gc.collect()

    def save(self, path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("model has not been fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        cast(_BoosterIO, self._booster).save_model(str(path))


def _output_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


#: One definition, in dataset, because the load is where the kills happen.
report_memory = dataset.report_memory


def load_partitions(
    config: Config, families: frozenset[str] | None = None
) -> Partitions[pd.DataFrame]:
    source = _DATA_ADAPTER.validate_python(_config_value(config, "data"))
    plan = _SPLIT_ADAPTER.validate_python(_config_value(config, "split"))
    if families is None:
        families = dataset.ALL_FAMILIES
    return split.split(dataset.load(source, families), plan)


def run_variant(
    config: Config,
    variant: Variant,
    partitions: Partitions[pd.DataFrame],
) -> tuple[_Result, PredictionArray]:
    plan = _FEATURE_ADAPTER.validate_python(_config_value(config, "features"))
    model = _MODEL_ADAPTER.validate_python(_config_value(config, "model"))
    stage1 = _STAGE1_ADAPTER.validate_python(_config_value(config, "stage1"))
    seed = _SEED_ADAPTER.validate_python(_config_value(config, "seed"))
    evaluation = stage1["evaluation"]

    print(f"\n=== {variant} ===")
    data = features.build_features(partitions, plan, variant, seed)

    output = _output_dir(stage1["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"xgboost_{variant}.json"
    predictions_path = output / f"test_predictions_{variant}.parquet"
    manifest_path = output / f"feature_manifest_{variant}.json"

    trainer = XGBoostTrainer(model, seed)
    trainer.fit(data)
    # X.train IS DEAD FROM HERE. `fit` retained the DMatrix built from it, which
    # is what the train prediction below uses, so the 19.6 GiB frame has no
    # remaining reader. FeatureDataset is a plain dataclass, so rebinding X is
    # allowed; only Partitions itself is frozen.
    data.X = Partitions(train=pd.DataFrame(), val=data.X.val, test=data.X.test)
    _ = gc.collect()
    before = trainer.predict(data.X.test, data.feature_names)
    trainer.save(model_path)

    # Metrics come from the reloaded artefact, so the number reported is the
    # number the saved model produces.
    reloaded = XGBoostTrainer(model, seed)
    reloaded.load(model_path)

    started = time.perf_counter()
    after = reloaded.predict(data.X.test, data.feature_names)
    elapsed = time.perf_counter() - started
    rows_per_second = len(after) / elapsed if elapsed > 0 else float("inf")

    difference = np.abs(before - after)
    max_abs_diff = _float(difference.max()) if difference.size else 0.0
    if max_abs_diff > 1.0e-12:
        raise RuntimeError(
            f"saved-model round trip changed predictions for {variant}: "
            f"max_abs_diff={max_abs_diff}"
        )

    # From the RETAINED matrix, not a rebuilt one: same numbers, ~39 GiB less.
    train_prediction = reloaded.predict_matrix(trainer.train_matrix)
    trainer.release_train_matrix()
    val_prediction = reloaded.predict(data.X.val, data.feature_names)

    labels: dict[Part, npt.NDArray[np.float64]] = {
        part: data.y[part].to_numpy(dtype="float64") for part in PARTS
    }
    scores: dict[Part, npt.NDArray[np.float64]] = {
        "train": train_prediction.astype("float64"),
        "val": val_prediction.astype("float64"),
        "test": after.astype("float64"),
    }

    # Tuned on validation, applied unchanged to test.
    threshold = metrics.best_f1_threshold(labels["val"], scores["val"])
    val_report = metrics.report(
        labels["val"],
        scores["val"],
        threshold=threshold,
        bootstrap_resamples=evaluation["bootstrap_resamples"],
        bootstrap_level=evaluation["bootstrap_level"],
        seed=seed,
    )
    test_report = metrics.report(
        labels["test"],
        scores["test"],
        threshold=threshold,
        bootstrap_resamples=evaluation["bootstrap_resamples"],
        bootstrap_level=evaluation["bootstrap_level"],
        seed=seed,
    )

    aucpr: _Scores = {
        "train": metrics.aucpr(labels["train"], scores["train"]),
        "val": val_report["aucpr"],
        "test": test_report["aucpr"],
    }
    roc_auc: _Scores = {
        "train": _float(metrics.roc_auc_score(labels["train"], scores["train"])),
        "val": val_report["roc_auc"],
        "test": test_report["roc_auc"],
    }

    prediction_frame = pd.DataFrame(
        {
            FACT_ID: partitions.test[FACT_ID].reset_index(drop=True).astype("string"),
            LABEL: data.y.test.to_numpy(dtype="int8"),
            "prediction": after,
        }
    )
    prediction_frame.to_parquet(predictions_path, index=False)

    _ = manifest_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "feature_names": data.feature_names,
                "dropped_columns": data.dropped_columns,
                "feature_plan": plan,
                "threshold_from_validation": threshold,
                "model_path": str(model_path),
                "test_predictions_path": str(predictions_path),
                "note": (
                    "The XGBoost JSON contains the Booster only. Feature "
                    "construction is defined by this repository, config.yaml "
                    "and the GSQL build that produced the matrix; the JSON "
                    "alone is not a standalone preprocessing pipeline."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result: _Result = {
        "variant": variant,
        "aucpr": aucpr,
        "roc_auc": roc_auc,
        "test_report": test_report,
        "val_report": val_report,
        "threshold_from_validation": threshold,
        "best_iteration": reloaded.best_iteration,
        "n_features": len(data.feature_names),
        "feature_names": data.feature_names,
        "dropped_columns": data.dropped_columns,
        "rows": {part: len(data.y[part]) for part in PARTS},  # type: ignore[typeddict-item]
        "positives": {part: _int(data.y[part]) for part in PARTS},  # type: ignore[typeddict-item]
        "prevalence": {  # type: ignore[typeddict-item]
            part: _float(data.y[part].mean()) for part in PARTS
        },
        "model_path": str(model_path),
        "test_predictions_path": str(predictions_path),
        "feature_manifest_path": str(manifest_path),
        "model_roundtrip_max_abs_diff": max_abs_diff,
        "evaluation_model": "reloaded_saved_json",
        "inference_rows_per_second": rows_per_second,
    }

    _ = (output / f"metrics_{variant}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    interval = test_report["aucpr_interval"]
    band = (
        f" [{interval['lower']:.6f}, {interval['upper']:.6f}]"
        if interval is not None
        else ""
    )
    print(
        f"[{variant}] features={len(data.feature_names)} "
        f"best_iteration={reloaded.best_iteration}"
    )
    print(
        f"  AUCPR train={aucpr['train']:.6f} val={aucpr['val']:.6f} "
        f"test={aucpr['test']:.6f}{band}"
    )
    print(
        f"  test ROC-AUC={roc_auc['test']:.6f} "
        f"F1@{threshold:.4f}={test_report['at_threshold']['f1']:.6f} "
        f"({rows_per_second:,.0f} rows/s)"
    )
    gap = aucpr["val"] - aucpr["test"]
    if abs(gap) > 0.05:
        print(
            f"  ! val-test AUCPR gap is {gap:+.4f}. A large gap between "
            "validation and a strictly later test window is a conventional "
            "leakage indicator (DECISIONS_LEAK_CONTROL D7.4). At this "
            "prevalence, read it against the bootstrap interval before "
            "concluding anything."
        )
    return result, after


def _lift(
    labels: npt.NDArray[np.float64],
    baseline: tuple[str, PredictionArray],
    candidate: tuple[str, PredictionArray],
    claim: str,
    *,
    resamples: int,
    level: float,
    seed: int,
) -> _Lift:
    comparison = metrics.paired_bootstrap(
        labels,
        baseline[1].astype("float64"),
        candidate[1].astype("float64"),
        resamples=resamples,
        level=level,
        seed=seed,
    )
    relative = (
        comparison.difference / comparison.baseline_aucpr
        if comparison.baseline_aucpr
        else None
    )
    return {
        "baseline": baseline[0],
        "candidate": candidate[0],
        "baseline_test_aucpr": comparison.baseline_aucpr,
        "candidate_test_aucpr": comparison.candidate_aucpr,
        "absolute": comparison.difference,
        "relative": relative,
        "paired_interval": comparison.interval,
        "probability_candidate_better": comparison.probability_candidate_better,
        "claim": claim,
    }


def run_all(
    config: Config,
    partitions: Partitions[pd.DataFrame] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> _Comparison:
    """Train every available arm and report each lift with a paired interval.

    ``exclude`` DROPS NAMED ARMS, and it exists so that skipping one does not
    require moving artifacts on disk. The alternative I suggested first was
    `mv artifacts/stage4/embeddings ...`, which is worse than it looks: the skip
    then depends on a filesystem state nobody records, the run's own output cannot
    say why the arm is missing, and putting the directory back is a step someone
    will forget. A flag is in the command line, in the log, and in the JSON.
    """
    stage1 = _STAGE1_ADAPTER.validate_python(_config_value(config, "stage1"))
    seed = _SEED_ADAPTER.validate_python(_config_value(config, "seed"))
    evaluation = stage1["evaluation"]

    results: dict[str, _Result] = {}
    predictions: dict[str, PredictionArray] = {}

    # ONE PASS PER EMBEDDING FAMILY, and the reason is memory rather than tidiness.
    # The R-GCN block is 128 float32 columns and the TGN block is 200; on 22.5M
    # rows that is 11.5 GB and 18 GB. Holding both in one frame is 29.5 GB and the
    # OOM killer took a lift run on 2026-08-04 the moment the second family
    # existed -- `Killed`, SIGKILL, no traceback to read. No arm reads more than
    # one family, so the frame is loaded once per family group and freed between.
    #
    # A caller that passes `partitions` explicitly (the offline smoke) keeps the
    # single-pass path: it builds a tiny synthetic frame where none of this bites.
    source = _DATA_ADAPTER.validate_python(_config_value(config, "data"))
    if partitions is not None:
        # A caller-supplied frame is already loaded, so what is available is
        # whatever its load REGISTERED -- not what is on disk. Reading the disk
        # here is what broke the offline smoke: it hands in a synthetic frame with
        # no embeddings, and probing found the developer's real artifacts,
        # planning two arms whose columns the frame does not carry.
        from tfgnn.features_meta import (
            EMBEDDING_FEATURES,
            TEMPORAL_EMBEDDING_FEATURES,
        )

        available = frozenset(
            ([dataset.RGCN_FAMILY] if EMBEDDING_FEATURES else [])
            + ([dataset.TGN_FAMILY] if TEMPORAL_EMBEDDING_FEATURES else [])
        )
    else:
        available = dataset.probe_embedding_families(source)

    # "all" means all AVAILABLE arms. Each embedding arm gates on ITS OWN family:
    # gating both on one registry ran raw_plus_temporal_embeddings whenever STAGE
    # 2 had exported, feeding it the R-GCN's vectors and reporting the result as
    # the TGN's. Skipping stays announced -- a silently absent arm reads as "that
    # stage added nothing".
    _RGCN_DIR, _RGCN_CMD = "artifacts/stage2/embeddings", "python -m tfgnn.stage2.run"
    _TGN_DIR = "artifacts/stage4/embeddings"
    _TGN_CMD = "python -m tfgnn.stage4.run --objective link"
    _REQUIRES: dict[str, tuple[str, str, str]] = {
        "raw_plus_embeddings": (dataset.RGCN_FAMILY, _RGCN_DIR, _RGCN_CMD),
        "raw_plus_temporal_embeddings": (dataset.TGN_FAMILY, _TGN_DIR, _TGN_CMD),
        # The replacement arm reads the same R-GCN tables, so it rides in the
        # rgcn group rather than forcing a fourth load.
        "raw_plus_aggregates_embeddings": (dataset.RGCN_FAMILY, _RGCN_DIR, _RGCN_CMD),
        "raw_plus_pair_scores": (dataset.RGCN_PAIR_FAMILY, _RGCN_DIR, _RGCN_CMD),
        "raw_plus_temporal_pair_scores": (dataset.TGN_PAIR_FAMILY, _TGN_DIR, _TGN_CMD),
    }
    planned = [
        variant
        for variant in VARIANTS
        if variant not in exclude
        and (variant not in _REQUIRES or _REQUIRES[variant][0] in available)
    ]
    for variant, (family, directory, command) in _REQUIRES.items():
        if variant in exclude:
            continue
        if family not in available:
            print(
                f"\nNOTE skipping {variant}: no embeddings found in "
                f"{directory}. Run `{command}` and re-run this trainer to add "
                "that arm."
            )
    for variant in sorted(exclude):
        print(f"\nNOTE excluding {variant}: asked for with --exclude.")

    # Group the plan: arms needing no family ride along with the first group.
    # GROUPED BY FAMILY, not one group per arm: several arms read the same tables,
    # and loading the matrix once per ARM would pay the read four or five times.
    # The pair-score families cost three columns, so they ride with their own
    # encoder's group rather than alone.
    by_family: dict[str, list[str]] = {}
    plain = [v for v in planned if v not in _REQUIRES]
    for variant in planned:
        if variant in _REQUIRES:
            by_family.setdefault(_REQUIRES[variant][0], []).append(variant)

    # rgcn and rgcn_pair read one directory; pairing them keeps the load count at
    # one per encoder rather than one per family.
    merged: dict[frozenset[str], list[str]] = {}
    for encoder, pair in (
        (dataset.RGCN_FAMILY, dataset.RGCN_PAIR_FAMILY),
        (dataset.TGN_FAMILY, dataset.TGN_PAIR_FAMILY),
    ):
        variants = by_family.pop(encoder, []) + by_family.pop(pair, [])
        if variants:
            merged[frozenset({encoder, pair})] = variants
    for family, variants in by_family.items():
        merged[frozenset({family})] = variants

    groups: list[tuple[frozenset[str], list[str]]] = list(merged.items())
    if groups:
        groups[0] = (groups[0][0], plain + groups[0][1])
    else:
        groups.append((frozenset(), plain))

    labels: PredictionArray | None = None
    for index, (families, variants) in enumerate(groups):
        if partitions is not None:
            shared = partitions
        else:
            if len(groups) > 1:
                print(
                    f"\n--- load {index + 1}/{len(groups)}: "
                    f"{', '.join(sorted(families)) or 'no embeddings'} "
                    f"for {', '.join(variants)} ---"
                )
            shared = load_partitions(config, families)
            report_memory(f"loaded + split ({','.join(sorted(families)) or 'base'})")
        if labels is None:
            labels = shared.test[LABEL].to_numpy(dtype="float64")
        for variant in variants:
            results[variant], predictions[variant] = run_variant(
                config, variant, shared
            )
            _ = gc.collect()
            report_memory(f"after {variant}")
        if partitions is None:
            del shared
            _ = gc.collect()

    if labels is None:
        raise RuntimeError("no arms were planned, so there is nothing to compare")
    lifts = [
        _lift(
            labels,
            ("raw", predictions["raw"]),
            ("raw_plus_aggregates", predictions["raw_plus_aggregates"]),
            "One-hop entity group-bys over the fact table. NOT graph lift: "
            "these need no traversal past one hop and must be shared byte for "
            "byte with any no-graph arm.",
            resamples=evaluation["bootstrap_resamples"],
            level=evaluation["bootstrap_level"],
            seed=seed,
        ),
        _lift(
            labels,
            ("raw_plus_aggregates", predictions["raw_plus_aggregates"]),
            ("raw_plus_graph", predictions["raw_plus_graph"]),
            "THE GRAPH LIFT. Centrality, community structure and resolved "
            "identity, on top of the same rows and the same group-bys.",
            resamples=evaluation["bootstrap_resamples"],
            level=evaluation["bootstrap_level"],
            seed=seed,
        ),
        _lift(
            labels,
            ("raw", predictions["raw"]),
            ("raw_plus_graph", predictions["raw_plus_graph"]),
            "Everything TigerGraph contributed. Reported for completeness; it "
            "is NOT the graph's contribution, because it includes the "
            "group-bys.",
            resamples=evaluation["bootstrap_resamples"],
            level=evaluation["bootstrap_level"],
            seed=seed,
        ),
    ]

    # THE GNN LIFT, and only when Stage 2 actually ran. Its base is
    # raw_plus_graph, NOT raw: the GNN's credit is what it adds ON TOP OF the
    # hand-specified graph features, and basing it on raw would fold the entire
    # +0.0286 graph lift into the GNN's number.
    # THE TEMPORAL LIFT (Stage 3). Same base as the GNN lift, deliberately:
    # raw_plus_temporal and raw_plus_embeddings are SIBLINGS off
    # raw_plus_graph, which is what makes the four-stage 2x2 legible -- one
    # cell adds time, the other adds a GNN, and they are directly comparable
    # because they are measured from the same point.
    #
    # NOT comparable against raw. The baseline already carries t_month, t_day,
    # t_hour, t_dayofweek and the forward-chaining target encodings, so this
    # is not "what time-awareness adds" -- it is what PER-ROW INTER-ARRIVAL
    # AND VELOCITY add on top of calendar components.
    if "raw_plus_temporal" in predictions:
        lifts.append(
            _lift(
                labels,
                ("raw_plus_graph", predictions["raw_plus_graph"]),
                ("raw_plus_temporal", predictions["raw_plus_temporal"]),
                "THE TEMPORAL LIFT. Per-row backward-looking time features on "
                "top of the same rows, the same group-bys AND the same graph "
                "features. Each window is anchored at the row's own "
                "unix_time, not at a build cutoff, so it carries no signal "
                "about which snapshot served the row.",
                resamples=evaluation["bootstrap_resamples"],
                level=evaluation["bootstrap_level"],
                seed=seed,
            )
        )

    # STAGE 4b. Based on raw_plus_temporal, so it isolates what the TGN adds
    # over the hand-built temporal block it consumes as its message vector --
    # not what temporal and the GNN add together.
    if "raw_plus_temporal_embeddings" in predictions:
        lifts.append(
            _lift(
                labels,
                ("raw_plus_temporal", predictions["raw_plus_temporal"]),
                (
                    "raw_plus_temporal_embeddings",
                    predictions["raw_plus_temporal_embeddings"],
                ),
                "THE TEMPORAL GNN LIFT (Stage 4b). TGN memory on top of the "
                "same rows and the same per-row temporal features the model "
                "was fed. Stage 4a's supervised head is NOT here: it trains "
                "on the label and is not comparable to arms that do not.",
                resamples=evaluation["bootstrap_resamples"],
                level=evaluation["bootstrap_level"],
                seed=seed,
            )
        )

    if "raw_plus_embeddings" in predictions:
        lifts.append(
            _lift(
                labels,
                ("raw_plus_graph", predictions["raw_plus_graph"]),
                ("raw_plus_embeddings", predictions["raw_plus_embeddings"]),
                "THE GNN LIFT. R-GCN embeddings on top of the same rows, the "
                "same group-bys AND the same graph features. This is what "
                "Stage 2 adds; comparing against raw instead would claim the "
                "graph features' contribution for the GNN.",
                resamples=evaluation["bootstrap_resamples"],
                level=evaluation["bootstrap_level"],
                seed=seed,
            )
        )

    # ---- the three arms added 2026-08-04, each answering a question the
    # ---- original six could not.
    def _optional(base: str, candidate: str, claim: str) -> None:
        if base in predictions and candidate in predictions:
            lifts.append(
                _lift(
                    labels,
                    (base, predictions[base]),
                    (candidate, predictions[candidate]),
                    claim,
                    resamples=evaluation["bootstrap_resamples"],
                    level=evaluation["bootstrap_level"],
                    seed=seed,
                )
            )

    # REPLACEMENT, not addition. Every other embedding arm carries the hand-built
    # graph block alongside the embeddings, so none of them can answer "could the
    # GNN stand in for PageRank and entity resolution". This one drops the graph
    # block and keeps the embeddings, so a positive lift means the GNN learned
    # what was hand-built, and a negative one means it did not.
    _optional(
        "raw_plus_graph",
        "raw_plus_aggregates_embeddings",
        "GNN INSTEAD OF the graph features, not on top of them. Positive means "
        "the R-GCN learned what PageRank, entity resolution and pair geography "
        "were hand-built to provide; negative means hand-building won.",
    )

    # PRESENTATION. Same encoder, same base, three pair statistics instead of the
    # raw dimensions. Against raw_plus_embeddings this isolates dilution -- a tree
    # cannot form the dot product the decoder was trained to produce, so the raw
    # dimensions may carry information it structurally cannot reach.
    _optional(
        "raw_plus_graph",
        "raw_plus_pair_scores",
        "R-GCN presented as three PAIR statistics (dot, cosine, L2) rather than "
        "128 raw dimensions. Compare with the R-GCN embedding lift off the same "
        "base: a gap between them is dilution, not information.",
    )
    _optional(
        "raw_plus_embeddings",
        "raw_plus_pair_scores",
        "Pair score versus raw dimensions, same encoder, same base. Positive "
        "means the 128 dimensions were diluting a signal three columns deliver.",
    )
    _optional(
        "raw_plus_temporal",
        "raw_plus_temporal_pair_scores",
        "TGN presented as three PAIR statistics rather than 200 raw dimensions, "
        "on the temporal base. The Stage 4b question with the dilution removed.",
    )
    _optional(
        "raw_plus_temporal_embeddings",
        "raw_plus_temporal_pair_scores",
        "Pair score versus raw dimensions for the TGN. Same reading as the R-GCN pair.",
    )

    comparison: _Comparison = {
        "results": results,
        "lifts": lifts,
        "notes": [
            f"{len(AGGREGATE_FEATURES)} aggregate columns and "
            f"{len(GRAPH_FEATURES)} graph columns are DEFINED in "
            "src/tfgnn/features_meta.py; each arm's n_features is what "
            "survived the train-fitted drop of constant and all-NaN columns, "
            "and dropped_columns names the rest.",
            "Every arm was trained and scored on identical rows, split by "
            "split_id from TigerGraph.",
            "Metrics come from the reloaded saved JSON, not the in-memory booster.",
            "The threshold behind F1 and the confusion matrix was chosen on "
            "validation and applied unchanged to test.",
            "Intervals are paired percentile bootstraps of the difference: both "
            "arms are resampled with the same row weights on each draw.",
            "This dataset is not TabFormer, so NVIDIA's 0.79 and 0.90 are not "
            "the reference numbers. The comparison here is within-dataset.",
        ],
    }

    output = _output_dir(stage1["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _ = (output / "stage1_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n=== test AUCPR ===")
    for variant, result in results.items():
        interval = result["test_report"]["aucpr_interval"]
        band = (
            f"  95% [{interval['lower']:.6f}, {interval['upper']:.6f}]"
            if interval is not None
            else ""
        )
        print(
            f"  {variant:<22s} {result['aucpr']['test']:.6f}"
            f"  ({result['n_features']} features){band}"
        )

    print("\n=== lifts (paired bootstrap on the difference) ===")
    for entry in lifts:
        interval = entry["paired_interval"]
        band = (
            f"  95% [{interval['lower']:+.6f}, {interval['upper']:+.6f}]"
            if interval is not None
            else ""
        )
        probability = entry["probability_candidate_better"]
        chance = f"  P(better)={probability:.3f}" if probability is not None else ""
        crosses = interval is not None and interval["lower"] <= 0.0 <= interval["upper"]
        print(
            f"  {entry['baseline']} -> {entry['candidate']}: "
            f"{entry['absolute']:+.6f}{band}{chance}"
        )
        if crosses:
            print(
                "    interval includes zero: this difference is not "
                "distinguishable from noise at this prevalence"
            )
    print(f"\nwrote {output / 'stage1_comparison.json'}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: XGBoost over the assembled TigerGraph matrix"
    )
    _ = parser.add_argument("--variant", choices=_CLI_VARIANTS, default="all")
    _ = parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        choices=[v for v in _CLI_VARIANTS if v != "all"],
        help="drop an arm from --variant all. Repeatable. Recorded in the run's "
        "output, unlike moving its artifacts out of the way.",
    )
    args = parser.parse_args()

    dataset.watch_memory()
    config = load_raw_config()

    if args.variant == "all":
        # NO PRE-LOAD. `run_all` loads ONCE PER EMBEDDING FAMILY, and handing it a
        # frame here defeated that completely: the `partitions is not None` branch
        # exists for the offline smoke, and passing a real load through it put
        # both families in one frame -- exactly the 29.5 GiB the grouping was
        # added to avoid.
        #
        # That is what killed the 2026-08-04 lift runs, and it is why the trace
        # ended just after `joined 128 Stage 2 R-GCN embedding columns`: the very
        # next thing was the 200-column TGN join on top of it, whose transient
        # took the process past the machine. The marker for it never printed
        # because SIGKILL arrived first, which made the death look like it
        # happened in code that allocates nothing.
        _ = run_all(config, exclude=frozenset(cast("list[str]", args.exclude)))
    else:
        # One variant needs at most one family; loading the other is pure cost.
        needs = {
            "raw_plus_embeddings": frozenset({dataset.RGCN_FAMILY}),
            "raw_plus_temporal_embeddings": frozenset({dataset.TGN_FAMILY}),
        }
        partitions = load_partitions(config, needs.get(str(args.variant), frozenset()))
        _, _ = run_variant(config, args.variant, partitions)


if __name__ == "__main__":
    main()
