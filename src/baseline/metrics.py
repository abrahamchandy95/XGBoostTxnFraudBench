from dataclasses import dataclass, field
from typing import SupportsFloat, TypedDict, cast

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score


type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


class Interval(TypedDict):
    lower: float
    upper: float
    level: float
    resamples: int


class ConfusionMatrix(TypedDict):
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


class ThresholdMetrics(TypedDict):
    threshold: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: ConfusionMatrix


class Report(TypedDict):
    rows: int
    positives: int
    prevalence: float
    aucpr: float
    aucpr_interval: Interval | None
    roc_auc: float
    at_threshold: ThresholdMetrics


@dataclass(frozen=True)
class _Grouped:
    """Predictions sorted descending, with tied scores collapsed into groups."""

    labels: FloatArray
    group_starts: IntArray
    positives: float

    @property
    def size(self) -> int:
        return int(self.labels.size)


def _group(labels: FloatArray, scores: FloatArray) -> _Grouped:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype("float64", copy=False)

    changed = np.ones(sorted_scores.size, dtype=bool)
    if sorted_scores.size > 1:
        changed[1:] = sorted_scores[1:] != sorted_scores[:-1]
    starts = np.flatnonzero(changed).astype("int64")
    return _Grouped(
        labels=sorted_labels,
        group_starts=starts,
        positives=float(sorted_labels.sum()),
    )


def _average_precision(grouped: _Grouped, weights: FloatArray | None = None) -> float:
    """Weighted average precision over pre-grouped, pre-sorted predictions."""
    labels = grouped.labels
    if weights is None:
        positive = labels
        total = np.ones_like(labels)
    else:
        positive = labels * weights
        total = weights

    tp = np.add.reduceat(positive, grouped.group_starts).cumsum()
    seen = np.add.reduceat(total, grouped.group_starts).cumsum()

    all_positives = tp[-1]
    if all_positives <= 0.0:
        return float("nan")

    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    recall = tp / all_positives
    gained = np.diff(recall, prepend=0.0)
    return float((gained * precision).sum())


def aucpr(labels: FloatArray, scores: FloatArray) -> float:
    return _average_precision(_group(labels, scores))


def verify_against_sklearn(
    labels: FloatArray, scores: FloatArray, tolerance: float = 1.0e-9
) -> float:
    """Confirm the fast path reproduces scikit-learn, then return the value."""
    fast = aucpr(labels, scores)
    reference = float(cast(SupportsFloat, average_precision_score(labels, scores)))
    if not np.isnan(fast) and abs(fast - reference) > tolerance:
        raise RuntimeError(
            "the fast average-precision path disagrees with scikit-learn: "
            f"{fast!r} vs {reference!r}. The bootstrap would be measuring a "
            "different quantity than the headline number."
        )
    return reference


def bootstrap_aucpr(
    labels: FloatArray,
    scores: FloatArray,
    *,
    resamples: int,
    level: float,
    seed: int,
) -> Interval | None:
    """Percentile bootstrap interval for AUCPR.

    Resampling with replacement is equivalent to reweighting by multinomial
    counts, so the sort and the tie grouping are done once for all resamples.
    """
    if resamples <= 0:
        return None
    grouped = _group(labels, scores)
    if grouped.positives <= 1.0:
        return None

    generator = np.random.default_rng(seed)
    size = grouped.size
    draws = np.empty(resamples, dtype="float64")
    for index in range(resamples):
        counts = generator.multinomial(size, np.full(size, 1.0 / size))
        draws[index] = _average_precision(grouped, counts.astype("float64"))

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return None
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "level": float(level),
        "resamples": int(finite.size),
    }


def best_f1_threshold(labels: FloatArray, scores: FloatArray) -> float:
    """The threshold maximising F1, computed on one partition only.

    Call this on VALIDATION and apply the result to test. Choosing it on test
    is selection contamination: it inflates the estimate, not the model.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype("float64", copy=False)

    tp = np.cumsum(sorted_labels)
    predicted = np.arange(1, sorted_labels.size + 1, dtype="float64")
    all_positives = tp[-1]
    if all_positives <= 0.0:
        return 0.5

    precision = tp / predicted
    recall = tp / all_positives
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    return float(sorted_scores[int(np.argmax(f1))])


def at_threshold(
    labels: FloatArray, scores: FloatArray, threshold: float
) -> ThresholdMetrics:
    predicted = scores >= threshold
    actual = labels > 0.5

    true_positives = int(np.count_nonzero(predicted & actual))
    false_positives = int(np.count_nonzero(predicted & ~actual))
    false_negatives = int(np.count_nonzero(~predicted & actual))
    true_negatives = int(np.count_nonzero(~predicted & ~actual))

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
        },
    }


def report(
    labels: FloatArray,
    scores: FloatArray,
    *,
    threshold: float,
    bootstrap_resamples: int = 0,
    bootstrap_level: float = 0.95,
    seed: int = 42,
) -> Report:
    positives = int(np.count_nonzero(labels > 0.5))
    return {
        "rows": int(labels.size),
        "positives": positives,
        "prevalence": positives / labels.size if labels.size else 0.0,
        "aucpr": verify_against_sklearn(labels, scores),
        "aucpr_interval": bootstrap_aucpr(
            labels,
            scores,
            resamples=bootstrap_resamples,
            level=bootstrap_level,
            seed=seed,
        ),
        "roc_auc": float(cast(SupportsFloat, roc_auc_score(labels, scores))),
        "at_threshold": at_threshold(labels, scores, threshold),
    }


@dataclass
class PairedComparison:
    """Paired bootstrap of the AUCPR difference between two arms.

    Both arms are resampled with the SAME row weights on each draw. Comparing
    two independent intervals answers a different and weaker question: two
    overlapping intervals are entirely compatible with a difference that is
    consistently positive on every resample.
    """

    baseline_aucpr: float
    candidate_aucpr: float
    difference: float
    interval: Interval | None
    probability_candidate_better: float | None
    resamples: int = field(default=0)


def paired_bootstrap(
    labels: FloatArray,
    baseline_scores: FloatArray,
    candidate_scores: FloatArray,
    *,
    resamples: int,
    level: float,
    seed: int,
) -> PairedComparison:
    baseline = _group(labels, baseline_scores)
    candidate = _group(labels, candidate_scores)

    baseline_order = np.argsort(-baseline_scores, kind="stable")
    candidate_order = np.argsort(-candidate_scores, kind="stable")

    point_baseline = _average_precision(baseline)
    point_candidate = _average_precision(candidate)
    difference = point_candidate - point_baseline

    if resamples <= 0:
        return PairedComparison(
            baseline_aucpr=point_baseline,
            candidate_aucpr=point_candidate,
            difference=difference,
            interval=None,
            probability_candidate_better=None,
        )

    generator = np.random.default_rng(seed)
    size = int(labels.size)
    uniform = np.full(size, 1.0 / size)
    draws = np.empty(resamples, dtype="float64")

    for index in range(resamples):
        counts = generator.multinomial(size, uniform).astype("float64")
        draws[index] = _average_precision(
            candidate, counts[candidate_order]
        ) - _average_precision(baseline, counts[baseline_order])

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return PairedComparison(
            baseline_aucpr=point_baseline,
            candidate_aucpr=point_candidate,
            difference=difference,
            interval=None,
            probability_candidate_better=None,
        )

    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return PairedComparison(
        baseline_aucpr=point_baseline,
        candidate_aucpr=point_candidate,
        difference=difference,
        interval={
            "lower": float(lower),
            "upper": float(upper),
            "level": float(level),
            "resamples": int(finite.size),
        },
        probability_candidate_better=float(np.mean(finite > 0.0)),
        resamples=int(finite.size),
    )
