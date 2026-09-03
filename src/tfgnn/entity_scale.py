from dataclasses import dataclass

import numpy as np
import pandas as pd

from tfgnn.features_meta import (
    AGGREGATE_FEATURES,
    GRAPH_FEATURES,
    PAIR_FEATURES,
    SENTINEL_AT_MINUS_ONE,
    SENTINEL_GROUPS,
)

ENTITY_FEATURE_COLUMNS: frozenset[str] = (
    frozenset(AGGREGATE_FEATURES + GRAPH_FEATURES) - PAIR_FEATURES
)

_MINUS_ONE: frozenset[str] = frozenset(SENTINEL_AT_MINUS_ONE)


@dataclass(frozen=True)
class ColumnReport:
    column: str
    distinct_before: int
    distinct_after: int
    missing: int
    rows: int

    @property
    def fingerprints(self) -> bool:
        """True when this column alone could still key the table.

        The check that caught the problem in the first place: 14,363
        distinct values over 14,363 cards is a primary key.
        """
        return self.rows > 0 and self.distinct_after >= self.rows


def _sentinel_to_nan(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The -1 -> NaN conversion, on one dimension table, before ranking.

    Mirrors ``assemble.apply_sentinels`` and must stay consistent with it.
    Two rules because they are not equally safe: counts and scores may be
    tested directly, amounts may not -- an amount of -1.00 is a legal
    refund, so each amount block is gated on the sibling count its
    producing query wrote in the same ``IF count > 0`` branch.
    """
    present = set(columns)

    for column in columns:
        if column in _MINUS_ONE:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.mask(values <= -1)

    for guard, dependents in SENTINEL_GROUPS:
        if guard not in present:
            continue
        unset = frame[guard].isna()
        for column in dependents:
            if column in present:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").mask(
                    unset
                )
    return frame


def rank_bin_table(
    frame: pd.DataFrame,
    bins: int,
    *,
    table: str,
) -> tuple[pd.DataFrame, list[ColumnReport]]:
    """Rank-and-bin every per-entity column of one build's dimension table.

    ``frame`` must already carry the FINAL prefixed feature names, so the
    sentinel rules in features_meta apply verbatim.
    """
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")

    columns = [name for name in frame.columns if name in ENTITY_FEATURE_COLUMNS]
    if not columns:
        return frame, []

    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = _sentinel_to_nan(out, columns)

    rows = len(out)
    reports: list[ColumnReport] = []
    for column in columns:
        values = out[column]
        before = int(values.nunique(dropna=True))

        ranked = values.rank(method="average", pct=True)
        binned = np.floor(ranked.to_numpy(dtype="float64") * bins)
        # rank == 1.0 lands on `bins`; clip it back into the top bin.
        binned = np.clip(binned, 0.0, float(bins - 1))
        binned[np.isnan(ranked.to_numpy(dtype="float64"))] = np.nan

        out[column] = pd.Series(binned, index=out.index, dtype="float64")
        reports.append(
            ColumnReport(
                column=column,
                distinct_before=before,
                distinct_after=int(out[column].nunique(dropna=True)),
                missing=int(values.isna().sum()),
                rows=rows,
            )
        )

    _ = table
    return out, reports


def describe(table: str, reports: list[ColumnReport]) -> list[str]:
    """Human-readable proof that the fingerprint is gone."""
    if not reports:
        return []
    lines = [f"  {table}: {reports[0].rows:,} entities"]
    for item in reports:
        flag = "  <-- STILL KEYS THE TABLE" if item.fingerprints else ""
        lines.append(
            f"    {item.column:<34s} {item.distinct_before:>7,} -> "
            f"{item.distinct_after:>4,} distinct"
            f"  ({item.missing:,} missing){flag}"
        )
    return lines
