"""Join the exported star schema into one training matrix.

THE JOIN
--------
The fact table carries three flat keys and the dimension tables carry two
more that only exist after a build::

    transaction.card_number  -> card_features.card_number
    transaction.merchant_id  -> merchant_features.id
    transaction.mer_cat      -> category_features.category
    card_features.c_id       -> community_features.cid    (entity_family card)
    merchant_features.c_id   -> community_features.cid    (entity_family merchant)
    card_features.re_id      -> resolved_entity_features.reid
    merchant_features.re_id  -> resolved_entity_features.reid

Community and Resolved_Entity are each reached twice, once through the card
and once through the merchant, so the two copies are prefixed ``ccom_`` /
``mcom_`` and ``cre_`` / ``mre_``. ``entity_family`` disambiguates the
community joins: a card-family and a merchant-family community are different
populations that share a key space.

WHICH BUILD SERVES WHICH ROW
----------------------------
Each build directory carries the ``serves_splits`` it was computed for, so a
transaction's features come from the snapshot assigned to its ``split_id`` and
from no other. That mapping is the whole point of the multi-build run: a test
row must read features fitted from train and val only.

``build_id`` is verified rather than assumed. ``export_card_features``' header
is explicit about why: ``The assembler asserts that the build_id on a feature
row equals the build assigned to the transaction consuming it. That assertion
is the only thing standing between you and assembling every row's features
from whichever build ran last.``

DO NOT LEFT-JOIN AND fillna(0)
------------------------------
Absence is information. ``export_card_features`` returns ``seen == TRUE`` rows
only, so a card missing from the table had no history under this build's
cutoff and every number for it is unknown -- not zero. Zero-filling turns
"unseen" into a value, and unseen correlates with "appears later in time", so
the model learns a calendar proxy. Missing stays NaN; XGBoost learns a default
direction per split, which is the honest handling.

MEMORY
------
The dimension tables are small (thousands of cards, hundreds of merchants) and
are held whole. The fact table is not, so assembly streams one fact part at a
time and writes one output part per input part. Feature columns are cast to
float32, which is what XGBoost uses internally anyway.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from common.config import load_config
from tfgnn.features_meta import (
    ALL_FEATURE_COLUMNS,
    ASSEMBLED_COLUMNS,
    CARD_COMMUNITY_PREFIX,
    CARD_ENTITY_PREFIX,
    CARD_PREFIX,
    CATEGORY_PREFIX,
    FACT_CATEGORICAL,
    LABEL,
    MERCHANT_COMMUNITY_PREFIX,
    MERCHANT_ENTITY_PREFIX,
    MERCHANT_PREFIX,
    SENTINEL_AT_MINUS_ONE,
    SENTINEL_GROUPS,
)
from tfgnn.entity_scale import describe, rank_bin_table
from tfgnn.tigergraph.export import resolve


class AssemblePlan(TypedDict):
    export_dir: str
    matrix_dir: str
    overwrite: bool
    # 0 disables and reproduces the raw-value matrix. See tfgnn.entity_scale:
    # a per-entity feature at full float precision is a primary key for its
    # dimension (avg_txn_amount alone keys all 14,363 cards), and the
    # snapshots disagree on scale. Ranking fixes the second, binning the
    # first, and NEITHER HALF IS OPTIONAL.
    entity_rank_bins: int


ASSEMBLE_ADAPTER = TypeAdapter(AssemblePlan)


@dataclass(frozen=True)
class BuildExport:
    build_id: str
    cutoff_event_seq: int
    cutoff_source: str
    serves_splits: tuple[int, ...]
    directory: Path
    # Forward chaining: the event_seq window inside those splits that this
    # snapshot serves. None on both means the whole split.
    serves_from: int | None = None
    serves_until: int | None = None

    @property
    def is_windowed(self) -> bool:
        return self.serves_from is not None or self.serves_until is not None

    def row_mask(self, split_ids: pd.Series, event_seq: pd.Series) -> pd.Series:
        mask = split_ids.isin(list(self.serves_splits))
        if self.serves_from is not None:
            mask &= event_seq >= self.serves_from
        if self.serves_until is not None:
            mask &= event_seq < self.serves_until
        return mask


def discover_builds(export_root: Path) -> list[BuildExport]:
    """Read the per-build manifests the pipeline wrote next to each export."""
    root = export_root / "dimensions"
    if not root.is_dir():
        raise FileNotFoundError(
            f"no dimension exports below {root}. Run "
            "`python -m tfgnn.tigergraph.pipeline` first; the dimension tables "
            "are written inside each build, because the next build's "
            "reset_build destroys them."
        )
    builds: list[BuildExport] = []
    for directory in sorted(root.iterdir()):
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        payload = cast(dict[str, Any], json.loads(manifest.read_text(encoding="utf-8")))
        window_from = payload.get("serves_from")
        window_until = payload.get("serves_until")
        builds.append(
            BuildExport(
                build_id=str(payload["build_id"]),
                cutoff_event_seq=int(payload["cutoff_event_seq"]),
                cutoff_source=str(payload.get("cutoff_source", "")),
                serves_splits=tuple(int(s) for s in payload["serves_splits"]),
                directory=directory,
                serves_from=None if window_from is None else int(window_from),
                serves_until=None if window_until is None else int(window_until),
            )
        )
    if not builds:
        raise FileNotFoundError(f"no build manifests below {root}")

    _assert_disjoint_coverage(builds)
    return builds


def _assert_disjoint_coverage(builds: Sequence[BuildExport]) -> None:
    """No row may be served by two exported snapshots.

    Two builds may share a split ONLY if both carry disjoint event_seq
    windows -- that is forward chaining. Anything else means a row would be
    joined twice and appear twice in the matrix with two different feature
    vectors, which nothing downstream would catch: the row counts would simply
    be larger and every per-split metric would be computed over duplicates.
    """
    for split_id in sorted({s for item in builds for s in item.serves_splits}):
        claimants = [item for item in builds if split_id in item.serves_splits]
        if len(claimants) < 2:
            continue

        whole = [item for item in claimants if not item.is_windowed]
        if whole:
            names = ", ".join(item.build_id for item in claimants)
            raise RuntimeError(
                f"split {split_id} is served by {len(claimants)} exported "
                f"builds ({names}) and {whole[0].build_id} claims the whole "
                "split. Each row must read exactly one snapshot."
            )

        ordered = sorted(claimants, key=lambda b: b.serves_from or 0)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if earlier.serves_until is None or later.serves_from is None:
                raise RuntimeError(
                    f"{earlier.build_id} and {later.build_id} both serve split "
                    f"{split_id} but one has an open-ended window"
                )
            if earlier.serves_until > later.serves_from:
                raise RuntimeError(
                    f"{earlier.build_id} and {later.build_id} serve overlapping "
                    f"event_seq ranges on split {split_id}: "
                    f"[{earlier.serves_from}, {earlier.serves_until}) and "
                    f"[{later.serves_from}, {later.serves_until})"
                )


@dataclass
class Dimensions:
    card: pd.DataFrame
    merchant: pd.DataFrame
    category: pd.DataFrame
    card_community: pd.DataFrame
    merchant_community: pd.DataFrame
    resolved_entity: pd.DataFrame
    # Keyed on the (card, merchant) PAIR, so not a dimension in the star-schema
    # sense. Carries the one column the entity dimensions can no longer supply.
    pair: pd.DataFrame


def _key_as_string(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    frame[column] = frame[column].astype("string").fillna("")
    return frame


def _key_as_int(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Nullable Int64, with the -1 sentinel turned into NA so it cannot join."""
    values = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame[column] = values.mask(values < 0)
    return frame


def _rename(frame: pd.DataFrame, prefix: str, skip: Sequence[str]) -> pd.DataFrame:
    keep = set(skip)
    return frame.rename(
        columns={name: f"{prefix}{name}" for name in frame.columns if name not in keep}
    )


def _rank_bin_dimensions(dims: Dimensions, bins: int, build_id: str) -> Dimensions:
    """Rank-and-bin every dimension's per-entity columns, within this build.

    See tfgnn.entity_scale for why: cross-snapshot scale drift and entity
    fingerprinting, one transform. Applied HERE rather than after the join
    because the quantiles must be computed over ENTITIES, one row each, not
    over fact rows weighted by transaction volume.

    resolved_entity is the awkward one: join_part renames that single table
    twice, into cre_* and mre_*, so it arrives here unprefixed. It is
    temporarily prefixed cre_ to make the features_meta sentinel rules
    apply, then renamed back. cre_ and mre_ carry identical rules -- both
    are SENTINEL_AT_MINUS_ONE only, neither appears in SENTINEL_GROUPS --
    so the choice of prefix does not change the result.
    """
    print(f"  [{build_id}] rank+bin per-entity features into {bins} bins")
    reports: list[str] = []

    def go(frame: pd.DataFrame, table: str) -> pd.DataFrame:
        out, items = rank_bin_table(frame, bins, table=table)
        reports.extend(describe(table, items))
        return out

    entity = dims.resolved_entity
    entity_features = [name for name in entity.columns if name != "reid"]
    prefixed = entity.rename(
        columns={name: f"{CARD_ENTITY_PREFIX}{name}" for name in entity_features}
    )
    prefixed = go(prefixed, "resolved_entity")
    entity = prefixed.rename(
        columns={f"{CARD_ENTITY_PREFIX}{name}": name for name in entity_features}
    )

    result = Dimensions(
        card=go(dims.card, "card"),
        merchant=go(dims.merchant, "merchant"),
        category=go(dims.category, "category"),
        card_community=go(dims.card_community, "card_community"),
        merchant_community=go(dims.merchant_community, "merchant_community"),
        resolved_entity=entity,
        # NOT rank-binned, and passed straight through. pair_home_distance_km is
        # a property of the PAIR: it cannot key either dimension, so neither half
        # of the rank+bin transform applies to it. entity_scale.PAIR_FEATURES
        # records the same exclusion for the assembled column.
        pair=dims.pair,
    )
    for line in reports:
        print(line)
    return result


def load_dimensions(build: BuildExport, entity_rank_bins: int = 0) -> Dimensions:
    def read(name: str) -> pd.DataFrame:
        path = build.directory / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing dimension export: {path}")
        return pd.read_parquet(path)

    card = read("export_card_features")
    merchant = read("export_merchant_features")
    category = read("export_category_features")
    community = read("export_community_features")
    entity = read("export_resolved_entity_features")
    pair = read("export_pair_features")

    for frame, column in (
        (card, "build_id"),
        (merchant, "build_id"),
        (category, "build_id"),
        (community, "build_id"),
        (entity, "build_id"),
    ):
        if len(frame) == 0:
            continue
        stamped = set(frame[column].astype("string").dropna().unique().tolist())
        if stamped - {build.build_id}:
            raise RuntimeError(
                f"{build.directory.name} contains feature rows stamped "
                f"{sorted(stamped)} but the build is {build.build_id!r}. That "
                "is cross-build contamination: some rows' features came from a "
                "different snapshot than the one assigned to them."
            )

    card = _key_as_string(card, "card_number")
    card = _key_as_int(card, "c_id")
    card = _key_as_int(card, "re_id")
    merchant = _key_as_string(merchant, "id")
    merchant = _key_as_int(merchant, "c_id")
    merchant = _key_as_int(merchant, "re_id")
    category = _key_as_string(category, "category")
    community = _key_as_int(community, "cid")
    entity = _key_as_int(entity, "reid")
    # Both halves of the composite key, cast the same way the fact table's are,
    # or the merge silently matches nothing and every distance becomes NaN.
    pair = _key_as_string(pair, "card_number")
    pair = _key_as_string(pair, "merchant_id")

    # entity_family separates the populations that share a key space.
    # "bipartite" components (wcc_bipartite) genuinely contain both types, so
    # the same community row serves as the card-side AND merchant-side
    # dimension, reached through either endpoint's c_id. For most rows
    # ccom_* and mcom_* then carry the same numbers -- correlated on purpose,
    # not duplicated by accident; the trainer's constant/importance handling
    # is the consumer that copes.
    family = community["entity_family"].astype("string")
    card_community = community.loc[family.isin(["card", "bipartite"])].drop(
        columns=["entity_family", "build_id"]
    )
    merchant_community = community.loc[family.isin(["merchant", "bipartite"])].drop(
        columns=["entity_family", "build_id"]
    )
    # "MIXED" is the projection-era stop-the-world label. community_stats no
    # longer emits it (both-sides communities are "bipartite" now), so any
    # occurrence means a stale export from before the bipartite rewrite.
    mixed = int((family == "MIXED").sum())
    if mixed:
        raise RuntimeError(
            f"{mixed} communities are MIXED, a label community_stats no longer "
            "writes. This export predates the bipartite feature path; re-run "
            "the feature pipeline before assembling."
        )

    # Drop the join keys the model must never see, and the per-table build_id
    # now that it has been verified.
    card = card.drop(columns=["build_id", "louvain_id"])
    merchant = merchant.drop(columns=["build_id", "louvain_id"])
    category = category.drop(columns=["build_id"])
    entity = entity.drop(columns=["build_id"])

    dims = Dimensions(
        card=_rename(card, CARD_PREFIX, ["card_number"]),
        merchant=_rename(merchant, MERCHANT_PREFIX, ["id"]),
        category=_rename(category, CATEGORY_PREFIX, ["category"]),
        card_community=_rename(card_community, CARD_COMMUNITY_PREFIX, ["cid"]),
        merchant_community=_rename(
            merchant_community, MERCHANT_COMMUNITY_PREFIX, ["cid"]
        ),
        resolved_entity=entity,
        pair=pair,
    )
    if entity_rank_bins:
        dims = _rank_bin_dimensions(dims, entity_rank_bins, build.build_id)
    return dims


def _join_pair_distance(frame: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    """Attach ``pair_home_distance_km`` from the exported pair table.

    THIS USED TO BE A HAVERSINE COMPUTED HERE, and that was the bug. Four raw
    coordinate columns rode in on the card and merchant dimensions and this
    function combined them, while ``card_home_distance`` computed the same
    quantity in GSQL for Stage 2. The old docstring named the hazard itself:
    "They must agree; if they ever disagree, one of the two traversals has
    picked a different address."

    Relocation made them disagree. The pandas copy read the card dimension's
    ``home_lat``/``home_lon``, which came from ``Street_Address`` -- collapsed by
    the loader to the party's LATEST tenure -- so every pre-move transaction was
    measured against a home the cardholder had not moved to yet. The GSQL copy
    picks the tenure live at the build cutoff. Two answers, and Stage 1 read the
    wrong one.

    So there is one traversal now. ``card_home_distance`` owns the arithmetic
    because it is the only place that has the cutoff, and this is a join.

    LEFT JOIN, NEVER INNER. A pair with no geography must keep its transaction
    row and lose only the distance: about a fifth of pairs have an endpoint with
    no coordinates, and an inner join would delete those rows from the matrix
    entirely, which would look like a smaller dataset rather than a missing
    feature.

    THE -1 SENTINEL, not -999. The exported column is
    ``Has_Interaction_With_Merchant.home_distance_km``, which schema r3 declares
    ``DOUBLE DEFAULT -1``, so it follows the house convention rather than the
    -999 the coordinate columns needed (a latitude of -1 is a real place; a
    distance of -1 is not).
    """
    merged = frame.merge(
        pair,
        how="left",
        on=["card_number", "merchant_id"],
    )
    if len(merged) != len(frame):
        raise RuntimeError(
            f"the pair join changed the row count: {len(frame):,} -> "
            f"{len(merged):,}. export_pair_features returned more than one row "
            "per (card, merchant), so transactions were multiplied rather than "
            "annotated."
        )
    distance = pd.to_numeric(merged["home_distance_km"], errors="coerce")
    merged["pair_home_distance_km"] = distance.where(distance >= 0)
    return merged.drop(columns=["home_distance_km"])


#: Which splits get measured against the training split, and what to call them.
_EVALUATED_SPLITS = ((1, "val"), (2, "test"))


def inductive_reach(matrix_root: Path) -> dict[str, object]:
    """How much of val and test is an entity the model never trained on.

    DECISIONS_LEAK_CONTROL D3d fixes the target setting as INDUCTIVE: "No
    learnable per-node ID embeddings. Every node's representation must be a
    function of its features and neighborhood, so an unseen 2019 card is
    scored by the same function as a 2016 one."

    The code satisfies that. The DATA has never tested it: measured on the
    2026-07-30 load, every card and merchant at test time also appears in
    training, so the inductive path is satisfied by construction and never
    exercised. An inductive claim on that data is an assumption, not a result.
    This turns it into a number that either backs the claim or does not.

    MEASURED OVER THE MATRIX, NOT THE FACT TABLE, and the difference matters:
    under forward chaining the warm-up fold is dropped during assembly. Those
    rows contribute topology and are never scored, so a card appearing only in
    warm-up is one the model never saw a labelled example of. The matrix is
    exactly the set of rows the model reads, which makes it the honest
    population for "seen in training".

    NOT A GATE, and that is deliberate. Zero unseen entities is not a failure --
    on the current data it is the correct answer and the run should proceed.
    Nor is a large share a failure: churn is the realistic setting and a high
    number is the point of regenerating. It is a REPORTED number, because what
    it qualifies is the WORDING of the inductive claim, not the validity of the
    build.

    Resolved entities are deliberately absent: ``join_part`` drops
    ``card_re_id`` / ``mer_re_id`` after the dimension join, so re_id is not in
    the matrix. Card and merchant are the node types D3d's ID-embedding
    prohibition is about, which is what the requirement turns on.
    """
    columns = ["split_id", "card_number", "merchant_id"]
    parts = sorted(matrix_root.glob("part-*.parquet"))

    # Pass 1: the distinct entities per split. Thousands of cards and hundreds
    # of merchants, so these sets are small however long the matrix is.
    entities: dict[int, dict[str, set[object]]] = {}
    for part in parts:
        frame = pd.read_parquet(part, columns=columns)
        for split_id, block in frame.groupby("split_id", sort=False):
            bucket = entities.setdefault(
                int(cast(int, split_id)), {"card": set(), "merchant": set()}
            )
            bucket["card"].update(block["card_number"].unique().tolist())
            bucket["merchant"].update(block["merchant_id"].unique().tolist())

    if 0 not in entities:
        # Without a training split there is no "seen in training" to measure
        # against, and every val/test row would score as unseen -- a 1.0 that
        # reads as total churn when it actually means the question was not
        # asked. plan.load_facts already refuses a load missing any split, so
        # this is unreachable from the pipeline; it is here because
        # inductive_reach is callable on any matrix directory.
        raise RuntimeError(
            "the matrix holds no split 0 rows, so 'unseen in training' is "
            "undefined. Every val/test row would report as unseen."
        )
    train = entities[0]

    # Pass 2: the ROW shares. "card or merchant unseen" is a joint over the two
    # columns and cannot be recovered from the two marginals, so it needs the
    # rows -- but only these three columns of them, and only after pass 1 has
    # the complete training set. Streaming twice is the price of not assuming
    # the parts arrive in split order.
    tally: dict[int, dict[str, int]] = {}
    for part in parts:
        frame = pd.read_parquet(part, columns=columns)
        cold_card = ~frame["card_number"].isin(train["card"])
        cold_merchant = ~frame["merchant_id"].isin(train["merchant"])
        block = pd.DataFrame(
            {
                "split_id": frame["split_id"],
                "card": cold_card,
                "merchant": cold_merchant,
                "either": cold_card | cold_merchant,
            }
        )
        for split_id, rows in block.groupby("split_id", sort=False):
            bucket = tally.setdefault(
                int(cast(int, split_id)),
                {"rows": 0, "card": 0, "merchant": 0, "either": 0},
            )
            bucket["rows"] += len(rows)
            for key in ("card", "merchant", "either"):
                bucket[key] += int(rows[key].sum())

    report: dict[str, object] = {
        "train_cards": len(train["card"]),
        "train_merchants": len(train["merchant"]),
    }
    for split_id, label in _EVALUATED_SPLITS:
        counts = tally.get(split_id)
        if counts is None or counts["rows"] == 0:
            continue
        rows = counts["rows"]
        present = entities.get(split_id, {"card": set(), "merchant": set()})
        report[label] = {
            "rows": rows,
            "unseen_card_row_share": round(counts["card"] / rows, 6),
            "unseen_merchant_row_share": round(counts["merchant"] / rows, 6),
            "unseen_either_row_share": round(counts["either"] / rows, 6),
            "cards": len(present["card"]),
            "unseen_cards": len(present["card"] - train["card"]),
            "merchants": len(present["merchant"]),
            "unseen_merchants": len(present["merchant"] - train["merchant"]),
        }
    return report


def join_part(
    fact: pd.DataFrame,
    build: BuildExport,
    dims: Dimensions,
) -> pd.DataFrame:
    """Join one fact slice against one build's dimension tables."""
    frame = fact.copy()
    for column in ("card_number", "merchant_id", "mer_cat"):
        frame = _key_as_string(frame, column)

    frame = frame.merge(dims.card, how="left", on="card_number")
    frame = frame.merge(
        dims.merchant, how="left", left_on="merchant_id", right_on="id"
    ).drop(columns=["id"])
    frame = frame.merge(
        dims.category, how="left", left_on="mer_cat", right_on="category"
    ).drop(columns=["category"])

    frame = frame.merge(
        dims.card_community,
        how="left",
        left_on=f"{CARD_PREFIX}c_id",
        right_on="cid",
    ).drop(columns=["cid"])
    frame = frame.merge(
        dims.merchant_community,
        how="left",
        left_on=f"{MERCHANT_PREFIX}c_id",
        right_on="cid",
    ).drop(columns=["cid"])

    card_entity = _rename(dims.resolved_entity, CARD_ENTITY_PREFIX, ["reid"])
    merchant_entity = _rename(dims.resolved_entity, MERCHANT_ENTITY_PREFIX, ["reid"])
    frame = frame.merge(
        card_entity, how="left", left_on=f"{CARD_PREFIX}re_id", right_on="reid"
    ).drop(columns=["reid"])
    frame = frame.merge(
        merchant_entity,
        how="left",
        left_on=f"{MERCHANT_PREFIX}re_id",
        right_on="reid",
    ).drop(columns=["reid"])

    frame = _join_pair_distance(frame, dims.pair)

    # No coordinate columns to drop any more: the exports stopped emitting them
    # when the distance became a pair-table column. features_meta still lists
    # them in FORBIDDEN_AS_FEATURES so reintroducing one cannot make it a
    # feature by accident.
    frame = frame.drop(
        columns=[
            f"{CARD_PREFIX}c_id",
            f"{CARD_PREFIX}re_id",
            f"{MERCHANT_PREFIX}c_id",
            f"{MERCHANT_PREFIX}re_id",
        ]
    )
    frame["build_id"] = build.build_id
    return frame


def apply_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the -1 "this build did not compute it" sentinels into NaN.

    Two rules, because they are not equally safe. See features_meta.
    """
    for column in SENTINEL_AT_MINUS_ONE:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.mask(values <= -1)

    for guard, dependents in SENTINEL_GROUPS:
        if guard not in frame.columns:
            continue
        # The guard has already been NaN-ed above where it held -1, so a null
        # guard is exactly "this block was never computed".
        unset = frame[guard].isna()
        for column in dependents:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").mask(
                    unset
                )
    return frame


def _tally(counts: dict[int, int], values: pd.Series) -> dict[int, int]:
    """Add one partition's split_id counts into a running tally.

    Goes through numpy rather than value_counts().items(): a pandas index
    label is Hashable, not int, so the loop form needs a cast that hides a
    real question about what the key actually is.
    """
    keys, tallies = np.unique(values.to_numpy(dtype="int64"), return_counts=True)
    merged = dict(counts)
    for key, tally in zip(keys.tolist(), tallies.tolist(), strict=True):
        merged[int(key)] = merged.get(int(key), 0) + int(tally)
    return merged


def _as_flag(series: pd.Series) -> pd.Series:
    """Coerce a BOOL attribute to int8 without silently zeroing it.

    TigerGraph returns BOOL as JSON true/false, which pandas reads as bool or
    object, but a CSV round trip through the export can leave the literal
    strings "true"/"false". ``pd.to_numeric`` maps those to NaN, so a naive
    ``fillna(0)`` would turn every card-not-present transaction into a
    card-present one and lose the column entirely.
    """
    if series.dtype == bool:
        return series.astype("int8")
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1,
            "t": 1,
            "1": 1,
            "yes": 1,
            "false": 0,
            "f": 0,
            "0": 0,
            "no": 0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    combined = mapped.astype("Float64").fillna(numeric.astype("Float64"))
    unresolved = int(combined.isna().sum())
    if unresolved:
        raise RuntimeError(
            f"is_online has {unresolved:,} values that are neither boolean nor "
            "numeric. Fix the export rather than defaulting them, because "
            "defaulting silently rewrites the feature."
        )
    return (combined > 0).astype("int8")


def finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Impose the column contract and the storage dtypes."""
    missing = sorted(set(ASSEMBLED_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"assembled matrix is missing columns: {missing}. Either a "
            "dimension export did not run or features_meta is stale."
        )
    frame = frame.loc[:, ASSEMBLED_COLUMNS]

    for column in FACT_CATEGORICAL:
        frame[column] = frame[column].astype("string").fillna("")
    frame["is_online"] = _as_flag(frame["is_online"])
    frame[LABEL] = pd.to_numeric(frame[LABEL], errors="coerce").astype("int8")
    for column in ("event_seq", "unix_time", "split_id", "causal_fold"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")

    numeric_features = [
        name
        for name in ALL_FEATURE_COLUMNS
        if name not in set(FACT_CATEGORICAL) and name != "is_online"
    ]
    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return frame


def load_temporal(export_root: Path) -> pd.DataFrame | None:
    """Stage 3's per-row temporal table, or None when it was never exported.

    NOT BUILD-SCOPED, which is why it is loaded once here rather than through
    ``load_dimensions``. Each window is anchored at the row's own unix_time,
    so one table serves every snapshot.

    MERGED ON ``event_seq``, NOT ON ``id``. Both sides carry both, but
    ``assert_cardinality`` gates that event_seq is a dense permutation of
    1..n -- unique by construction -- so an int64 join key is available and a
    string one is not worth its memory at this row count. ``id`` is dropped
    on load for the same reason.

    Returns None rather than raising when the directory or its manifest is
    absent: Stage 3 is additive, and a run that predates it must still
    assemble. ``finalise`` is what fails closed if the columns are then
    missing from the contract.
    """
    directory = export_root / "temporal"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return None

    payload = cast(
        "dict[str, object]", json.loads(manifest.read_text(encoding="utf-8"))
    )
    expected = int(cast("int", payload.get("rows", 0)))
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        return None

    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if expected and len(frame) != expected:
        raise RuntimeError(
            f"the temporal export's manifest records {expected:,} rows but "
            f"{len(frame):,} are on disk. A part was added or removed after "
            "the export; re-run `python -m tfgnn.tigergraph.temporal`."
        )
    if frame["event_seq"].duplicated().any():
        raise RuntimeError(
            "the temporal table has duplicate event_seq values, so the join "
            "below would MULTIPLY transaction rows rather than annotate "
            "them. event_seq is supposed to be a dense unique rank -- run "
            "assert_cardinality."
        )
    frame = frame.drop(columns=["id"], errors="ignore")
    frame["event_seq"] = pd.to_numeric(frame["event_seq"]).astype("int64")
    # INDEXED ONCE, HERE, AND THAT IS A PERFORMANCE FIX NOT A STYLE CHOICE.
    # _join_temporal runs once per fact part -- over a hundred times -- and a
    # plain `merge(..., on="event_seq")` rebuilds a hash table over this whole
    # frame on every one of them. At tens of millions of rows that is minutes
    # of hashing repeated per part. Setting the index once and using .join()
    # builds the hashtable a single time and turns each part into a lookup.
    frame = frame.set_index("event_seq")
    print(f"temporal features: {len(frame):,} rows from {len(parts)} part(s)")
    return frame


def _join_temporal(fact: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN, never inner -- same rule as the pair distance.

    A transaction with no temporal row would keep its place in the matrix and
    lose only these columns. In practice that cannot happen (the export
    refuses to write a manifest unless it produced one row per transaction),
    which is exactly why an inner join here would be a silent row-count
    change rather than a visible failure.
    """
    keyed = fact.copy()
    keyed["event_seq"] = pd.to_numeric(keyed["event_seq"]).astype("int64")
    # .join(on=...) against a frame already indexed on event_seq -- see
    # load_temporal. A left join, so a fact row with no temporal match keeps
    # its place and loses only these columns.
    merged = keyed.join(temporal, on="event_seq")
    if len(merged) != len(fact):
        raise RuntimeError(
            f"the temporal join changed the row count: {len(fact):,} -> "
            f"{len(merged):,}."
        )
    return merged


def assemble(plan: AssemblePlan) -> Path:
    export_root = resolve(plan["export_dir"])
    matrix_root = resolve(plan["matrix_dir"])
    fact_dir = export_root / "fact"

    parts = sorted(fact_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"no fact parts below {fact_dir}. Run "
            "`python -m tfgnn.tigergraph.export` first."
        )

    if matrix_root.exists() and any(matrix_root.iterdir()):
        if not plan["overwrite"]:
            raise FileExistsError(
                f"matrix directory is not empty: {matrix_root}. Set "
                "assemble.overwrite true or move the previous run aside."
            )
        for stale in matrix_root.glob("part-*.parquet"):
            stale.unlink()
    matrix_root.mkdir(parents=True, exist_ok=True)

    builds = discover_builds(export_root)
    print(f"builds found: {len(builds)}")
    dimensions: dict[str, Dimensions] = {}
    for build in builds:
        dimensions[build.build_id] = load_dimensions(
            build, entity_rank_bins=plan["entity_rank_bins"]
        )
        window = ""
        if build.is_windowed:
            window = (
                f"  rows [{build.serves_from:,}, {build.serves_until:,})"
                if build.serves_until is not None
                else f"  rows from {build.serves_from:,}"
            )
        print(
            f"  {build.build_id:<18s} cutoff={build.cutoff_event_seq:<12,} "
            f"serves splits {list(build.serves_splits)}{window}"
        )
    if any(item.is_windowed for item in builds):
        print(
            "  forward chaining is ON: each training snapshot serves only rows "
            "at or after\n  its own cutoff, so no training row reads a feature "
            "fitted over a window\n  containing itself. Rows below the FIRST "
            "fold boundary are warm-up and are\n  reported as unassigned below "
            "-- they contribute topology and are never scored."
        )

    # Under forward chaining the first fold is warm-up: it contributes topology
    # to the first real snapshot and is never scored. Rows in split 0 below the
    # earliest served rank are therefore expected to be unrouted.
    train_windows = [
        item.serves_from
        for item in builds
        if 0 in item.serves_splits and item.serves_from is not None
    ]
    warmup_below = min(train_windows) if train_windows else None

    rows_total = 0
    unassigned_total = 0
    warmup_total = 0
    rows_per_split: dict[int, int] = {}
    frauds_per_split: dict[int, int] = {}
    null_counts = pd.Series(0, index=ALL_FEATURE_COLUMNS, dtype="int64")
    # Running min/max per feature column, so constancy is known across the
    # whole matrix without ever holding it. min == max with a non-zero count
    # means one value on every row -- which the saturated card projection
    # produces for c_size and for every Community aggregate.
    running_min: pd.Series | None = None
    running_max: pd.Series | None = None

    temporal = load_temporal(export_root)

    for index, part in enumerate(parts):
        fact = pd.read_parquet(part)
        # BEFORE routing, so every build's piece carries the columns. The
        # table is not build-scoped, so this is one join per fact part rather
        # than one per (part, build).
        if temporal is not None:
            fact = _join_temporal(fact, temporal)
        pieces: list[pd.DataFrame] = []

        split_ids = pd.to_numeric(fact["split_id"], errors="coerce").astype("int64")
        event_seq = pd.to_numeric(fact["event_seq"], errors="coerce").astype("int64")

        # Route by (split_id, event_seq window). A row matching no build is
        # DROPPED, not defaulted: under forward chaining that is the warm-up
        # fold, and silently attaching it to some snapshot would give it
        # features fitted after it happened.
        routed = pd.Series(False, index=fact.index)
        for build in builds:
            mask = build.row_mask(split_ids, event_seq)
            if not bool(mask.any()):
                continue
            routed |= mask
            pieces.append(join_part(fact.loc[mask], build, dimensions[build.build_id]))

        # Two very different reasons a row can go unrouted, and conflating them
        # would either hide a coverage bug or make forward chaining impossible:
        #   WARM-UP -- a valid split_id below the first fold boundary. Expected,
        #              reported, and exactly what causal_fold -2 means.
        #   UNSERVED -- anything else, e.g. split_id -1. A real bug: the row
        #              would vanish from the matrix silently.
        stray = ~routed
        if warmup_below is not None:
            warmup_mask = stray & split_ids.isin([0]) & (event_seq < warmup_below)
            warmup_total += int(warmup_mask.sum())
            stray = stray & ~warmup_mask
        unassigned_total += int(stray.sum())
        if not pieces:
            continue

        joined = finalise(apply_sentinels(pd.concat(pieces, ignore_index=True)))
        target = matrix_root / f"part-{index:06d}.parquet"
        joined.to_parquet(target, index=False)

        rows_total += len(joined)
        block = joined[ALL_FEATURE_COLUMNS]
        null_counts = null_counts.add(block.isna().sum().astype("int64"), fill_value=0)
        part_min = block.min(numeric_only=False)
        part_max = block.max(numeric_only=False)
        running_min = (
            part_min if running_min is None else running_min.combine(part_min, min)
        )
        running_max = (
            part_max if running_max is None else running_max.combine(part_max, max)
        )
        rows_per_split = _tally(rows_per_split, joined["split_id"])
        frauds_per_split = _tally(
            frauds_per_split, joined.loc[joined[LABEL] == 1, "split_id"]
        )

        print(f"  {part.name} -> {target.name}  {len(joined):>10,} rows")

    if unassigned_total:
        raise RuntimeError(
            f"{unassigned_total:,} fact rows have a split_id no build serves. "
            "They would silently vanish from the matrix. Check the build plan's "
            "serves_splits coverage."
        )
    if warmup_total:
        print(
            f"\nwarm-up rows withheld from training: {warmup_total:,} "
            f"(split 0, event_seq < {warmup_below:,})\n"
            "  They contribute topology to the first training snapshot and are "
            "never scored,\n  which is what schema r3 means by causal_fold -2. "
            "A snapshot serving them would\n  have its cutoff at the very first "
            "row, so every entity feature would be -1 ->\n  NaN and those rows "
            "would teach the model to read 'unknown' as normal.\n"
            "  Lower stage1_pipeline.train_folds to withhold fewer."
        )
    if rows_total == 0:
        raise RuntimeError("assembly produced no rows")

    fully_null = sorted(
        column
        for column in ALL_FEATURE_COLUMNS
        if int(null_counts.get(column, 0)) == rows_total
    )
    constant = sorted(
        column
        for column in ALL_FEATURE_COLUMNS
        if column not in set(fully_null)
        and running_min is not None
        and running_max is not None
        and pd.notna(running_min.get(column))
        and bool(running_min.get(column) == running_max.get(column))
    )
    inductive = inductive_reach(matrix_root)

    manifest = {
        "rows": rows_total,
        "rows_per_split": {str(k): v for k, v in sorted(rows_per_split.items())},
        "frauds_per_split": {str(k): v for k, v in sorted(frauds_per_split.items())},
        # DECISIONS_LEAK_CONTROL D3d's inductive requirement, measured rather
        # than assumed. All zeros means every evaluated entity was also trained
        # on, so the inductive path is satisfied in the code and never exercised
        # by the data -- say that next to any inductive claim instead of implying
        # the setting was tested. Reported, never gated: see inductive_reach.
        "inductive_reach": inductive,
        "builds": [
            {
                "build_id": b.build_id,
                "cutoff_event_seq": b.cutoff_event_seq,
                "cutoff_source": b.cutoff_source,
                "serves_splits": list(b.serves_splits),
                "serves_from": b.serves_from,
                "serves_until": b.serves_until,
            }
            for b in builds
        ],
        # Non-zero means forward chaining was on and this many split-0 rows were
        # withheld as warm-up. A row count read without this looks like data loss.
        "warmup_rows_withheld": warmup_total,
        "columns": ASSEMBLED_COLUMNS,
        "null_fraction": {
            column: round(float(null_counts.get(column, 0)) / rows_total, 6)
            for column in ALL_FEATURE_COLUMNS
        },
        "all_null_columns": fully_null,
        # One value on every row. wcc_card's header: "A constant feature is not
        # a weak feature, it is a column of noise that absorbs regularisation
        # budget." baseline.features drops these, refitting the decision on the
        # training rows alone rather than trusting this whole-matrix list.
        "constant_columns": constant,
        # 0 means raw values. Any other value means every AGGREGATE_FEATURES
        # and GRAPH_FEATURES column is a within-build percentile bin index,
        # not the quantity its name suggests. A metric read against the wrong
        # assumption here is not comparable, so it is recorded per matrix.
        "entity_rank_bins": plan["entity_rank_bins"],
        "parts": len(list(matrix_root.glob("part-*.parquet"))),
    }
    _ = (matrix_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _ = (matrix_root / "_SUCCESS").write_text("\n", encoding="utf-8")

    print(f"\nassembled {rows_total:,} rows -> {matrix_root}")
    for split_id in sorted(rows_per_split):
        print(
            f"  split {split_id}: {rows_per_split[split_id]:>12,} rows, "
            f"{frauds_per_split.get(split_id, 0):>8,} frauds"
        )
    print(
        f"\ninductive reach, against {inductive['train_cards']:,} training "
        f"cards and {inductive['train_merchants']:,} training merchants:"
    )
    for _, label in _EVALUATED_SPLITS:
        block = cast(dict[str, Any] | None, inductive.get(label))
        if block is None:
            continue
        print(
            f"  {label:<5s} {block['unseen_either_row_share']:.4f} of rows "
            f"have an unseen card or merchant "
            f"(card {block['unseen_card_row_share']:.4f}, "
            f"merchant {block['unseen_merchant_row_share']:.4f}); "
            f"{block['unseen_cards']:,} of {block['cards']:,} cards and "
            f"{block['unseen_merchants']:,} of {block['merchants']:,} "
            "merchants are new"
        )
    scored = [label for _, label in _EVALUATED_SPLITS if label in inductive]
    # `all()` over an empty sequence is True, so the membership guard alone
    # would print "every evaluated entity was trained on" when NOTHING was
    # evaluated. Require at least one measured split.
    if scored and all(
        cast(dict[str, Any], inductive[label])["unseen_either_row_share"] == 0.0
        for label in scored
    ):
        print(
            "  every evaluated card and merchant was also trained on. D3d's "
            "inductive requirement is\n  satisfied in the CODE and NOT "
            "EXERCISED by this data -- an inductive claim here is an\n  "
            "assumption. Entity churn in a regenerated load is what makes it "
            "a result."
        )

    if fully_null:
        print(
            "\ncolumns that are NaN on every row (the producing query did not "
            "run, which is expected for anything you disabled):"
        )
        for column in fully_null:
            print(f"  {column}")
    if constant:
        print(
            f"\n{len(constant)} columns hold ONE value on every row. A "
            "constant is not a weak feature, it is a column of noise that "
            "absorbs regularisation budget; the trainer drops them:"
        )
        for column in constant:
            value = running_min.get(column) if running_min is not None else "?"
            print(f"  {column:<40s} = {value}")
    worst = pd.Series(manifest["null_fraction"]).sort_values(ascending=False).head(8)
    print("\nhighest null fractions:")
    for column, fraction in worst.items():
        print(f"  {str(column):<40s} {float(cast(float, fraction)):.4f}")
    return matrix_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join the exported star schema into the Stage-1 training matrix"
    )
    _ = parser.parse_args()
    _ = np.seterr(all="ignore")
    plan = load_config("assemble", ASSEMBLE_ADAPTER)
    _ = assemble(plan)


if __name__ == "__main__":
    main()
