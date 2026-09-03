#!/usr/bin/env python
"""End-to-end Stage-1 smoke test against a synthetic TigerGraph.

WHY THIS EXISTS
---------------
The static validator checks names and signatures. It cannot check that the
pipeline's gate handling, the star-schema join, the -1 -> NaN sentinel rules and
the three-arm ablation actually work, and none of that should first be exercised
against a 27M-row Savanna instance forty minutes into a run.

``FakeGraph`` answers installed-query calls with TigerGraph-shaped payloads over
a small generated dataset that follows schema r3: dense ``event_seq`` from 1,
signed ``split_id``, one Card and one Merchant per transaction, and dimension
rows only for entities the build actually saw. Fraud is made mildly
structure-correlated so the graph arm has something to find; the AUCPR numbers
this prints are meaningless as results and are only evidence that the plumbing
carries signal at all.

Run it after any change to the pipeline, the export, the assembly or the
contract:

    python scripts/smoke_stage1_offline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfgnn.features_meta import (  # noqa: E402
    CARD_DIM_COLUMNS,
    CATEGORY_DIM_COLUMNS,
    COMMUNITY_DIM_COLUMNS,
    MERCHANT_DIM_COLUMNS,
    RESOLVED_ENTITY_DIM_COLUMNS,
)

TRANSACTIONS = 6000
CARDS = 220
MERCHANTS = 40
CATEGORIES = 7
# 1, not 9: this reproduces the saturation the live run actually hit --
# wcc_card returned ONE component holding all cards, so c_size is a constant and
# every transaction joins the same community aggregate row. That makes the
# constant-column drop in baseline.features a tested path rather than a claim.
COMMUNITIES = 1
ENTITIES = 60


def _assert_entity_binning(export_root: Path, bins: int) -> None:
    """Prove the transform destroys the entity fingerprint, per dimension.

    The live measurement that started this: card_avg_txn_amount took 14,363
    distinct values over 14,363 cards, so one continuous column was a primary
    key for the card dimension and XGBoost could isolate any single card. The
    check here is that property, at smoke scale: after rank+bin, no per-entity
    column may have as many distinct values as the table has rows.
    """
    import pandas as pd

    from tfgnn.entity_scale import ENTITY_FEATURE_COLUMNS, rank_bin_table
    from tfgnn.tigergraph import assemble as assemble_module

    for build in assemble_module.discover_builds(export_root):
        dims = assemble_module.load_dimensions(build, entity_rank_bins=0)
        for table, frame in (
            ("card", dims.card),
            ("merchant", dims.merchant),
            ("card_community", dims.card_community),
        ):
            columns = [c for c in frame.columns if c in ENTITY_FEATURE_COLUMNS]
            if not columns or len(frame) <= bins:
                continue
            binned, reports = rank_bin_table(frame, bins, table=table)
            for item in reports:
                if item.fingerprints:
                    raise AssertionError(
                        f"{build.build_id}/{table}.{item.column} still keys its "
                        f"dimension after binning: {item.distinct_after} distinct "
                        f"over {item.rows} rows. The fingerprint survived."
                    )
            for column in columns:
                values = pd.to_numeric(binned[column], errors="coerce").dropna()
                if len(values) and (values.min() < 0 or values.max() > bins - 1):
                    raise AssertionError(
                        f"{build.build_id}/{table}.{column} left bin range "
                        f"[0, {bins - 1}]: [{values.min()}, {values.max()}]"
                    )


def _assert_tuning_inputs_recorded(manifest_path: Path, *, bipartite: bool) -> None:
    """The tuning-provenance inputs reach the MANIFEST, not just the console.

    An attachment share is derived -- a count out of ``cards_seen`` -- so it
    appears in no query output. §0f's threshold pick has to cite the bipartite
    pair (``interaction_max_txn_count`` and
    ``cards_attached_share_at_min_txn_count``), and the share used to be
    computed inside ``advise()`` and dropped, which turned that citation into a
    hand recomputation from two other keys. Asserted per mode, because each
    mode runs only its own histogram.
    """
    import json

    query = (
        "interaction_strength_histogram" if bipartite else "projection_weight_histogram"
    )
    wanted = (
        {"interaction_max_txn_count", "cards_attached_share_at_min_txn_count"}
        if bipartite
        else {"cards_attached_share_at_min_weight"}
    )
    share_key = (
        "cards_attached_share_at_min_txn_count"
        if bipartite
        else "cards_attached_share_at_min_weight"
    )

    payload = cast(
        dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    histograms = [
        cast(dict[str, Any], record)
        for record in cast(list[Any], payload["records"])
        if cast(dict[str, Any], record).get("query") == query
    ]
    if not histograms:
        raise AssertionError(f"{manifest_path.name} has no {query} record")
    for record in histograms:
        recorded = cast(dict[str, Any], record.get("tuning_inputs") or {})
        missing = sorted(wanted - set(recorded))
        if missing:
            raise AssertionError(
                f"{query} record in {manifest_path.name} does not carry "
                f"{missing}; the A/B would have to recompute them by hand"
            )
        share = float(cast(float, recorded[share_key]))
        if not 0.0 < share <= 1.0:
            raise AssertionError(
                f"{share_key} is {share}, which is not a share of cards_seen"
            )
    print(f"  manifest carries the {query} tuning inputs")


def _assert_inductive_reach(tmp_root: Path) -> None:
    """inductive_reach against a matrix with KNOWN churn.

    The end-to-end smoke matrix has no churn -- every card and merchant appears
    in every split -- so it can only ever exercise the all-zero answer. A
    counter that is structurally always zero reads as a passing check, which is
    a failure this repo already has a worked example of in pagerank_card's
    header. So the non-zero path gets its own matrix, laid out by hand:

        split 0 (train)  cards c0 c1        merchants m0
        split 1 (val)    cards c0           merchants m0
        split 2 (test)   cards c0 c2 c2     merchants m0 m0 m1

    Test therefore holds 3 rows, of which 2 carry the unseen card c2 and 1
    carries the unseen merchant m1, and all 3 have one or the other. Val holds
    one fully-seen row, which is the zero case in the same assertion.
    """
    import pandas as pd

    from tfgnn.tigergraph.assemble import inductive_reach

    root = tmp_root / "inductive"
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "split_id": [0, 0, 1, 2, 2, 2],
            "card_number": ["c0", "c1", "c0", "c0", "c2", "c2"],
            "merchant_id": ["m0", "m0", "m0", "m1", "m0", "m0"],
        }
    )
    # Two parts, split across the boundary, so the measurement cannot depend on
    # a single part holding a whole split or on the parts arriving in order.
    frame.iloc[3:].to_parquet(root / "part-000000.parquet", index=False)
    frame.iloc[:3].to_parquet(root / "part-000001.parquet", index=False)

    report = inductive_reach(root)
    expected = {
        "train_cards": 2,
        "train_merchants": 1,
        "val": {
            "rows": 1,
            "unseen_card_row_share": 0.0,
            "unseen_merchant_row_share": 0.0,
            "unseen_either_row_share": 0.0,
            "cards": 1,
            "unseen_cards": 0,
            "merchants": 1,
            "unseen_merchants": 0,
        },
        "test": {
            "rows": 3,
            "unseen_card_row_share": round(2 / 3, 6),
            "unseen_merchant_row_share": round(1 / 3, 6),
            "unseen_either_row_share": 1.0,
            "cards": 2,
            "unseen_cards": 1,
            "merchants": 2,
            "unseen_merchants": 1,
        },
    }
    if report != expected:
        raise AssertionError(
            f"inductive reach mismatch\n  expected {expected}\n  got      {report}"
        )
    print("  inductive reach: unseen card/merchant shares correct under churn")


def _assert_pair_distance(matrix_dir: Path) -> None:
    """pair_home_distance_km must round-trip from the pair export, exactly.

    This is the feature the whole geography change exists for, and the join that
    delivers it is keyed on TWO columns. A mis-keyed composite join does not
    raise -- it quietly produces NaN, or worse, matches the wrong pair and
    returns a plausible number. So every non-null value is recomputed from the
    fake's own rule and compared.

    The fake sets -1 (-> NaN) when card % 7 == 0 or merchant % 5 == 0, which is
    how the left join is proved to KEEP those transaction rows rather than drop
    them: an inner join would delete about a third of the matrix and look like a
    smaller dataset rather than a missing feature.
    """
    import pandas as pd

    parts = sorted(matrix_dir.glob("part-*.parquet"))
    if not parts:
        raise AssertionError(f"no matrix parts below {matrix_dir}")
    frame = pd.concat(
        [
            pd.read_parquet(
                p, columns=["card_number", "merchant_id", "pair_home_distance_km"]
            )
            for p in parts
        ],
        ignore_index=True,
    )

    card = frame["card_number"].str.removeprefix("card_").astype("int64")
    merchant = frame["merchant_id"].str.removeprefix("mer_").astype("int64")
    missing = (card % 7 == 0) | (merchant % 5 == 0)
    expected = (5.0 + (card * 7 + merchant) % 4000).astype("float64")

    observed = pd.to_numeric(frame["pair_home_distance_km"], errors="coerce")

    # 1. Nothing the fake marked missing may carry a distance.
    leaked = int((~observed.isna() & missing).sum())
    if leaked:
        raise AssertionError(
            f"{leaked} rows carry a distance for a pair the fake marked -1. "
            "The -1 sentinel is reaching the matrix as a real value."
        )

    # 2. Every distance present must be the RIGHT one for that exact pair.
    present = ~observed.isna()
    if not bool(present.any()):
        raise AssertionError(
            "every pair_home_distance_km is NaN: the composite join matched "
            "nothing. Check the card_number / merchant_id dtypes on both sides."
        )
    wrong = int((observed[present] != expected[present]).sum())
    if wrong:
        raise AssertionError(
            f"{wrong} of {int(present.sum())} distances do not match the pair "
            "they are attached to. The (card_number, merchant_id) join matched "
            "the wrong row."
        )

    # 3. A pair with no interaction edge under its build's cutoff has no row in
    #    the pair table, so its distance is NaN too. That is expected and is why
    #    this is a bound rather than an equality.
    share = float(observed.isna().mean())
    if not 0.0 < share < 0.75:
        raise AssertionError(
            f"pair_home_distance_km is {share:.4f} NaN, outside the plausible "
            "band. All-NaN means the join failed; near-zero means the -1 "
            "sentinel is being read as a distance."
        )
    print(
        f"  pair distance: {int(present.sum()):,} joined values all correct, "
        f"{share:.4f} NaN"
    )


def _assert_binned_matrix(raw_dir: Path, binned_dir: Path, bins: int) -> None:
    """The binned matrix must differ from the raw one, and only where intended.

    Guards the two ways this could silently do nothing: the flag not being
    plumbed through (matrices identical), and the transform reaching
    FACT_FEATURES, whose values are row-level observations that must survive
    byte for byte across all four project stages.
    """
    import pandas as pd

    from tfgnn.features_meta import (
        AGGREGATE_FEATURES,
        FACT_ID,
        FACT_NUMERIC,
        GRAPH_FEATURES,
        PAIR_FEATURES,
    )

    def load(directory: Path) -> pd.DataFrame:
        # Glob the parts: the directory also holds manifest.json and _SUCCESS,
        # which read_parquet on the directory would try to parse as parquet.
        parts = sorted(directory.glob("part-*.parquet"))
        if not parts:
            raise AssertionError(f"no matrix parts below {directory}")
        frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        return frame.sort_values(FACT_ID).reset_index(drop=True)

    raw = load(raw_dir)
    binned = load(binned_dir)
    if len(raw) != len(binned):
        raise AssertionError(f"row count changed: {len(raw)} -> {len(binned)}")

    for column in FACT_NUMERIC:
        if column in raw.columns and not raw[column].equals(binned[column]):
            raise AssertionError(
                f"{column} is a FACT feature and must not be transformed. "
                "entity_scale.ENTITY_FEATURE_COLUMNS has leaked."
            )

    changed = 0
    for column in AGGREGATE_FEATURES + GRAPH_FEATURES:
        if column not in raw.columns:
            continue
        if column in PAIR_FEATURES:
            # Deliberately NOT binned: a per-pair distance cannot key a
            # dimension and does not drift between snapshots, so neither
            # half of the transform applies. Assert it was left ALONE,
            # which is the property that matters here.
            if not raw[column].equals(binned[column]):
                raise AssertionError(
                    f"{column} is a PAIR feature and must pass through the "
                    "rank+bin transform untouched."
                )
            continue
        values = pd.to_numeric(binned[column], errors="coerce").dropna()
        if len(values) and (values.min() < 0 or values.max() > bins - 1):
            raise AssertionError(
                f"{column} left bin range [0, {bins - 1}]: "
                f"[{values.min()}, {values.max()}]"
            )
        if not raw[column].equals(binned[column]):
            changed += 1
    if changed == 0:
        raise AssertionError(
            "entity_rank_bins changed no per-entity column. The flag is not "
            "reaching load_dimensions."
        )
    print(f"  rank+bin verified: {changed} per-entity columns transformed")


def _vertex_payload(
    alias: str, columns: list[str], rows: list[dict[str, Any]], build_id: str
) -> list[object]:
    """Shape a result the way a TigerGraph vertex-set PRINT does.

    Attributes are prefixed with the vertex-set variable name, which is the
    behaviour ``export.vertex_rows`` has to tolerate.
    """
    return [
        {
            alias: [
                {
                    "v_id": str(row[columns[0]]),
                    "v_type": alias,
                    "attributes": {f"{alias}.{name}": row[name] for name in columns},
                }
                for row in rows
            ]
        },
        {"rows_returned": len(rows), "build_id": build_id},
    ]


class FakeGraph:
    """A synthetic TF_GNN instance with just enough behaviour to drive Stage 1.

    ``bipartite`` mirrors graph_features.source: it decides which community
    shape the export answers with. In bipartite mode a component spans both
    types, so cards AND merchants share community ids and the community rows
    carry entity_family "bipartite" -- which is what exercises assemble's
    both-frames mapping rather than the per-family split.
    """

    graphname = "TF_GNN"

    def __init__(self, seed: int = 7, bipartite: bool = False) -> None:
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.bipartite = bipartite
        # Instance-level, NOT class-level: two FakeGraphs exist in one smoke
        # run (one per graph_features source), and a shared class dict would
        # let the first pass's registrations silently serve the second.
        self._cutoffs: dict[str, int] = {}
        n = TRANSACTIONS

        self.event_seq = np.arange(1, n + 1, dtype="int64")
        # thirds, in time order: train | val | test
        self.split_id = np.where(
            self.event_seq <= int(0.6 * n),
            0,
            np.where(self.event_seq <= int(0.8 * n), 1, 2),
        ).astype("int64")
        self.causal_fold = np.where(self.split_id == 0, 0, -1).astype("int64")

        self.card_index = rng.integers(0, CARDS, n)
        self.merchant_index = rng.integers(0, MERCHANTS, n)
        self.amount = np.round(np.exp(rng.normal(3.2, 1.1, n)), 2)

        # A handful of cards are structurally bad, and they also sit in the
        # low-numbered communities, so the graph arm has a real (if synthetic)
        # edge over the fact-only arm.
        self.bad_card = rng.random(CARDS) < 0.09
        risk = 0.004 + 0.09 * self.bad_card[self.card_index]
        self.is_fraud = (rng.random(n) < risk).astype("int64")
        # guarantee positives in every split
        for split in (0, 1, 2):
            mask = self.split_id == split
            if self.is_fraud[mask].sum() < 12:
                candidates = np.flatnonzero(mask)[:12]
                self.is_fraud[candidates] = 1

        self.mer_cat = np.array(
            [f"cat_{index % CATEGORIES}" for index in self.merchant_index]
        )
        self.use_chip = rng.choice(
            ["Swipe Transaction", "Online Transaction", "Chip Transaction"], n
        )
        self.error = rng.choice(
            ["", "Bad PIN", "Insufficient Balance"], n, p=[0.9, 0.05, 0.05]
        )
        self.is_online = self.use_chip == "Online Transaction"
        self.unix_time = (1_500_000_000 + self.event_seq * 900).astype("int64")

        self.card_ids = [f"card_{index:05d}" for index in range(CARDS)]
        self.merchant_ids = [f"mer_{index:04d}" for index in range(MERCHANTS)]
        self.calls: list[str] = []

    # -- the Client interface the pipeline and exporter use ----------------

    def run_installed_with_timeout(
        self,
        query_name: str,
        params: dict[str, object],
        timeout_s: float | None = None,
        size_limit: int | None = None,
    ) -> list[object]:
        del timeout_s, size_limit
        return self._dispatch(query_name, params)

    def run_installed_detached(
        self,
        query_name: str,
        params: dict[str, object],
        **_: object,
    ) -> list[object]:
        return self._dispatch(query_name, params)

    def gsql(self, statement: str) -> str:
        return f"Successfully created queries: {statement[:40]}"

    # -- query answers -----------------------------------------------------

    def _dispatch(self, name: str, params: dict[str, object]) -> list[object]:
        self.calls.append(name)
        handler = getattr(self, f"_q_{name}", None)
        if handler is None:
            return self._generic(name)
        return cast(list[object], handler(params))

    def _generic(self, name: str) -> list[object]:
        """A query with no gates: report that it did something."""
        return [{"query": name, "rows_touched": TRANSACTIONS}]

    def _q_derive_build_plan(self, params: dict[str, object]) -> list[object]:
        del params
        first = {
            int(split): int(self.event_seq[self.split_id == split].min())
            for split in (0, 1, 2)
        }
        last = {
            int(split): int(self.event_seq[self.split_id == split].max())
            for split in (0, 1, 2)
        }
        rows = {int(split): int((self.split_id == split).sum()) for split in (0, 1, 2)}
        frauds = {
            int(split): int(self.is_fraud[self.split_id == split].sum())
            for split in (0, 1, 2)
        }
        return [
            {
                "total_transactions": TRANSACTIONS,
                "unknown_event_seq_MUST_BE_ZERO": 0,
                "unset_split_id_MUST_BE_ZERO": 0,
                "is_fraud_outside_0_1_MUST_BE_ZERO": 0,
            },
            {
                "fold_first_event_seq": {"0": first[0]},
                "fold_last_event_seq": {"0": last[0]},
                "fold_rows": {"0": rows[0]},
                "fold_frauds": {"0": frauds[0]},
            },
            {
                "split_first_event_seq": {str(k): v for k, v in first.items()},
                "split_last_event_seq": {str(k): v for k, v in last.items()},
                "split_rows": {str(k): v for k, v in rows.items()},
                "split_frauds": {str(k): v for k, v in frauds.items()},
            },
        ]

    def _q_assert_edge_stamp(self, params: dict[str, object]) -> list[object]:
        # Healthy load: the edge stamp equals the transaction's event_seq on
        # every edge, and none sit at the 0 default. Returned with the real
        # gate key names so gates.evaluate actually evaluates them -- a
        # handler that omitted them would let the load gate pass vacuously,
        # which is the failure mode this file exists to prevent.
        del params
        return [
            {
                "card_send_transaction_stamp_mismatch_MUST_BE_ZERO": 0,
                "card_send_transaction_stamp_zero_MUST_BE_ZERO": 0,
                "transaction_to_merchant_stamp_mismatch_MUST_BE_ZERO": 0,
                "transaction_to_merchant_stamp_zero_MUST_BE_ZERO": 0,
            },
            {
                "card_send_transaction_edges": TRANSACTIONS,
                "transaction_to_merchant_edges": TRANSACTIONS,
                "card_send_transaction_max_abs_difference": 0,
                "transaction_to_merchant_max_abs_difference": 0,
            },
        ]

    def _q_assert_cardinality(self, params: dict[str, object]) -> list[object]:
        del params
        n = TRANSACTIONS
        return [
            {
                "transaction_count": n,
                "min_event_seq_MUST_BE_1": 1,
                "max_event_seq_MUST_EQUAL_transaction_count": n,
                "sum_event_seq": n * (n + 1) / 2,
                "expected_sum_if_dense_and_unique": n * (n + 1) / 2,
                "sum_gap_MUST_BE_ZERO": 0,
                "zero_event_seq_MUST_BE_ZERO": 0,
            },
            {
                "transactions_without_exactly_one_card_MUST_BE_ZERO": 0,
                "transactions_without_exactly_one_merchant_MUST_BE_ZERO": 0,
            },
            {
                "transactions_missing_card_number_MUST_BE_ZERO": 0,
                "transactions_missing_merchant_id_MUST_BE_ZERO": 0,
            },
            {
                "is_fraud_outside_0_1_MUST_BE_ZERO": 0,
                "rows_per_label": {
                    "0": int(n - self.is_fraud.sum()),
                    "1": int(self.is_fraud.sum()),
                },
            },
            {
                "split_id_out_of_range_MUST_BE_ZERO": 0,
                "causal_fold_inconsistent_with_split_MUST_BE_ZERO": 0,
            },
        ]

    def _q_reset_identity(self, params: dict[str, object]) -> list[object]:
        dry = bool(params.get("dry_run"))
        return [
            {
                "dry_run": dry,
                "same_as_edges": 0 if dry else 41,
                "resolved_entity_vertices": 0 if dry else ENTITIES,
                "parties_with_stale_re_id": 0 if dry else 12,
                "cards_with_stale_re_id": 0 if dry else 8,
                "merchants_with_stale_re_id": 0,
            }
        ]

    def _q_match_parties(self, params: dict[str, object]) -> list[object]:
        return [
            {
                "parties_with_at_least_one_match": 46,
                "same_as_edges_inserted": 31,
                "source_parties_under_cutoff": 180,
                "cutoff_used": params["cutoff_event_seq"],
                "threshold_used": params["threshold"],
            },
            # Buckets are score * 10. Read by a human to set threshold, not by
            # any code: repeated values of ONE PII type accumulate, so the
            # reachable scores are not the powerset of the weights and adding
            # them up does not tell you where to cut.
            {"score_histogram_tenths": {"10": 18, "11": 9, "16": 4}},
            # weights_used x scored_relations is the pair advise() cross-reads
            # for inert weights. This fake used to return
            # parties_with_multiple_email / _name / _phone instead -- three keys
            # the real query STOPPED PRINTING when the SumAccum<STRING> bug was
            # removed, so the fake was asserting a contract that no longer
            # existed and the advisory reading them could never fire.
            #
            # Every weighted relation carries evidence here, so a passing smoke
            # run is quiet. The firing branch is covered directly in
            # validate_stage1_static.check_identity_advice.
            {
                "weights_used": {
                    "Party_Has_Birthdate": 0.5,
                    "Party_Has_Street_Address": 0.2,
                    "Party_Has_Email_Address_Hash": 0.5,
                    "Party_Has_Name_Hash": 0.2,
                    "Party_Has_Phone_Hash": 0.5,
                },
                "scored_relations": {
                    "Party_Has_Birthdate": 22,
                    "Party_Has_Street_Address": 6,
                    "Party_Has_Email_Address_Hash": 4,
                    "Party_Has_Name_Hash": 345,
                    "Party_Has_Phone_Hash": 3,
                },
            },
        ]

    def _q_stamp_same_as_provenance(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "same_as_edges_seen": 31,
                "same_as_edges_stamped": 31,
                "same_as_edges_with_no_shared_pii_MUST_BE_ZERO": 0,
            },
            {"shared_pii_count_histogram": {"1": 22, "2": 9}},
        ]

    def _q_unify_parties(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "same_as_edges_total": 31,
                "same_as_edges_unstamped_MUST_BE_ZERO": 0,
                "same_as_edges_under_cutoff": 27,
            },
            {
                "parties_resolved": 180,
                "resolved_entity_count": ENTITIES,
                "singleton_resolved_entities": 38,
                "largest_component_size": 4,
            },
        ]

    def _q_assert_resolved_entity(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "ABORT_largest_entity_over_max_member_count": False,
                "ABORT_largest_entity_over_max_fraction": False,
                "largest_entity_member_count": 4,
                "largest_entity_fraction": 0.022,
            },
            {"parties_resolved": 180, "resolved_entity_count": ENTITIES},
        ]

    def _q_resolved_entity_stats(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "resolved_entities_with_members_under_cutoff": ENTITIES,
                "resolved_entities_empty_under_cutoff": 3,
            },
            {
                "same_as_unstamped_MUST_BE_ZERO": 0,
                "same_as_evidence_under_cutoff": 27,
            },
        ]

    def _q_stamp_resolved_entity_keys(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "cards_stamped": 150,
                "cards_with_conflicting_re_id": 2,
                "cards_dropped_as_conflicting": 0,
                "merchants_stamped": 20,
            }
        ]

    def _q_build_interaction_edges(self, params: dict[str, object]) -> list[object]:
        # Per-invocation batched: SLICE keys only, no gate key -- the gates
        # live in verify_interaction_build, and the slices must sum to its
        # recount or the equality gate there is being faked rather than
        # exercised.
        cutoff = int(cast(int, params["cutoff_event_seq"]))
        batch = int(cast(int, params.get("source_batch", 0)))
        batches = int(cast(int, params.get("num_of_source_batches", 1)))
        under = int((self.event_seq < cutoff).sum())
        slice_count = under // batches + (1 if batch < under % batches else 0)
        return [
            {
                "interaction_edges_inserted_in_batch": 900 // batches,
                "transactions_under_cutoff_in_batch": slice_count,
                "cutoff_used": cutoff,
                "source_batch": batch,
                "source_batch_count": batches,
            }
        ]

    def _q_verify_interaction_build(self, params: dict[str, object]) -> list[object]:
        # The independent recount. The equality gate holds by construction
        # here because the txn sum is derived from the same fake event_seq
        # array the writer slices -- which is the contract the REAL pair
        # (writer batches + verifier) must satisfy on the instance.
        cutoff = int(cast(int, params["cutoff_event_seq"]))
        under = int((self.event_seq < cutoff).sum())
        return [
            {
                "transactions_under_cutoff_MUST_BE_BELOW_TOTAL": under,
                "transactions_under_cutoff": under,
                "interaction_txn_sum_MUST_EQUAL_transactions_under_cutoff": under,
                "interaction_edges_for_build": 900,
                "cutoff_used": cutoff,
            }
        ]

    def _q_reset_build(self, params: dict[str, object]) -> list[object]:
        # Batched reset: per-slice counts plus source_batch_count, which the
        # pipeline's expect_present names as the stale-endpoint tripwire.
        batch = int(cast(int, params.get("source_batch", 0)))
        batches = int(cast(int, params.get("num_of_source_batches", 1)))
        return [
            {
                "card_card_edges_deleted": 0,
                "merchant_merchant_edges_deleted": 0,
                "interaction_edges_deleted": 900 // batches,
                "community_vertices_deleted": 0,
                "cards_reset": CARDS // batches,
                "merchants_reset": MERCHANTS // batches,
                "merchant_categories_reset": 0,
                "source_batch": batch,
                "source_batch_count": batches,
            }
        ]

    def _q_card_home_distance(self, params: dict[str, object]) -> list[object]:
        # Batched like the builder: card-derived counters are slices, the
        # merchant census (and its gate) is full on every call.
        batch = int(cast(int, params.get("source_batch", 0)))
        batches = int(cast(int, params.get("num_of_source_batches", 1)))
        return [
            {
                "merchants_with_multiple_locations_MUST_BE_ZERO": 0,
                "edges_written": 600 // batches,
                "edges_missing_geography": 300 // batches,
                "cards_with_home_coordinates": (CARDS - 20) // batches,
                "cards_with_multiple_tenures": 0,
                "merchants_with_location": MERCHANTS - 1,
                "max_distance_km": 1_500.0,
                "distance_sum_km": 90_000.0 / batches,
                "source_batch": batch,
                "source_batch_count": batches,
            }
        ]

    def _q_card_merchant_degree_stats(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "cards_seen": CARDS - 12,
                "cards_unseen": 12,
                "merchants_seen": MERCHANTS,
                "merchants_unseen": 0,
            },
            {
                "max_locations_per_merchant": 1,
                "merchants_with_multiple_locations": 0,
                # The geography gate. Geocoded, not raw: see the query header.
                "merchants_with_multiple_geocoded_locations_MUST_BE_ZERO": 0,
            },
        ]

    def _q_projection_density_check(self, params: dict[str, object]) -> list[object]:
        probe_cap = int(cast(int, params.get("probe_cap", 0)))
        pair_budget = float(cast(float, params.get("pair_budget", 0.0)))
        capped = 39_000.0 if probe_cap > 0 else 0.0
        # Armed only in projection mode (the pipeline passes 0/0.0 in
        # bipartite mode). Under budget here so the smoke passes; the gate
        # machinery still EVALUATES the ABORT_* key on every projection-mode
        # smoke run, which is what keeps an inverted condition or a dropped
        # print from shipping green.
        over_budget = 1 if (probe_cap > 0 and 0 < pair_budget < capped) else 0
        return [
            {
                "cards_with_interactions": CARDS - 12,
                "merchants_with_interactions": MERCHANTS,
                "card_card_candidate_pairs": 41_000.0,
                "card_pairs_possible": 21_000.0,
                "card_card_saturation_hint": 0.21,
                "merchant_merchant_saturation_hint": 0.28,
                # In band against tuning_provenance, so a passing smoke run is
                # quiet. The DRIFTED branch is covered directly by
                # validate_stage1_static's check_tuning_provenance -- an
                # advisory that only ever fires on real data is an advisory
                # nobody has seen work.
                "max_merchant_reach_share": 0.49,
                "hub_concentration_S": 2.72,
            },
            {
                "card_card_candidate_pairs_at_cap": capped,
                "merchants_excluded_by_cap": 1 if probe_cap > 0 else 0,
                "probe_cap_used": probe_cap,
                "pair_budget_used": pair_budget,
                "ABORT_capped_candidate_pairs_over_budget": over_budget,
            },
            {"top_merchants_by_degree": []},
        ]

    def _q_projection_weight_histogram(self, params: dict[str, object]) -> list[object]:
        probe = int(cast(int, params["probe_e"]))
        seen = CARDS - 12
        return [
            {
                "card_card_edges": 5100.0,
                "card_card_min_weight": 1,
                "card_card_max_weight": 39,
                "card_card_avg_weight": 3.26,
                "cards_seen": seen,
            },
            {
                # Keyed by the probe the pipeline passed, which is the whole
                # point of passing it: the attachment share is read at the
                # threshold in force, not at a fixed one.
                "card_card_surviving_edges_by_threshold": {str(probe): 1_340.0},
                "cards_with_degree_by_threshold": {
                    str(probe): int(round(0.918 * seen))
                },
            },
        ]

    def _q_card_card_with_weights(self, params: dict[str, object]) -> list[object]:
        del params
        return [{"card_card_edges_inserted": 5100, "hub_merchants_skipped": 0}]

    def _q_merchant_merchant_with_weights(
        self, params: dict[str, object]
    ) -> list[object]:
        del params
        return [{"merchant_merchant_edges_inserted": 480, "hub_cards_skipped": 0}]

    def _q_wcc_card(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "component_count": COMMUNITIES,
                "cards_in_components": CARDS - 30,
                "cards_isolated_in_projection": 18,
                "cards_unseen_excluded": 12,
                "largest_component_size": 60,
            },
            {
                "component_size_histogram_MUST_HAVE_NO_KEY_1": {
                    "60": 1,
                    "24": 3,
                    "12": 5,
                }
            },
        ]

    def _q_wcc_merchant(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "component_count": 3,
                "merchants_in_components": MERCHANTS - 4,
                "merchants_isolated_in_projection": 4,
                "merchants_unseen_excluded": 0,
                "largest_component_size": 20,
            },
            {"component_size_histogram_MUST_HAVE_NO_KEY_1": {"20": 1, "8": 2}},
        ]

    def _q_community_stats(self, params: dict[str, object]) -> list[object]:
        del params
        if self.bipartite:
            return [
                {
                    "card_communities": 0,
                    "merchant_communities": 0,
                    "bipartite_communities": COMMUNITIES,
                    "mode_mixture_communities_MUST_BE_ZERO": 0,
                    "communities_with_no_txns_under_cutoff": 0,
                }
            ]
        return [
            {
                "card_communities": COMMUNITIES,
                "merchant_communities": 3,
                "bipartite_communities": 0,
                "mode_mixture_communities_MUST_BE_ZERO": 0,
                "communities_with_no_txns_under_cutoff": 1,
            }
        ]

    # -- bipartite feature path (graph_features.source: bipartite) ---------

    def _q_interaction_strength_histogram(
        self, params: dict[str, object]
    ) -> list[object]:
        probe = int(cast(int, params["probe_e"]))
        seen_cards = CARDS - 12
        return [
            {
                "interaction_edges": 900.0,
                "interaction_min_txn_count": 1,
                "interaction_max_txn_count": 34,
                "interaction_avg_txn_count": 4.2,
                "cards_seen": seen_cards,
                "merchants_seen": MERCHANTS,
                "build_id": params["build_id"],
            },
            {
                "interaction_txn_count_bands": {
                    "1": 480,
                    "2": 190,
                    "3": 90,
                    "4": 80,
                    "5": 60,
                }
            },
            {
                # Keyed by the probe the pipeline passed, same contract as
                # projection_weight_histogram: attachment is read at the
                # threshold in force. Non-empty at the probe, so the
                # zero-surviving advisory stays quiet on a passing run.
                "surviving_edges_by_threshold": {str(probe): 620.0},
                "cards_attached_by_threshold": {
                    str(probe): int(round(0.93 * seen_cards))
                },
                "merchants_attached_by_threshold": {str(probe): MERCHANTS - 2},
            },
        ]

    def _q_wcc_bipartite(self, params: dict[str, object]) -> list[object]:
        del params
        return [
            {
                "component_count": COMMUNITIES,
                "cards_in_components": CARDS - 30,
                "merchants_in_components": MERCHANTS - 4,
                "cards_isolated_at_threshold": 18,
                "merchants_isolated_at_threshold": 4,
                "cards_unseen_excluded": 12,
                "merchants_unseen_excluded": 0,
                "largest_component_size": 96,
            },
            # Sizes count cards + merchants; a bipartite component is >= 2 by
            # construction (one card + one merchant), so no key 1 and no key
            # that could be a singleton.
            {
                "component_size_histogram_MUST_HAVE_NO_KEY_1": {
                    "96": 1,
                    "48": 2,
                    "16": 2,
                }
            },
        ]

    def _q_pagerank_bipartite(self, params: dict[str, object]) -> list[object]:
        return [
            {
                "cards_scored": CARDS - 30,
                "merchants_scored": MERCHANTS - 4,
                "cards_seen_but_isolated_left_at_sentinel": 18,
                "merchants_seen_but_isolated_left_at_sentinel": 4,
                "cards_unseen_excluded": 12,
                "merchants_unseen_excluded": 0,
                "iterations_run": 11,
                "iteration_limit": params["maximum_iteration"],
                "final_max_change": 0.0007,
            }
        ]

    def _q_louvain_bipartite(self, params: dict[str, object]) -> list[object]:
        return [
            {
                "louvain_community_count": 9,
                "largest_community_size": 70,
                # Above the 0.3 advisory line so a passing smoke run is quiet;
                # the firing branch is a pipeline advisory, not a gate.
                "modularity_Q": 0.41,
                "iterations_run": 7,
                "iteration_limit": params["max_iteration"],
                "cards_unseen_excluded": 12,
                "merchants_unseen_excluded": 0,
            },
            {"louvain_size_histogram": {"70": 1, "20": 4, "10": 4}},
        ]

    def _q_pagerank_card(self, params: dict[str, object]) -> list[object]:
        return [
            {
                "cards_scored": CARDS - 30,
                "cards_seen_but_isolated_left_at_sentinel": 18,
                "iterations_run": 9,
                "iteration_limit": params["maximum_iteration"],
                "final_max_change": 0.0008,
            }
        ]

    def _q_pagerank_merchant(self, params: dict[str, object]) -> list[object]:
        return [
            {
                "merchants_scored": MERCHANTS - 4,
                "merchants_seen_but_isolated_left_at_sentinel": 4,
                "iterations_run": 5,
                "iteration_limit": params["maximum_iteration"],
                "final_max_change": 0.0004,
            }
        ]

    # -- exports -----------------------------------------------------------

    def _seen_cards(self, cutoff: int) -> np.ndarray:
        under = self.event_seq < cutoff
        return np.unique(self.card_index[under])

    def _q_export_transaction_rows(self, params: dict[str, object]) -> list[object]:
        lo = int(cast(int, params["from_event_seq"]))
        hi = int(cast(int, params["to_event_seq"]))
        mask = (self.event_seq >= lo) & (self.event_seq < hi)
        indices = np.flatnonzero(mask)
        rows = [
            {
                "id": f"txn_{self.event_seq[i]:08d}",
                "card_number": self.card_ids[int(self.card_index[i])],
                "merchant_id": self.merchant_ids[int(self.merchant_index[i])],
                "mer_cat": str(self.mer_cat[i]),
                "event_seq": int(self.event_seq[i]),
                "unix_time": int(self.unix_time[i]),
                "split_id": int(self.split_id[i]),
                "causal_fold": int(self.causal_fold[i]),
                "amount": float(self.amount[i]),
                "use_chip": str(self.use_chip[i]),
                "error": str(self.error[i]),
                "is_online": bool(self.is_online[i]),
                "is_fraud": int(self.is_fraud[i]),
            }
            for i in indices
        ]
        payload = _vertex_payload(
            "Rows",
            [
                "id",
                "card_number",
                "merchant_id",
                "mer_cat",
                "event_seq",
                "unix_time",
                "split_id",
                "causal_fold",
                "amount",
                "use_chip",
                "error",
                "is_online",
                "is_fraud",
            ],
            rows,
            "-",
        )
        payload[1] = {
            "rows_returned": len(rows),
            "from_event_seq": lo,
            "to_event_seq": hi,
        }
        return payload

    def _q_export_card_features(self, params: dict[str, object]) -> list[object]:
        build_id = str(params["build_id"])
        cutoff = self._cutoff_for(build_id)
        seen = self._seen_cards(cutoff)
        under = self.event_seq < cutoff
        rows: list[dict[str, Any]] = []
        for card in seen:
            mask = under & (self.card_index == card)
            count = int(mask.sum())
            if count == 0:
                continue
            amounts = self.amount[mask]
            bad = bool(self.bad_card[card])
            community = int(card % COMMUNITIES)
            rows.append(
                {
                    "card_number": self.card_ids[int(card)],
                    "build_id": build_id,
                    "pagerank": float(0.5 + (2.4 if bad else 0.4) * self.rng.random()),
                    # constant on purpose: one saturated component
                    "c_size": CARDS,
                    "cc_degree": int(3 + (22 if bad else 2)),
                    # never run in the default plan -> sentinel -> NaN
                    "louvain_size": -1,
                    "core_number": -1,
                    "distinct_merchant_count": int(
                        np.unique(self.merchant_index[mask]).size
                    ),
                    "repeated_merchant_count": int(
                        max(0, np.unique(self.merchant_index[mask]).size - 2)
                    ),
                    "txn_count": count,
                    "total_amount": float(amounts.sum()),
                    "max_txn_amount": float(amounts.max()),
                    "min_txn_amount": float(amounts.min()),
                    "avg_txn_amount": float(amounts.mean()),
                    "max_amount_in_interval": -1,
                    "max_txn_count_in_interval": -1,
                    "c_id": community,
                    "louvain_id": -1,
                    "re_id": int(card % ENTITIES),
                }
            )
        return _vertex_payload("Cards", CARD_DIM_COLUMNS, rows, build_id)

    def _q_export_merchant_features(self, params: dict[str, object]) -> list[object]:
        build_id = str(params["build_id"])
        cutoff = self._cutoff_for(build_id)
        under = self.event_seq < cutoff
        rows: list[dict[str, Any]] = []
        for merchant in range(MERCHANTS):
            mask = under & (self.merchant_index == merchant)
            count = int(mask.sum())
            if count == 0:
                continue
            amounts = self.amount[mask]
            rows.append(
                {
                    "id": self.merchant_ids[merchant],
                    "build_id": build_id,
                    # Every 5th merchant has no location, so a row can be
                    # missing geography from either side.
                    "pagerank": float(0.5 + self.rng.random()),
                    "c_size": 20 if merchant % 2 else 8,
                    "cc_degree": int(4 + merchant % 7),
                    "louvain_size": -1,
                    "core_number": -1,
                    "distinct_card_count": int(np.unique(self.card_index[mask]).size),
                    "repeated_card_count": int(
                        max(0, np.unique(self.card_index[mask]).size - 3)
                    ),
                    "txn_count": count,
                    "total_amount": float(amounts.sum()),
                    "max_txn_amount": float(amounts.max()),
                    "min_txn_amount": float(amounts.min()),
                    "avg_txn_amount": float(amounts.mean()),
                    # In bipartite mode a component spans both types, so
                    # merchants share the cards' community id space.
                    "c_id": (
                        int(merchant % COMMUNITIES)
                        if self.bipartite
                        else 1000 + merchant % 3
                    ),
                    "louvain_id": -1,
                    "re_id": -1,
                }
            )
        return _vertex_payload("Merchants", MERCHANT_DIM_COLUMNS, rows, build_id)

    def _q_export_category_features(self, params: dict[str, object]) -> list[object]:
        build_id = str(params["build_id"])
        cutoff = self._cutoff_for(build_id)
        under = self.event_seq < cutoff
        rows: list[dict[str, Any]] = []
        for index in range(CATEGORIES):
            label = f"cat_{index}"
            mask = under & (self.mer_cat == label)
            count = int(mask.sum())
            amounts = self.amount[mask]
            rows.append(
                {
                    "category": label,
                    "build_id": build_id,
                    "distinct_merchant_count": int(
                        np.unique(self.merchant_index[mask]).size
                    )
                    or -1,
                    "txn_count": count if count else -1,
                    "total_amount": float(amounts.sum()) if count else -1.0,
                    "max_txn_amount": float(amounts.max()) if count else -1.0,
                    "min_txn_amount": float(amounts.min()) if count else -1.0,
                    "avg_txn_amount": float(amounts.mean()) if count else -1.0,
                }
            )
        return _vertex_payload("Cats", CATEGORY_DIM_COLUMNS, rows, build_id)

    def _q_export_community_features(self, params: dict[str, object]) -> list[object]:
        build_id = str(params["build_id"])
        cutoff = self._cutoff_for(build_id)
        under = self.event_seq < cutoff
        rows: list[dict[str, Any]] = []
        for community in range(COMMUNITIES):
            members = np.flatnonzero(np.arange(CARDS) % COMMUNITIES == community)
            mask = under & np.isin(self.card_index, members)
            count = int(mask.sum())
            amounts = self.amount[mask]
            merchant_members = (
                np.flatnonzero(np.arange(MERCHANTS) % COMMUNITIES == community)
                if self.bipartite
                else np.array([], dtype="int64")
            )
            rows.append(
                {
                    "cid": community,
                    "build_id": build_id,
                    # A bipartite community holds both types; its aggregates
                    # still follow CARD membership only (community_stats'
                    # double-count rule), which is exactly what `mask` counts.
                    "entity_family": "bipartite" if self.bipartite else "card",
                    "member_count": int(members.size + merchant_members.size),
                    "txn_count": count if count else -1,
                    "total_amount": float(amounts.sum()) if count else -1.0,
                    "max_amount": float(amounts.max()) if count else -1.0,
                    "min_amount": float(amounts.min()) if count else -1.0,
                    "avg_amount": float(amounts.mean()) if count else -1.0,
                }
            )
        if self.bipartite:
            # No single-family rows: every component in a bipartite build has
            # one card end and one merchant end by construction, and a mixed
            # population would trip mode_mixture_communities_MUST_BE_ZERO.
            return _vertex_payload("Comms", COMMUNITY_DIM_COLUMNS, rows, build_id)
        for offset in range(3):
            merchants = np.flatnonzero(np.arange(MERCHANTS) % 3 == offset)
            mask = under & np.isin(self.merchant_index, merchants)
            count = int(mask.sum())
            amounts = self.amount[mask]
            rows.append(
                {
                    "cid": 1000 + offset,
                    "build_id": build_id,
                    "entity_family": "merchant",
                    "member_count": int(merchants.size),
                    "txn_count": count if count else -1,
                    "total_amount": float(amounts.sum()) if count else -1.0,
                    "max_amount": float(amounts.max()) if count else -1.0,
                    "min_amount": float(amounts.min()) if count else -1.0,
                    "avg_amount": float(amounts.mean()) if count else -1.0,
                }
            )
        return _vertex_payload("Comms", COMMUNITY_DIM_COLUMNS, rows, build_id)

    def _q_export_resolved_entity_features(
        self, params: dict[str, object]
    ) -> list[object]:
        build_id = str(params["build_id"])
        rows = [
            {
                "reid": index,
                "build_id": build_id,
                "member_count": 1 + index % 3,
                "distinct_pii_count": 2 + index % 5,
                "connected_card_count": 1 + index % 4,
                "connected_merchant_count": index % 2,
                "max_pair_score": float(1.0 + 0.1 * (index % 6)),
            }
            for index in range(ENTITIES)
        ]
        return _vertex_payload("Entities", RESOLVED_ENTITY_DIM_COLUMNS, rows, build_id)

    def _q_export_pair_features(self, params: dict[str, object]) -> list[object]:
        """One row per (card, merchant) pair seen under this build's cutoff.

        Deliberately a FLAT list rather than a _vertex_payload: the real query
        PRINTs a global ListAccum<TUPLE>, which serialises with no "attributes"
        wrapper, and export.vertex_rows takes a different branch for that shape.
        Faking the vertex shape would leave that branch untested.

        Every 7th card and every 5th merchant gets NO distance, so the
        -1 -> NaN path in _join_pair_distance is exercised rather than assumed,
        and the left join is proved to keep those transaction rows.
        """
        build_id = str(params["build_id"])
        cutoff = self._cutoff_for(build_id)
        under = self.event_seq < cutoff
        seen: set[tuple[int, int]] = set(
            zip(
                self.card_index[under].tolist(),
                self.merchant_index[under].tolist(),
                strict=True,
            )
        )
        rows: list[dict[str, Any]] = []
        for card, merchant in sorted(seen):
            missing = (card % 7 == 0) or (merchant % 5 == 0)
            rows.append(
                {
                    "card_number": self.card_ids[card],
                    "merchant_id": self.merchant_ids[merchant],
                    "home_distance_km": (
                        -1.0 if missing else float(5.0 + (card * 7 + merchant) % 4000)
                    ),
                }
            )
        return [
            {"pair_features": rows},
            {"rows_returned": len(rows), "build_id": build_id},
        ]

    # The exporter passes build_id only, so the fake needs the cutoff behind
    # it. The dict itself is created per-instance in __init__.
    def register(self, build_id: str, cutoff: int) -> None:
        self._cutoffs[build_id] = cutoff

    def _cutoff_for(self, build_id: str) -> int:
        if build_id not in self._cutoffs:
            raise KeyError(f"unregistered build {build_id!r}")
        return self._cutoffs[build_id]


def main() -> None:
    from common.config import load_raw_config
    from tfgnn.tigergraph import assemble as assemble_module
    from tfgnn.tigergraph import export as export_module
    from tfgnn.tigergraph import pipeline as pipeline_module
    from tfgnn.tigergraph.client import Client
    from tfgnn.tigergraph.plan import build_plan, deduplicate, describe, load_facts

    workspace = Path(tempfile.mkdtemp(prefix="tfgnn-smoke-"))
    print(f"workspace {workspace}\n")
    try:
        config = load_raw_config()
        plan = pipeline_module.PIPELINE_ADAPTER.validate_python(
            config["stage1_pipeline"]
        )
        export_plan = export_module.EXPORT_PLAN_ADAPTER.validate_python(
            {
                **config["tigergraph_export"],
                "output_dir": str(workspace / "export"),
                "fact_page_rows": 1500,
            }
        )

        configured_source = plan["graph_features"]["source"]
        graph = FakeGraph(bipartite=configured_source == "bipartite")
        client = cast(Client, graph)

        print("--- 1. build plan -------------------------------------------")
        facts = load_facts(graph.run_installed_with_timeout("derive_build_plan", {}))
        builds = build_plan(plan["builds"], facts)  # type: ignore[arg-type]
        builds = deduplicate(builds)
        print(describe(facts, builds))
        for build in builds:
            graph.register(build.build_id, build.cutoff_event_seq)

        print("\n--- 2. pipeline ---------------------------------------------")
        runner = pipeline_module.Runner(
            client=client,
            plan=plan,
            export_plan=export_plan,
            facts=facts,
            output_root=workspace / "export",
            manifest_path=workspace / "manifest.json",
        )
        index = 0
        runner.run_call(
            index,
            pipeline_module.QueryCall("assert_cardinality", {}, "validate", "-", False),
            None,
        )
        # BOTH load gates, matching pipeline.main's validate_calls. Listing
        # only one here is how a gate ends up untested offline: its FakeGraph
        # handler is never invoked, so nothing checks that the query names its
        # invariants in keys the gate machinery recognises.
        index += 1
        runner.run_call(
            index,
            pipeline_module.QueryCall("assert_edge_stamp", {}, "validate", "-", False),
            None,
        )
        for build in builds:
            for call in pipeline_module.build_calls(plan, build):
                index += 1
                runner.run_call(index, call, build)
            print(f"  exporting dimension tables for {build.build_id}")
            export_module.export_dimensions(
                client, build, workspace / "export", export_plan
            )
        _assert_tuning_inputs_recorded(
            workspace / "manifest.json", bipartite=configured_source == "bipartite"
        )

        print("\n--- 3. fact export ------------------------------------------")
        manifest = export_module.export_fact(
            client, facts, workspace / "export", export_plan
        )
        print(
            f"  {manifest.rows:,} rows, {manifest.frauds:,} frauds, {manifest.parts} parts"
        )

        _write_fake_temporal(workspace / "export")

        print("\n--- 4. assembly ---------------------------------------------")
        # Assemble TWICE, so the rank+bin transform is a tested path rather
        # than something only the live run exercises. The raw pass goes first
        # and is thrown away; the binned pass is what training consumes,
        # matching config.yaml. The assertions below are the real test: after
        # binning, no per-entity column may still key its dimension, which is
        # the property the whole transform exists to establish.
        raw_matrix = assemble_module.assemble(
            {
                "export_dir": str(workspace / "export"),
                "matrix_dir": str(workspace / "matrix-raw"),
                "overwrite": True,
                "entity_rank_bins": 0,
            }
        )
        _assert_temporal_joined(raw_matrix)
        _assert_entity_binning(workspace / "export", bins=8)
        matrix = assemble_module.assemble(
            {
                "export_dir": str(workspace / "export"),
                "matrix_dir": str(workspace / "matrix"),
                "overwrite": True,
                "entity_rank_bins": 8,
            }
        )
        _assert_binned_matrix(raw_matrix, matrix, bins=8)
        _assert_pair_distance(matrix)
        _assert_inductive_reach(workspace)

        print("\n--- 5. training ---------------------------------------------")
        from baseline import train_xgboost

        smoke_config = dict(config)
        smoke_config["data"] = {
            "matrix_dir": str(matrix),
            "require_all_feature_columns": True,
            # POINT AT THE TEMP WORKSPACE, WHICH HAS NO EMBEDDINGS. This used
            # to be hardcoded to the project's artifacts/stage2/embeddings, so
            # on any machine that had run Stage 2 the smoke joined REAL
            # embedding tables -- keyed on real card_numbers and real
            # build_ids -- onto its synthetic matrix. Nothing matched, 128
            # all-NaN columns went in, the embedding arm trained on them, and
            # the smoke passed having tested nothing. Absent embeddings make
            # the arm skip with a NOTE, which is the honest offline state.
            "embeddings_dir": str(workspace / "stage2-embeddings"),
            # AND THE SAME FOR STAGE 4. Adding a second embedding family
            # reopened exactly the hole described above by a different door:
            # without this key the smoke would read the developer's real
            # artifacts/stage4/embeddings and train the TGN arm on a column of
            # NaN. One override per family, or the hermeticity is per-family too.
            "temporal_embeddings_dir": str(workspace / "stage4-embeddings"),
        }
        smoke_config["stage1"] = {
            "output_dir": str(workspace / "artifacts"),
            "evaluation": {"bootstrap_resamples": 40, "bootstrap_level": 0.95},
        }
        smoke_config["model"] = {
            **config["model"],
            "params": {**config["model"]["params"], "n_estimators": 120},
        }
        partitions = train_xgboost.load_partitions(smoke_config)
        _ = train_xgboost.run_all(smoke_config, partitions)

        print("\n--- 6. the other graph_features source ----------------------")
        # The A/B's other arm, through pipeline calls + dimension export +
        # assembly. Training it is the live A/B's job, not the smoke's; what
        # this pass proves is that the mode branch emits a runnable call plan,
        # its gates and advisories parse, and assemble handles the other
        # community family shape.
        other_source = "projection" if configured_source == "bipartite" else "bipartite"
        other_plan = cast(Any, dict(plan))
        other_plan["graph_features"] = {"source": other_source}
        other_graph = FakeGraph(bipartite=other_source == "bipartite")
        other_client = cast(Client, other_graph)
        for build in builds:
            other_graph.register(build.build_id, build.cutoff_event_seq)
        other_root = workspace / "export-other"
        other_runner = pipeline_module.Runner(
            client=other_client,
            plan=other_plan,
            export_plan=export_plan,
            facts=facts,
            output_root=other_root,
            manifest_path=workspace / "manifest-other.json",
        )
        index = 0
        for build in builds:
            for call in pipeline_module.build_calls(other_plan, build):
                index += 1
                other_runner.run_call(index, call, build)
            export_module.export_dimensions(
                other_client, build, other_root, export_plan
            )
        _assert_tuning_inputs_recorded(
            workspace / "manifest-other.json", bipartite=other_source == "bipartite"
        )
        _ = export_module.export_fact(other_client, facts, other_root, export_plan)
        _write_fake_temporal(other_root)
        _ = assemble_module.assemble(
            {
                "export_dir": str(other_root),
                "matrix_dir": str(workspace / "matrix-other"),
                "overwrite": True,
                "entity_rank_bins": 8,
            }
        )
        print(f"  {other_source} mode: pipeline, export and assembly all pass")

        print("\nSMOKE TEST PASSED")
        print(
            f"  queries exercised: "
            f"{len(set(graph.calls) | set(other_graph.calls))} across both modes"
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _write_fake_temporal(export_root: Path) -> None:
    """Stage 3's per-row table, in the exact shape tfgnn.tigergraph.temporal
    writes: one part per batch plus a manifest whose row count must match.

    Written from the FACT parts rather than invented, so the join key is real
    and ``_join_temporal``'s row-count assertion is a live check. A fake that
    made up its own event_seq values would left-join to all-NaN and still
    "pass" -- the same vacuous shape as a gate nothing evaluates.
    """
    import json as _json

    import pandas as pd

    from tfgnn.tigergraph.temporal import FEATURE_COLUMNS

    fact_parts = sorted((export_root / "fact").glob("part-*.parquet"))
    # event_seq ONLY. The fact table's `id` is a pyarrow large_string and
    # reading it back here trips the reader on some pyarrow builds -- and the
    # join key is event_seq anyway, since load_temporal drops `id`.
    seqs = pd.concat(
        [pd.read_parquet(p, columns=["event_seq"]) for p in fact_parts],
        ignore_index=True,
    )
    frame = pd.DataFrame({"event_seq": seqs["event_seq"]})
    frame["id"] = "t" + frame["event_seq"].astype(str)
    # A card's first transaction has undefined gaps (-1) and genuinely zero
    # counts -- the sentinel split the query header argues for. Alternating
    # rows exercise both branches.
    first = frame.index % 3 == 0
    frame["row_card_gap_seconds"] = np.where(first, -1.0, 120.0)
    frame["row_pair_gap_seconds"] = np.where(first, -1.0, 300.0)
    frame["row_card_txn_count_1h"] = np.where(first, 0, 2)
    frame["row_card_txn_count_24h"] = np.where(first, 0, 5)
    frame["row_card_amount_24h"] = np.where(first, 0.0, 42.5)
    frame["row_card_tenure_seconds"] = np.where(first, 0.0, 86400.0)
    # gap_ratio shares the -1 sentinel; amount_z is guarded by the gap, so its
    # value on a first row is irrelevant but must be present; pair_txn_count is
    # a genuine 0 on a first visit.
    frame["row_card_gap_ratio"] = np.where(first, -1.0, 1.4)
    frame["row_amount_z"] = np.where(first, -1.0, 0.8)
    frame["row_pair_txn_count"] = np.where(first, 0, 3)

    target = export_root / "temporal"
    target.mkdir(parents=True, exist_ok=True)
    half = len(frame) // 2
    frame.iloc[:half].to_parquet(target / "part-00000.parquet", index=False)
    frame.iloc[half:].to_parquet(target / "part-00001.parquet", index=False)
    _ = (target / "manifest.json").write_text(
        _json.dumps(
            {
                "query": "card_row_temporal",
                "rows": len(frame),
                "transactions": len(frame),
                "key_columns": ["id", "event_seq"],
                "feature_columns": list(FEATURE_COLUMNS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  temporal: {len(frame):,} per-row features across 2 parts")


def _assert_temporal_joined(matrix_dir: Path) -> None:
    """The temporal columns reached the matrix with real values.

    Presence alone is not enough: a left join on a mismatched key produces
    the columns and fills them with NaN, and ``finalise`` would pass. So
    check the sentinel split actually survived -- some rows at -1 (undefined
    gap on a first transaction) AND some rows positive.
    """
    import pandas as pd

    from tfgnn.tigergraph.temporal import FEATURE_COLUMNS

    parts = sorted(matrix_dir.glob("part-*.parquet"))
    frame = pd.concat(
        [pd.read_parquet(p, columns=list(FEATURE_COLUMNS)) for p in parts],
        ignore_index=True,
    )
    gaps = pd.to_numeric(frame["row_card_gap_seconds"], errors="coerce")
    if gaps.isna().all():
        raise AssertionError(
            "every row_card_gap_seconds is NaN, so the temporal join matched "
            "nothing. The key is event_seq on both sides -- check the fact "
            "export and the temporal export describe the same load."
        )
    # The sentinel is NaN BY THIS POINT, not -1: row_card_gap_seconds is in
    # SENTINEL_AT_MINUS_ONE, so apply_sentinels has already converted it. So
    # the two branches to see here are "some NaN" (undefined gap on a card's
    # first transaction, converted) and "some positive" (a real measured
    # gap). A matrix with only one of them means either the join matched
    # nothing or the sentinel conversion silently ate every value.
    if (gaps == -1).any():
        raise AssertionError(
            "row_card_gap_seconds still holds a literal -1 in the assembled "
            "matrix. It is registered in SENTINEL_AT_MINUS_ONE, so "
            "apply_sentinels should have converted it to NaN -- the "
            "registration is not taking effect."
        )
    if not (gaps.isna().any() and (gaps > 0).any()):
        raise AssertionError(
            "row_card_gap_seconds has no NaN sentinel or no positive value, "
            "so the join collapsed to one branch and the sentinel handling "
            "is untested."
        )
    print(
        f"  temporal join: {int(gaps.isna().sum()):,} rows at the sentinel "
        f"(NaN), {int((gaps > 0).sum()):,} with a measured gap"
    )


if __name__ == "__main__":
    main()
