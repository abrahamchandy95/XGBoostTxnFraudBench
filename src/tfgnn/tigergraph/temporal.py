"""Stage 3's per-row temporal export. Drives ``card_row_temporal``.

======================================================================
WHY THIS IS A MODULE AND NOT A PHASE OF pipeline.build_calls
======================================================================
``card_row_temporal`` takes no ``build_id`` and no ``cutoff_event_seq``.
Its features are windows anchored at each transaction's OWN unix_time, so
they are causal by construction and identical under every build -- running
it six times would produce six byte-identical copies.

So it belongs where ``target_edges`` belongs: a run-once step invoked from
``run_all.sh`` outside the per-build loop, not a ``QueryCall`` in the flat
list. That also keeps it out of ``reset_build``, out of ``build_fingerprint``
and out of the resume guards, none of which have anything to say about a
table that is not build-scoped.

======================================================================
THE ROW-COUNT RECONCILIATION IS THE POINT
======================================================================
Every transaction with ``event_seq > 0`` must produce exactly one temporal
row, because the query emits one per element of each card's sorted history
and every transaction has exactly one card. So the export is complete iff

    sum of rows over all batches == the transaction count

and anything less means a card fell in no residue class, or a HeapAccum
truncated a busy card's tail, or a batch silently failed. That equality is
checked here and refuses to write a manifest when it fails -- the same
"writer does not grade its own work" rule ``verify_interaction_build``
follows for the interaction edges.

``cards_truncated_MUST_BE_ZERO`` is enforced per batch on top of it, because
truncation is the one failure with a specific, actionable fix (raise
``--heap-capacity``) and the aggregate count alone would not name it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from common.config import project_root
from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.export import vertex_rows
from tfgnn.tigergraph.settings import Settings

QUERY = "card_row_temporal"

#: Emitted column order, and the contract features_meta.TEMPORAL_FEATURES
#: must agree with. ``id`` and ``event_seq`` are join/ordering keys, not
#: features: ``id`` matches the fact table's own key column, and
#: ``event_seq`` exists so the parts can be aligned to the fact export's
#: pages rather than merged as one 30M-row frame.
KEY_COLUMNS = ("id", "event_seq")
FEATURE_COLUMNS = (
    "row_card_gap_seconds",
    "row_pair_gap_seconds",
    "row_card_txn_count_1h",
    "row_card_txn_count_24h",
    "row_card_amount_24h",
    "row_card_tenure_seconds",
    "row_card_gap_ratio",
    "row_amount_z",
    "row_pair_txn_count",
)

#: One 1-hop expansion plus a per-card sort, per batch. Detached, so this is
#: the poll deadline rather than an HTTP read timeout.
DEFAULT_TIMEOUT_S = 3600.0

#: Sized by ROWS, not by cards -- see the query header. Every batch returns
#: one row per transaction it covers, so the binding constraint is
#: serialisation. 169 batches puts a batch near the fact exporter's
#: 200k-row page at the currently loaded corpus size; it is a starting
#: point, not a measurement, and --num-batches overrides it.
DEFAULT_BATCHES = 169


class TemporalExportError(RuntimeError):
    pass


def _scalar(result: list[object], key: str) -> int:
    for block in result:
        if isinstance(block, dict) and key in block:
            return int(cast("dict[str, Any]", block)[key])
    return 0


def _rows(result: list[object]) -> list[dict[str, Any]]:
    """The ``temporal_rows`` block, flattened.

    ``vertex_rows`` takes the WHOLE result and finds the single list-valued
    entry itself. An earlier version of this function dug out the
    ``temporal_rows`` key first and passed the extracted list in, which made
    ``vertex_rows`` scan each ROW for list-valued fields, find none, and
    return [] -- so a batch that had counted 166,185 rows reported zero back
    over REST and the reconciliation blamed the transport. The query was
    fine both times.

    The query's second PRINT is scalars only, so exactly one populated list
    exists and ``vertex_rows``' own "expected one row-list" check still means
    what it says. This is the same call the pair-table export makes.
    """
    return vertex_rows(result)


def export_temporal(
    client: Client,
    output_dir: Path,
    num_batches: int = DEFAULT_BATCHES,
    heap_capacity: int = 5000,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> int:
    """Run every batch, write one parquet per batch, return the row count."""
    import pandas as pd

    if num_batches < 1:
        raise TemporalExportError("--num-batches must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Same reasoning as Stage 2's embedding directory: the reader globs this
    # directory, so a leftover part from a run with a different batch count
    # would be concatenated in as though this run produced it. A stale part
    # here duplicates transaction ids rather than missing them, which the
    # row-count check would catch -- but catching it after a full export is
    # the expensive way to find out.
    for stale in sorted(output_dir.glob("part-*.parquet")):
        stale.unlink()

    total = 0
    for batch in range(num_batches):
        result = client.run_installed_detached(
            QUERY,
            {
                "source_batch": batch,
                "num_of_source_batches": num_batches,
                "heap_capacity": heap_capacity,
            },
            timeout_s=timeout_s,
        )
        truncated = _scalar(result, "cards_truncated_MUST_BE_ZERO")
        if truncated:
            raise TemporalExportError(
                f"batch {batch}: {truncated:,} card(s) have more transactions "
                f"than --heap-capacity {heap_capacity:,}, so their history was "
                "truncated and their trailing rows are missing. Raise "
                "--heap-capacity above the busiest card's transaction count "
                "and re-run; the export is rewritten from scratch, so a "
                "partial run costs nothing but time."
            )

        rows = _rows(result)
        if rows:
            frame = pd.DataFrame(rows)
            missing = [
                column
                for column in (*KEY_COLUMNS, *FEATURE_COLUMNS)
                if column not in frame.columns
            ]
            if missing:
                raise TemporalExportError(
                    f"batch {batch} returned rows without {missing}. The query "
                    "signature and temporal.FEATURE_COLUMNS have diverged; "
                    "TigerGraph ignores unknown parameters silently, so check "
                    "the installed endpoint is current."
                )
            frame = frame[[*KEY_COLUMNS, *FEATURE_COLUMNS]]
            frame.to_parquet(output_dir / f"part-{batch:05d}.parquet", index=False)
            total += len(frame)

        reported = _scalar(result, "rows_returned")
        if reported != len(rows):
            raise TemporalExportError(
                f"batch {batch}: the query counted {reported:,} rows but "
                f"{len(rows):,} came back over REST. A payload was truncated "
                "in transit; raise --num-batches so each response is smaller."
            )
        print(
            f"  batch {batch + 1:>4d}/{num_batches}  {len(rows):>9,} rows  "
            f"(total {total:,})"
        )

    return total


def _transaction_count(client: Client) -> int:
    raw = cast("object", client.conn.getVertexCount("Payment_Transaction"))
    if isinstance(raw, dict):
        values = cast("dict[str, Any]", raw).values()
        return int(sum(int(v) for v in values))
    return int(cast("int", raw))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: export per-row temporal features. Runs ONCE for the "
            "whole run -- the features take no build cutoff."
        )
    )
    _ = parser.add_argument(
        "--output-dir", type=Path, default=Path("data/stage1/export/temporal")
    )
    _ = parser.add_argument("--num-batches", type=int, default=DEFAULT_BATCHES)
    _ = parser.add_argument("--heap-capacity", type=int, default=5000)
    _ = parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    output_dir = cast("Path", args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir

    client = Client(Settings())
    expected = _transaction_count(client)
    print(f"  {expected:,} Payment_Transaction vertices; one row each expected")

    total = export_temporal(
        client,
        output_dir,
        num_batches=cast("int", args.num_batches),
        heap_capacity=cast("int", args.heap_capacity),
        timeout_s=cast("float", args.timeout_s),
    )

    if total != expected:
        raise TemporalExportError(
            f"exported {total:,} temporal rows against {expected:,} "
            "transactions. Every transaction with event_seq > 0 must yield "
            "exactly one row, so a shortfall means a card fell in no residue "
            "class or a batch failed silently, and an excess means a stale "
            "part survived. The manifest is deliberately NOT written, so "
            "assemble will refuse rather than join a partial table."
        )

    manifest = output_dir / "manifest.json"
    _ = manifest.write_text(
        json.dumps(
            {
                "query": QUERY,
                "rows": total,
                "transactions": expected,
                "key_columns": list(KEY_COLUMNS),
                "feature_columns": list(FEATURE_COLUMNS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {total:,} rows and {manifest}")


if __name__ == "__main__":
    main()
