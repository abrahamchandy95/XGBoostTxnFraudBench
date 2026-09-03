import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd
from pydantic import TypeAdapter

from common.config import load_config, project_root
from tfgnn.features_meta import DIMENSION_TABLES, FACT_COLUMNS, PAIR_TABLES
from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.plan import Build, LoadFacts, load_facts
from tfgnn.tigergraph.settings import Settings


class ExportPlan(TypedDict):
    output_dir: str
    fact_page_rows: int
    min_page_rows: int
    timeout_s: float
    size_limit_bytes: int
    overwrite: bool


EXPORT_PLAN_ADAPTER = TypeAdapter(ExportPlan)

_FACT_RENAMES = {"id": "transaction_id"}


@dataclass(frozen=True)
class FactManifest:
    rows: int
    frauds: int
    parts: int
    rows_per_split: dict[int, int]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _bare(key: str) -> str:
    """``Cards.pagerank`` -> ``pagerank``.

    TigerGraph prefixes projected attributes with the vertex-set variable
    name, and whether it does so varies by version and by whether the PRINT
    carried an ``AS`` alias. Column names never contain a dot, so taking the
    tail is correct under either behaviour.
    """
    return key.rsplit(".", 1)[-1]


def vertex_rows(result: Sequence[object]) -> list[dict[str, Any]]:
    """Pull the single row-list out of a query result and flatten it.

    A vertex-set PRINT returns ``[{"alias": [{"v_id":..., "attributes":
    {...}}, ...]}, {scalars...}]``. Exactly one list-valued entry is expected;
    two would mean the query grew a second projection and the caller no
    longer knows which one it is reading.
    """
    lists: list[list[object]] = []
    for item in result:
        if not isinstance(item, Mapping):
            continue
        for value in cast(Mapping[object, object], item).values():
            if isinstance(value, list):
                lists.append(cast(list[object], value))

    populated = [candidate for candidate in lists if candidate]
    if not populated:
        return []
    if len(populated) > 1:
        raise ValueError(
            f"expected one row-list in the query result; found {len(populated)}"
        )

    rows: list[dict[str, Any]] = []
    for element in populated[0]:
        if not isinstance(element, Mapping):
            raise ValueError(f"unexpected row shape: {element!r}")
        record = cast(Mapping[str, object], element)
        attributes = record.get("attributes")
        flat: dict[str, Any] = {}
        if isinstance(attributes, Mapping):
            for key, value in cast(Mapping[object, object], attributes).items():
                flat[_bare(str(key))] = value
        else:
            for key, value in record.items():
                if key not in {"v_id", "v_type"}:
                    flat[_bare(str(key))] = value
        if "v_id" in record:
            flat.setdefault("v_id", record["v_id"])
        rows.append(flat)
    return rows


def _scalars(result: Sequence[object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for item in result:
        if not isinstance(item, Mapping):
            continue
        for key, value in cast(Mapping[object, object], item).items():
            if isinstance(key, str) and not isinstance(value, list):
                merged[key] = value
    return merged


def _frame(
    rows: list[dict[str, Any]],
    expected: list[str],
    label: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=expected)
    frame = pd.DataFrame.from_records(rows)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{label} did not return columns {missing}. Either the GSQL "
            f"projection changed or features_meta is stale. Returned: "
            f"{sorted(frame.columns)}"
        )
    return frame.loc[:, expected]


# ======================================================================
# dimension tables -- must run inside their own build
# ======================================================================


def build_dir(root: Path, build_id: str) -> Path:
    return root / "dimensions" / build_id


def prune_stale_builds(root: Path, keep: Sequence[str]) -> list[str]:
    """Move build directories not in `keep` out of dimensions/.

    THE BUG THIS EXISTS TO PREVENT, which cost a 3h20m rebuild once:
    ``assemble.discover_builds`` globs every directory under ``dimensions/``,
    so a build directory left over from a PREVIOUS run with a different plan is
    indistinguishable from a current one. Switching train_folds from 1 to 5
    created b_train_f1..f4 and left the old whole-split b_train in place; the
    old manifest claimed ``serves_splits: [0, 1]`` with no window, so it
    collided with all four folds AND with b_val. The coverage guard caught it,
    correctly, but only after the entire feature build had finished.

    The pipeline owns ``dimensions/``, so pruning belongs here rather than in
    assemble, which cannot know which run is current.

    MOVED, NOT DELETED. A stale export is still the only copy of a previous
    run's dimension tables, and those are 20 minutes of TigerGraph time each.
    They go to ``_stale/<build_id>-<n>`` so a mistake is recoverable.
    """
    dimensions = root / "dimensions"
    if not dimensions.is_dir():
        return []

    keeping = set(keep)
    stale_root = root / "_stale"
    moved: list[str] = []
    for directory in sorted(dimensions.iterdir()):
        if not directory.is_dir() or directory.name in keeping:
            continue
        # Only touch things that actually look like a build export, so an
        # unrelated stray directory is reported rather than silently relocated.
        if not (directory / "manifest.json").is_file():
            print(
                f"  NOTE {directory} is not a build export (no manifest.json) "
                "and was left alone. assemble skips it."
            )
            continue
        stale_root.mkdir(parents=True, exist_ok=True)
        target = stale_root / directory.name
        suffix = 1
        while target.exists():
            suffix += 1
            target = stale_root / f"{directory.name}-{suffix}"
        directory.rename(target)
        moved.append(directory.name)
        print(f"  pruned stale build export {directory.name} -> {target}")
    return moved


def export_dimensions(
    client: Client,
    build: Build,
    root: Path,
    plan: ExportPlan,
    fingerprint: str = "",
) -> dict[str, int]:
    """Export all five dimension tables for one resident build.

    ``fingerprint`` identifies the configuration these tables were produced
    under and is recorded in the manifest, so a later run can tell whether
    they are still valid for its own plan. Empty means "unknown", which the
    reuse gate treats as not reusable.
    """
    target = build_dir(root, build.build_id)
    target.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for query, (columns, key) in DIMENSION_TABLES.items():
        result = client.run_installed_with_timeout(
            query,
            {"build_id": build.build_id},
            timeout_s=plan["timeout_s"],
            size_limit=plan["size_limit_bytes"],
        )
        frame = _frame(vertex_rows(result), columns, query)
        reported = _scalars(result).get("rows_returned")

        if reported is not None and int(cast(int, reported)) != len(frame):
            raise RuntimeError(
                f"{query} reported rows_returned={reported} but returned "
                f"{len(frame)} rows"
            )
        if frame[key].duplicated().any():
            raise RuntimeError(f"{query} returned duplicate {key} values")

        path = target / f"{query}.parquet"
        frame.to_parquet(path, index=False)
        counts[query] = len(frame)
        print(f"    {query:36s} {len(frame):>12,} rows -> {path.name}")

    # The pair table. Kept out of the loop above because its key is COMPOSITE:
    # DIMENSION_TABLES' contract is one primary key per table and the
    # duplicated() check below has to run over both columns together. It is
    # exported here rather than alongside the fact table because it is
    # BUILD-SCOPED like the dimensions -- reset_build deletes
    # Has_Interaction_With_Merchant, so the distance goes with it.
    for query, (columns, key_columns) in PAIR_TABLES.items():
        result = client.run_installed_with_timeout(
            query,
            {"build_id": build.build_id},
            timeout_s=plan["timeout_s"],
            size_limit=plan["size_limit_bytes"],
        )
        frame = _frame(vertex_rows(result), columns, query)
        reported = _scalars(result).get("rows_returned")

        if reported is not None and int(cast(int, reported)) != len(frame):
            raise RuntimeError(
                f"{query} reported rows_returned={reported} but returned "
                f"{len(frame)} rows"
            )
        if frame.duplicated(subset=key_columns).any():
            raise RuntimeError(
                f"{query} returned duplicate {key_columns} pairs. "
                "Has_Interaction_With_Merchant is supposed to be deduplicated "
                "per (card, merchant, build), so this means the interaction "
                "edge grew a second row per pair and the join below would "
                "MULTIPLY transaction rows rather than annotate them."
            )

        path = target / f"{query}.parquet"
        frame.to_parquet(path, index=False)
        counts[query] = len(frame)
        print(f"    {query:36s} {len(frame):>12,} rows -> {path.name}")

    meta = {
        "build_id": build.build_id,
        "cutoff_event_seq": build.cutoff_event_seq,
        "cutoff_source": build.cutoff_source,
        "serves_splits": list(build.serves_splits),
        # The event_seq window this snapshot serves, for forward chaining.
        # null on both means "every row of the listed splits". assemble reads
        # these to route rows, so a manifest written before they existed still
        # loads and still means the same thing.
        "serves_from": build.serves_from,
        "serves_until": build.serves_until,
        "rows": counts,
        # What configuration produced these tables. The reuse gate in
        # pipeline.main compares it against the current plan's fingerprint;
        # a mismatch means the features would differ and the build must run
        # again. Absent (an older manifest) reads as "unknown" and never
        # matches, so the conservative direction is the default.
        "fingerprint": fingerprint,
    }
    _ = (target / "manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return counts


# ======================================================================
# fact table -- exported once, no build dependency
# ======================================================================


def _fact_page(
    client: Client,
    lo: int,
    hi: int,
    plan: ExportPlan,
) -> list[dict[str, Any]]:
    """Fetch [lo, hi) with adaptive halving on transport failure."""
    try:
        result = client.run_installed_with_timeout(
            "export_transaction_rows",
            {"from_event_seq": lo, "to_event_seq": hi},
            timeout_s=plan["timeout_s"],
            size_limit=plan["size_limit_bytes"],
        )
    except Exception:
        width = hi - lo
        if width <= plan["min_page_rows"]:
            raise
        middle = lo + width // 2
        print(f"      page [{lo}, {hi}) failed; halving at {middle}")
        return _fact_page(client, lo, middle, plan) + _fact_page(
            client, middle, hi, plan
        )

    rows = vertex_rows(result)
    reported = _scalars(result).get("rows_returned")
    if reported is not None and int(cast(int, reported)) != len(rows):
        raise RuntimeError(
            f"export_transaction_rows page [{lo}, {hi}) reported "
            f"rows_returned={reported} but returned {len(rows)} rows"
        )
    return rows


def export_fact(
    client: Client,
    facts: LoadFacts,
    root: Path,
    plan: ExportPlan,
) -> FactManifest:
    target = root / "fact"
    existing = sorted(target.glob("part-*.parquet")) if target.is_dir() else []
    if existing and not plan["overwrite"]:
        raise FileExistsError(
            f"{len(existing)} fact parts already exist under {target}. Set "
            "tigergraph_export.overwrite true, or move them aside: a partial "
            "overlay of two exports would pass the row-count reconciliation "
            "and still mix two pulls."
        )
    target.mkdir(parents=True, exist_ok=True)
    for stale in existing:
        stale.unlink()

    # FACT_COLUMNS is the contract's naming; export_transaction_rows projects
    # Rows.id, so the raw list is the contract with the renames undone.
    undo = {new: old for old, new in _FACT_RENAMES.items()}
    raw_expected = [undo.get(name, name) for name in FACT_COLUMNS]
    expected = list(FACT_COLUMNS)

    total = facts.total_transactions
    page = max(1, plan["fact_page_rows"])
    rows_written = 0
    frauds = 0
    per_split: dict[int, int] = {}
    part = 0

    # event_seq is dense from 1 to total inclusive.
    lo = 1
    while lo <= total:
        hi = min(lo + page, total + 1)
        rows = _fact_page(client, lo, hi, plan)
        if rows:
            frame = _frame(rows, raw_expected, "export_transaction_rows")
            frame = frame.rename(columns=_FACT_RENAMES).loc[:, expected]
            path = target / f"part-{part:06d}-{lo}-{hi}.parquet"
            frame.to_parquet(path, index=False)

            rows_written += len(frame)
            frauds += int(frame["is_fraud"].astype("int64").sum())
            for split_id, count in (
                frame["split_id"].astype("int64").value_counts().items()
            ):
                key = int(split_id)
                per_split[key] = per_split.get(key, 0) + int(count)
            print(
                f"    part-{part:06d} event_seq [{lo:,}, {hi:,}) {len(frame):>10,} rows"
            )
            part += 1

        width = hi - lo
        if hi <= total and len(rows) != width:
            raise RuntimeError(
                f"page [{lo}, {hi}) returned {len(rows)} of {width} expected "
                "rows. event_seq is not dense; assert_cardinality should have "
                "caught this."
            )
        lo = hi

    if rows_written != total:
        raise RuntimeError(
            f"fact export wrote {rows_written:,} rows, derive_build_plan "
            f"reports {total:,}"
        )

    manifest = FactManifest(
        rows=rows_written, frauds=frauds, parts=part, rows_per_split=per_split
    )
    _ = (target / "manifest.json").write_text(
        json.dumps(
            {
                "rows": manifest.rows,
                "frauds": manifest.frauds,
                "parts": manifest.parts,
                "rows_per_split": {str(k): v for k, v in sorted(per_split.items())},
                "columns": expected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the Stage-1 fact table from TigerGraph"
    )
    _ = parser.add_argument(
        "--what",
        choices=("fact",),
        default="fact",
        help=(
            "Only the fact table is exportable standalone. Dimension tables "
            "must be exported inside their own build and are written by "
            "tfgnn.tigergraph.pipeline."
        ),
    )
    _ = parser.parse_args()

    settings = Settings()
    plan = load_config("tigergraph_export", EXPORT_PLAN_ADAPTER)
    root = resolve(plan["output_dir"])
    client = Client(settings)

    facts = load_facts(client.run_installed_with_timeout("derive_build_plan", {}))
    print(f"fact table: {facts.total_transactions:,} rows -> {root / 'fact'}")
    manifest = export_fact(client, facts, root, plan)
    print(
        f"fact export complete: {manifest.rows:,} rows, {manifest.frauds:,} "
        f"frauds, {manifest.parts} parts"
    )


if __name__ == "__main__":
    main()
