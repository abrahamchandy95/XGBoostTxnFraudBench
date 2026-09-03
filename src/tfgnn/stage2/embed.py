from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class EmbedError(RuntimeError):
    pass


#: Column prefixes in the joined matrix. Deliberately distinct from Stage 1's
#: card_/mer_ so a feature list is unambiguous about which stage produced a
#: column, and so features_meta's contract check can tell them apart.
CARD_EMBED_PREFIX = "cemb_"
MERCHANT_EMBED_PREFIX = "memb_"


@dataclass(frozen=True, slots=True)
class EmbeddingTable:
    """One node type's embeddings for ONE build.

    ``build_id`` is not decoration. It is half the join key, and the reason
    Stage 1's forward-chaining fix carries over to Stage 2 unchanged.
    """

    node_type: str
    build_id: str
    keys: pd.Series
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if len(self.keys) != self.vectors.shape[0]:
            raise EmbedError(
                f"{self.node_type}/{self.build_id}: {len(self.keys)} keys for "
                f"{self.vectors.shape[0]} vectors"
            )
        if self.keys.duplicated().any():
            raise EmbedError(
                f"{self.node_type}/{self.build_id}: duplicate keys. A duplicated "
                "key fans the fact table out on join and silently multiplies "
                "row counts."
            )
        if not np.isfinite(self.vectors).all():
            raise EmbedError(
                f"{self.node_type}/{self.build_id}: non-finite values. NaN would "
                "be indistinguishable from Stage 1's 'never computed' sentinel, "
                "and inf poisons every tree split."
            )

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def to_frame(self, prefix: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            self.vectors,
            columns=[f"{prefix}{i}" for i in range(self.dim)],
        )
        frame.insert(0, "build_id", self.build_id)
        frame.insert(0, "_key", self.keys.to_numpy())
        return frame


def normalize_within_build(table: EmbeddingTable) -> EmbeddingTable:
    """Percentile-rank each dimension WITHIN this build.

    The same transform Stage 1 applies to scale-dependent entity features, and
    for the same measured reason: two snapshots do not agree on scale, so a raw
    value learned from one misapplies to another. Stage 1's rank half is what
    turned the aggregate arm from +0.0026 (P=0.62) into +0.0204 with a CI
    excluding zero.

    NOT binned, unlike Stage 1's scalar features. Binning was added there to
    destroy per-entity fingerprints and MEASURABLY FAILED at it -- 4 binned
    columns still identified 96.9% of cards. So it buys nothing here and would
    throw away resolution the GNN spent capacity learning. Forward chaining is
    the fingerprint defence; ranking is only the comparability fix.
    """
    ranked = (
        pd.DataFrame(table.vectors)
        .rank(pct=True, axis=0, method="average")
        .to_numpy(dtype=np.float32)
    )
    return EmbeddingTable(
        node_type=table.node_type,
        build_id=table.build_id,
        keys=table.keys,
        vectors=ranked,
    )


def write_tables(tables: list[EmbeddingTable], directory: Path) -> Path:
    """One parquet per (node_type, build_id), mirroring the dimension exports.

    THE DIRECTORY IS CLEARED FIRST, AND THAT IS A CORRECTNESS FIX, NOT TIDYING
    (2026-08-02). It used to ``mkdir(exist_ok=True)`` and write over whatever
    was there. The reader globs ``{node_type}__*.parquet`` and concatenates
    everything it finds (baseline/dataset.py), so files from an EARLIER run
    survived and were joined as if this run had produced them.

    The join is on ``(key, build_id)``, which defends the easy case: a stale
    file for a build_id this plan no longer has simply misses the join and
    comes back NaN. What it CANNOT defend is a stale file carrying the SAME
    build_id -- b_train_f3 from before a query rewrite, say. That joins
    cleanly and silently attaches embeddings trained on a graph that no longer
    exists. ``build_id`` is a NAME, not a fingerprint; Stage 1 has
    ``build_fingerprint`` for exactly this class of problem and the embeddings
    had no equivalent.

    Clearing makes the directory the product of exactly one Stage 2 run, which
    is the invariant the reader was already assuming.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stale = sorted(directory.glob("*__*.parquet"))
    for path in stale:
        path.unlink()
    if stale:
        print(
            f"  cleared {len(stale)} embedding table(s) from a previous run; "
            "the reader concatenates every parquet it finds, so a leftover "
            "with a matching build_id would have joined silently"
        )
    for table in tables:
        prefix = (
            CARD_EMBED_PREFIX if table.node_type == "Card" else MERCHANT_EMBED_PREFIX
        )
        path = directory / f"{table.node_type}__{table.build_id}.parquet"
        table.to_frame(prefix).to_parquet(path, index=False)
        print(
            f"  {table.node_type:<10s} {table.build_id:<14s} "
            f"{len(table.keys):>8,} rows x {table.dim} dims -> {path.name}"
        )
    return directory


def join_onto_matrix(
    matrix_part: pd.DataFrame,
    card_tables: dict[str, EmbeddingTable],
    merchant_tables: dict[str, EmbeddingTable],
) -> pd.DataFrame:
    """Join embeddings onto one Stage 1 matrix partition.

    *** JOINS ON (key, build_id), NEVER ON key ALONE. ***
    The matrix carries the build_id that served each row -- that is what
    assemble's forward-chaining routing wrote. Joining on card_number alone
    would attach whichever build's embedding happened to sort first, which is a
    silent leak whenever that build is later than the row.

    Rows whose (key, build_id) has no embedding get NaN, which is correct and
    honest: an entity the snapshot never saw has no representation, and XGBoost
    learns a default direction. That is also the inductive case, so it must not
    be filled with zeros.
    """
    if "build_id" not in matrix_part.columns:
        raise EmbedError(
            "matrix partition has no build_id column, so the join cannot be "
            "made snapshot-correct. Re-run assemble."
        )

    out = matrix_part
    for key_column, tables, prefix in (
        ("card_number", card_tables, CARD_EMBED_PREFIX),
        ("merchant_id", merchant_tables, MERCHANT_EMBED_PREFIX),
    ):
        if not tables:
            continue
        stacked = pd.concat(
            [table.to_frame(prefix) for table in tables.values()],
            ignore_index=True,
        )
        stacked = stacked.rename(columns={"_key": key_column})
        before = len(out)
        out = out.merge(stacked, how="left", on=[key_column, "build_id"])
        if len(out) != before:
            raise EmbedError(
                f"joining {key_column} embeddings changed the row count "
                f"{before:,} -> {len(out):,}. Some (key, build_id) is duplicated "
                "in the embedding tables."
            )
    return out


def embedding_columns(dim: int) -> list[str]:
    """The column names the trainer should add for the Stage 2 arm."""
    return [f"{CARD_EMBED_PREFIX}{i}" for i in range(dim)] + [
        f"{MERCHANT_EMBED_PREFIX}{i}" for i in range(dim)
    ]


def assert_snapshot_coverage(
    matrix_build_ids: set[str], tables: dict[str, EmbeddingTable]
) -> None:
    """Every build serving rows must have embeddings, or those rows are all-NaN.

    A silently missing snapshot does not error anywhere downstream -- the arm
    simply trains with a fifth of its embedding columns blank and reports a
    weaker lift, which reads as "the GNN did not help".
    """
    missing = sorted(matrix_build_ids - set(tables))
    if missing:
        raise EmbedError(
            f"no embeddings for builds {missing}, but the matrix has rows served "
            "by them. Those rows would carry NaN for every embedding column and "
            "the arm would understate the GNN."
        )
