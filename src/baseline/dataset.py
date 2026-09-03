import json
from collections.abc import Iterable
from pathlib import Path
from typing import NotRequired, SupportsFloat, SupportsInt, TypedDict, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
from pydantic import TypeAdapter

from common.config import load_config, project_root
from tfgnn.features_meta import (
    ALL_FEATURE_COLUMNS,
    ASSEMBLED_COLUMNS,
    FACT_CATEGORICAL,
    FACT_ID,
    FACT_ORDERING,
    LABEL,
    METADATA_COLUMNS,
    SPLIT,
)


class DataSource(TypedDict):
    matrix_dir: str
    require_all_feature_columns: bool
    # Where Stage 2's embedding tables live. Optional, defaulting to the
    # project-anchored path, because production has exactly one location and
    # config.yaml should not have to name it.
    #
    # IT IS OVERRIDABLE SO THE OFFLINE SMOKE CAN BE HERMETIC. It used to be
    # hardcoded, so the smoke -- which builds a synthetic matrix in a temp
    # workspace -- read the developer's REAL artifacts/stage2/embeddings.
    # Those tables key on real card_numbers and real build_ids, so on any
    # machine that had run Stage 2 the smoke joined 128 all-NaN columns and
    # trained its embedding arm on them. It passed, every time, having tested
    # nothing. The null-share gate below is what surfaced it.
    embeddings_dir: NotRequired[str]

    # Stage 4's TGN read-out, overridable for the same hermeticity reason. A
    # SEPARATE key because the two families feed different arms: pointing one
    # directory at both is what made `raw_plus_temporal_embeddings` report Stage
    # 2's R-GCN vectors under the TGN's name.
    temporal_embeddings_dir: NotRequired[str]


DATASOURCE_ADAPTER = TypeAdapter(DataSource)

# An uninitialised MinAccum<FLOAT> holds ~3.4e38.
_ACCUMULATOR_DEFAULT = 1.0e30

# Above this share of rows with no Stage 2 embedding, the tables are assumed to
# describe a different matrix rather than an unusually inductive one. See
# _attach_embeddings for why it is loose: the honest value is 0.0% and the
# failure value is near 1.0, with nothing meaningful in between.
_EMBEDDING_NULL_LIMIT = 0.60


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root() / path


def _parquet_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(
            f"the assembled matrix does not exist: {root}. Run "
            "`python -m tfgnn.tigergraph.assemble` first."
        )
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet parts below: {root}")
    return files


def _require(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"the assembled matrix is missing {label}: {missing}")


def report_memory(label: str) -> None:
    """Print resident and peak RSS.

    THE FAILURE MODE IN THIS FILE IS SIGKILL. The OOM killer leaves no traceback
    and no partial output, so the only evidence is which line printed last -- and
    across four attempts on 2026-08-04 that evidence was used to blame the merge,
    the split, `build_features` and `_validate` in turn. Printing RSS at every
    step of the load replaces inference with a number.
    """
    import resource
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is KiB on Linux and BYTES on macOS -- so the divisor to reach GiB
    # differs by 1024 between them, and getting it wrong prints a number that is
    # confidently off by three orders of magnitude. The first version of this
    # function did exactly that and reported 206.8 GiB for a fresh interpreter.
    peak = usage.ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2)
    current = peak
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            current = int(handle.read().split()[1]) * 4096 / 1024**3
    except OSError:
        pass
    print(f"  [mem] {label:<46s} now {current:6.1f} GiB  peak {peak:6.1f} GiB")


def _resident_gib() -> float | None:
    """Resident set size in GiB from /proc, or None where /proc is absent."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            return int(handle.read().split()[1]) * 4096 / 1024**3
    except OSError:
        return None


def watch_memory(step_gib: float = 2.0, interval: float = 0.25) -> None:
    """Print every new resident high-water mark, from a background thread.

    NEEDED BECAUSE THE KILL IS SIGKILL. `report_memory` prints the high-water mark
    as of the moment it is called, so a spike between two markers never appears --
    and that is precisely the window this project kept dying in. A 2026-08-04 run
    reported `peak 67.0 GiB` and was then killed in code that allocates nothing,
    which only makes sense if the real peak arrived after the print.

    SIGKILL runs no handlers and flushes no buffers, so an atexit report or a
    final summary is worthless here. The only thing that survives is output
    already written, which is why this prints as it climbs rather than at the end.
    Idempotent and daemonised, so it never holds the interpreter open.
    """
    if _resident_gib() is None:
        return
    if getattr(watch_memory, "_started", False):
        return
    watch_memory._started = True  # type: ignore[attr-defined]

    import threading

    def sample() -> None:
        import time

        high = 0.0
        while True:
            current = _resident_gib()
            if current is None:
                return
            if current >= high + step_gib:
                high = current
                print(
                    f"  [mem] high-water                                  {high:6.1f} GiB"
                )
            time.sleep(interval)

    threading.Thread(target=sample, daemon=True, name="mem-watch").start()


def _validate(frame: pd.DataFrame, require_features: bool) -> None:
    report_memory("validate: start")
    _require(frame, METADATA_COLUMNS + [LABEL, SPLIT], "required columns")
    if require_features:
        _require(frame, ALL_FEATURE_COLUMNS, "feature columns")

    if frame[FACT_ID].isna().any():
        raise ValueError(f"{FACT_ID} contains null values")
    # `duplicated()` hashes 22.5M strings; it is the one step here that is not
    # per-column, so it gets its own marker.
    report_memory(f"validate: before {FACT_ID} uniqueness")
    duplicates = int(cast(SupportsInt, frame[FACT_ID].duplicated().sum()))
    if duplicates:
        raise ValueError(f"{FACT_ID} is not unique: {duplicates:,} duplicates")
    report_memory(f"validate: after {FACT_ID} uniqueness")

    labels = set(frame[LABEL].dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{LABEL} must be binary; observed {sorted(labels)}")

    present = [name for name in ALL_FEATURE_COLUMNS if name in frame.columns]
    numeric = [name for name in present if name not in set(FACT_CATEGORICAL)]
    if not numeric:
        return

    # ============================================================
    # ONE COLUMN AT A TIME. THIS WAS THE BIGGEST ALLOCATION IN THE PROJECT.
    # ============================================================
    # It used to be `frame[numeric].to_numpy(dtype="float64", copy=False)` over
    # the whole feature matrix. `copy=False` is a HINT pandas cannot honour on a
    # mixed-dtype frame -- it must copy, and it upcasts every float32 embedding
    # column to float64 while doing it. Then isfinite/~finite/~isnan and their
    # `&` add four bool arrays of the same shape, and np.where followed by np.abs
    # add two more float64 copies:
    #
    #   R-GCN pass (196 numeric cols)   32.9 + 16.4 + 65.7 = 115 GiB
    #   TGN pass   (268 numeric cols)   44.9 + 22.5 + 89.9 = 157 GiB
    #
    # On a 138 GiB box that is the OOM, and it is why the kill landed between the
    # embedding join and the null report with no traceback. Three earlier
    # diagnoses in this session blamed the merge, the split and build_features;
    # all three were downstream of a check that allocated more than everything it
    # was checking.
    #
    # Per column the same temporaries are ~180 MB. float64 is kept deliberately:
    # casting to float32 would turn a genuine 1e40 into inf and report it as the
    # wrong fault. The offending columns are now collected in the SAME pass that
    # counts, replacing a second sweep that recomputed np.where per column.
    report_memory(f"validate: before {len(numeric)} numeric column checks")
    infinite = 0
    huge = 0
    huge_columns: list[str] = []
    for name in numeric:
        column = frame[name].to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(column)
        infinite += int((~finite & ~np.isnan(column)).sum())
        over = int((np.abs(column[finite]) > _ACCUMULATOR_DEFAULT).sum())
        if over:
            huge += over
            huge_columns.append(name)

    if infinite:
        raise ValueError(
            f"{infinite:,} numeric feature values are +/-inf. A count of zero "
            "reached a divisor somewhere; find it rather than clipping it."
        )

    if huge:
        raise ValueError(
            f"{huge:,} values exceed 1e30 in {huge_columns}. That is an "
            "uninitialised MinAccum<FLOAT> reaching the matrix, and which rows "
            "get it correlates with which entities are unseen, which "
            "correlates with time."
        )

    report_memory("validate: before null fractions")
    # Per column for the same reason: `frame[numeric].isna()` is a bool frame the
    # shape of the whole matrix, 4.1 GiB on the R-GCN pass, to produce one number
    # per column.
    nulls = pd.Series(
        {name: float(frame[name].isna().mean()) for name in numeric},
        dtype="float64",
    )
    empty = sorted(nulls.index[nulls >= 1.0].astype(str).tolist())
    if empty:
        print(
            f"  {len(empty)} feature columns are NaN on every row (their "
            "producing query did not run):"
        )
        for name in empty:
            print(f"    {name}")
    partial = nulls[(nulls > 0.0) & (nulls < 1.0)].sort_values(ascending=False)
    if len(partial):
        print("  highest partial null fractions:")
        for name, fraction in partial.head(8).items():
            print(f"    {str(name):<40s} {float(cast(SupportsFloat, fraction)):.4f}")


def _attach_embedding_family(
    frame: pd.DataFrame,
    embeddings_dir: str,
    card_prefix: str,
    merchant_prefix: str,
    family: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Join ONE embedding family if it exists, and return the columns it added.

    ======================================================================
    TWO FAMILIES SHARE THIS BODY, AND THEY USED TO SHARE THE REGISTRY TOO
    ======================================================================
    Stage 2's R-GCN tables are `cemb_`/`memb_` under artifacts/stage2/embeddings;
    Stage 4's TGN memory read-out is `temb_c`/`temb_m` under
    artifacts/stage4/embeddings. This function knew only the first: one
    directory, Stage 2's prefixes, one global registry. So
    `raw_plus_temporal_embeddings` -- the arm named for the TGN -- was fed Stage
    2's R-GCN vectors, and the number reported as the TGN's contribution was
    `raw_plus_temporal + R-GCN`. Parameterised, each arm gets its own encoder.

    Absent embeddings are NOT an error: three of the four arms do not use them,
    and Stage 1 must keep running before Stage 2 has ever been trained. When the
    tables are missing this returns the frame untouched and
    `raw_plus_embeddings` then fails with a message telling you to run Stage 2 --
    which is better than silently training a duplicate of raw_plus_graph and
    reporting it as the GNN arm.

    *** THE JOIN IS ON (key, build_id). *** The embedding for a card differs per
    snapshot, and joining on card_number alone would attach whichever build
    sorted first -- a leak whenever that build is later than the row, and it
    would silently undo the forward-chaining fix that turned the graph lift
    positive.
    """
    # Project-anchored, not cwd-anchored, and that distinction is load-bearing
    # HERE specifically: a miss returns the frame untouched, so reading the
    # wrong directory is indistinguishable from "Stage 2 has not run" -- the
    # embedding arm would quietly report a duplicate of raw_plus_graph. Every
    # other path in this module already resolves through _resolve; this one
    # was the exception.
    directory = _resolve(embeddings_dir)
    if not directory.is_dir():
        return frame, []

    added: list[str] = []
    for node_type, key_column, prefix in (
        ("Card", "card_number", card_prefix),
        ("Merchant", "merchant_id", merchant_prefix),
    ):
        parts = sorted(directory.glob(f"{node_type}__*.parquet"))
        if not parts:
            continue
        stacked = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        stacked = stacked.rename(columns={"_key": key_column})
        # NAMED, not a KeyError from .astype two lines down. Stage 4 exported a
        # single global table with no build_id at all until 2026-08-03, and the
        # failure that produced was an unexplained KeyError deep in a join.
        if "build_id" not in stacked.columns:
            raise ValueError(
                f"{family} embedding tables in {directory} have no `build_id` "
                "column. The join is on (key, build_id) because a card's "
                "embedding differs per snapshot -- an export with one global "
                "state cannot be joined without attaching post-cutoff "
                "information to earlier rows. Re-export with per-build "
                "snapshots."
            )
        stacked[key_column] = stacked[key_column].astype("string")
        stacked["build_id"] = stacked["build_id"].astype("string")

        # ONE reindex, ONE concat. Neither `frame.merge` nor a per-column loop.
        #
        # `merge` builds a whole second frame before releasing the first, so peak
        # is ~2x the result -- and the result is large: 22.5M rows x 200 float32
        # columns is 18 GB for the TGN family alone. That OOM-killed a lift run on
        # 2026-08-04 (`Killed`, SIGKILL, no traceback to read).
        #
        # The obvious replacement -- assigning one column at a time -- fixes the
        # peak and breaks something else: it inserts 200 separate blocks into the
        # BlockManager and pandas raises `PerformanceWarning: DataFrame is highly
        # fragmented ... result of calling frame.insert many times`. Every later
        # operation then pays for 200 blocks. `frame[cols] = matrix` fragments
        # identically; it is the same insert path.
        #
        # Measured on 400k x 200 float32 (pandas 3.0.3):
        #   per-column loop   203 blocks   peak +320.3 MB   PerformanceWarning
        #   frame[cols] = m   203 blocks   peak +320.2 MB   PerformanceWarning
        #   concat(axis=1)      4 blocks   peak +320.0 MB   none
        # Copy-on-write lets concat reference the block instead of copying it, so
        # the tidy form is also the cheap one. The block is 320 MB and peak is
        # 320 MB: no duplication at all.
        cols = [c for c in stacked.columns if c.startswith(prefix)]
        if not cols:
            raise ValueError(
                f"{family} tables in {directory} have no columns starting with "
                f"{prefix!r}. The families are distinguished by prefix, so a "
                "mismatch here means the wrong stage wrote this directory."
            )
        keyed = stacked.set_index([key_column, "build_id"])
        if keyed.index.has_duplicates:
            raise RuntimeError(
                f"{family}/{node_type}: a (key, build_id) is duplicated in the "
                "embedding tables, so the join would be ambiguous."
            )
        target = pd.MultiIndex.from_arrays([frame[key_column], frame["build_id"]])
        # float32 explicitly: a miss reindexes to NaN, and without the cast that
        # can promote the block to float64 -- doubling the largest thing in the
        # process for precision the model never uses.
        block = keyed[cols].reindex(target).astype("float32")
        del keyed, target, stacked
        # reindex carries `target` as the index; the concat needs frame's.
        block.index = frame.index
        before = len(frame)
        frame = pd.concat([frame, block], axis=1)
        del block
        if len(frame) != before:
            raise RuntimeError(
                f"attaching {node_type} embeddings changed the row count "
                f"{before:,} -> {len(frame):,}, which should be impossible on an "
                "index-aligned concat."
            )
        added.extend(cols)

    if added:
        null_share = float(frame[added[0]].isna().mean())
        # SIZE PRINTED, because this block is the largest thing in the process and
        # its failure mode is silent: the OOM killer sends SIGKILL, so there is no
        # traceback to read afterwards, only `Killed`.
        gib = len(frame) * len(added) * 4 / 1024**3
        print(
            f"joined {len(added)} {family} embedding columns from {directory} "
            f"(~{gib:.1f} GiB float32; {null_share:.1%} of rows have no "
            "embedding -- an entity the snapshot never saw, which is the "
            "inductive case and stays NaN)"
        )
        report_memory(f"after {family} join")
        # A GATE, not a note (2026-08-02). This number was computed and
        # printed and nothing acted on it, which made it useless in the one
        # case it exists for: embeddings that do not correspond to this
        # matrix. The join is on (key, build_id), so a mismatched set does not
        # error -- it MISSES, and a near-total miss reads as "the inductive
        # case" in the line above while the arm trains on a column of NaN and
        # reports a number.
        #
        # The threshold is deliberately loose. A genuine inductive share is
        # entities the snapshot never saw; a wrong-embeddings share is close
        # to everything. There is a wide gap between those and no reason to
        # tune inside it -- the observed value on a correct run is 0.0%.
        if null_share > _EMBEDDING_NULL_LIMIT:
            raise ValueError(
                f"{null_share:.1%} of rows got no {family} embedding, above "
                f"the {_EMBEDDING_NULL_LIMIT:.0%} limit. The join is on "
                "(card_number, build_id) and (merchant_id, build_id), so this "
                "means the embedding tables in "
                f"{directory} do not describe this matrix's "
                "builds -- most likely the exporting stage ran against a "
                "different feature build, or only some builds were exported. "
                "Re-run that stage, which clears the directory, or delete "
                f"{directory} to drop the arm entirely."
            )
    return frame, added


#: The two encoder families, by the registry each one feeds.
RGCN_FAMILY = "rgcn"
TGN_FAMILY = "tgn"
#: The pair-score views of the same two encoders. Separate families because they
#: read the same directories but attach 3 columns instead of 128/200, so an arm
#: that wants only the scores must not pay for the wide block.
RGCN_PAIR_FAMILY = "rgcn_pair"
TGN_PAIR_FAMILY = "tgn_pair"
ALL_FAMILIES = frozenset({RGCN_FAMILY, TGN_FAMILY, RGCN_PAIR_FAMILY, TGN_PAIR_FAMILY})


def probe_embedding_families(source: DataSource) -> frozenset[str]:
    """Which families have tables on disk, WITHOUT joining anything.

    The arm plan needs to know what is available before the join, because the
    join is the expensive part: attaching both families to a 22.5M-row frame is
    ~29.5 GB of float32 and no single arm reads more than one of them. Reading
    one filename per family costs nothing and lets the caller attach only what
    the planned arms will actually select.
    """
    present: set[str] = set()
    for family, pair_family, key, default in (
        (
            RGCN_FAMILY,
            RGCN_PAIR_FAMILY,
            "embeddings_dir",
            "artifacts/stage2/embeddings",
        ),
        (
            TGN_FAMILY,
            TGN_PAIR_FAMILY,
            "temporal_embeddings_dir",
            "artifacts/stage4/embeddings",
        ),
    ):
        directory = _resolve(source.get(key, default))
        if directory.is_dir() and any(directory.glob("*__*.parquet")):
            present.add(family)
            # The pair scores are DERIVED from the same tables, so availability is
            # identical -- but they are attached separately.
            present.add(pair_family)
    return frozenset(present)


def _attach_pair_scores(
    frame: pd.DataFrame,
    embeddings_dir: str,
    card_prefix: str,
    merchant_prefix: str,
    family: str,
    out_prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Add THREE pair-interaction columns instead of the whole embedding block.

    ======================================================================
    WHY THREE NUMBERS AND NOT TWO HUNDRED
    ======================================================================
    A GNN trained on link prediction encodes what it knows about a (card,
    merchant) pair in the INTERACTION between the two vectors -- for a DistMult
    decoder, literally their dot product. A gradient-boosted tree splits one
    dimension at a time and cannot form that product, so handing it the raw
    dimensions asks it to rediscover a dot product through axis-aligned splits.
    It structurally cannot, and the measured consequence is the R-GCN arm
    destroying 53% of the gain the hand-built features had earned.

    This project already learned that lesson on the neural side. Stage 4a's
    `FraudHead` takes `card_memory * merchant_memory` as an explicit input,
    with the note that "a tree cannot form an interaction between two embedding
    blocks, so handing the model the product rather than hoping it is discovered
    is the one structural lesson that transfers." Nobody applied it to the
    XGBoost arms.

    The three statistics are RELATION-FREE on purpose. A trained DistMult score
    needs the decoder's relation vector, which is not in the exported tables, so
    using it would mean re-exporting. The dot product is DistMult with the
    relation set to ones -- which is how Stage 4's `LinkHead` initialises it --
    and cosine and L2 add scale-invariant and metric views of the same pair.

    ======================================================================
    IT IS ALSO WHY THIS ARM FITS IN MEMORY
    ======================================================================
    The wide block is never joined. Both embedding tables are tiny (68,618 and
    1,027 rows), so they are read into small matrices and gathered per row in
    chunks: three output columns instead of 200, and a bounded transient rather
    than 16.8 GiB.
    """
    directory = _resolve(embeddings_dir)
    if not directory.is_dir():
        return frame, []

    vectors: dict[str, tuple[pd.Index, np.ndarray]] = {}
    for node_type, prefix in (("Card", card_prefix), ("Merchant", merchant_prefix)):
        parts = sorted(directory.glob(f"{node_type}__*.parquet"))
        if not parts:
            return frame, []
        stacked = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if "build_id" not in stacked.columns:
            raise ValueError(
                f"{family} tables in {directory} have no `build_id`; the pair "
                "score is per (key, build_id) like the embeddings themselves."
            )
        columns = [c for c in stacked.columns if c.startswith(prefix)]
        keys = pd.MultiIndex.from_arrays(
            [stacked["_key"].astype("string"), stacked["build_id"].astype("string")]
        )
        vectors[node_type] = (keys, stacked[columns].to_numpy(dtype="float32"))
        del stacked

    card_keys, card_matrix = vectors["Card"]
    merchant_keys, merchant_matrix = vectors["Merchant"]
    if card_matrix.shape[1] != merchant_matrix.shape[1]:
        raise ValueError(
            f"{family}: card vectors are {card_matrix.shape[1]}-dim and merchant "
            f"vectors {merchant_matrix.shape[1]}-dim, so no pair score is defined."
        )

    build = frame["build_id"].astype("string")
    card_row = card_keys.get_indexer(
        pd.MultiIndex.from_arrays([frame["card_number"].astype("string"), build])
    )
    merchant_row = merchant_keys.get_indexer(
        pd.MultiIndex.from_arrays([frame["merchant_id"].astype("string"), build])
    )

    rows = len(frame)
    dot = np.full(rows, np.nan, dtype="float32")
    cosine = np.full(rows, np.nan, dtype="float32")
    distance = np.full(rows, np.nan, dtype="float32")
    # A miss on either side leaves all three NaN, which is the same convention the
    # embedding block uses for an entity a snapshot never saw.
    both = (card_row >= 0) & (merchant_row >= 0)

    chunk = 1_000_000
    present = np.flatnonzero(both)
    for start in range(0, present.size, chunk):
        index = present[start : start + chunk]
        left = card_matrix[card_row[index]]
        right = merchant_matrix[merchant_row[index]]
        products = left * right
        inner = products.sum(axis=1)
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        scale = left_norm * right_norm
        dot[index] = inner
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine[index] = np.where(scale > 0, inner / scale, np.nan)
        distance[index] = np.linalg.norm(left - right, axis=1)
        del left, right, products

    added = [f"{out_prefix}_dot", f"{out_prefix}_cosine", f"{out_prefix}_l2"]
    block = pd.DataFrame(
        np.stack([dot, cosine, distance]).T,
        columns=added,
        index=frame.index,
        copy=False,
    )
    frame = pd.concat([frame, block], axis=1)
    miss = 1.0 - float(both.mean())
    print(
        f"derived {len(added)} {family} pair-score columns from {directory} "
        f"({miss:.1%} of rows miss one side and stay NaN)"
    )
    report_memory(f"after {family} pair scores")
    return frame, added


def _attach_embeddings(
    frame: pd.DataFrame,
    matrix_root: Path,
    embeddings_dir: str,
    temporal_embeddings_dir: str,
    families: frozenset[str] = ALL_FAMILIES,
) -> pd.DataFrame:
    """Join the requested embedding families and register each in its own slot.

    Stage 2's R-GCN and Stage 4's TGN are DIFFERENT ENCODERS feeding DIFFERENT
    arms, so they get different registries. Either may be absent -- four of the
    six arms use neither, and Stage 1 has to keep running before Stage 2 or
    Stage 4 has ever been trained. A missing family leaves its arm to fail with a
    message naming the stage to run, which beats silently training a duplicate of
    a neighbouring arm and reporting it as the GNN's.

    ``families`` EXISTS FOR MEMORY, and it is not a micro-optimisation. The R-GCN
    block is 128 float32 columns and the TGN block is 200; on 22.5M rows that is
    11.5 GB and 18 GB. Attaching both put 29.5 GB in one frame and the OOM killer
    took the process. No arm reads more than one family, so the trainer loads once
    per family instead.
    """
    from tfgnn.features_meta import (
        RGCN_PAIR_PREFIX,
        TEMPORAL_CARD_EMBED_PREFIX,
        TEMPORAL_MERCHANT_EMBED_PREFIX,
        TGN_PAIR_PREFIX,
        set_embedding_features,
        set_rgcn_pair_features,
        set_temporal_embedding_features,
        set_tgn_pair_features,
    )
    from tfgnn.stage2.embed import CARD_EMBED_PREFIX, MERCHANT_EMBED_PREFIX

    # Registries are CLEARED for a family that was not requested, so a stale
    # registration from an earlier load in the same process cannot make an arm
    # select columns this frame does not carry.
    if RGCN_FAMILY in families:
        frame, rgcn = _attach_embedding_family(
            frame,
            embeddings_dir,
            CARD_EMBED_PREFIX,
            MERCHANT_EMBED_PREFIX,
            "Stage 2 R-GCN",
        )
        set_embedding_features(rgcn)
    else:
        set_embedding_features([])

    if TGN_FAMILY in families:
        frame, tgn = _attach_embedding_family(
            frame,
            temporal_embeddings_dir,
            TEMPORAL_CARD_EMBED_PREFIX,
            TEMPORAL_MERCHANT_EMBED_PREFIX,
            "Stage 4 TGN",
        )
        set_temporal_embedding_features(tgn)
    else:
        set_temporal_embedding_features([])

    if RGCN_PAIR_FAMILY in families:
        frame, pair = _attach_pair_scores(
            frame,
            embeddings_dir,
            CARD_EMBED_PREFIX,
            MERCHANT_EMBED_PREFIX,
            "Stage 2 R-GCN",
            RGCN_PAIR_PREFIX,
        )
        set_rgcn_pair_features(pair)
    else:
        set_rgcn_pair_features([])

    if TGN_PAIR_FAMILY in families:
        frame, pair = _attach_pair_scores(
            frame,
            temporal_embeddings_dir,
            TEMPORAL_CARD_EMBED_PREFIX,
            TEMPORAL_MERCHANT_EMBED_PREFIX,
            "Stage 4 TGN",
            TGN_PAIR_PREFIX,
        )
        set_tgn_pair_features(pair)
    else:
        set_tgn_pair_features([])

    return frame


def load(source: DataSource, families: frozenset[str] = ALL_FAMILIES) -> pd.DataFrame:
    root = _resolve(source["matrix_dir"])
    files = _parquet_files(root)
    print(f"reading {len(files):,} assembled parts from {root}")

    manifest = root / "manifest.json"
    if manifest.is_file():
        payload = cast(
            dict[str, object], json.loads(manifest.read_text(encoding="utf-8"))
        )
        builds = payload.get("builds")
        if isinstance(builds, list):
            print("feature snapshots behind this matrix:")
            for entry in cast(list[dict[str, object]], builds):
                print(
                    f"  {entry.get('build_id')}: cutoff="
                    f"{entry.get('cutoff_event_seq')} "
                    f"({entry.get('cutoff_source')}) "
                    f"serves splits {entry.get('serves_splits')}"
                )

    columns = [name for name in ASSEMBLED_COLUMNS]
    source_dataset = pads.dataset([str(path) for path in files], format="parquet")

    # ============================================================
    # FEATURES ARE READ AS float32, NOT float64
    # ============================================================
    # Measured on 2026-08-04: the arrow table was 13.2 GiB and the frame it
    # produced 27.3 GiB, before a single embedding column. Half of that is
    # precision the model never sees -- `features.build_features` casts every
    # feature to float32 on its way to XGBoost regardless, so a float64 column in
    # this frame is a copy of a number that will be truncated later.
    #
    # ORDERING AND KEY COLUMNS ARE EXCLUDED, and that exclusion is the whole
    # safety argument. `unix_time` near 1.5e9 needs more than float32's 24 bits of
    # mantissa, and `event_seq` is the ordering every temporal guarantee in the
    # project rests on. Casting either would corrupt the split silently.
    protected = set(METADATA_COLUMNS) | set(FACT_ORDERING)
    fields: list[pa.Field] = []
    for name in columns:
        field = source_dataset.schema.field(name)
        if name not in protected and pa.types.is_float64(field.type):
            field = field.with_type(pa.float32())
        fields.append(field)
    narrowed = pa.schema(fields)
    downcast = sum(
        1
        for name in columns
        if narrowed.field(name).type != source_dataset.schema.field(name).type
    )
    if downcast:
        print(f"reading {downcast} float64 feature columns as float32")
        source_dataset = pads.dataset(
            [str(path) for path in files], format="parquet", schema=narrowed
        )

    table = source_dataset.to_table(columns=columns)
    report_memory("arrow table read")
    # self_destruct + split_blocks RELEASE EACH ARROW CHUNK AS IT CONVERTS.
    # Without them `to_pandas` holds the whole arrow table alongside the whole
    # frame -- the measured 13.2 -> 27.3 GiB step was exactly that double. The
    # table is unusable afterwards, which is why it is deleted immediately.
    frame = table.to_pandas(self_destruct=True, split_blocks=True)
    del table
    report_memory("arrow -> pandas")

    for column in [FACT_ID, "card_number", "merchant_id", "mer_cat", "build_id"]:
        if column in frame:
            frame[column] = frame[column].fillna("").astype("string")
    for column in FACT_CATEGORICAL:
        if column in frame:
            frame[column] = frame[column].fillna("").astype("string")
    frame[LABEL] = frame[LABEL].astype("int8")
    report_memory("dtypes coerced")

    frame = _attach_embeddings(
        frame,
        root,
        source.get("embeddings_dir", "artifacts/stage2/embeddings"),
        # embeddings/, never embeddings_fraud/. A fraud-objective run writes the
        # latter precisely so the lift table cannot pick up label-trained
        # vectors by default.
        source.get("temporal_embeddings_dir", "artifacts/stage4/embeddings"),
        families,
    )
    _validate(frame, source["require_all_feature_columns"])
    return frame


def main() -> None:
    source = load_config("data", DATASOURCE_ADAPTER)
    frame = load(source)
    frauds = int(cast(SupportsInt, frame[LABEL].sum()))
    rate = float(cast(SupportsFloat, frame[LABEL].mean()))
    print(f"loaded rows={len(frame):,}, columns={frame.shape[1]:,}")
    print(f"frauds={frauds:,} ({100.0 * rate:.6f}%)")
    print(
        f"contract: {len(ASSEMBLED_COLUMNS)} columns, {len(ALL_FEATURE_COLUMNS)} features"
    )


if __name__ == "__main__":
    main()
