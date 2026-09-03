"""The transaction stream Stage 4 trains on, read from the assembled matrix.

======================================================================
WHY THE MATRIX AND NOT TIGERGRAPH
======================================================================
Everything Stage 4 needs is already in `data/stage1/matrix`: the ordering key
(`event_seq`), both endpoints (`card_number`, `merchant_id`), the label, and --
critically -- Stage 3's six per-row temporal columns. Re-querying TigerGraph
would recompute what an export already reconciled row-for-row against the
transaction count.

======================================================================
STAGE 3'S FEATURES ARE THE MESSAGE VECTOR. THIS IS THE DESIGN.
======================================================================
A synthetic check on 2026-08-03 built a stream where fraud concentrated on a
card's FIRST visit to a merchant -- the 11.3x-enriched pattern Stage 3
measured -- and the memory-only TGN learned nothing from it (ROC-AUC 0.50).

The reason is structural and worth stating plainly: **TGN's memory is per-NODE,
not per-PAIR.** It tracks when this card last transacted and when this merchant
was last active. It has no slot for "when did this card last use THIS
merchant", which is `row_pair_gap_seconds` -- the single strongest feature in
the project at 15.3% of gain.

So the raw message vector carries Stage 3's columns. Pair recency then enters
the memory update directly instead of being hoped for, node recency comes free
from TGN's time encoding on top, and Stage 4 becomes a genuine SUPERSET of
Stage 3 rather than a parallel attempt -- which is what makes arm 6's lift
interpretable.

======================================================================
ORDER IS THE GUARANTEE
======================================================================
Rows are sorted by `event_seq` once, here, and never shuffled. event_seq is a
dense deterministic rank over (unix_time, id) (schema.gsql:181), so sorting on
it is sorting on time with ties already broken. Splits are contiguous ranges of
it, so train -> val -> test is also the stream order and forward chaining holds
by construction rather than by a cutoff predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import project_root
from tfgnn.features_meta import GRAPH_FEATURES, TEMPORAL_FEATURES
from tfgnn.stage4.model import NodeSpace
from tfgnn.stage4.train import StreamBatch

#: Transaction-own columns that join the message vector alongside Stage 3's.
#: Deliberately NOT the aggregate or graph families: those are per-entity and
#: build-scoped, and TGN's memory is the thing that is supposed to learn
#: per-entity state. Feeding it a precomputed entity summary would both
#: duplicate that and drag a build cutoff into a stream that has none.
FACT_MESSAGE_COLUMNS = ["amount", "is_online"]

#: WIDENED 2026-08-03, and the previous version was an UNFAIR COMPARISON I ran
#: and then reported. The TGN saw 11 columns (amount, is_online + 9 temporal)
#: while raw_plus_temporal saw 57, and I published the 0.13 AUC-PR gap as
#: though the architectures were being compared. They were not.
#:
#: The exclusion was reasoned -- aggregates and graph features are per-entity
#: and build-scoped, and TGN's memory is supposed to LEARN per-entity state, so
#: feeding it a precomputed summary duplicates that. That argument holds for the
#: aggregates. It does NOT hold for the graph block: PageRank,
#: cre_distinct_pii_count (28.1% of the graph arm's gain) and
#: pair_home_distance_km cannot be derived from an event stream by any amount of
#: memory. Withholding them and then reporting the model as losing was the
#: error.
#:
#: Streaming is unaffected: these are per-ROW after the star-schema join, so the
#: footprint stays O(batch). Only the width changes.
MESSAGE_COLUMNS = FACT_MESSAGE_COLUMNS + list(GRAPH_FEATURES) + list(TEMPORAL_FEATURES)


@dataclass(frozen=True)
class Stream:
    space: NodeSpace
    train: list[StreamBatch]
    val: list[StreamBatch]
    test: list[StreamBatch]
    raw_dim: int
    n_events: int
    # Where the matrix was read from, so the caller can find its manifest.json
    # without re-deriving this module's default. 4b's per-build export needs the
    # build cutoffs recorded there.
    matrix_dir: Path
    # The ORIGINAL keys, in index order. Carried so the 4b export can label
    # each memory row -- a positional assumption at export time would give
    # every entity another entity's vector, which trains and scores cleanly.
    card_keys: list[str]
    merchant_keys: list[str]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def _batches(
    frame: pd.DataFrame,
    columns: list[str],
    batch_size: int,
    centre: np.ndarray,
    scale: np.ndarray,
) -> list[StreamBatch]:
    if frame.empty:
        return []
    card = torch.from_numpy(frame["_card_ix"].to_numpy(dtype="int64"))
    merchant = torch.from_numpy(frame["_merchant_ix"].to_numpy(dtype="int64"))
    # LONG, not float: TGNMemory index-puts into a Long last_update buffer.
    event = torch.from_numpy(frame["event_seq"].to_numpy(dtype="int64"))
    label = torch.from_numpy(frame["is_fraud"].to_numpy(dtype="int64"))

    # NaN is meaningful here -- row_pair_gap_seconds is NaN exactly when this
    # is a first visit, which is the 11.3x-enriched case. A neural net cannot
    # consume NaN, so it becomes 0 PLUS an explicit indicator column, the same
    # trick model.encode_with_missingness uses in Stage 2. Dropping the
    # indicator would tell the model a first visit had a zero-second gap.
    # STANDARDISED, and the statistics come from TRAIN ONLY (see load_stream).
    # Unscaled, row_card_tenure_seconds reaches ~1e6 while is_online is 0/1,
    # and the larger column dominates the first linear layer's gradients: the
    # first run of this loop showed a loss of 834 falling to 442 while AUC-PR
    # went nowhere. Fitting the scaler on the whole stream would read val and
    # test distributions, which is the same class of leak this project spends
    # its effort avoiding, so the centre/scale are passed in.
    values = frame[columns].to_numpy(dtype="float32")
    missing = np.isnan(values).astype("float32")
    filled = (np.nan_to_num(values, nan=0.0) - centre) / scale
    raw = torch.from_numpy(np.concatenate([filled, missing], axis=1))

    return [
        StreamBatch(
            card=card[start : start + batch_size],
            merchant=merchant[start : start + batch_size],
            event_time=event[start : start + batch_size],
            raw=raw[start : start + batch_size],
            label=label[start : start + batch_size],
        )
        for start in range(0, len(frame), batch_size)
    ]


def load_stream(
    matrix_dir: str | Path = "data/stage1/matrix",
    batch_size: int = 4096,
    limit: int | None = None,
) -> Stream:
    """Read the matrix, index the entities, and cut it into ordered batches."""
    root = _resolve(matrix_dir)
    parts = sorted(root.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"no matrix parts below {root}. Run "
            "`python -m tfgnn.tigergraph.assemble` first."
        )

    keep = ["card_number", "merchant_id", "event_seq", "split_id", "is_fraud"]
    missing_cols = [c for c in MESSAGE_COLUMNS if c not in _columns(parts[0])]
    if missing_cols:
        raise ValueError(
            f"the matrix has no {missing_cols}. Stage 4's message vector is "
            "Stage 3's temporal block -- export it with "
            "`python -m tfgnn.tigergraph.temporal` and re-assemble."
        )

    frame = pd.concat(
        [pd.read_parquet(p, columns=keep + MESSAGE_COLUMNS) for p in parts],
        ignore_index=True,
    )
    # ONE sort, here, and never a shuffle. See the module header.
    frame = frame.sort_values("event_seq", kind="mergesort").reset_index(drop=True)
    if limit is not None:
        frame = frame.iloc[:limit].copy()

    cards = pd.Index(frame["card_number"].astype("string").unique())
    merchants = pd.Index(frame["merchant_id"].astype("string").unique())
    frame["_card_ix"] = cards.get_indexer(frame["card_number"].astype("string"))
    frame["_merchant_ix"] = merchants.get_indexer(frame["merchant_id"].astype("string"))
    if int(frame["_card_ix"].min()) < 0 or int(frame["_merchant_ix"].min()) < 0:
        raise RuntimeError("an endpoint failed to index; check for null keys")

    space = NodeSpace(n_cards=len(cards), n_merchants=len(merchants))
    split = frame["split_id"].to_numpy(dtype="int64")

    # DROP UNINFORMATIVE COLUMNS, exactly as the XGBoost arms do. Widening the
    # message vector to all 37 GRAPH_FEATURES on 2026-08-03 handed the model 23
    # dead columns -- 7 all-NaN (louvain, k-core, card_interval are disabled)
    # and 16 constant (the WCC community collapse). With the missingness
    # doubling that was ~46 of 96 input slots carrying nothing, and the run
    # scored 0.1888 against memory-only's 0.3964. The `Mean of empty slice`
    # warning was the all-NaN columns announcing themselves.
    #
    # baseline.train_xgboost drops these per arm and prints what it dropped;
    # not doing the same here was the difference between a fair comparison and
    # a diluted one.
    train_values = frame.loc[split == 0, MESSAGE_COLUMNS].to_numpy(dtype="float32")
    with np.errstate(invalid="ignore"):
        col_min = np.nanmin(train_values, axis=0)
        col_max = np.nanmax(train_values, axis=0)
    all_nan = np.isnan(col_min)
    constant = ~all_nan & (col_max - col_min < 1e-9)
    keep = ~(all_nan | constant)
    dropped = [c for c, k in zip(MESSAGE_COLUMNS, keep, strict=True) if not k]
    columns = [c for c, k in zip(MESSAGE_COLUMNS, keep, strict=True) if k]
    if dropped:
        print(
            f"  dropping {len(dropped)} uninformative message columns, "
            f"{len(columns)} remain:"
        )
        print(f"    {', '.join(dropped)}")

    train_values = train_values[:, keep]
    centre = np.nan_to_num(np.nanmean(train_values, axis=0), nan=0.0)
    scale = np.nan_to_num(np.nanstd(train_values, axis=0), nan=1.0)
    scale[scale < 1e-6] = 1.0

    stream = Stream(
        space=space,
        train=_batches(frame[split == 0], columns, batch_size, centre, scale),
        val=_batches(frame[split == 1], columns, batch_size, centre, scale),
        test=_batches(frame[split == 2], columns, batch_size, centre, scale),
        # filled values + missingness indicators
        raw_dim=len(columns) * 2,
        n_events=len(frame),
        matrix_dir=root,
        card_keys=[str(k) for k in cards],
        merchant_keys=[str(k) for k in merchants],
    )
    print(
        f"stream: {stream.n_events:,} events, {space.n_cards:,} cards, "
        f"{space.n_merchants:,} merchants, raw_dim {stream.raw_dim}"
    )
    print(
        f"  batches  train {len(stream.train):,}  val {len(stream.val):,}  "
        f"test {len(stream.test):,}"
    )
    return stream


def _columns(part: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(part).schema.names)
