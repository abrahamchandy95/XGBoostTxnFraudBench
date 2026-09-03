import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import SupportsFloat, SupportsInt, TypedDict, cast

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from baseline.types import PARTS, Part, Partitions, Variant
from common.config import load_config
from tfgnn.features_meta import FACT_FEATURES, LABEL, variant_columns


class TargetEncodingPlan(TypedDict):
    columns: list[str]
    smoothing: float
    n_folds: int


class FeaturePlan(TypedDict):
    amount: bool
    is_online: bool
    time_components: list[str]
    onehot: list[str]
    target_encode: TargetEncodingPlan


@dataclass
class FeatureDataset:
    X: Partitions[pd.DataFrame]
    y: Partitions[pd.Series]
    feature_names: list[str]
    # Columns removed because they carry no information on the TRAINING rows.
    # Reported per arm so the feature count in the write-up is the count the
    # model actually saw.
    dropped_columns: dict[str, list[str]] = field(
        default_factory=lambda: {"all_null": [], "constant": []}
    )


_FEATURE_PLAN_ADAPTER = TypeAdapter(FeaturePlan)
TARGET = LABEL


def load_feature_plan(path: str | Path = "config.yaml") -> FeaturePlan:
    return load_config("features", _FEATURE_PLAN_ADAPTER, path)


def _mean(series: pd.Series) -> float:
    return float(cast(SupportsFloat, series.mean()))


def _sanitize(name: object) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    return cleaned or "blank"


def _time_block(frame: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    stamps = pd.to_datetime(frame["unix_time"], unit="s", utc=True)
    output: dict[str, pd.Series] = {}
    for component in components:
        if component == "dayofweek":
            values = stamps.dt.dayofweek
        else:
            values = cast(pd.Series, getattr(stamps.dt, component))
        output[f"t_{component}"] = values.astype("int16")
    return pd.DataFrame(output, index=frame.index)


def _onehot_block(
    frame: pd.DataFrame, column: str, template: pd.Index | None
) -> pd.DataFrame:
    values = frame[column].fillna("").astype(str)
    block = pd.get_dummies(values, prefix=column, prefix_sep="=", dtype="int8")
    block.columns = pd.Index(
        [f"{column}={_sanitize(name.split('=', 1)[-1])}" for name in block.columns]
    )
    if template is not None:
        block = block.reindex(columns=template, fill_value=0)
    return block


def _te_map(
    frame: pd.DataFrame, column: str, smoothing: float, prior: float
) -> pd.Series:
    aggregate = frame.groupby(column, dropna=False)[TARGET].agg(["sum", "count"])
    return (aggregate["sum"] + smoothing * prior) / (aggregate["count"] + smoothing)


def _encode(values: pd.Series, mapping: pd.Series, default: float) -> pd.Series:
    return values.map(mapping).astype("float64").fillna(default)


def _forward_chaining_encode(
    train: pd.DataFrame,
    column: str,
    smoothing: float,
    n_folds: int,
) -> pd.Series:
    """Encode training rows from their own past only.

    Rows are ordered by event_seq and cut into ``n_folds`` contiguous blocks.
    Block k is encoded from blocks 0..k-1; block 0 has no history and stays
    NaN. Positional, not label-based, so it is safe on any index.
    """
    if "event_seq" not in train.columns:
        raise ValueError(
            "forward-chaining target encoding needs event_seq, which is the "
            "authoritative ordering. Without it the encoding cannot be causal."
        )
    order = np.argsort(train["event_seq"].to_numpy(dtype="int64"), kind="stable")
    encoded = np.full(len(train), np.nan, dtype="float64")
    blocks = np.array_split(order, max(2, n_folds))

    # NARROWED TO THE TWO COLUMNS THIS ACTUALLY READS, once, before the loop.
    #
    # The loop used to do `history = train.iloc[np.concatenate(seen)]` on the FULL
    # partition frame and then touch only `history[TARGET]` and `history[column]`.
    # `train` here is the whole assembled training partition -- every fact,
    # aggregate, graph, temporal and EMBEDDING column -- so the copy was the frame
    # width, and it grew with the fold: 10% of rows on the first pass, 90% on the
    # last. On the TGN arm that is ~33 GiB to read two columns, repeated for every
    # fold and every encoded column.
    #
    # It is also why the cost scaled with the ARM: the partition frame carries 128
    # embedding columns on the R-GCN pass and 200 on the TGN pass, so the same
    # loop got 56% more expensive for a variable it never looks at.
    narrow = train[[column, TARGET]]

    seen: list[np.ndarray] = []
    for block in blocks:
        if seen:
            history = narrow.iloc[np.concatenate(seen)]
            prior = _mean(history[TARGET])
            mapping = _te_map(history, column, smoothing, prior)
            values = train[column].iloc[block]
            encoded[block] = _encode(values, mapping, prior).to_numpy(dtype="float64")
            del history
        seen.append(block)

    unencoded = int(np.isnan(encoded).sum())
    if unencoded:
        print(
            f"    te_{column}: {unencoded:,} of {len(train):,} training rows "
            f"({100.0 * unencoded / len(train):.1f}%) are in the first block and "
            "have no history to encode from; they stay NaN. Raise "
            "features.target_encode.n_folds to shrink that share."
        )
    return pd.Series(encoded, index=train.index)


def _target_encode(
    frames: Partitions[pd.DataFrame], plan: TargetEncodingPlan
) -> tuple[Partitions[pd.DataFrame], list[str]]:
    train = frames.train
    blocks = Partitions(
        train=pd.DataFrame(index=frames.train.index),
        val=pd.DataFrame(index=frames.val.index),
        test=pd.DataFrame(index=frames.test.index),
    )
    names: list[str] = []

    for column in plan["columns"]:
        name = f"te_{column}"
        names.append(name)

        # ============================================================
        # STORED AS float32, AND THIS IS A MEMORY DECISION, NOT A PRECISION ONE.
        # ============================================================
        # The encoding is COMPUTED in float64 -- the aggregation and smoothing in
        # `_te_map` need it -- and narrowed only once it is a column.
        #
        # These two columns used to be the only float64 in the assembled feature
        # matrix, and a single float64 column promotes the WHOLE dense array
        # XGBoost builds from the frame. On the TGN arm (19.6M x 260) that is
        # 38.0 GiB instead of 19.0 GiB -- 19 GiB of padding around two columns of
        # real float64. It is what took the 2026-08-04 lift run to a 131.7 GiB
        # high-water mark and got it killed on the last of six arms.
        #
        # float32 holds ~7 significant digits; a smoothed fraud rate near 0.0014
        # keeps every digit that a tree split could use.
        blocks.train[name] = _forward_chaining_encode(
            train, column, plan["smoothing"], plan["n_folds"]
        ).astype("float32")

        # Val and test both encode from the full training window and nothing
        # later. See the module note on model-selection coupling.
        prior = _mean(train[TARGET])
        mapping = _te_map(train, column, plan["smoothing"], prior)
        blocks.val[name] = _encode(frames.val[column], mapping, prior).astype("float32")
        blocks.test[name] = _encode(frames.test[column], mapping, prior).astype(
            "float32"
        )

    return blocks, names


def _drop_uninformative(
    X: Partitions[pd.DataFrame],
    names: list[str],
    variant: Variant,
) -> tuple[Partitions[pd.DataFrame], list[str], dict[str, list[str]]]:
    """Remove columns that carry no information on the TRAINING rows.

    WHY THIS IS NOT JUST A MEMORY SAVING
    ------------------------------------
    wcc_card's header, on a saturated projection: "A constant feature is not a
    weak feature, it is a column of noise that absorbs regularisation budget."
    On the loaded data the card projection does saturate, so c_size is one
    number for every card and the whole Community family is one number for
    every transaction. Those columns cannot inform a split; they can only
    dilute the importance table and the regularisation.

    THE DECISION IS FITTED ON TRAIN ONLY
    ------------------------------------
    A column set derived from the whole matrix is a quantity fitted over data
    containing the scored rows -- leak rule 3, and the same argument that puts
    every cutoff in the GSQL. So the constancy test reads the training
    partition, and the resulting column set is APPLIED to val and test
    unchanged. A column that is constant in train and varies in test is
    dropped, correctly: the model could never have learned anything from it.

    min == max RATHER THAN nunique()
    -------------------------------
    nunique() hashes every value in every column; on 24.4M rows across ~58
    columns that is the slowest thing in this module. Every column here is
    numeric by construction, so `count == 0` is all-NaN and `min == max` is
    constant, and both come from one cheap aggregate.
    """
    train = X.train
    if train.empty:
        return X, names, {"all_null": [], "constant": []}

    # ONE PER-COLUMN PASS, not three whole-frame reductions.
    #
    # `train.count()`, `train.min()` and `train.max()` each walk the entire frame
    # and each allocate full-width intermediates -- measured at ~9.8, ~24.1 and
    # ~24.1 GiB on the widest arm, about 58 GiB to produce three numbers per
    # column. The same shape as the checks in `_validate` and the null report,
    # both of which are already per column for exactly this reason.
    #
    # min/max are computed ONLY when the column has at least one value, which
    # preserves the original short-circuit: an all-NaN column was classified by
    # `count() == 0` and never reached the `min == max` comparison.
    all_null: list[str] = []
    constant: list[str] = []
    for name in names:
        column = train[name]
        if int(cast(SupportsInt, column.count())) == 0:
            all_null.append(name)
        elif bool(column.min() == column.max()):
            constant.append(name)

    dropped = {"all_null": all_null, "constant": constant}
    remove = set(all_null) | set(constant)
    if not remove:
        return X, names, dropped

    kept = [name for name in names if name not in remove]
    if not kept:
        raise ValueError(
            f"variant {variant!r} has no informative features left: every one "
            "of its columns is constant or all-NaN on the training rows."
        )

    print(
        f"  [{variant}] dropping {len(remove)} uninformative columns, "
        f"{len(kept)} remain:"
    )
    if all_null:
        print(
            f"    all-NaN on train ({len(all_null)}): {', '.join(all_null)}\n"
            "      the producing query did not run, or the entity was never seen"
        )
    if constant:
        print(
            f"    constant on train ({len(constant)}): {', '.join(constant)}\n"
            "      one value for every training row, so no split can use it"
        )

    # DELETED IN PLACE, not selected into a new frame. `X.train.loc[:, kept]`
    # takes every kept column into fresh arrays -- ~21.2 GiB on the widest arm to
    # discard 23 columns out of 283. `__delitem__` routes through
    # `BlockManager.idelete` with `only_slice=True`, so the survivors stay views
    # on the existing blocks and only the dropped columns are released.
    #
    # Column ORDER is unchanged by deletion, and `kept` preserves `names` order,
    # so the resulting frames carry exactly the columns and the order that
    # `X.train.loc[:, kept]` produced.
    for part in PARTS:
        frame = X[part]
        for name in sorted(remove):
            del frame[name]
    return X, kept, dropped


def build_features(
    splits: Partitions[pd.DataFrame],
    plan: FeaturePlan,
    variant: Variant,
    seed: int,
) -> FeatureDataset:
    del seed  # the encoding is deterministic in event_seq order, not sampled
    frames = Partitions(
        train=splits.train.reset_index(drop=True),
        val=splits.val.reset_index(drop=True),
        test=splits.test.reset_index(drop=True),
    )
    blocks: Partitions[list[pd.DataFrame]] = Partitions(train=[], val=[], test=[])
    names: list[str] = []

    if plan["amount"]:
        for part in PARTS:
            blocks[part].append(frames[part][["amount"]].astype("float32"))
        names.append("amount")

    if plan["is_online"]:
        for part in PARTS:
            blocks[part].append(frames[part][["is_online"]].astype("float32"))
        names.append("is_online")

    if plan["time_components"]:
        for part in PARTS:
            blocks[part].append(_time_block(frames[part], plan["time_components"]))
        names.extend(f"t_{component}" for component in plan["time_components"])

    for column in plan["onehot"]:
        train_block = _onehot_block(frames.train, column, None)
        blocks.train.append(train_block)
        names.extend(train_block.columns.tolist())
        blocks.val.append(_onehot_block(frames.val, column, train_block.columns))
        blocks.test.append(_onehot_block(frames.test, column, train_block.columns))

    te_blocks, te_names = _target_encode(frames, plan["target_encode"])
    for part in PARTS:
        blocks[part].append(te_blocks[part])
    names.extend(te_names)

    # Everything the arm adds beyond the fact-table attributes handled above.
    dimension_columns = [
        column
        for column in variant_columns(variant)
        if column not in set(FACT_FEATURES)
    ]
    if dimension_columns:
        missing = sorted(set(dimension_columns) - set(frames.train.columns))
        if missing:
            raise ValueError(
                f"variant {variant!r} needs columns absent from the matrix: "
                f"{missing}. Re-run the assembly, or check that the producing "
                "queries were enabled."
            )
        for part in PARTS:
            # FILLED COLUMN BY COLUMN INTO ONE PREALLOCATED float32 ARRAY, not
            # `frames[part][cols].astype("float32")`.
            #
            # That expression makes TWO full copies: the selection materialises a
            # mixed-dtype frame, then astype materialises a float32 one. On the
            # train split of the TGN arm -- 19.6M rows x 264 dimension columns,
            # 200 of them already float32 and 64 of them float64 -- that is
            # 23.9 GiB followed by 19.3 GiB. Together with the concat, the
            # reindex below and XGBoost's own DMatrix it summed to ~134 GiB on a
            # 138 GiB box, and the OOM killer took the process with no traceback.
            # One allocation of 19.3 GiB replaces the pair.
            # ALLOCATED (ncols, nrows) AND TRANSPOSED, which is the layout pandas
            # stores internally. Built the natural way round, `pd.DataFrame(values,
            # ...)` has to copy to reach its own block layout -- a second 19.3 GiB
            # on the widest arm. `values.T` with `copy=False` hands the array over
            # directly: measured 0 MB of growth, one block, C-contiguous, and
            # `np.shares_memory` True.
            #
            # `copy=False` on the (nrows, ncols) form also avoids the copy but
            # leaves the block F-contiguous, which makes every column a strided
            # read for XGBoost's per-column extraction. The transpose avoids both.
            values = np.empty(
                (len(dimension_columns), len(frames[part])), dtype="float32"
            )
            # THE CHECKS RUN PER COLUMN, INSIDE THIS LOOP. They used to run once
            # over the whole assembled array, and that was the largest allocation
            # in the process by a wide margin -- larger than the block being
            # checked. On the R-GCN arm's train split (19.6M x 192 float32, a
            # 14 GiB block) the one-shot form allocated:
            #
            #   isfinite bool  3.5 GiB   ~finite bool  3.5 GiB
            #   ~isnan   bool  3.5 GiB   their &  bool  3.5 GiB
            #   np.where       14 GiB    np.abs        14 GiB
            #   ------------------------------------------- 56 GiB of temporaries
            #
            # That is what the OOM killer was taking, and it explains why the
            # R-GCN pass died at 10.7 GiB of embeddings on a 138 GiB box. Per
            # column the same temporaries are ~20 MB each.
            #
            # It also improves the diagnosis: the offending COLUMN is now named,
            # where the one-shot check could only say "somewhere in this block".
            for index, column in enumerate(dimension_columns):
                col = frames[part][column].to_numpy(dtype="float32", na_value=np.nan)
                finite = np.isfinite(col)
                # NaN is expected and passed through. inf and accumulator-default
                # magnitudes are not: both are silent and time-correlated.
                if bool((~finite & ~np.isnan(col)).any()):
                    raise ValueError(f"{part} column {column!r} contains +/-inf")
                if bool((np.abs(col[finite]) > 1.0e30).any()):
                    raise ValueError(
                        f"{part} column {column!r} contains "
                        "accumulator-default-sized values (>1e30)"
                    )
                values[index, :] = col
            # `values` is fully filled here and never written afterwards -- the
            # next partition rebinds it to a fresh np.empty -- so handing it over
            # without a copy is safe, and a hypothetical later write to the frame
            # would trigger copy-on-write rather than corrupt the array.
            block = pd.DataFrame(
                values.T,
                columns=dimension_columns,
                index=frames[part].index,
                copy=False,
            )
            blocks[part].append(block)
        names.extend(dimension_columns)

    def _combine(part: Part) -> pd.DataFrame:
        """Concatenate one part's blocks, reordering only if it is needed.

        `names` is extended in the same order the blocks are appended, so the
        concat already produces that column order and the unconditional
        `.reindex(columns=names)` this replaces was a no-op that copied the whole
        matrix -- 19.3 GiB on the TGN arm's train split. Checked rather than
        assumed: if a future edit appends a block without extending `names` in
        step, the reindex still runs and the arm still gets the right columns.
        """
        combined = pd.concat(blocks[part], axis=1)
        blocks[part].clear()  # the blocks are now aliased by `combined`
        if list(combined.columns) != names:
            combined = combined.reindex(columns=names)
        # A GATE, because the cost is invisible and the cause is one column.
        # XGBoost builds a DENSE array from this frame, so its dtype is the
        # promotion of every column's: one float64 doubles the whole thing.
        # 19.0 -> 38.0 GiB on the TGN arm, which is what got a run killed.
        wide = sorted(
            {
                str(name)
                for name, dtype in combined.dtypes.items()
                if dtype == np.float64
            }
        )
        if wide:
            raise ValueError(
                f"{part} feature block has float64 columns {wide}. XGBoost "
                "materialises this frame as one dense array, so a single float64 "
                "column promotes every other column and doubles the allocation. "
                "Narrow them where they are produced -- compute in float64 if the "
                "arithmetic needs it, store float32."
            )
        return combined

    X = Partitions(train=_combine("train"), val=_combine("val"), test=_combine("test"))
    y = Partitions(
        train=frames.train[TARGET].astype("int8"),
        val=frames.val[TARGET].astype("int8"),
        test=frames.test[TARGET].astype("int8"),
    )

    # PER COLUMN, for the same reason as the finiteness checks above:
    # `X.train.isna()` materialises a bool frame the shape of the whole matrix,
    # 3.5 GiB on the R-GCN arm's train split, purely to print six numbers.
    nulls = pd.Series(
        {
            name: float(X.train[name].isna().mean())
            for name in cast("list[str]", X.train.columns.tolist())
        },
        dtype="float64",
    )
    populated = nulls[(nulls > 0.0) & (nulls < 1.0)].sort_values(ascending=False)
    if len(populated):
        print(f"  [{variant}] training-set null fractions (passed to XGBoost as NaN):")
        for name, fraction in populated.head(6).items():
            print(f"    {str(name):<40s} {float(cast(SupportsFloat, fraction)):.4f}")

    # Rebound rather than reassigned: pyright treats an all-caps name as a
    # constant, and X is the conventional name for a design matrix.
    reduced, kept, dropped = _drop_uninformative(X, names, variant)
    return FeatureDataset(X=reduced, y=y, feature_names=kept, dropped_columns=dropped)
