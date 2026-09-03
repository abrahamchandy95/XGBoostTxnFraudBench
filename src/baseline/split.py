from typing import TypedDict, cast

import pandas as pd

from baseline.types import PARTS, Part, Partitions
from tfgnn.features_meta import LABEL, SPLIT, SPLIT_IDS


class SplitPlan(TypedDict):
    train_id: int
    val_id: int
    test_id: int


def split(frame: pd.DataFrame, plan: SplitPlan) -> Partitions[pd.DataFrame]:
    if SPLIT not in frame.columns:
        raise ValueError(
            f"the assembled matrix has no {SPLIT!r} column. Schema r3 replaced "
            "the is_train/is_val/is_test triplet with one signed split_id."
        )
    ids = frame[SPLIT].astype("int64")

    unset = int((ids < 0).sum())
    if unset:
        raise ValueError(
            f"{unset:,} rows have split_id < 0, which means the loader did not "
            "assign them. Under the old UINT column they would have been "
            "silently trained on as split 0. Fix the loader; do not default "
            "them here."
        )

    wanted = {"train": plan["train_id"], "val": plan["val_id"], "test": plan["test_id"]}
    unknown = sorted(set(ids.unique().tolist()) - set(wanted.values()))
    if unknown:
        raise ValueError(
            f"split_id values {unknown} are not in the configured plan "
            f"{wanted}. Every row must land in exactly one partition."
        )

    frames: dict[Part, pd.DataFrame] = {}
    for part in PARTS:
        # NO `.copy()`, AND THAT ONE CALL WAS THE LARGEST AVOIDABLE ALLOCATION IN
        # THE PIPELINE -- measured at roughly 50 GiB on the widest arm.
        #
        # `frame.loc[boolean_mask]` is ALREADY a take: it allocates fresh arrays
        # and shares nothing with the source. The trailing `.copy()` then made a
        # SECOND copy of the part, and because `BlockManager.copy(deep=True)`
        # consolidates afterwards it also built one merged float32 array while the
        # fresh per-column copies were still referenced. Measured transient for
        # the copy alone: 2.26x-2.65x the size of the part it produced, and block
        # counts confirm the consolidation (79 blocks before, 9 after).
        #
        # Verified exactly equal with assert_frame_equal(check_exact=True,
        # check_dtype=True) over this pipeline's real layout -- arrow strings,
        # int8 label, int64 ordering, float32-with-NaN, and the concatenated
        # embedding block. Nothing downstream re-pays for the block layout:
        # build_features does reset_index(drop=True) (shallow under CoW), selects
        # single columns, reads one column at a time into a preallocated array,
        # and narrows to two columns before the target-encoding .iloc. Nothing
        # writes into these frames, and under copy-on-write a later write would
        # copy only the written column, so there is no aliasing hazard either.
        frames[part] = frame.loc[ids == wanted[part]]

    partitions = Partitions(
        train=frames["train"], val=frames["val"], test=frames["test"]
    )

    covered = sum(len(partitions[part]) for part in PARTS)
    if covered != len(frame):
        raise ValueError(
            f"partitions cover {covered:,} of {len(frame):,} rows. "
            "A row is missing from every split."
        )

    print("split summary (split_id from TigerGraph):")
    for part in PARTS:
        current = partitions[part]
        positives = int(cast(int, current[LABEL].sum()))
        rate = positives / len(current) if len(current) else 0.0
        print(
            f"  {part:5s} split_id={wanted[part]} rows={len(current):>12,} "
            f"fraud={positives:>8,} ({100.0 * rate:.4f}%)"
        )
        if current.empty:
            raise ValueError(f"{part} partition is empty")
        if positives == 0:
            raise ValueError(
                f"{part} partition has no fraud examples, so its AUCPR is undefined"
            )

    ordering = [
        (part, int(cast(int, partitions[part]["event_seq"].min())))
        for part in PARTS
        if "event_seq" in frame.columns
    ]
    if len(ordering) == 3:
        starts = [value for _, value in ordering]
        if starts != sorted(starts):
            raise ValueError(
                f"splits are not in time order by event_seq: {ordering}. The "
                "comparison assumes train precedes val precedes test."
            )
        print(f"  event_seq starts in order: {ordering}")

    default_ids = dict(SPLIT_IDS)
    if wanted != default_ids:
        print(f"  ! non-default split ids in use: {wanted} (schema says {default_ids})")
    return partitions
