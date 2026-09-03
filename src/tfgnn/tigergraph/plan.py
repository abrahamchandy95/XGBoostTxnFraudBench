import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast


VAL_BOUNDARY = "val_boundary"
TEST_BOUNDARY = "test_boundary"
END_OF_DATA = "end_of_data"
_FOLD = re.compile(r"^fold:(\d+)$")


class BuildPlanError(RuntimeError):
    """derive_build_plan's output cannot support the requested plan."""


@dataclass(frozen=True)
class LoadFacts:
    """The read-only facts derive_build_plan reports about the loaded data."""

    total_transactions: int
    split_first_event_seq: dict[int, int]
    split_last_event_seq: dict[int, int]
    split_rows: dict[int, int]
    split_frauds: dict[int, int]
    fold_first_event_seq: dict[int, int]
    fold_rows: dict[int, int]
    fold_frauds: dict[int, int]


@dataclass(frozen=True)
class Build:
    """One topology snapshot and the rows that read its features.

    ``serves_from`` / ``serves_until`` narrow a build to an event_seq WINDOW
    inside its splits, which is what forward chaining needs: several training
    snapshots each serving a slice of split 0 rather than one snapshot serving
    all of it. ``None`` on both means "every row of the listed splits", the
    single-snapshot behaviour.
    """

    build_id: str
    cutoff_event_seq: int
    cutoff_source: str
    serves_splits: tuple[int, ...]
    serves_from: int | None = None
    serves_until: int | None = None

    def serves_row(self, split_id: int, event_seq: int) -> bool:
        if split_id not in self.serves_splits:
            return False
        if self.serves_from is not None and event_seq < self.serves_from:
            return False
        if self.serves_until is not None and event_seq >= self.serves_until:
            return False
        return True

    @property
    def is_windowed(self) -> bool:
        return self.serves_from is not None or self.serves_until is not None


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError as exc:
            raise BuildPlanError(f"{label} is not numeric: {value!r}") from exc
    raise BuildPlanError(f"{label} is not numeric: {value!r}")


def _int_map(value: object, label: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise BuildPlanError(f"{label} is not a map: {value!r}")
    mapping = cast(Mapping[object, object], value)
    return {
        _as_int(key, f"{label} key"): _as_int(item, f"{label}[{key!r}]")
        for key, item in mapping.items()
    }


def _merge_records(result: Sequence[object]) -> dict[str, object]:
    """derive_build_plan emits three PRINT blocks; flatten them into one dict."""
    merged: dict[str, object] = {}
    for item in result:
        if not isinstance(item, Mapping):
            continue
        for key, value in cast(Mapping[object, object], item).items():
            if isinstance(key, str):
                merged[key] = value
    if not merged:
        raise BuildPlanError(f"derive_build_plan returned nothing usable: {result!r}")
    return merged


def load_facts(result: Sequence[object]) -> LoadFacts:
    """Parse derive_build_plan's output and enforce its three zero gates."""
    record = _merge_records(result)

    for key in (
        "unknown_event_seq_MUST_BE_ZERO",
        "unset_split_id_MUST_BE_ZERO",
        "is_fraud_outside_0_1_MUST_BE_ZERO",
    ):
        if key not in record:
            raise BuildPlanError(f"derive_build_plan did not report {key}")
        count = _as_int(record[key], key)
        if count != 0:
            raise BuildPlanError(
                f"{key} is {count:,}. Every cutoff below would be computed "
                "over a subset of the data, silently. Fix the loader before "
                "building anything."
            )

    facts = LoadFacts(
        total_transactions=_as_int(record["total_transactions"], "total_transactions"),
        split_first_event_seq=_int_map(
            record["split_first_event_seq"], "split_first_event_seq"
        ),
        split_last_event_seq=_int_map(
            record["split_last_event_seq"], "split_last_event_seq"
        ),
        split_rows=_int_map(record["split_rows"], "split_rows"),
        split_frauds=_int_map(record["split_frauds"], "split_frauds"),
        fold_first_event_seq=_int_map(
            record.get("fold_first_event_seq", {}), "fold_first_event_seq"
        ),
        fold_rows=_int_map(record.get("fold_rows", {}), "fold_rows"),
        fold_frauds=_int_map(record.get("fold_frauds", {}), "fold_frauds"),
    )

    if facts.total_transactions <= 0:
        raise BuildPlanError("total_transactions is not positive")
    for split_id in (0, 1, 2):
        if facts.split_rows.get(split_id, 0) <= 0:
            raise BuildPlanError(
                f"split_id {split_id} has no rows. Stage 1 needs all three."
            )
        if facts.split_frauds.get(split_id, 0) <= 0:
            raise BuildPlanError(
                f"split_id {split_id} has no frauds, so its AUCPR is undefined."
            )
    return facts


def resolve_cutoff(token: object, facts: LoadFacts) -> tuple[int, str]:
    """Resolve a symbolic or literal cutoff against the load facts."""
    if isinstance(token, bool):
        raise BuildPlanError(f"cutoff must not be a boolean: {token!r}")
    if isinstance(token, int):
        return token, f"literal:{token}"
    if not isinstance(token, str):
        raise BuildPlanError(f"unsupported cutoff: {token!r}")

    if token == VAL_BOUNDARY:
        value = facts.split_first_event_seq.get(1)
        if value is None:
            raise BuildPlanError("no split_id 1 rows, so val_boundary is undefined")
        return value, f"{VAL_BOUNDARY}={value}"
    if token == TEST_BOUNDARY:
        value = facts.split_first_event_seq.get(2)
        if value is None:
            raise BuildPlanError("no split_id 2 rows, so test_boundary is undefined")
        return value, f"{TEST_BOUNDARY}={value}"
    if token == END_OF_DATA:
        value = facts.total_transactions + 1
        return value, f"{END_OF_DATA}={value}"

    fold = _FOLD.match(token)
    if fold is not None:
        index = int(fold.group(1))
        value = facts.fold_first_event_seq.get(index)
        if value is None:
            known = sorted(facts.fold_first_event_seq)
            raise BuildPlanError(
                f"causal_fold {index} is not in the data; folds present: {known}"
            )
        return value, f"fold:{index}={value}"

    try:
        return int(token), f"literal:{int(token)}"
    except ValueError as exc:
        raise BuildPlanError(
            f"unrecognised cutoff {token!r}; expected an integer, "
            f"{VAL_BOUNDARY}, {TEST_BOUNDARY}, {END_OF_DATA} or fold:<k>"
        ) from exc


FOLD_SOURCE_AUTO = "auto"
FOLD_SOURCE_CAUSAL = "causal_fold"
FOLD_SOURCE_EVENT_SEQ = "event_seq"
FOLD_SOURCES = (FOLD_SOURCE_AUTO, FOLD_SOURCE_CAUSAL, FOLD_SOURCE_EVENT_SEQ)


def _training_span(facts: LoadFacts) -> tuple[int, int]:
    """``[first, end)`` event_seq range of split 0. ``end`` is the val boundary."""
    first = facts.split_first_event_seq.get(0)
    if first is None:
        raise BuildPlanError("no split_id 0 rows, so training folds are undefined")
    end = facts.split_first_event_seq.get(1)
    if end is None:
        last = facts.split_last_event_seq.get(0)
        if last is None:
            raise BuildPlanError("split 0 has no last event_seq")
        end = last + 1
    return first, end


def _derived_boundaries(facts: LoadFacts, folds: int) -> tuple[list[int], str]:
    """Equal-row event_seq boundaries. The fallback, and today's behaviour.

    EQUAL-WIDTH IS EQUAL-ROW HERE, and that is a guarantee rather than an
    approximation: schema.gsql defines event_seq as "a dense deterministic rank
    over Payment_Transaction from 1, ordered by (unix_time, id)". Dense means no
    gaps, so a range of width w contains exactly w rows. No distribution needs
    to be read and no quantile needs to be estimated.
    """
    first, end = _training_span(facts)
    span = end - first
    if span < folds:
        raise BuildPlanError(
            f"split 0 holds {span} rows, which cannot be divided into {folds} "
            "folds with at least one row each"
        )
    edges = [first + round(index * span / folds) for index in range(folds)]
    edges.append(end)
    return edges, "event_seq"


def _loader_boundaries(facts: LoadFacts, folds: int) -> tuple[list[int], str]:
    """Boundaries read off the loader's ``causal_fold`` stamps.

    Preferred when available, because the loader can respect boundaries
    event_seq cannot see -- keeping a fraud burst inside one fold rather than
    cutting it in half. Every check below exists because getting one wrong
    changes which rows are withheld from training, silently.
    """
    present = sorted(facts.fold_first_event_seq)
    if present != list(range(len(present))):
        raise BuildPlanError(
            f"causal_fold indices are not contiguous from 0: {present}. A gap "
            "means the loader skipped a fold, so a boundary read off this map "
            "would not be the boundary it looks like."
        )
    if len(present) != folds:
        raise BuildPlanError(
            f"the loader stamped {len(present)} causal folds but "
            f"train_folds={folds}. These are two different requests and "
            "neither one silently wins: set train_folds to "
            f"{len(present)} to use the loader's boundaries, or "
            f"stage1_pipeline.fold_source: {FOLD_SOURCE_EVENT_SEQ} to keep the "
            "event_seq quantiles that produced the archived runs."
        )

    first, end = _training_span(facts)
    edges = [facts.fold_first_event_seq[index] for index in present]
    if edges != sorted(set(edges)):
        raise BuildPlanError(
            f"causal_fold first-event_seq ranks are not strictly increasing: "
            f"{edges}. Folds must be windows in time, not overlapping sets."
        )
    if edges[0] < first or edges[-1] >= end:
        raise BuildPlanError(
            f"causal folds span [{edges[0]:,}, {edges[-1]:,}] but split 0 is "
            f"[{first:,}, {end:,}). A fold outside the training split would "
            "give a snapshot a cutoff that reads validation rows."
        )

    covered = sum(facts.fold_rows.get(index, 0) for index in present)
    split_rows = facts.split_rows.get(0, 0)
    if covered != split_rows:
        raise BuildPlanError(
            f"the loader's causal folds cover {covered:,} rows but split 0 "
            f"holds {split_rows:,}. The {split_rows - covered:,} row "
            "difference is presumably stamped -2 as warm-up, which means every "
            "fold here wants a build -- the opposite of the convention this "
            "code implements, where fold 0 IS the warm-up. Withholding the "
            "wrong rows is silent and it changes the reported metric, so this "
            "stops rather than picks. Confirm the loader's intent, then either "
            f"set stage1_pipeline.fold_source: {FOLD_SOURCE_EVENT_SEQ} or "
            "teach _expand_train_folds the second convention."
        )

    edges.append(end)
    return edges, "causal_fold"


def fold_boundaries(
    facts: LoadFacts,
    folds: int,
    source: str = FOLD_SOURCE_AUTO,
) -> tuple[list[int], str]:
    """Training-fold boundaries, and which authority supplied them.

    Returns ``folds + 1`` ranks and a provenance label. The first rank is the
    start of the training window and the last is the val boundary, i.e. the
    first rank NOT in split 0.

    ``source`` picks the authority:

    ``causal_fold``  the loader's own fold stamps. Fails if they are unusable.
    ``event_seq``    always derive equal-row boundaries. Reproduces the archived
                     runs byte for byte, whatever the loader now stamps.
    ``auto``         prefer the loader's folds, fall back to event_seq.

    """
    if folds < 1:
        raise BuildPlanError(f"train_folds must be at least 1, got {folds}")
    if source not in FOLD_SOURCES:
        raise BuildPlanError(
            f"unknown fold_source {source!r}; expected one of {list(FOLD_SOURCES)}"
        )

    if source == FOLD_SOURCE_EVENT_SEQ:
        return _derived_boundaries(facts, folds)
    if source == FOLD_SOURCE_CAUSAL:
        if len(facts.fold_first_event_seq) < 2:
            raise BuildPlanError(
                f"fold_source: {FOLD_SOURCE_CAUSAL} was requested but the "
                f"loader stamped {len(facts.fold_first_event_seq)} causal "
                "folds. Every training row is presumably still -2. Use "
                f"fold_source: {FOLD_SOURCE_AUTO} or {FOLD_SOURCE_EVENT_SEQ}."
            )
        return _loader_boundaries(facts, folds)

    if len(facts.fold_first_event_seq) < 2:
        return _derived_boundaries(facts, folds)
    return _loader_boundaries(facts, folds)


def _expand_train_folds(
    entry: Mapping[str, object],
    facts: LoadFacts,
    folds: int,
    fold_source: str = FOLD_SOURCE_AUTO,
) -> list[Build]:
    """One training spec -> `folds - 1` windowed snapshots.

    Fold 0 is WARM-UP and gets no build. Its rows contribute topology to the
    first real snapshot and are never scored, which is exactly what schema r3
    means by causal_fold -2. There is no honest alternative: a snapshot serving
    the first fold would have a cutoff at the very first row, so every feature
    would be -1 -> NaN, and those rows would train the model to read "unknown"
    as normal. The cost is 1/folds of the training rows, and it is reported.

    ``cutoff_source`` carries which authority placed the boundary, because
    event_seq quantiles and loader folds are not interchangeable: two runs whose
    manifests both say ``train_fold:1/5`` but disagree on the rank are not
    comparable, and the manifest is the only place that difference is visible.
    """
    build_id = str(entry["build_id"])
    edges, origin = fold_boundaries(facts, folds, fold_source)
    builds: list[Build] = []
    for index in range(1, folds):
        builds.append(
            Build(
                build_id=f"{build_id}_f{index}",
                cutoff_event_seq=edges[index],
                cutoff_source=(f"train_fold:{index}/{folds}={edges[index]}({origin})"),
                serves_splits=(0,),
                serves_from=edges[index],
                serves_until=edges[index + 1],
            )
        )
    return builds


def build_plan(
    specs: Sequence[Mapping[str, object]],
    facts: LoadFacts,
    train_folds: int = 1,
    fold_source: str = FOLD_SOURCE_AUTO,
) -> list[Build]:
    """Turn the configured build list into resolved Build objects.

    ``train_folds`` > 1 replaces any spec serving ONLY split 0 with that many
    forward-chaining snapshots. See Build's docstring for the measured reason.
    ``fold_source`` picks which authority places their boundaries; see
    ``fold_boundaries``.
    """
    builds: list[Build] = []
    for entry in specs:
        if (
            train_folds > 1
            and isinstance(entry.get("serves_splits"), Sequence)
            and not isinstance(entry.get("serves_splits"), (str, bytes))
            and [
                _as_int(v, "serves_splits")
                for v in cast(Sequence[object], entry["serves_splits"])
            ]
            == [0]
        ):
            builds.extend(_expand_train_folds(entry, facts, train_folds, fold_source))
            continue
        build_id = entry.get("build_id")
        if not isinstance(build_id, str) or not build_id:
            raise BuildPlanError(f"build_id must be a non-empty string: {entry!r}")
        if "cutoff" not in entry:
            raise BuildPlanError(f"build {build_id} has no cutoff")
        cutoff, source = resolve_cutoff(entry["cutoff"], facts)

        raw_splits = entry.get("serves_splits")
        if not isinstance(raw_splits, Sequence) or isinstance(raw_splits, (str, bytes)):
            raise BuildPlanError(
                f"build {build_id} needs serves_splits as a list of split ids"
            )
        splits = tuple(
            _as_int(item, f"{build_id}.serves_splits") for item in raw_splits
        )
        if not splits:
            raise BuildPlanError(f"build {build_id} serves no split")
        for split_id in splits:
            if split_id not in (0, 1, 2):
                raise BuildPlanError(
                    f"build {build_id} serves unknown split_id {split_id}"
                )
        builds.append(
            Build(
                build_id=build_id,
                cutoff_event_seq=cutoff,
                cutoff_source=source,
                serves_splits=splits,
            )
        )

    _assert_one_snapshot_per_row(builds)
    missing = sorted({0, 1, 2} - {s for b in builds for s in b.serves_splits})
    if missing:
        raise BuildPlanError(
            f"no build serves split ids {missing}; those rows would have no "
            "features at all"
        )
    _assert_no_future_fitting(builds, facts)
    return builds


def _assert_one_snapshot_per_row(builds: Sequence[Build]) -> None:
    """No row may read two snapshots, and no snapshot pair may overlap.

    Unwindowed builds claim their whole split, so two of them on one split is
    always a conflict. Windowed builds may share a split provided their
    event_seq ranges are disjoint -- that is the whole point of forward
    chaining. A row served twice would appear twice in the matrix with two
    different feature vectors, which no downstream check would catch.
    """
    for split_id in (0, 1, 2):
        claimants = [item for item in builds if split_id in item.serves_splits]
        if len(claimants) < 2:
            continue

        unwindowed = [item for item in claimants if not item.is_windowed]
        if unwindowed and len(claimants) > 1:
            names = ", ".join(item.build_id for item in claimants)
            raise BuildPlanError(
                f"split {split_id} is served by {len(claimants)} builds "
                f"({names}) and at least one of them "
                f"({unwindowed[0].build_id}) claims the whole split. Each row "
                "must read features from exactly one snapshot."
            )

        ordered = sorted(claimants, key=lambda b: b.serves_from or 0)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            end = earlier.serves_until
            start = later.serves_from
            if end is None or start is None or end > start:
                raise BuildPlanError(
                    f"builds {earlier.build_id} and {later.build_id} both serve "
                    f"split {split_id} over overlapping event_seq ranges "
                    f"[{earlier.serves_from}, {end}) and [{start}, "
                    f"{later.serves_until}). Those rows would read two "
                    "different feature vectors."
                )


def _assert_no_future_fitting(builds: Sequence[Build], facts: LoadFacts) -> None:
    """A build must not fit features from rows at or after its splits' start.

    This is the machine-checkable half of leak rule 1. A build serving split
    k whose cutoff exceeds split k's first event_seq has fitted at least one
    feature from a row it is about to score. The check is intentionally
    permissive about the first split a build serves (a training snapshot
    legitimately spans its own window); it fires when the cutoff reaches past
    the START of a LATER split than any the build serves.
    """
    for item in builds:
        if item.serves_from is not None and item.cutoff_event_seq > item.serves_from:
            raise BuildPlanError(
                f"build {item.build_id} fits features below event_seq "
                f"{item.cutoff_event_seq} but serves rows from "
                f"{item.serves_from}. A served row's features would be fitted "
                "over a window containing that row."
            )

        latest_served = max(item.serves_splits)
        for split_id in sorted(facts.split_first_event_seq):
            if split_id <= latest_served:
                continue
            boundary = facts.split_first_event_seq[split_id]
            if item.cutoff_event_seq > boundary:
                raise BuildPlanError(
                    f"build {item.build_id} (cutoff {item.cutoff_event_seq}, "
                    f"{item.cutoff_source}) serves splits {item.serves_splits} "
                    f"but fits features from split {split_id}, which begins at "
                    f"event_seq {boundary}. That is leak rule 1: silent, and it "
                    "improves the metric. Lower the cutoff or widen "
                    "serves_splits."
                )


def deduplicate(builds: Sequence[Build]) -> list[Build]:
    """Merge builds that share a cutoff; identical cutoffs mean identical features.

    Two builds at the same cutoff compute byte-identical features, so running
    both is pure wall clock. The merged build keeps the first build_id and the
    union of the splits, and the caller is told.
    """
    merged: dict[int, Build] = {}
    order: list[int] = []
    for item in builds:
        existing = merged.get(item.cutoff_event_seq)
        if existing is None:
            merged[item.cutoff_event_seq] = item
            order.append(item.cutoff_event_seq)
            continue
        if existing.is_windowed or item.is_windowed:
            raise BuildPlanError(
                f"builds {existing.build_id} and {item.build_id} share cutoff "
                f"{item.cutoff_event_seq} but at least one serves an event_seq "
                "window. Windowed builds must not be merged."
            )
        print(
            f"  merging build {item.build_id} into {existing.build_id}: "
            f"both fit at cutoff {item.cutoff_event_seq} "
            f"({existing.cutoff_source}), so their features are identical"
        )
        merged[item.cutoff_event_seq] = Build(
            build_id=existing.build_id,
            cutoff_event_seq=existing.cutoff_event_seq,
            cutoff_source=existing.cutoff_source,
            serves_splits=tuple(
                sorted(set(existing.serves_splits) | set(item.serves_splits))
            ),
        )
    return [merged[cutoff] for cutoff in order]


def offline_plan(specs: Sequence[Mapping[str, object]]) -> list[Build]:
    """The build list with cutoffs left as their config tokens.

    For printing the call order without a connection. The real ranks come from
    ``derive_build_plan``, so ``cutoff_event_seq`` is a placeholder here and is
    deliberately 0: a plan built this way must never be executed, and 0 would
    make every strict ``event_seq < cutoff`` filter select nothing rather than
    silently select the wrong window.
    """
    builds: list[Build] = []
    for entry in specs:
        build_id = entry.get("build_id")
        raw_splits = entry.get("serves_splits")
        if not isinstance(build_id, str) or not isinstance(raw_splits, Sequence):
            raise BuildPlanError(f"malformed build entry: {entry!r}")
        builds.append(
            Build(
                build_id=build_id,
                cutoff_event_seq=0,
                cutoff_source=f"{entry.get('cutoff')} (unresolved)",
                serves_splits=tuple(
                    _as_int(item, f"{build_id}.serves_splits")
                    for item in cast(Sequence[object], raw_splits)
                ),
            )
        )
    return builds


def describe(facts: LoadFacts, builds: Sequence[Build]) -> str:
    lines = [
        f"total_transactions        {facts.total_transactions:,}",
        "split                     rows            frauds     first_event_seq",
    ]
    for split_id in sorted(facts.split_rows):
        lines.append(
            f"  {split_id}                       "
            f"{facts.split_rows[split_id]:<15,} "
            f"{facts.split_frauds.get(split_id, 0):<10,} "
            f"{facts.split_first_event_seq.get(split_id, 0):,}"
        )
    lines.append("builds")
    for item in builds:
        lines.append(
            f"  {item.build_id:<18s} cutoff={item.cutoff_event_seq:<12,} "
            f"({item.cutoff_source})  serves splits {list(item.serves_splits)}"
        )
    return "\n".join(lines)
