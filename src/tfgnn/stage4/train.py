"""Stage 4 training: an ordered pass over the transaction stream.

======================================================================
NO SHUFFLING. NO SAMPLING. BOTH ABSENCES ARE DELIBERATE.
======================================================================
Every other trainer in this repo shuffles -- Stage 2 draws a fresh permutation
per epoch, and that is correct there because its sampler enforces the temporal
rule per seed. Here the temporal rule IS THE STREAM ORDER, so a shuffle does
not merely reduce quality, it destroys the guarantee silently and leaves an
excellent loss curve behind. `assert_ordered` runs on every batch.

There is also no neighbour sampling, and that is not an omission. TGN's memory
is an O(1) lookup of two rows per transaction: no k-hop expansion, no fan-out
cap, no hub problem. Stage 2 needed caps because one merchant reaches 51% of
cards and any 2-hop walk from a card touches most of the graph. That failure
mode does not exist for a memory read. If the paper's graph-attention module is
added later it uses PyG's LastNeighborLoader, which keeps the last N neighbours
per node -- bounded by construction rather than by a tuned cap.

======================================================================
pos_weight IS NOT OPTIONAL AT 0.13% PREVALENCE
======================================================================
Unweighted BCE on this label spends its capacity confidently predicting zero,
reaches a low loss, and ranks nothing. `pos_weight = negatives / positives` is
computed from the TRAIN split only -- computing it over the whole stream would
read val and test prevalence, which is a leak of exactly the kind this project
spends most of its effort avoiding.

======================================================================
THE METRIC STAGE 2 NEVER HAD
======================================================================
Stage 2 reported BCE loss and nothing else, so its -0.0231 could not be read as
"the model learned nothing" versus "the model learned well and XGBoost already
had the signal". This one reports AUC-PR and ROC-AUC on val and test directly,
so a zero lift in the 4b arm is interpretable rather than ambiguous.

AUC-PR is the honest metric at 0.13% prevalence; ROC-AUC is reported beside it
because it is the one everyone asks for, and the gap between them on imbalanced
data is itself informative.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from tfgnn.stage4.model import TemporalFraudModel, assert_ordered


@dataclass
class StreamBatch:
    """One time-ordered slice of the transaction stream."""

    card: Tensor
    merchant: Tensor
    event_time: Tensor
    raw: Tensor
    label: Tensor

    def __len__(self) -> int:
        return int(self.card.numel())

    def to(self, device: torch.device) -> StreamBatch:
        return StreamBatch(
            card=self.card.to(device),
            merchant=self.merchant.to(device),
            event_time=self.event_time.to(device),
            raw=self.raw.to(device),
            label=self.label.to(device),
        )


@dataclass
class EpochReport:
    loss: float
    steps: int
    events: int
    val_aucpr: float = float("nan")
    val_roc_auc: float = float("nan")
    # THE NUMBER THIS PROJECT HAS ARGUED FROM WITHOUT EVER MEASURING. Every
    # latency verdict since Stage 2 -- including the one that rejected a GSQL
    # sampler as "I/O bound against a ~10 ms train step" -- rested on a 10 ms
    # figure nobody timed. Measured, it is what decides whether TigerGraph can
    # serve a sampler per batch or only per window.
    median_step_ms: float = float("nan")
    wall_seconds: float = float("nan")
    extra: dict[str, float] = field(default_factory=dict)


def positive_weight(labels: Tensor) -> float:
    """negatives / positives, from the TRAIN split only.

    Guarded at both ends: a split with no positives would divide by zero, and a
    split with no negatives would return zero and silently disable the
    weighting.
    """
    positives = float(labels.sum())
    negatives = float(labels.numel()) - positives
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError(
            f"cannot weight a split with {positives:.0f} positives and "
            f"{negatives:.0f} negatives; check the label column reached the "
            "stream."
        )
    return negatives / positives


def _metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """AUC-PR and ROC-AUC, via the shared metrics module.

    Imported lazily so this file can be read and type-checked without sklearn,
    and reused rather than reimplemented so Stage 4's numbers are computed the
    same way as every other stage's -- which is the whole point of
    baseline.metrics existing.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if labels.min() == labels.max():
        return float("nan"), float("nan")
    return (
        float(average_precision_score(labels, scores)),
        float(roc_auc_score(labels, scores)),
    )


def train_epoch(
    model: TemporalFraudModel,
    batches: list[StreamBatch],
    optimizer: torch.optim.Optimizer,
    pos_weight: float,
    device: torch.device,
    log_every: int = 200,
) -> EpochReport:
    """One ordered pass. Memory is reset first and detached between batches."""
    model.train()
    model.reset()

    # Balanced under the link objective, so pos_weight would distort it.
    criterion = (
        nn.BCEWithLogitsLoss()
        if model.objective == "link"
        else nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    )

    total_loss = 0.0
    steps = 0
    events = 0
    step_ms: list[float] = []
    started = time.perf_counter()

    for batch in batches:
        step_began = time.perf_counter()
        item = batch.to(device)
        # The stream order IS the guarantee -- check it every batch rather
        # than trusting the loader.
        assert_ordered(item.event_time)

        logits = model.step(item.card, item.merchant, item.event_time, item.raw)
        if model.objective == "link":
            # D5: the fraud label is NEVER read here. Targets are structural --
            # the first half are real pairs, the second half sampled negatives,
            # in the order model.step concatenates them.
            n = len(item)
            target = torch.cat(
                [torch.ones(n, device=device), torch.zeros(n, device=device)]
            )
        else:
            target = item.label.float()
        loss = criterion(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Cut the graph, keep the values. Without this, backprop walks every
        # event since the epoch began and the run dies partway through.
        model.detach()

        if device.type == "cuda":
            # Kernels are async; without this the timing measures queue-time.
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - step_began) * 1000.0)

        total_loss += float(loss.detach())
        steps += 1
        events += len(item)
        if log_every and steps % log_every == 0:
            print(
                f"    step {steps:>6d}  loss {total_loss / steps:.4f}  "
                f"events {events:>10,}"
            )

    if steps == 0:
        raise ValueError("the training stream produced no batches")
    ordered = sorted(step_ms)
    return EpochReport(
        loss=total_loss / steps,
        steps=steps,
        events=events,
        median_step_ms=ordered[len(ordered) // 2] if ordered else float("nan"),
        wall_seconds=time.perf_counter() - started,
    )


@torch.no_grad()
def evaluate(
    model: TemporalFraudModel,
    batches: list[StreamBatch],
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Score a held-out stream. MEMORY KEEPS UPDATING -- deliberately.

    This looks wrong and is not. Evaluation continues the stream: a val
    transaction is scored from memory built out of everything strictly before
    it, which includes earlier val transactions. That is exactly the
    at-inference-time situation the user asked for -- "catch the fraud when it
    happens" means the model has seen everything up to that moment.

    What must NOT happen is a gradient step, and eval mode plus no_grad is what
    stops it. The label is never read here; it is only returned for scoring.
    """
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in batches:
        item = batch.to(device)
        assert_ordered(item.event_time)
        logits = model.step(item.card, item.merchant, item.event_time, item.raw)
        if model.objective == "link":
            n = len(item)
            all_scores.append(logits.detach().cpu().numpy())
            all_labels.append(np.concatenate([np.ones(n), np.zeros(n)]))
        else:
            all_scores.append(logits.detach().cpu().numpy())
            all_labels.append(item.label.detach().cpu().numpy())

    scores = np.concatenate(all_scores) if all_scores else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    aucpr, roc_auc = _metrics(labels, scores)
    return aucpr, roc_auc, labels, scores


def rebuild_memory(
    model: TemporalFraudModel,
    batches: list[StreamBatch],
    device: torch.device,
) -> None:
    """Replay a stream through memory with no gradient and no scoring.

    NEEDED AFTER EARLY STOPPING, and this is the TGN-specific trap. Restoring
    the best WEIGHTS is not enough: memory is state, and the memory sitting in
    the model was built by the LAST epoch's weights, not the best ones. The 4b
    embedding export reads that memory directly, so without this replay it
    would export vectors produced by weights that were thrown away -- which
    trains, scores and exports without any error.
    """
    model.eval()
    model.reset()
    with torch.no_grad():
        for batch in batches:
            item = batch.to(device)
            _ = model.step(item.card, item.merchant, item.event_time, item.raw)


def plan_snapshots(
    batches: list[StreamBatch],
    cutoffs: list[tuple[str, int, int | None]],
) -> list[tuple[str, int, int | None, int | None]]:
    """Resolve where each build's read-out will fire, WITHOUT training.

    ADDED BECAUSE THE CHECK USED TO RUN AFTER `fit`. A 2026-08-04 link run spent
    22 epochs -- about two and a half hours -- and then aborted in the export
    pass on an ordering violation that is a pure function of the batch boundaries
    and the manifest. Nothing about it needed a trained model. Called from run.py
    before the model is built, it fails in seconds instead.

    Returns (build_id, cutoff, serves_from, firing_boundary), where the boundary
    is the event_seq that memory will stand strictly below when the read-out is
    taken, or None if the cutoff sits past the end of the replayed stream (that
    build gets the final state, which is safe -- the replay covers train and val
    only).
    """
    bounds = [(int(b.event_time[0]), int(b.event_time[-1])) for b in batches]
    resolved: list[tuple[str, int, int | None, int | None]] = []
    for build_id, cutoff, serves_from in sorted(cutoffs, key=lambda i: i[1]):
        boundary: int | None = None
        for first, last in bounds:
            if cutoff <= last:
                boundary = first
                break
        resolved.append((build_id, cutoff, serves_from, boundary))
    return resolved


@torch.no_grad()
def replay_with_snapshots(
    model: TemporalFraudModel,
    batches: list[StreamBatch],
    cutoffs: list[tuple[str, int, int | None]],
    device: torch.device,
    score_from: int | None = None,
) -> tuple[dict[str, tuple[Tensor, Tensor]], float, float]:
    """One ordered replay that reads memory out as the stream crosses each
    build cutoff.

    ======================================================================
    WHY 4b NEEDS THIS AND A SINGLE FINAL EXPORT WILL NOT DO
    ======================================================================
    The lift table joins embeddings on ``(key, build_id)``, never on key alone,
    because a card's embedding differs per snapshot and joining on the key
    attaches whichever build sorted first -- a leak whenever that build is later
    than the row (baseline/dataset.py:194). Stage 4 was exporting ONE global
    table with no ``build_id`` at all, which cannot satisfy that contract.

    Worse, it exported that table AFTER scoring test, so the memory it wrote had
    already consumed every test event. Joined back onto the matrix, every row's
    embedding columns encoded the whole test period -- future information
    relative to the row, on the one arm whose entire purpose is to be comparable.

    Here memory is read out at each cutoff as the ordered stream reaches it, so
    build ``b``'s vectors depend only on events below ``b``'s cutoff. That is the
    same forward-chaining rule the GSQL feature builds follow.

    BATCH GRANULARITY IS THE ONE APPROXIMATION, and it usually errs safe: the
    snapshot fires on the first batch whose opening event_seq is at or past the
    cutoff, so memory is typically missing up to one batch of events that sit
    just below it rather than containing any above it.

    "USUALLY" IS NOT "ALWAYS", WHICH IS WHY ``serves_from`` IS CHECKED. If the
    stream has a gap wider than one batch just above a cutoff, the firing point
    can land past the first row that build serves -- and then the build's vectors
    encode events at or after the rows they are attached to, which is a leak that
    joins cleanly and reports a better number. ``plan.py`` already guarantees
    ``cutoff <= serves_from`` for the GSQL features; this asserts the same
    property for the memory read-out, which is a different mechanism and needs
    its own check.

    ``score_from`` scores the tail of the stream from that batch index onward, so
    the val split can be replayed and scored in the same pass rather than in a
    second one.
    """
    model.eval()
    model.reset()

    pending = sorted(cutoffs, key=lambda item: item[1])
    snapshots: dict[str, tuple[Tensor, Tensor]] = {}

    def take(build_id: str) -> None:
        card, merchant = model.entity_embeddings()
        snapshots[build_id] = (
            card.detach().to("cpu", copy=True),
            merchant.detach().to("cpu", copy=True),
        )

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for position, batch in enumerate(batches):
        first = int(batch.event_time[0])
        last = int(batch.event_time[-1])
        # FIRE BEFORE PROCESSING ANY BATCH THAT REACHES THE CUTOFF, not on the
        # first batch that opens past it. The difference is the whole gate.
        #
        # Opening-based firing overshoots whenever a cutoff falls mid-batch, and
        # for these builds cutoff == serves_from exactly, so ANY overshoot trips
        # the check: a 2026-08-04 link run aborted on b_train_f2 at cutoff
        # 9,797,235 having fired at 9,797,434 -- 199 event_seq units, which is
        # ordinary batch granularity, not a stream gap. event_seq is a rank over
        # all 27.4M loaded transactions while the matrix holds 22.5M, so a
        # 4,096-row batch spans roughly 4,990 event_seq units and a cutoff lands
        # mid-batch almost every time. "Lower --batch-size" was bad advice: it
        # shrinks the overshoot and never removes it, at 10x the runtime.
        #
        # Firing before the containing batch is EXACT, not merely smaller.
        # Memory then holds only events at or below the previous batch's last
        # event, and the previous batch did not reach the cutoff, so every event
        # in memory is strictly below it. Same one-batch conservatism as before,
        # zero overshoot, no cost.
        while pending and pending[0][1] <= last:
            build_id, cutoff, serves_from = pending.pop(0)
            if serves_from is not None and first > serves_from:
                # Now a genuine tripwire rather than a near-certainty: reaching
                # this means memory crossed the first row the build describes,
                # which the firing rule above should make impossible.
                raise RuntimeError(
                    f"build {build_id}: memory reached event_seq {first:,} "
                    f"before the read-out, but the build serves rows from "
                    f"{serves_from:,} (cutoff {cutoff:,}). Those rows would get "
                    "embeddings encoding their own future. The firing rule "
                    "should prevent this, so treat it as a bug in the snapshot "
                    "ordering rather than something to tune around."
                )
            take(build_id)
            print(
                f"  snapshot {build_id} at cutoff {cutoff:,} (memory as of < {first:,})"
            )

        item = batch.to(device)
        assert_ordered(item.event_time)
        logits = model.step(item.card, item.merchant, item.event_time, item.raw)

        if score_from is not None and position >= score_from:
            all_scores.append(logits.detach().cpu().numpy())
            if model.objective == "link":
                n = len(item)
                all_labels.append(np.concatenate([np.ones(n), np.zeros(n)]))
            else:
                all_labels.append(item.label.detach().cpu().numpy())

    # A cutoff at or beyond the end of the replayed stream gets the final state.
    # Safe by construction: the replay covers train and val only, so "the final
    # state" contains no test event, and a build whose cutoff sits above that is
    # getting strictly less information than its GSQL features had, never more.
    while pending:
        build_id, cutoff, _ = pending.pop(0)
        take(build_id)
        print(f"  snapshot {build_id} at cutoff {cutoff:,} (end of stream)")

    if all_labels:
        aucpr, roc_auc = _metrics(
            np.concatenate(all_labels), np.concatenate(all_scores)
        )
    else:
        aucpr, roc_auc = float("nan"), float("nan")
    return snapshots, aucpr, roc_auc


def fit(
    model: TemporalFraudModel,
    train_batches: list[StreamBatch],
    val_batches: list[StreamBatch],
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 5,
    min_delta: float = 1e-4,
    rebuild: bool = True,
) -> list[EpochReport]:
    """Train until val AUC-PR stops improving, then restore the best weights.

    EARLY STOPPING ON val AUC-PR, not on loss. Loss falls monotonically here
    whether or not anything transferable is being learned -- Stage 2's header
    makes the same point -- and at 0.13% prevalence a falling BCE is entirely
    compatible with a model that ranks nothing. AUC-PR is the quantity the
    project reports, so it is the quantity to stop on.

    `patience` epochs without an improvement of at least `min_delta` ends the
    run. The best state_dict is kept and restored, then MEMORY IS REPLAYED
    under those weights -- see rebuild_memory for why that second step is not
    optional.

    With no val stream there is nothing to stop on, so it runs the full
    `epochs` and says so rather than silently stopping on training loss.
    """
    # NOT COMPUTED UNDER THE LINK OBJECTIVE, and that is a D5 point rather than
    # an optimisation. `train_epoch` already ignores pos_weight for link (the
    # positives-vs-negatives target is balanced), so computing it only read the
    # fraud label in order to print it -- a needless label touch on the one path
    # whose whole claim is that it never touches the label. It also meant a
    # split with no positives aborted a run that did not need the column.
    if model.objective == "link":
        weight = 1.0
        print("  objective=link: pos_weight not used, fraud label never read")
    else:
        labels = torch.cat([b.label for b in train_batches])
        weight = positive_weight(labels)
        print(
            f"  pos_weight {weight:,.1f} "
            f"({int(labels.sum()):,} positives in {labels.numel():,} train events)"
        )
    if not val_batches:
        print(
            "  no val stream: early stopping is DISABLED and all "
            f"{epochs} epochs will run"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[EpochReport] = []

    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(epochs):
        report = train_epoch(model, train_batches, optimizer, weight, device)
        if val_batches:
            aucpr, roc_auc, _, _ = evaluate(model, val_batches, device)
            report.val_aucpr = aucpr
            report.val_roc_auc = roc_auc
        history.append(report)

        marker = ""
        if val_batches and report.val_aucpr > best_score + min_delta:
            best_score = report.val_aucpr
            best_epoch = epoch
            # TO CPU, not .clone() on the same device. TGNMemory registers
            # three num_nodes-sized buffers, so a device-side clone of the
            # state_dict doubles the resident memory table for the whole run:
            # measured at 421 B/node, which alone caps a 24 GB card at ~2M
            # extra nodes it did not need to spend.
            best_state = {
                k: v.detach().to("cpu", copy=True)
                for k, v in model.state_dict().items()
            }
            stale = 0
            marker = "  <- best"
        elif val_batches:
            stale += 1

        # LABELLED BY OBJECTIVE. Under `link` these are LINK-prediction scores on
        # a BALANCED target -- base rate 0.5, so chance is AUC-PR 0.5 and ROC-AUC
        # 0.5 -- while under `fraud` they are fraud scores at a 0.129% base rate
        # where chance is AUC-PR 0.00129. A bare "val AUC-PR 0.62" from a link run
        # reads as a huge improvement on a fraud run's 0.38 and is in fact barely
        # above chance on an easier task. The numbers must carry their task.
        task = "link" if model.objective == "link" else "fraud"
        print(
            f"  epoch {epoch + 1:>3d}/{epochs}  loss {report.loss:.4f}  "
            f"val {task} AUC-PR {report.val_aucpr:.4f}  "
            f"ROC-AUC {report.val_roc_auc:.4f}  "
            f"{report.median_step_ms:.1f} ms/step  "
            f"{report.wall_seconds / 60:.1f} min{marker}"
        )

        if val_batches and stale >= patience:
            print(
                f"  early stop: {patience} epochs without a val AUC-PR gain "
                f"above {min_delta:g}. Best was epoch {best_epoch + 1} at "
                f"{best_score:.4f}."
            )
            break
    else:
        if val_batches and best_epoch == epochs - 1:
            print(
                "  ! val AUC-PR was still improving at the last epoch; raise --epochs"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored epoch {best_epoch + 1} (val AUC-PR {best_score:.4f})")
        # Weights changed, so the memory they produced is stale. Replay.
        #
        # `rebuild=False` is for a caller that is about to make its own ordered
        # pass and wants to control where it stops -- run.py snapshots memory at
        # each build cutoff on the way through, so a replay here would be
        # thrown away work rather than a correctness step.
        if rebuild:
            rebuild_memory(model, train_batches, device)

    return history
