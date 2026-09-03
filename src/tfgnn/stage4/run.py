"""Stage 4 entry point: train the TGN, score it, export both heads.

    python -m tfgnn.stage4.run                      # 4a, supervised fraud head
    python -m tfgnn.stage4.run --objective link     # 4b, D5-compliant

WRITES
  artifacts/stage4/stage4_metrics_<objective>.json   scores + per-epoch history
  artifacts/stage4/embeddings/         4b tables, per (node type, build_id)
  artifacts/stage4/embeddings_fraud/   the SAME read-out from a fraud run, kept
                                       for inspection and deliberately NOT on
                                       the path the lift table reads

======================================================================
WHAT THE TWO NUMBERS MEAN, AND WHY THEY ARE NOT INTERCHANGEABLE
======================================================================
4a is the user's goal -- score a transaction as it arrives -- and it BREAKS D5
by training on the label. It is reported here, on its own, never inside the
XGBoost lift table, because an arm trained on the label is not comparable to
arms that were not.

4a's AUC-PR is also not directly comparable to an XGBoost arm's for a second
reason: scoring keeps updating memory, because "catch it when it happens" means
the model has seen everything up to that moment. An XGBoost arm scores a frozen
model. Both are legitimate; they answer different questions, and the difference
must be stated wherever the number is.

4b is the comparable one, and ONLY under `--objective link`. The embedding
read-out never sees a label either way, but under `--objective fraud` the memory
it reads was shaped by gradients from BCE on is_fraud, so the vectors are
label-shaped even though the export is not. That is why a fraud run writes to
`embeddings_fraud/` and a link run writes to `embeddings/`.

======================================================================
PASS ORDER IS A CORRECTNESS PROPERTY OF THIS FILE
======================================================================
It used to be: fit -> score test -> export embeddings. Both halves of that were
wrong.

  * `evaluate` advances memory, so exporting after scoring test wrote vectors
    that had consumed every test event. Joined back onto the matrix, each row's
    embedding columns encoded the whole test period.
  * `fit` left memory at the end of TRAIN, and test was then scored directly
    from that state -- skipping the entire val split. Every node's last_update
    was stale by the val span and the neighbour table was missing val's edges,
    which is safe in the leak direction but is NOT the at-arrival situation this
    file claims to measure, and it depresses the test number.

The order below is: fit -> ordered replay over train then val, snapshotting
memory at each build cutoff -> score test. Each build's vectors see only events
below its cutoff, and test is scored from a memory that has consumed train and
val and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.config import project_root
from tfgnn.stage4 import cugraph_sampler
from tfgnn.stage4.model import TemporalFraudModel
from tfgnn.stage4.stream import load_stream
from tfgnn.stage4.train import evaluate, fit, plan_snapshots, replay_with_snapshots

# Imported, not redeclared. baseline.dataset reads these to recognise the TGN
# embedding family, and it cannot import this module (torch). One definition.
from tfgnn.features_meta import (
    TEMPORAL_CARD_EMBED_PREFIX as CARD_EMBED_PREFIX,
    TEMPORAL_MERCHANT_EMBED_PREFIX as MERCHANT_EMBED_PREFIX,
)


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_cutoffs(matrix_dir: Path) -> list[tuple[str, int, int | None]]:
    """(build_id, cutoff_event_seq, serves_from) for every snapshot behind the matrix.

    REQUIRED, not optional. The lift table joins embeddings on
    ``(key, build_id)``, so an export without build_id cannot be consumed at all
    -- and one with a build_id but the wrong memory state is worse, because it
    joins cleanly and reports a number. The manifest is the only record of which
    cutoff each build was computed at, so a missing one is a hard stop rather
    than a guess.
    """
    manifest = matrix_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"no {manifest}. Stage 4b exports one embedding table per build and "
            "needs each build's cutoff_event_seq to know where in the stream to "
            "read memory out. Re-run `python -m tfgnn.tigergraph.assemble`, "
            "which writes it."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    builds = payload.get("builds")
    if not isinstance(builds, list) or not builds:
        raise ValueError(
            f"{manifest} has no `builds` list, so there is no cutoff to snapshot "
            "at. Re-assemble the matrix."
        )
    out: list[tuple[str, int, int | None]] = []
    for entry in builds:
        build_id = entry.get("build_id")
        cutoff = entry.get("cutoff_event_seq")
        if build_id is None or cutoff is None:
            raise ValueError(
                f"a build entry in {manifest} is missing build_id or "
                f"cutoff_event_seq: {entry!r}"
            )
        # serves_from is the first event_seq this build's rows start at. Optional
        # in the manifest (a whole-split build has none), and where it exists it
        # is what `replay_with_snapshots` checks the firing point against.
        serves = entry.get("serves_from")
        out.append(
            (str(build_id), int(cutoff), None if serves is None else int(serves))
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: temporal GNN (TGN)")
    # 50 with early stopping, not a fixed 3. The old 3 was inherited from Stage
    # 2's default and was arbitrary; every XGBoost arm in this project stops on
    # val with early_stopping_rounds: 50, and the TGN is held to the same rule.
    _ = parser.add_argument("--epochs", type=int, default=50)
    _ = parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="epochs without a val AUC-PR gain before stopping",
    )
    _ = parser.add_argument("--batch-size", type=int, default=4096)
    _ = parser.add_argument(
        "--no-attention",
        action="store_true",
        help="reduce to memory-only, which does NOT use the graph. Attention is "
        "the canonical TGN and is on by default.",
    )
    _ = parser.add_argument("--neighbours", type=int, default=10)
    _ = parser.add_argument("--memory-dim", type=int, default=100)
    _ = parser.add_argument("--time-dim", type=int, default=100)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    # SEEDED, and it was not until 2026-08-03. Stage 2 has taken a --seed since
    # it was written; Stage 4 did not, which made its runs unrepeatable and any
    # A/B between two configurations uninterpretable. It cost a real conclusion:
    # the attention time-encoder fix was compared against a prior run at a
    # different seed AND a different scoring regime, so the 0.3824 -> 0.3241 move
    # could not be attributed to either change.
    #
    # This does NOT buy bitwise determinism. TGN's memory update is a scatter
    # reduction, and scatter-add on CUDA is order-nondeterministic, so residual
    # run-to-run variance survives the seed. It buys a controlled comparison:
    # hold the seed, change one thing. To claim an effect, vary the seed and
    # check the direction holds -- a single pair of runs cannot separate a 0.06
    # AUC-PR change from noise.
    _ = parser.add_argument("--seed", type=int, default=13)
    _ = parser.add_argument("--device", type=str, default="auto")
    _ = parser.add_argument(
        "--limit", type=int, default=None, help="first N events only, for a smoke"
    )
    _ = parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage4"))
    _ = parser.add_argument(
        "--objective",
        choices=("fraud", "link"),
        default="fraud",
        help="fraud = 4a, the supervised head, BREAKS D5. link = 4b, "
        "self-supervised link prediction, D5-COMPLIANT and the only "
        "version whose embeddings belong in the XGBoost lift table.",
    )
    args = parser.parse_args()
    objective = str(args.objective)

    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _device(str(args.device))
    print(f"device: {device}  seed: {seed}")
    # PRINTED, not assumed. This project believed cuGraph was in use while it
    # was PyG throughout; a run log should say which.
    print(cugraph_sampler.describe())

    stream = load_stream(batch_size=int(args.batch_size), limit=args.limit)
    cutoffs = _build_cutoffs(stream.matrix_dir)
    # RESOLVED AND CHECKED HERE, before a model exists. This is a pure function
    # of the batch boundaries and the manifest, and it used to be checked only
    # inside the export pass -- which meant a violation surfaced after a full
    # training run rather than in the first few seconds.
    print("  build snapshots (memory stands strictly below the boundary):")
    for build_id, cutoff, serves_from, boundary in plan_snapshots(
        stream.train + stream.val, cutoffs
    ):
        if boundary is None:
            print(
                f"    {build_id:<14s} cutoff {cutoff:>12,}  -> final state "
                "(cutoff past the end of train+val)"
            )
            continue
        margin = "" if serves_from is None else f"  margin {serves_from - boundary:>8,}"
        print(f"    {build_id:<14s} cutoff {cutoff:>12,}  -> < {boundary:,}{margin}")
        if serves_from is not None and boundary > serves_from:
            raise RuntimeError(
                f"build {build_id}: the read-out would fire with memory at "
                f"event_seq {boundary:,}, past the {serves_from:,} where its rows "
                "start, so those rows would get embeddings encoding their own "
                "future. Refusing to train for hours before failing on this."
            )

    model = TemporalFraudModel(
        stream.space,
        raw_dim=stream.raw_dim,
        memory_dim=int(args.memory_dim),
        time_dim=int(args.time_dim),
        neighbours=int(args.neighbours),
        attention=not bool(args.no_attention),
    ).to(device)
    model.objective = objective

    if objective == "fraud":
        print("\n=== 4a: supervised fraud head (BREAKS D5 by design) ===")
    else:
        print("\n=== 4b: self-supervised link objective (D5-compliant) ===")

    # rebuild=False: the snapshot replay below makes its own ordered pass and
    # controls where it stops, so fit's train-only replay would be discarded.
    history = fit(
        model,
        stream.train,
        stream.val,
        device,
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        patience=int(args.patience),
        rebuild=False,
    )

    # ---- the export pass: train then val, in order, snapshotting at cutoffs ----
    print("\n=== embedding read-out per build (no label on this path) ===")
    snapshots, val_aucpr, val_roc = replay_with_snapshots(
        model,
        stream.train + stream.val,
        cutoffs,
        device,
        score_from=len(stream.train),
    )
    print(
        f"  val under the restored weights: AUC-PR {val_aucpr:.4f}  "
        f"ROC-AUC {val_roc:.4f}"
    )

    # Memory now stands at the test boundary, having consumed train and val and
    # nothing later -- which is what makes the next line the at-arrival number.
    test_aucpr, test_roc, labels, scores = evaluate(model, stream.test, device)
    base = float(labels.mean()) if labels.size else float("nan")
    label = "4a" if objective == "fraud" else "4b"
    if objective == "fraud":
        print(
            f"\n{label} TEST  AUC-PR {test_aucpr:.4f}  ROC-AUC {test_roc:.4f}  "
            f"base rate {base:.5f}  lift {test_aucpr / base:.1f}x"
        )
    else:
        # No base rate and no lift here: the link target is balanced by
        # construction, so `base` is ~0.5 and a lift against it means nothing.
        print(
            f"\n{label} TEST link  AUC-PR {test_aucpr:.4f}  "
            f"ROC-AUC {test_roc:.4f}  (balanced target, no base-rate lift)"
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # A fraud run's vectors are label-shaped even though the export reads no
    # label, so they go somewhere the lift table does not look.
    if objective == "link":
        embed_dir = out_dir / "embeddings"
    else:
        embed_dir = out_dir / "embeddings_fraud"
        print(
            "\n  NOTE these vectors come from label-trained weights. They are "
            "written to embeddings_fraud/ and are NOT read by the lift table. "
            "Run `--objective link` for the arm that belongs there."
        )
    embed_dir.mkdir(parents=True, exist_ok=True)
    # Cleared first, for the same reason Stage 2's export is: the reader
    # concatenates whatever it finds, so a leftover from an earlier build plan
    # joins silently.
    for stale in sorted(embed_dir.glob("*.parquet")):
        stale.unlink()
    _write_embeddings(embed_dir, snapshots, stream)

    payload = {
        "model": "TGN",
        "objective": objective,
        "breaks_d5": objective == "fraud",
        "cugraph": cugraph_sampler.available(),
        "cugraph_note": (
            "true means cugraph and cugraph_pyg IMPORT, not that they are used. "
            "cugraph_sampler.build_store() is not called on this path; "
            "neighbour sampling is PyG's LastNeighborLoader."
        ),
        "d5_note": (
            "4a trains on the fraud label. Not comparable to Stage 2's arms on "
            "that axis, and never reported inside the XGBoost lift table."
        ),
        "scoring_note": (
            "Scoring keeps updating memory, which is the at-arrival situation. "
            "Memory has consumed train and val when test is scored, and nothing "
            "later. An XGBoost arm scores a frozen model; the two answer "
            "different questions."
        ),
        "embeddings_dir": str(embed_dir.relative_to(project_root())),
        "seed": seed,
        "seed_note": (
            "TGN's memory update is a scatter reduction and scatter-add on CUDA "
            "is order-nondeterministic, so the seed gives a controlled "
            "comparison, not bitwise reproducibility. Vary it before claiming "
            "an effect."
        ),
        "build_snapshots": [
            {"build_id": b, "cutoff_event_seq": c, "serves_from": s}
            for b, c, s in sorted(cutoffs, key=lambda i: i[1])
        ],
        "epochs_max": int(args.epochs),
        "epochs_run": len(history),
        "patience": int(args.patience),
        "memory_dim": int(args.memory_dim),
        "time_dim": int(args.time_dim),
        "attention": not bool(args.no_attention),
        "neighbours": int(args.neighbours),
        "raw_dim": stream.raw_dim,
        "events": stream.n_events,
        "n_cards": stream.space.n_cards,
        "n_merchants": stream.space.n_merchants,
        "val_aucpr_restored": val_aucpr,
        "val_roc_auc_restored": val_roc,
        "test_aucpr": test_aucpr,
        "test_roc_auc": test_roc,
        "test_base_rate": base,
        "history": [
            {
                "loss": r.loss,
                "steps": r.steps,
                "events": r.events,
                "val_aucpr": r.val_aucpr,
                "val_roc_auc": r.val_roc_auc,
                "median_step_ms": r.median_step_ms,
                "wall_seconds": r.wall_seconds,
            }
            for r in history
        ],
    }
    # NAMED BY OBJECTIVE. A fixed filename meant `--objective link` silently
    # overwrote the fraud run's record with a link AUC-PR under the same
    # `test_aucpr` key, so two incomparable numbers shared one slot.
    metrics = out_dir / f"stage4_metrics_{objective}.json"
    _ = metrics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {metrics}")


def _write_embeddings(
    directory: Path,
    snapshots: dict[str, tuple[torch.Tensor, torch.Tensor]],
    stream: object,
) -> None:
    """One parquet per (node_type, build_id), mirroring Stage 2's export.

    ``build_id`` IS HALF THE JOIN KEY and is not decoration: the reader merges on
    ``(key, build_id)`` so that a card's embedding is the one computed at its
    row's snapshot. Without the column the join raises; with the column but a
    single global state it would attach post-cutoff information to every row.

    Keys are the ORIGINAL card_number / merchant_id, recovered from the stream's
    index order -- a positional assumption here would give every entity another
    entity's memory, which trains and scores without error.
    """
    import pandas as pd

    from tfgnn.stage4.stream import Stream

    assert isinstance(stream, Stream)
    if not snapshots:
        raise RuntimeError(
            "no memory snapshots were taken, so there is nothing to export. "
            "Check that the matrix manifest's build cutoffs fall inside the "
            "event_seq range of the stream."
        )
    for build_id, (card_memory, merchant_memory) in sorted(snapshots.items()):
        for node_type, memory, prefix, keys in (
            ("Card", card_memory, CARD_EMBED_PREFIX, stream.card_keys),
            (
                "Merchant",
                merchant_memory,
                MERCHANT_EMBED_PREFIX,
                stream.merchant_keys,
            ),
        ):
            block = memory.detach().cpu().numpy().astype(np.float32)
            if block.shape[0] != len(keys):
                raise RuntimeError(
                    f"{node_type}/{build_id}: {block.shape[0]} memory rows for "
                    f"{len(keys)} keys. The index space and the key list have "
                    "drifted apart."
                )
            frame = pd.DataFrame(
                block, columns=[f"{prefix}_{i}" for i in range(block.shape[1])]
            )
            frame.insert(0, "_key", pd.Series(keys, dtype="string"))
            frame.insert(0, "build_id", build_id)
            path = directory / f"{node_type}__{build_id}.parquet"
            frame.to_parquet(path, index=False)
            print(
                f"  {node_type:<10s} {build_id:<14s} {len(frame):>8,} rows "
                f"x {block.shape[1]} dims"
            )


if __name__ == "__main__":
    main()
