import argparse
import json
from pathlib import Path

import torch

from common.config import project_root
from tfgnn.stage2.embed import write_tables
from tfgnn.stage2.fanout import NeighborFanout, describe as describe_fanout
from tfgnn.stage2.graph import ResidentGraph, build_timed_csr
from tfgnn.stage2.link_pred import DistMultDecoder, NegativeRegime, PairIndex
from tfgnn.stage2.load import (
    ALL_RELATIONS,
    PAIR_ONLY,
    export_graph,
    load_edges,
    load_node_features,
    load_vertex_maps,
)
from tfgnn.stage2.model import RGCN
from tfgnn.stage2.sampler import NeighborSampler, Strategy
from tfgnn.stage2.schema_spec import TARGET_RELATION
from tfgnn.stage2.train import (
    Seeds,
    evaluate_epoch,
    export_embeddings,
    train_epoch,
)


def _resolve(path: Path) -> Path:
    """Anchor a relative path to the PROJECT, never to the shell's cwd.

    The defaults below name project-relative locations, and an unanchored
    relative default silently means a different directory per working
    directory: run from ``src/`` and Stage 2 writes ``src/artifacts/stage2``,
    finds no Stage-1 dimension manifests (so ``builds`` is empty, the seed
    mask degrades to unmasked and the run trains on everything), and exports
    embeddings where ``baseline.dataset`` will not look for them. The shell
    scripts all cd to the repo root, which is exactly what hid this; a direct
    ``python -m tfgnn.stage2.run`` does not.

    Same rule the Stage-1 modules already apply (``common.config._resolve``,
    ``pipeline._resolve_manifest``, ``export``, ``dataset``): an ABSOLUTE
    path is honoured untouched, so pointing any of these at scratch space on
    another machine still works.
    """
    return path if path.is_absolute() else project_root() / path


def _build_ids(export_root: Path) -> list[tuple[str, int, tuple[int, ...]]]:
    """(build_id, cutoff, serves_splits) from the Stage 1 dimension manifests,
    in cutoff order.

    Read from Stage 1's own exports rather than recomputed, so the embedding
    tables are keyed to exactly the snapshots assemble routed rows to. Deriving
    them independently is how the two halves drift apart.
    """
    root = export_root / "dimensions"
    out: list[tuple[str, int, tuple[int, ...]]] = []
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        manifest = directory / "manifest.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            splits = tuple(int(s) for s in payload.get("serves_splits", []))
            out.append(
                (str(payload["build_id"]), int(payload["cutoff_event_seq"]), splits)
            )
    return sorted(out, key=lambda item: item[1])


def _boundary_builds(
    builds: list[tuple[str, int, tuple[int, ...]]],
) -> tuple[tuple[str, int] | None, tuple[str, int] | None]:
    """The (val, test) builds, selected BY CONTRACT, never by position.

    Each dimension manifest records which splits its build serves. Positional
    selection (builds[-2] as val) breaks silently the moment the on-disk set
    is not exactly [folds..., b_val, b_test] -- an end_of_data build for the
    kit-parity or permutation arms sorts above b_test and would shift every
    index, admitting the val window into training with no error anywhere.
    serves_splits is written by export_dimensions for exactly this kind of
    routing, so it is the identity to trust.

    Returns (None, None) when either build cannot be identified unambiguously;
    the caller degrades LOUDLY to unmasked training rather than masking at a
    wrong boundary.
    """
    vals = [(b, c) for b, c, splits in builds if 1 in splits and 2 not in splits]
    tests = [(b, c) for b, c, splits in builds if 2 in splits]
    if len(vals) == 1 and len(tests) == 1:
        return vals[0], tests[0]
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: R-GCN link prediction")
    _ = parser.add_argument("--graph-dir", type=Path, default=Path("data/stage2/graph"))
    _ = parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage2"))
    _ = parser.add_argument(
        "--stage1-export", type=Path, default=Path("data/stage1/export")
    )
    _ = parser.add_argument("--export-only", action="store_true")
    _ = parser.add_argument("--overwrite-export", action="store_true")
    _ = parser.add_argument("--relations", choices=["all", "pair"], default="all")
    # torch, not cugraph. The torch sampler is per-SEED exact, leak-asserted and
    # tested; the cuGraph one is per-FOLD and is not wired to the training loop
    # at all -- see the gate below, which explains and refuses rather than
    # letting it TypeError several minutes into a run.
    _ = parser.add_argument("--backend", choices=["cugraph", "torch"], default="torch")
    _ = parser.add_argument("--epochs", type=int, default=3)
    _ = parser.add_argument("--batch-size", type=int, default=1024)
    _ = parser.add_argument("--hidden", type=int, default=128)
    _ = parser.add_argument("--out-dim", type=int, default=64)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    _ = parser.add_argument(
        "--negatives",
        choices=[r.value for r in NegativeRegime],
        default=NegativeRegime.RANDOM.value,
    )
    _ = parser.add_argument("--max-batch-bytes", type=float, default=4e9)
    _ = parser.add_argument(
        "--eval-edges",
        type=int,
        default=50_000,
        help="held-out val-window edges scored after each epoch (0 disables). "
        "A fixed slice, so the number is comparable across epochs.",
    )
    _ = parser.add_argument("--log-every", type=int, default=50)
    _ = parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Anchor before anything reads them, so every downstream module
    # (load, embed, the manifest writers) receives an absolute path and
    # cannot disagree with this one about where the run lives.
    args.graph_dir = _resolve(args.graph_dir)
    args.out_dir = _resolve(args.out_dir)
    args.stage1_export = _resolve(args.stage1_export)

    builds = _build_ids(args.stage1_export)
    if not builds:
        print(
            f"  WARNING no Stage-1 dimension manifests under {args.stage1_export}. "
            "Run Stage 1 first, or pass --stage1-export. Training would fall "
            "back to unmasked seeds, which is a leak."
        )
    # The LAST build, not the first: reset_build deletes every
    # Has_Interaction_With_Merchant edge at the start of each build and
    # build_interaction_edges re-stamps them with the current build_id, so
    # after a multi-build Stage-1 run only the max-cutoff build's edges are
    # resident -- exporting with an earlier build_id returns zero rows. The
    # superset is leak-correct: admission is per seed on the edge's own
    # first_event_seq, so a pair born after a seed's time is filtered out
    # regardless of which build exported it.
    build_id = builds[-1][0] if builds else ""
    build_cutoff = builds[-1][1] if builds else 0

    # ---- export (cached) ----
    # The cutoff rides along as cache identity: build NAMES are fixed config
    # strings ('b_test' on every standard run), so a name-only check passes
    # across a data regeneration and silently reuses the previous load's
    # parquets. The cutoff is data-derived and moves with the load.
    export_graph(
        args.graph_dir,
        build_id=build_id,
        cutoff_event_seq=build_cutoff,
        relations=PAIR_ONLY if args.relations == "pair" else ALL_RELATIONS,
        overwrite=args.overwrite_export,
    )
    if args.export_only:
        print("--export-only: graph cached, stopping.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    if device.type != "cuda":
        print(
            "  NOTE no CUDA. The torch backend will run on CPU; cuGraph cannot. "
            "Forcing --backend torch."
        )
        args.backend = "torch"

    if args.backend == "cugraph":
        # Refuse HERE, before loading 200M edges, rather than letting the
        # training loop TypeError minutes later. Three gaps, all confirmed by
        # reading the call sites -- none is a cuGraph limitation, all are
        # unfinished wiring in this repo:
        raise SystemExit(
            "--backend cugraph is not wired to the training loop and would "
            "fail or silently leak. Three specific gaps:\n"
            "  1. SIGNATURE. train_epoch calls "
            "sample(cards, times, extra_seeds={'Merchant': ...}) but "
            "CuGraphSampler.sample(seed_global, seed_type) accepts neither the "
            "times nor extra_seeds -> TypeError. Without extra_seeds the "
            "positive/negative merchants are not guaranteed into the batch, so "
            "the decoder cannot score them.\n"
            "  2. ISOLATED SEEDS. CuGraphSampler registers only vertices that "
            "appear in sampled rows, so a card with no admissible edges is "
            "absent from the id map and to_local raises. The torch sampler "
            "registers seeds unconditionally.\n"
            "  3. LEAK. run.py builds ONE fold graph at max(cutoffs) for ALL "
            "seeds, not one graph per fold as cugraph_backend's docstring "
            "specifies. A seed at t < max(cutoffs) then finds its own target "
            "edge in the message-passing graph -- the D3a/D3c leak the "
            "per-fold design exists to prevent, and it is SILENT.\n\n"
            "Use --backend torch (the default). It is GPU-resident too, and "
            "its guarantee is STRONGER: per-seed exact "
            "(edge_event_seq < that seed's event_seq) rather than per-fold, so "
            "it does not pay cugraph_backend's documented ~10-month "
            "within-fold coarsening either."
        )

    # dense_ids is what aligns the vertex map and the Stage-1 feature tables
    # to THIS graph's node ids. It is not derivable from either of them: the
    # dense space holds only getvids that appear in exported edges.
    edges, node_counts, dense_ids = load_edges(args.graph_dir, device)
    print(f"  node counts: {node_counts}")
    print(f"  relations loaded: {len(edges)}")

    fanout = NeighborFanout()
    print(describe_fanout(fanout, seeds=args.batch_size))

    # ---- resident graph ----
    # ONE path. The cuGraph branch that used to sit here was unreachable once
    # the backend gate above started refusing it, and an unreachable branch
    # constructing a sampler the training loop cannot call is worse than no
    # branch: it reads as a supported option. cugraph_backend.py documents what
    # has to be finished before it comes back.
    relations_csr = {
        triple: build_timed_csr(
            triple,
            src,
            dst,
            time,
            node_counts[triple[0]],
            node_counts[triple[2]],
            device,
        )
        for triple, (src, dst, time) in edges.items()
    }
    graph = ResidentGraph(
        relations=relations_csr, node_counts=node_counts, device=device
    )
    graph.validate()
    print()
    print(graph.describe())
    sampler = NeighborSampler(graph=graph, fanout=fanout, strategy=Strategy.MOST_RECENT)
    guarantee = "per-seed (edge_event_seq < seed event_seq)"

    # ---- seeds: the target relation's edges ----
    # An explicit failure, because `next()` on an empty generator raised a bare
    # StopIteration that named neither the relation nor either of its two real
    # causes.
    target = next((t for t in edges if t[1] == TARGET_RELATION), None)
    if target is None:
        raise SystemExit(
            f"no {TARGET_RELATION} edges are loaded, so there are no seeds to "
            "train on. Two causes, both actionable:\n"
            "  * --relations pair EXCLUDES the target by design (it is the 33.7M "
            "-edge relation pair mode exists to avoid). Pair mode proves the "
            "export path; it cannot train, because the seeds ARE the target "
            "relation's edges. Drop --pair-graph.\n"
            f"  * {TARGET_RELATION} is empty in TigerGraph. It is BUILT, not "
            "loaded: run `python -m tfgnn.tigergraph.target_edges` "
            "(scripts/run_all.sh does this automatically before the export)."
        )
    src, dst, time = edges[target]
    # PairIndex over the FULL window on purpose: a "negative" that is a real
    # pair later in time is still a real pair, and admitting it as a negative
    # would train against the exact structure the model is asked to predict.
    pairs = PairIndex.build(src, dst, node_counts["Merchant"])

    # TRAINING SEEDS COME FROM THE TRAINING WINDOW ONLY. The target relation
    # is materialised over the whole corpus (its builder's header delegates
    # cutoff discipline to the sampler), but the sampler only guards MESSAGE
    # PASSING per seed -- the model and decoder WEIGHTS are fitted on whatever
    # link-existence targets appear here. Unmasked, that includes val- and
    # test-window edges: weights fitted on future link structure, which
    # train.py's own contract ("seeds are real card-merchant edges from the
    # TRAINING window") and embed.py's leak rule both forbid. The boundary
    # builds are selected BY CONTRACT (serves_splits in each manifest), never
    # by position -- an end_of_data build for the kit-parity or permutation
    # arms sorts above b_test and would silently shift a positional pick.
    val_build, test_build = _boundary_builds(builds)
    if val_build is not None and test_build is not None:
        train_cutoff = val_build[1]
        test_cutoff = test_build[1]
        train_mask = time < train_cutoff
        val_mask = (time >= train_cutoff) & (time < test_cutoff)
        seeds = Seeds(
            card=src[train_mask], merchant=dst[train_mask], time=time[train_mask]
        )
        heldout_val_edges = int(val_mask.sum().item())
        seed_time_cutoff = train_cutoff
        # The held-out set train.py:47 promises. Capped: the val window runs
        # to ~1.4M edges, and re-sampling all of them every epoch would cost
        # more than the training pass it is meant to comment on. A fixed slice
        # (not a fresh random draw) keeps the number COMPARABLE ACROSS
        # EPOCHS, which is the whole point of watching it.
        val_seeds = Seeds(
            card=src[val_mask][: args.eval_edges],
            merchant=dst[val_mask][: args.eval_edges],
            time=time[val_mask][: args.eval_edges],
        )
    else:
        print(
            "  WARNING could not identify the val and test builds from "
            "serves_splits in the dimension manifests: training on ALL "
            "target edges. Acceptable only for offline smoke runs -- a "
            "full run reaching this warning is a leak, stop and look at "
            f"the manifests under {args.stage1_export}/dimensions."
        )
        seeds = Seeds(card=src, merchant=dst, time=time)
        heldout_val_edges = 0
        seed_time_cutoff = 0
        val_seeds = None
    print(
        f"\nseeds: {len(seeds):,} train-window target edges "
        f"(cutoff {seed_time_cutoff:,}; {heldout_val_edges:,} val-window edges "
        f"held out); {pairs.keys.numel():,} distinct pairs of "
        f"{node_counts['Card'] * node_counts['Merchant']:,} possible "
        f"({pairs.keys.numel() / max(node_counts['Card'] * node_counts['Merchant'], 1):.2%} density)"
    )
    # The densified per-relation edge dict is dead weight from here on: the
    # CSRs hold their own sorted copies and the seeds/pair index above hold
    # theirs. On the full graph this frees ~5 GB of GPU memory that would
    # otherwise sit next to the CSRs for the whole run.
    del edges, src, dst, time

    # ---- features: Stage 1's dimension block, shared by construction ----
    # Taken from Stage 1's own exports, not re-derived, so the GNN's node inputs
    # and Stage 1's columns cannot diverge. D5 requires that: otherwise
    # raw_plus_graph -> raw_plus_embeddings differs in two ways at once and
    # measures nothing.
    # Training reads the VAL build's tables (selected by contract above):
    # fitted strictly below val_boundary, i.e. on exactly the window the
    # training seeds come from -- never b_test's, whose aggregates have seen
    # the val window. (Residual approximation, documented: an early-fold seed
    # reads features fitted over the whole training window. The exact
    # per-fold routing is what Stage 1's forward chaining does; here it
    # would need per-fold feature tensors inside the batch loop.)
    features_build = val_build[0] if val_build is not None else build_id
    print(f"\nnode features from Stage 1 dimension tables ({features_build}):")
    features, raw_dims = load_node_features(
        args.stage1_export,
        args.graph_dir,
        features_build,
        node_counts,
        device,
        dense_ids,
    )
    if not raw_dims:
        print(
            "  WARNING no node features found. The GNN will learn from "
            "STRUCTURE ONLY via type-level vectors, which is a different (and "
            "weaker) experiment than D5 specifies -- a poor result would not "
            "distinguish 'the GNN failed' from 'the GNN was starved'."
        )

    model = RGCN(raw_dims=raw_dims, hidden=args.hidden, out=args.out_dim).to(device)
    decoder = DistMultDecoder(args.out_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=args.lr
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        print(f"\nepoch {epoch}")
        # Fresh permutation per epoch. The export writes seeds in vid-page
        # order, so unshuffled batches are runs of consecutive cards -- every
        # epoch then replays identical, card-correlated batches and the
        # negative sampler keeps drawing against the same local neighborhoods.
        perm = torch.randperm(len(seeds), generator=generator, device=device)
        epoch_seeds = Seeds(
            card=seeds.card[perm],
            merchant=seeds.merchant[perm],
            time=seeds.time[perm],
        )
        del perm
        metrics = train_epoch(
            model,
            decoder,
            sampler,
            epoch_seeds,
            features,
            pairs,
            optimizer,
            batch_size=args.batch_size,
            num_merchants=node_counts["Merchant"],
            num_cards=node_counts["Card"],
            regime=NegativeRegime(args.negatives),
            max_batch_bytes=int(args.max_batch_bytes),
            log_every=args.log_every,
            generator=generator,
        )
        # Held-out link-prediction loss on the val window: the ONLY selection
        # signal this stage is allowed (an early stop on downstream AUCPR
        # would be leak rule 4). Training loss alone falls with epochs
        # regardless of whether anything transferable is being learned, so a
        # val loss that stops falling while the train loss keeps going is the
        # overfitting signature to watch for.
        if val_seeds is not None and len(val_seeds) > 0:
            metrics["val_loss"] = evaluate_epoch(
                model,
                decoder,
                sampler,
                val_seeds,
                features,
                pairs,
                batch_size=args.batch_size,
                num_merchants=node_counts["Merchant"],
                num_cards=node_counts["Card"],
                regime=NegativeRegime(args.negatives),
                generator=generator,
            )
        history.append(metrics)
        val_note = f"  val {metrics['val_loss']:.4f}" if "val_loss" in metrics else ""
        print(
            f"  loss {metrics['loss']:.4f}{val_note}  "
            f"steps {int(metrics['steps'])}  "
            f"peak batch {metrics['peak_batch_bytes'] / 1e6:.1f} MB"
        )

    # ---- embeddings, one table per build snapshot ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = load_vertex_maps(args.graph_dir, dense_ids)
    tables = []
    for build, cutoff, _splits in builds:
        print(f"\nembedding at {build} (cutoff {cutoff:,})")
        # Each build's embeddings read THAT build's dimension tables, exactly
        # as the sampler reads that build's cutoff. One shared feature tensor
        # here would hand every snapshot the same (latest) aggregates and the
        # per-build tables would differ only by neighborhood -- half the
        # forward-chaining contract.
        build_features, build_dims = load_node_features(
            args.stage1_export,
            args.graph_dir,
            build,
            node_counts,
            device,
            dense_ids,
        )
        if build_dims != raw_dims:
            raise SystemExit(
                f"{build}: feature dims {build_dims} differ from the training "
                f"build's {raw_dims}; the model cannot consume them. The "
                "dimension exports disagree across builds -- re-run Stage 1."
            )
        tables.extend(
            export_embeddings(
                model,
                sampler,
                build_features,
                keys,
                build,
                cutoff,
                batch_size=args.batch_size,
            )
        )
        del build_features
    write_tables(tables, args.out_dir / "embeddings")

    (args.out_dir / "stage2_metrics.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "temporal_guarantee": guarantee,
                "negative_regime": args.negatives,
                "epochs": args.epochs,
                "out_dim": args.out_dim,
                "history": history,
                "node_counts": node_counts,
                "seeds": len(seeds),
                "seed_time_cutoff": seed_time_cutoff,
                "heldout_val_edges": heldout_val_edges,
                "export_build": build_id,
                "features_build": features_build,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir / 'stage2_metrics.json'}")


if __name__ == "__main__":
    main()
