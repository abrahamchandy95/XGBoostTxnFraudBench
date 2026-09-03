import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from tfgnn.stage2.embed import EmbeddingTable, normalize_within_build
from tfgnn.stage2.fanout import NeighborFanout, describe as describe_fanout
from tfgnn.stage2.link_pred import (
    DistMultDecoder,
    NegativeRegime,
    PairIndex,
    bce_loss,
    sample_negatives,
    verify_no_collisions,
)
from tfgnn.stage2.model import RGCN
from tfgnn.stage2.sampler import Batch, NeighborSampler, Strategy


class TrainError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Seeds:
    """Target edges to train on: (card, merchant, time), all global ids."""

    card: Tensor
    merchant: Tensor
    time: Tensor

    def __post_init__(self) -> None:
        if not (self.card.numel() == self.merchant.numel() == self.time.numel()):
            raise TrainError("seed card, merchant and time must be the same length")

    def __len__(self) -> int:
        return int(self.card.numel())

    def slice(self, lo: int, hi: int) -> "Seeds":
        return Seeds(self.card[lo:hi], self.merchant[lo:hi], self.time[lo:hi])


def _gather_features(batch: Batch, features: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per node type, the feature rows for this batch's nodes, in local order.

    Indexes the resident feature table by GLOBAL id -- which is what
    ``id_map.local_to_global`` returns, in local order -- so no per-batch feature
    copy of the whole table is ever made.
    """
    out: dict[str, Tensor] = {}
    for node_type, global_ids in batch.global_ids.items():
        table = features.get(node_type)
        if table is not None:
            out[node_type] = table[global_ids]
    return out


def estimate_step_bytes(model: RGCN, node_counts: Mapping[str, int]) -> int:
    """Activation bytes a forward+backward will hold for this batch.

    ``Batch.bytes_held()`` prices the SAMPLER's tensors -- dst ids, times,
    edge index -- which on a capped batch is tens of megabytes. The step's
    real peak is dominated by the per-node REPRESENTATIONS, which the sampler
    never sees: every node carries an input projection, two convolution
    outputs and a LayerNorm, autograd saves most of them for the backward
    pass, and the optimizer then holds a gradient of the same width.

    Widths per node, in float32 units: ``hidden`` (projection) + ``hidden``
    (conv1) + ``out`` (conv2). Doubled for the gradients autograd retains.
    That is a LOWER BOUND -- it ignores dropout masks, LayerNorm's saved
    statistics and the decoder's scores -- and it is deliberately a bound
    rather than a fitted constant: the number exists to stop a run before
    the allocator does, and an estimate that overshoots would stop runs that
    would have fit.
    """
    nodes = sum(node_counts.values())
    per_node_floats = 2 * model.hidden_dim + model.out_dim
    return nodes * per_node_floats * 4 * 2


def train_epoch(
    model: RGCN,
    decoder: DistMultDecoder,
    sampler: NeighborSampler,
    seeds: Seeds,
    features: dict[str, Tensor],
    real_pairs: PairIndex,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_merchants: int,
    num_cards: int,
    regime: NegativeRegime,
    max_batch_bytes: int | None,
    log_every: int,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    model.train()
    decoder.train()
    total_loss = 0.0
    steps = 0
    peak_batch = 0
    dropped = 0

    for start in range(0, len(seeds), batch_size):
        chunk = seeds.slice(start, start + batch_size)
        if len(chunk) == 0:
            continue
        # Sample the NEGATIVES FIRST, so their merchants can be seeded into the
        # same batch. Otherwise a negative merchant absent from the batch has no
        # embedding and the negative set silently shrinks -- which weakens the
        # objective without weakening anything a caller can see.
        neg_card, neg_merchant = sample_negatives(
            chunk.card,
            num_cards,
            num_merchants,
            real_pairs,
            regime=regime,
            generator=generator,
        )
        verify_no_collisions(neg_card, neg_merchant, real_pairs)

        # Both endpoints of every positive AND every negative enter as seed
        # NODES. See NeighborSampler.sample: the time filter withholds the target
        # edge, so the merchant is otherwise unreachable and unscoreable.
        merchant_seeds = torch.cat([chunk.merchant, neg_merchant])
        merchant_times = torch.cat([chunk.time, chunk.time[: neg_merchant.numel()]])
        batch = sampler.sample(
            chunk.card,
            chunk.time,
            extra_seeds={"Merchant": (merchant_seeds, merchant_times)},
        )
        try:
            # Sampler tensors PLUS the activations the forward pass is about
            # to allocate. Checked BEFORE the forward, because a guard that
            # only measures what already exists cannot stop an allocation
            # that has not happened yet -- and the activations are the larger
            # term by an order of magnitude, so the old check could not fire
            # before a CUDA OOM killed the run.
            sampler_bytes = batch.bytes_held()
            activation_bytes = estimate_step_bytes(model, batch.node_counts())
            held = sampler_bytes + activation_bytes
            peak_batch = max(peak_batch, held)
            if max_batch_bytes is not None and held > max_batch_bytes:
                raise TrainError(
                    f"batch would hold {held / 1e9:.2f} GB "
                    f"({sampler_bytes / 1e9:.2f} GB sampled + "
                    f"{activation_bytes / 1e9:.2f} GB activations, a lower "
                    f"bound), above the {max_batch_bytes / 1e9:.2f} GB limit. "
                    "Tighten the fan-out caps in tfgnn.stage2.fanout -- on "
                    "this dataset one merchant reaches 49% of cards, so a cap "
                    "is what bounds the batch -- or lower --batch-size."
                )
            dropped += batch.edges_dropped_by_time

            embeddings = model(
                _gather_features(batch, features),
                batch.edge_index,
                batch.node_counts(),
            )

            # Seeds and their negatives, in the batch's local space. A seed whose
            # endpoint did not make it into the batch cannot be scored; that
            # happens only if the sampler dropped it, which would be a bug, so it
            # raises inside to_local rather than being filtered away here.
            card_local = batch.id_map.to_local("Card", chunk.card)
            merchant_local = batch.id_map.to_local("Merchant", chunk.merchant)
            positive = decoder(
                embeddings["Card"][card_local],
                embeddings["Merchant"][merchant_local],
            )

            # Both endpoints were seeded above, so every negative is scoreable.
            negative = decoder(
                embeddings["Card"][batch.id_map.to_local("Card", neg_card)],
                embeddings["Merchant"][batch.id_map.to_local("Merchant", neg_merchant)],
            )

            loss = bce_loss(positive, negative)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach())
            steps += 1
            if log_every and steps % log_every == 0:
                print(
                    f"    step {steps:>5d}  loss {total_loss / steps:.4f}  "
                    f"batch {held / 1e6:>7.1f} MB  "
                    f"nodes {sum(batch.node_counts().values()):>8,}"
                )
        finally:
            # Fresh ids next batch, nothing retained. In a finally block because
            # an exception mid-batch would otherwise leak the id map into the
            # next step and make a rising-peak diagnosis impossible.
            batch.free()

    if steps == 0:
        raise TrainError("no batch produced a usable loss")
    return {
        "loss": total_loss / steps,
        "steps": float(steps),
        "peak_batch_bytes": float(peak_batch),
        "edges_dropped_by_time": float(dropped),
    }


@torch.no_grad()
def evaluate_epoch(
    model: RGCN,
    decoder: DistMultDecoder,
    sampler: NeighborSampler,
    seeds: Seeds,
    features: dict[str, Tensor],
    real_pairs: PairIndex,
    batch_size: int,
    num_merchants: int,
    num_cards: int,
    regime: NegativeRegime,
    generator: torch.Generator | None = None,
) -> float:
    """Link-prediction loss on HELD-OUT edges. The module contract's validation.

    ``train_epoch``'s loss falls monotonically with epochs whether or not the
    model is learning anything transferable, so on its own it cannot answer
    "is this overfitting". This scores the val-window target edges the seed
    mask withheld -- edges the weights never saw -- with the same objective,
    which is the only signal this file is allowed to select on: an early stop
    on the downstream XGBoost metric would be leak rule 4 (see the module
    docstring's "WHAT IS DELIBERATELY NOT HERE").

    Sampling still respects each seed's own time, so a val edge reads a
    message-passing neighbourhood strictly before itself; this measures
    generalisation, not a second training pass.

    ``model.eval()`` for the duration: dropout during scoring would make the
    number noisy across epochs for a reason that has nothing to do with the
    model. Restored to train mode by the caller's next ``train_epoch``.
    """
    model.eval()
    decoder.eval()
    total = 0.0
    steps = 0
    for start in range(0, len(seeds), batch_size):
        chunk = seeds.slice(start, start + batch_size)
        if len(chunk) == 0:
            continue
        neg_card, neg_merchant = sample_negatives(
            chunk.card,
            num_cards,
            num_merchants,
            real_pairs,
            regime=regime,
            generator=generator,
        )
        merchant_seeds = torch.cat([chunk.merchant, neg_merchant])
        merchant_times = torch.cat([chunk.time, chunk.time[: neg_merchant.numel()]])
        batch = sampler.sample(
            chunk.card,
            chunk.time,
            extra_seeds={"Merchant": (merchant_seeds, merchant_times)},
        )
        try:
            embeddings = model(
                _gather_features(batch, features),
                batch.edge_index,
                batch.node_counts(),
            )
            positive = decoder(
                embeddings["Card"][batch.id_map.to_local("Card", chunk.card)],
                embeddings["Merchant"][
                    batch.id_map.to_local("Merchant", chunk.merchant)
                ],
            )
            negative = decoder(
                embeddings["Card"][batch.id_map.to_local("Card", neg_card)],
                embeddings["Merchant"][batch.id_map.to_local("Merchant", neg_merchant)],
            )
            total += float(bce_loss(positive, negative))
            steps += 1
        finally:
            batch.free()

    model.train()
    decoder.train()
    return total / steps if steps else float("nan")


@torch.no_grad()
def export_embeddings(
    model: RGCN,
    sampler: NeighborSampler,
    features: dict[str, Tensor],
    keys: dict[str, pd.Series],
    build_id: str,
    cutoff: int,
    batch_size: int,
    normalize: bool = True,
) -> list[EmbeddingTable]:
    """Embed every Card and Merchant AT one build's cutoff.

    The cutoff is the build's, not per-seed, because this produces the table that
    build's ROWS will join to -- and assemble already guarantees those rows sit
    at or after it. Passing a later cutoff here would hand every row of that
    build an embedding computed from its own future.
    """
    model.eval()
    tables: list[EmbeddingTable] = []

    for node_type in ("Card", "Merchant"):
        key_series = keys[node_type]
        count = len(key_series)
        chunks: list[np.ndarray] = []
        for start in range(0, count, batch_size):
            ids = torch.arange(
                start, min(start + batch_size, count), device=sampler.graph.device
            )
            times = torch.full_like(ids, cutoff)
            # Seeds are Cards in the sampler's POSITIONAL contract, so
            # merchants enter through extra_seeds -- the same mechanism
            # train_epoch uses for the link-prediction targets. The previous
            # version passed merchant ids AS card seeds and read the merchant
            # side out of whatever those arbitrary cards happened to reach:
            # coverage was luck, and any merchant absent from the batch raised
            # IdMapError below. extra_seeds puts every requested merchant in
            # the id map by construction, expanded from its own past.
            if node_type == "Card":
                batch = sampler.sample(ids, times)
            else:
                no_cards = ids.new_empty(0)
                batch = sampler.sample(
                    no_cards,
                    no_cards,
                    extra_seeds={"Merchant": (ids, times)},
                )
            try:
                out = model(
                    _gather_features(batch, features),
                    batch.edge_index,
                    batch.node_counts(),
                )
                vectors = out[node_type]
                local = batch.id_map.to_local(node_type, ids)
                chunks.append(vectors[local].detach().cpu().numpy())
            finally:
                batch.free()

        table = EmbeddingTable(
            node_type=node_type,
            build_id=build_id,
            keys=key_series,
            vectors=np.concatenate(chunks).astype("float32"),
        )
        tables.append(normalize_within_build(table) if normalize else table)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Stage 2 R-GCN and export embeddings"
    )
    _ = parser.add_argument("--graph-dir", required=True, type=Path)
    _ = parser.add_argument("--out-dir", required=True, type=Path)
    _ = parser.add_argument("--epochs", type=int, default=3)
    _ = parser.add_argument("--batch-size", type=int, default=1024)
    _ = parser.add_argument("--hidden", type=int, default=128)
    _ = parser.add_argument("--out-dim", type=int, default=64)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    _ = parser.add_argument(
        "--negatives",
        choices=[r.value for r in NegativeRegime],
        default=NegativeRegime.RANDOM.value,
        help="D5 requires random and historical be REPORTED SEPARATELY",
    )
    _ = parser.add_argument(
        "--sampling",
        choices=[s.value for s in Strategy],
        default=Strategy.MOST_RECENT.value,
    )
    _ = parser.add_argument(
        "--max-batch-bytes",
        type=float,
        default=4e9,
        help="abort if one batch exceeds this; the fan-out caps are the fix",
    )
    _ = parser.add_argument("--log-every", type=int, default=50)
    _ = parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "WARNING: no CUDA device. This will run, but the resident-graph "
            "design assumes a GPU and cuGraph is unavailable on CPU."
        )
    print(f"device: {device}")
    print(describe_fanout(NeighborFanout(), seeds=args.batch_size))
    raise SystemExit(
        "graph loading is wired in tfgnn.stage2.graph.load_from_parquet; this "
        "entry point needs the edge exporter (see docs) to have produced "
        f"{args.graph_dir}. Run the exporter first."
    )


if __name__ == "__main__":
    main()
