#!/usr/bin/env python
"""Offline Stage 2 sampler + model smoke. CPU-only; no TigerGraph, no cuGraph.

Proves, on a five-node synthetic graph, the two properties the derived
co-use relations depend on -- and which the sampler did NOT have before the
orientation fix (see sampler.py's module docstring):

  1. ORIENTATION. Sampled edges enter the batch pointing AT the frontier.
     Hop 1 from seed card C0 discovers its owner party P0; hop 2 expands P0
     over Party_Shares_Device and finds ring partner P1. The batch must
     contain the edge (P1 -> P0), because that is the direction a conv
     layer needs to move P1's existence into P0's representation and then
     into C0's. Under the old storage the same walk produced (P0 -> P1),
     which no 2-layer model can route back to C0.

  2. CAPTURE. With a fixed-weight RGCN, C0's embedding must CHANGE when the
     Party_Shares_Device edge is removed from the batch. This is the
     end-to-end check: the ring signal does not merely land in the batch,
     it reaches the seed embedding. This assertion FAILS on the
     pre-fix sampler, which is the point -- it is the firing branch.

  3. LEAK. A sharing edge stamped AT or AFTER the seed's cutoff must not be
     sampled at all. Sharing observable at rank 15 is invisible to a seed
     at rank 10.

Run:  .venv/bin/python scripts/smoke_stage2_sampler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tfgnn.stage2.fanout import NeighborFanout
from tfgnn.stage2.graph import ResidentGraph, TimedCSR, build_timed_csr
from tfgnn.stage2.sampler import Batch, NeighborSampler
from tfgnn.stage2.schema_spec import EdgeTriple, pyg_metadata

Rows = list[tuple[int, int, int]]

DEVICE = torch.device("cpu")

# Card C0 (seed) is owned by party P0; card C1 by party P1. P0 and P1 share
# a device, observable from rank `share_time`. Seed time is 10.
NODE_COUNTS = {
    "Card": 2,
    "Party": 2,
    "Merchant": 1,
    "Payment_Transaction": 1,
    "Merchant_Category": 1,
}
PSD: EdgeTriple = ("Party", "Party_Shares_Device", "Party")
PHC: EdgeTriple = ("Party", "Party_Has_Card", "Card")
CBP: EdgeTriple = ("Card", "Card_Belongs_To_Party", "Party")


def _csr(triple: EdgeTriple, rows: Rows) -> TimedCSR:
    src, dst, time = (
        (torch.tensor(col, dtype=torch.int64) for col in zip(*rows))
        if rows
        else (torch.empty(0, dtype=torch.int64) for _ in range(3))
    )
    return build_timed_csr(
        triple,
        *list((src, dst, time)),
        num_src=NODE_COUNTS[triple[0]],
        num_dst=NODE_COUNTS[triple[2]],
        device=DEVICE,
    )


def build_graph(share_time: int) -> ResidentGraph:
    rows: dict[EdgeTriple, Rows] = {
        PHC: [(0, 0, 1), (1, 1, 1)],
        CBP: [(0, 0, 1), (1, 1, 1)],
        # ONE CSR holding BOTH orientations, exactly as the export emits
        # a self-relation.
        PSD: [(0, 1, share_time), (1, 0, share_time)],
    }
    _, triples = pyg_metadata()
    relations = {t: _csr(t, rows.get(t, [])) for t in triples}
    return ResidentGraph(
        relations=relations, node_counts=dict(NODE_COUNTS), device=DEVICE
    )


def edge_pairs(
    batch: Batch, triple: EdgeTriple, src_type: str, dst_type: str
) -> set[tuple[int, int]]:
    """Sampled edges of `triple` as (src_global, dst_global) pairs."""
    index = batch.edge_index.get(triple)
    if index is None:
        return set()
    src = batch.id_map.to_global(src_type, index[0])
    dst = batch.id_map.to_global(dst_type, index[1])
    return set(zip(src.tolist(), dst.tolist()))


def main() -> int:
    seeds = torch.tensor([0], dtype=torch.int64)
    seed_time = torch.tensor([10], dtype=torch.int64)

    sampler = NeighborSampler(graph=build_graph(share_time=5), fanout=NeighborFanout())
    batch = sampler.sample(seeds, seed_time)

    # 1. ORIENTATION: messages point at the frontier.
    psd = edge_pairs(batch, PSD, "Party", "Party")
    assert (1, 0) in psd, (
        f"ring edge (P1 -> P0) missing from the batch; got {psd}. The sampler "
        "is storing walk direction instead of message direction."
    )
    assert (0, 1) not in psd, (
        "the outward orientation (P0 -> P1) was stored; P1 is a hop-2 "
        "discovery and is never expanded, so this edge should not exist."
    )
    phc = edge_pairs(batch, PHC, "Party", "Card")
    assert (0, 0) in phc, (
        f"ownership edge (P0 -> C0) missing; got {phc}. Hop-1 edges must "
        "point into the seeds."
    )

    # 2. CAPTURE: the ring partner reaches the seed embedding.
    try:
        from tfgnn.stage2.model import RGCN
    except ImportError as exc:  # torch_geometric absent: sampler checks stand
        print(f"NOTE model check skipped (torch_geometric unavailable: {exc})")
    else:
        torch.manual_seed(0)
        model = RGCN(raw_dims={}, hidden=16, out=8)
        model.eval()
        num_nodes = batch.node_counts()
        with torch.no_grad():
            with_ring = model({}, batch.edge_index, num_nodes)["Card"]
            without = model(
                {},
                {k: v for k, v in batch.edge_index.items() if k != PSD},
                num_nodes,
            )["Card"]
        seed_local = int(batch.id_map.to_local("Card", seeds)[0])
        delta = float((with_ring[seed_local] - without[seed_local]).abs().max())
        assert delta > 1e-7, (
            "removing the Party_Shares_Device edge did not change the seed "
            "card's embedding: the ring signal is sampled but not captured. "
            "This is the pre-orientation-fix behaviour."
        )
        print(f"ok   capture: seed embedding moves by {delta:.2e} with the ring edge")

    # 3. LEAK: sharing observable at rank 15 is invisible to a seed at 10.
    late = NeighborSampler(graph=build_graph(share_time=15), fanout=NeighborFanout())
    late_batch = late.sample(seeds, seed_time)
    late_psd = edge_pairs(late_batch, PSD, "Party", "Party")
    assert not late_psd, (
        f"sharing stamped at 15 was sampled for a seed at 10: {late_psd}. "
        "The per-seed cutoff is not being applied to the derived relation."
    )

    print("ok   orientation: (P1 -> P0) stored, (P0 -> P1) not")
    print("ok   leak: future sharing invisible to the seed")
    print("SMOKE STAGE 2 SAMPLER: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
