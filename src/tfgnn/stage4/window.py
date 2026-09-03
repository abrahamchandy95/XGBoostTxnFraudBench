"""Seed Stage 4's adjacency from TigerGraph, once per window.

======================================================================
THIS IS WHERE TIGERGRAPH BECOMES THE BACKEND
======================================================================
Without it the attention module starts every run from an empty
neighbourhood and only fills in as the stream is consumed -- correct, but
cold, and the graph is never actually consulted. With it, the last-N
adjacency as of a window boundary comes out of TigerGraph and the client
advances it forward from there.

SIX CALLS PER RUN, at the forward-chaining cutoffs the pipeline already
resolves. Per-batch sampling was costed at 1.05 full relation passes per
step -- 21-72 days per epoch. Per window it is the same shape and cost as
build_interaction_edges, which the run already pays.

======================================================================
WHY THE MERGE IS EXACT AND NOT A COARSENING
======================================================================
Everything this query returns is strictly EARLIER than the window bound;
everything the trainer then inserts is at or after it. So last-N of the
union is true last-N -- per-seed exact. That is strictly stronger than
Stage 2's cuGraph route, whose per-fold graph was only "temporally correct
by construction" and let a seed at the start of a fold see edges from its
end (cugraph_backend.py's own header calls that a real coarsening).

======================================================================
GETVID IS NOT THE MODEL'S INDEX SPACE
======================================================================
The query returns raw getvid values. The model indexes cards [0, n_cards)
and merchants [n_cards, total) in NodeSpace order, built from the matrix.
Mapping between them is this module's whole remaining job, and a wrong map
would attach every node's history to some other node -- which trains and
scores without any error at all. Hence `unmapped_*`, reported rather than
silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from tfgnn.stage4.model import TemporalFraudModel
from tfgnn.tigergraph.client import Client

QUERY = "tgn_window_neighbors"


@dataclass(frozen=True)
class WindowLoad:
    pairs: int
    inserted: int
    unmapped_seed: int
    unmapped_neighbour: int


def _arrays(result: list[object]) -> tuple[list[int], list[int]]:
    """The two flat vid columns. Three parallel ListAccum<INT>, by design.

    Flat arrays rather than rows-of-dicts: measured at 1.16 MB / 9.5 ms
    against 2.36 MB / 23.4 ms for the dict form on a 50k-triple payload.
    edge_eseq is returned by the query and deliberately not read here --
    LastNeighborLoader.insert carries no time, and the ORDER of insertion is
    what encodes recency.
    """
    seed: list[int] = []
    nbr: list[int] = []
    for block in result:
        if not isinstance(block, dict):
            continue
        item = cast("dict[str, Any]", block)
        if "seed_vid" in item and "nbr_vid" in item:
            seed = [int(v) for v in cast("list[Any]", item["seed_vid"])]
            nbr = [int(v) for v in cast("list[Any]", item["nbr_vid"])]
    return seed, nbr


def load_window(
    client: Client,
    model: TemporalFraudModel,
    bound_event_seq: int,
    vid_to_index: dict[int, int],
    last_n: int = 10,
    num_batches: int = 8,
    timeout_s: float = 1800.0,
) -> WindowLoad:
    """Fetch last-N-before-bound and seed the model's neighbour loader.

    ``vid_to_index`` maps TigerGraph getvid -> the model's flat index. Build it
    once from a vid export; a vid this run has never seen is COUNTED, not
    dropped quietly, because a systematically empty map would look exactly like
    a graph with no history.
    """
    pairs = 0
    inserted = 0
    unmapped_seed = 0
    unmapped_nbr = 0
    src_all: list[int] = []
    dst_all: list[int] = []

    for batch in range(num_batches):
        result = client.run_installed_detached(
            QUERY,
            {
                "bound_event_seq": bound_event_seq,
                "last_n": last_n,
                "source_batch": batch,
                "num_of_source_batches": num_batches,
            },
            timeout_s=timeout_s,
        )
        seed, nbr = _arrays(result)
        pairs += len(seed)
        for s_vid, n_vid in zip(seed, nbr, strict=True):
            s_ix = vid_to_index.get(s_vid)
            n_ix = vid_to_index.get(n_vid)
            if s_ix is None:
                unmapped_seed += 1
                continue
            if n_ix is None:
                unmapped_nbr += 1
                continue
            src_all.append(s_ix)
            dst_all.append(n_ix)

    if src_all:
        # ONE insert, in the query's returned order. LastNeighborLoader keeps
        # the most RECENT `size` per node by insertion order, and the query
        # emits each seed's heap oldest-to-newest, so a single ordered insert
        # reproduces last-N exactly. Splitting it per batch would still be
        # correct; doing it in ascending event_seq order is what matters.
        model.neighbours.insert(
            torch.tensor(src_all, dtype=torch.long),
            torch.tensor(dst_all, dtype=torch.long),
        )
        inserted = len(src_all)

    load = WindowLoad(pairs, inserted, unmapped_seed, unmapped_nbr)
    print(
        f"  window <{bound_event_seq:,}: {pairs:,} pairs, {inserted:,} inserted"
        + (
            f", UNMAPPED seed {unmapped_seed:,} / nbr {unmapped_nbr:,}"
            if unmapped_seed or unmapped_nbr
            else ""
        )
    )
    if pairs and inserted == 0:
        raise RuntimeError(
            f"{QUERY} returned {pairs:,} pairs and none mapped into the "
            "model's index space. vid_to_index describes a different load "
            "than the matrix -- re-export the vid map."
        )
    return load
