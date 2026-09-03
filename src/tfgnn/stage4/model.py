"""Stage 4: a TGN that scores a transaction as it arrives.

======================================================================
WHY TGN, AND WHY IT IS NOT A PREFERENCE
======================================================================
Stage 3 measured `row_pair_gap_seconds` -- per-(card, merchant) recency -- at
**15.3% of total gain**, and its arm lifted test AUC-PR by +0.3115 where the
whole 37-column graph family gave +0.0098. That feature is the hand-built
scalar version of exactly what TGN's memory computes natively: a per-node state
updated on every interaction, whose principal input is a time encoding of the
gap since that node's last event.

It also diagnoses why Stage 2 lost 0.0231. The R-GCN produced STATIC
per-snapshot embeddings from a link-prediction objective with no notion of
recency, then handed 128 dense dimensions to a tree that can only threshold one
at a time. TGN inverts all three: the state is dynamic, the signal is recency,
and (4a below) the objective is the task.

======================================================================
TWO HEADS, ONE MEMORY. 4a IS THE POINT; 4b KEEPS THE COMPARISON HONEST.
======================================================================
4a  FraudHead      -- supervised, scores THIS transaction. The user's goal:
                      "actually catch the payment fraud transactions when they
                      happen."
4b  embeddings     -- memory read out per entity, exported for XGBoost as
                      `raw_plus_temporal_embeddings`, sibling to Stage 2's
                      arm and Stage 3's, so the 2x2 stays comparable.

*** 4a DELIBERATELY BREAKS D5. *** D5 is "the label never enters the GNN", and
it is why Stage 2 trained on link prediction instead. The user overrode it for
this arm on 2026-08-03. That is a legitimate decision and it is NOT a licence to
be quiet about it: any 4a number must be reported as breaking D5, because it is
not comparable to Stage 2's on that axis. 4b does not break it -- the embedding
export never sees a label -- which is precisely why 4b is the arm that belongs
in the lift table.

======================================================================
THE TEMPORAL GUARANTEE IS STRUCTURAL HERE, NOT A FILTER
======================================================================
Stage 2 had to FILTER sampled edges against each seed's event_seq (D3a) because
its graph was a static snapshot that contained everything. TGN has no such
graph. Memory is updated by a STRICTLY ORDERED STREAM of events, and a
transaction is scored from memory as it stood BEFORE that transaction was
applied. So "no future information" is a property of the update order rather
than a predicate someone can forget to write.

That is a stronger guarantee, and it is also more fragile in one specific way:
**the stream must be processed in non-decreasing time order, and the score must
be taken BEFORE update_state.** Reverse those two lines and the model reads its
own answer. `_assert_ordered` and the score-then-update sequencing in
`TemporalFraudModel.step` exist for that and must not be "tidied".

event_seq is a dense deterministic rank ordered by (unix_time, id)
(schema.gsql:181), so ordering by event_seq is ordering by time, with ties
broken deterministically. Order on event_seq, not unix_time: same order, no
ties to resolve.

======================================================================
ONE ID SPACE, AND THE REASON IT IS NOT OPTIONAL
======================================================================
`TGNMemory` takes a single `num_nodes` and indexes one flat memory table, so
cards and merchants share an index space here even though the rest of this repo
keeps them typed. Cards occupy [0, n_cards) and merchants
[n_cards, n_cards + n_merchants). `NodeSpace` owns that arithmetic so the
offset appears in exactly one place -- an off-by-one would silently give every
merchant a card's memory, which trains and scores without error and produces a
quietly meaningless model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    TGNMemory,
)


class ChunkedTGNMemory(TGNMemory):
    """TGNMemory with the num_nodes-sized transients bounded.

    PyG's TGNMemory is correct but allocates as though num_nodes is small. A
    2026-08-03 measurement put the SHIPPED cost at 11,406 B/node against an
    irreducible 400 B/node for the memory table itself -- capping a 24 GB card
    at ~1.9M nodes when the table alone would allow ~54M. Almost all of the
    difference is transient, and this subclass bounds the worst of it.

    THE FLUSH. `train(mode=False)` calls `_update_memory(arange(num_nodes))`,
    materialising every node's message in one shot. Measured at 8,249 B/node
    and FLAT in N -- 24x the resident table, allocated once per epoch by the
    eval switch. Chunking makes it a constant instead of a term in N.

    Left alone deliberately: `msg_store`'s per-node dict (a packing change that
    would fork more of PyG than is wise) and the scatter `dim_size` at
    tgn.py:147 (a batch-local dim_size changes index semantics and needs its
    own measurement). Both are recorded in SUMMARY.md as open.
    """

    #: Nodes flushed per slice. 100k x ~8.2 KB is ~825 MB of transient,
    #: constant regardless of graph size.
    flush_chunk: int = 100_000

    def train(self, mode: bool = True) -> "ChunkedTGNMemory":
        if self.training and not mode:
            # Same flush PyG performs, in slices. Order does not matter: each
            # node's update reads only its own message store.
            total = self.memory.size(0)
            for start in range(0, total, self.flush_chunk):
                stop = min(start + self.flush_chunk, total)
                index = torch.arange(start, stop, device=self.memory.device)
                self._update_memory(index)
            self._reset_message_store()
        # Skip TGNMemory.train's own flush; nn.Module.train does the rest.
        return nn.Module.train(self, mode)  # type: ignore[return-value]


@dataclass(frozen=True)
class NodeSpace:
    """The flat index space TGNMemory requires. Cards first, then merchants."""

    n_cards: int
    n_merchants: int

    @property
    def total(self) -> int:
        return self.n_cards + self.n_merchants

    def card(self, index: Tensor) -> Tensor:
        return index

    def merchant(self, index: Tensor) -> Tensor:
        return index + self.n_cards


class GraphAttentionEmbedding(nn.Module):
    """THE PIECE THAT MAKES STAGE 4 A GRAPH MODEL RATHER THAN A SEQUENCE MODEL.

    Memory alone reads two vectors per transaction and updates them. It uses
    NO neighbourhood: graph structure enters only through which pairs happen to
    co-occur. That is a temporal model with per-entity state, not a GNN, and
    saying otherwise would have been the easiest overclaim in this project.

    This attends over each node's recent neighbours, so a card's representation
    depends on WHICH merchants it has lately touched and when. The edge feature
    is a time encoding of (t_seed - t_edge), which is what lets attention
    prefer recent neighbours over old ones -- the neural generalisation of
    row_pair_gap_seconds, the 15.3%-of-gain feature Stage 3 measured.

    NO FAN-OUT CAP IS NEEDED, and that is the structural difference from Stage
    2. The neighbourhood is LAST-N by construction, so the merchant reaching
    51% of cards contributes at most N edges like any other node. Stage 2 had
    to cap because a k-hop expansion from any card reached that hub and then
    most of the graph.
    """

    def __init__(self, memory_dim: int, time_dim: int, out_dim: int) -> None:
        super().__init__()
        self.time_encoder = nn.Linear(1, time_dim)
        self.conv = TransformerConv(
            memory_dim, out_dim // 2, heads=2, dropout=0.1, edge_dim=time_dim
        )

    def forward(
        self, memory: Tensor, last_update: Tensor, edge_index: Tensor, t: Tensor
    ) -> Tensor:
        # Relative time, not absolute: an absolute stamp is a calendar proxy on
        # a temporally split dataset, which is the mistake t_year made in
        # Stage 1 and cost 0.011 AUC-PR to undo.
        # t is shaped PER EDGE, not per node: edge_index[0] indexes edges, so a
        # per-node t mismatches the moment a batch has more nodes than edges
        # -- which the very first batch does, with zero neighbours. Its VALUES
        # are batch-constant ("now" = this batch's first event_seq), which is
        # the standard TGN batching approximation; at 4,096 events out of 22.5M
        # a batch spans a narrow slice, so the error is small. The recency that
        # matters comes from last_update varying per neighbour, not from t.
        # *** THE SUBTRACTION ORDER IS NOT COSMETIC. *** It was backwards until
        # 2026-08-03 and the clamp then destroyed the entire time signal.
        #
        # `t` is NOW (this batch's event_seq); `last_update[neighbour]` is when
        # that neighbour was last active. The stream is non-decreasing and the
        # read happens BEFORE update_state, so last_update <= t ALWAYS. The old
        # `last_update - t` was therefore <= 0 everywhere, `.clamp(min=0)` sent
        # every edge to exactly 0, and `time_encoder(0)` is just its bias --
        # one identical vector for every edge in the batch.
        #
        # Measured on a synthetic ordered stream: 0 of 20,507 edges got a
        # non-zero time feature, raw delta ranging [-708, -1]. So the attention
        # module was aggregating neighbour memory with a CONSTANT edge
        # attribute -- structure but no recency -- while this class's docstring
        # claimed the time encoding was "what lets attention prefer recent
        # neighbours over old ones". It could not. That is the honest reading of
        # why attention + 27 clean columns (0.3824) landed at parity with
        # memory-only + 11 columns (0.3964) instead of above it.
        #
        # Flipped, `t - last_update[neighbour]` is the AGE of that neighbour's
        # state: non-negative by construction, and it varies across a node's
        # neighbours, which is exactly the per-edge recency the attention needs.
        # An unseen neighbour has last_update 0 and so reads as maximally
        # stale, which is correct.
        #
        # log1p stays, and it is load-bearing: event_seq is a dense rank
        # reaching ~2.7e7, and feeding a raw delta of that magnitude into a
        # linear layer is what diverged the first attention run (loss
        # 45.99 -> 4.07 -> 115.93, ROC-AUC exactly 0.5000). log1p bounds it to
        # ~17. The clamp now guards only the tie case (last_update == t).
        age = (t - last_update[edge_index[0]]).to(memory.dtype).clamp(min=0)
        rel = torch.log1p(age).unsqueeze(-1)
        return self.conv(memory, edge_index, self.time_encoder(rel))


class FraudHead(nn.Module):
    """4a. Score one transaction from both endpoints' memory plus its own row.

    Takes the card memory, the merchant memory, their elementwise product, and
    the transaction's own features. The product term is there deliberately:
    Stage 2's post-mortem was that a tree cannot form an interaction between
    two embedding blocks, so handing the model the product rather than hoping
    it is discovered is the one structural lesson that transfers.
    """

    def __init__(self, memory_dim: int, raw_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim * 3 + raw_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self, card_memory: Tensor, merchant_memory: Tensor, raw: Tensor
    ) -> Tensor:
        """Returns LOGITS, not probabilities.

        Logits so the caller can use BCEWithLogitsLoss, which is numerically
        stable and takes pos_weight -- both of which matter at 0.13%
        prevalence, where a plain BCE on probabilities spends its time
        confidently predicting zero.
        """
        features = torch.cat(
            [card_memory, merchant_memory, card_memory * merchant_memory, raw],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class TemporalFraudModel(nn.Module):
    """TGN memory + a fraud head, driven by a time-ordered transaction stream."""

    def __init__(
        self,
        space: NodeSpace,
        raw_dim: int,
        memory_dim: int = 100,
        time_dim: int = 100,
        neighbours: int = 10,
        # ON by default. The TGN paper's model IS memory + graph attention;
        # memory-only is a REDUCED variant, and it is also the variant that
        # does not use the graph at all. Defaulting it off made Stage 4 a
        # temporal sequence model, which defeats the point.
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.space = space
        self.attention = attention
        #: "fraud" = 4a (breaks D5). "link" = 4b (D5-compliant).
        self.objective = "fraud"
        self.memory = ChunkedTGNMemory(
            num_nodes=space.total,
            raw_msg_dim=raw_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        # LAST-N adjacency, advanced by insert() as the stream is consumed and
        # seeded per window from tgn_window_neighbors. O(nodes * N), which the
        # 2026-08-03 sizing put at 168 B/node at N=10 -- affordable up to ~31M
        # nodes, and the project has no intention of exceeding that.
        self.neighbours = LastNeighborLoader(space.total, size=neighbours)
        self.embedder = (
            GraphAttentionEmbedding(memory_dim, time_dim, memory_dim)
            if attention
            else None
        )
        self.head = FraudHead(memory_dim, raw_dim)
        # 4b's D5-compliant head. Both exist on one model so the memory can be
        # trained under either objective without a second architecture.
        self.link_head = LinkHead(memory_dim)

    def reset(self) -> None:
        """Clear memory. Call at the start of every epoch.

        Not optional: memory carried across epochs would let epoch 2 begin with
        state derived from the whole stream, including its own future.
        """
        self.memory.reset_state()
        self.neighbours.reset_state()

    def detach(self) -> None:
        """Cut the graph between batches so backprop does not walk the whole
        stream. TGN's standard idiom; without it memory grows a gradient path
        back to the first event and the run OOMs partway through an epoch."""
        self.memory.detach()

    def step(
        self,
        card_index: Tensor,
        merchant_index: Tensor,
        event_time: Tensor,
        raw: Tensor,
    ) -> Tensor:
        """Score a batch of transactions, THEN fold them into memory.

        *** THE ORDER OF THE TWO BLOCKS BELOW IS THE TEMPORAL GUARANTEE. ***
        Reading memory before update_state is what makes the score depend only
        on strictly earlier events. Swapping them lets each transaction
        contribute to the state used to score itself, which produces an
        excellent loss curve and a worthless model.
        """
        # PyG's TGNMemory keeps `last_update` as a LONG buffer and index-puts
        # into it, so a float time raises "Index put requires the source and
        # destination dtypes match" from inside update_state -- a confusing
        # place to learn it. Caught here instead, and the fix is free: pass
        # event_seq, a dense integer rank over (unix_time, id), which is the
        # ordering this model wants anyway.
        if torch.is_floating_point(event_time):
            raise TypeError(
                "event_time must be an INTEGER tensor -- TGNMemory stores "
                "last_update as Long. Pass event_seq (a dense rank over "
                "(unix_time, id)) rather than a float unix_time; it is the "
                "same order with ties already broken."
            )

        src = self.space.card(card_index)
        dst = self.space.merchant(merchant_index)

        # --- read: memory as it stood BEFORE these events ---
        seeds = torch.cat([src, dst])
        if self.embedder is None:
            memory, _ = self.memory(seeds)
            card_memory = memory[: src.numel()]
            merchant_memory = memory[src.numel() :]
        else:
            # n_id is the seeds PLUS their sampled neighbours; the loader's
            # assoc maps global ids into this batch-local block, which is the
            # "fresh ids every pass" rule -- a cached global table would pin
            # memory proportional to the whole graph.
            n_id, edge_index, e_id = self.neighbours(seeds.cpu())
            n_id = n_id.to(seeds.device)
            memory, last_update = self.memory(n_id)
            embedded = self.embedder(
                memory,
                last_update,
                edge_index.to(seeds.device),
                event_time[0].repeat(edge_index.size(1)),
            )
            # _assoc, not assoc: PyG names it private but __call__ populates
            # it as the global -> batch-local map and there is no public
            # accessor. Pinned here so a PyG upgrade fails loudly.
            local = self.neighbours._assoc[seeds.cpu()].to(seeds.device)
            card_memory = embedded[local[: src.numel()]]
            merchant_memory = embedded[local[src.numel() :]]

        if self.objective == "link":
            # Positives are the real pairs; negatives shuffle the merchant
            # side within the batch. No label is read on this path.
            shuffled = merchant_memory[torch.randperm(merchant_memory.size(0))]
            logits = torch.cat(
                [
                    self.link_head(card_memory, merchant_memory),
                    self.link_head(card_memory, shuffled),
                ]
            )
        else:
            logits = self.head(card_memory, merchant_memory, raw)

        # --- write: only now do these events exist ---
        self.memory.update_state(src, dst, event_time, raw)
        self.neighbours.insert(src.cpu(), dst.cpu())
        return logits

    @torch.no_grad()
    def entity_embeddings(self) -> tuple[Tensor, Tensor]:
        """4b. Current memory, split back into (cards, merchants).

        Read AFTER a pass over a build's stream, so it is the state implied by
        every event below that build's cutoff and nothing later.
        `replay_with_snapshots` is what calls this at each cutoff.

        "NO LABEL HAS TOUCHED THIS PATH" USED TO BE WRITTEN HERE AND IT WAS ONLY
        HALF TRUE. The read is label-free, but under `--objective fraud` the
        memory being read was shaped by BCE gradients on is_fraud, so the vectors
        are label-shaped even though the export is not. Only `--objective link`
        gives a D5-compliant table; run.py sends a fraud run's export to
        embeddings_fraud/, which the lift table does not read.
        """
        index = torch.arange(self.space.total, device=self.memory.memory.device)
        memory, _ = self.memory(index)
        return memory[: self.space.n_cards], memory[self.space.n_cards :]


class LinkHead(nn.Module):
    """4b's objective. D5-COMPLIANT: the fraud label never enters it.

    THIS EXISTS BECAUSE 4b WAS NOT D5-COMPLIANT AND WAS DOCUMENTED AS THOUGH IT
    WERE. The embedding EXPORT never reads a label -- which is what was
    checked -- but the memory it exports was shaped by gradients from
    BCEWithLogitsLoss on is_fraud. Label-trained weights produce label-shaped
    embeddings, so `raw_plus_temporal_embeddings` was being compared against
    Stage 2's genuinely self-supervised arm on unequal terms.

    Scores whether a (card, merchant) interaction is real, against a sampled
    negative -- the same DistMult-free formulation as Stage 2's decoder, over
    TGN memory instead of static embeddings. Trained with BCE on
    positives-vs-negatives, so the only supervision is the graph's own
    structure.
    """

    def __init__(self, memory_dim: int) -> None:
        super().__init__()
        self.relation = nn.Parameter(torch.ones(memory_dim))

    def forward(self, card_memory: Tensor, merchant_memory: Tensor) -> Tensor:
        return (card_memory * self.relation * merchant_memory).sum(dim=-1)


def assert_ordered(event_time: Tensor) -> None:
    """Refuse an out-of-order stream.

    The temporal guarantee is the update ORDER, so a shuffled stream silently
    destroys it -- and a per-epoch shuffle is the reflex on every other trainer
    in this repo. This is the tripwire for someone adding one here.
    """
    if event_time.numel() > 1 and bool((event_time[1:] < event_time[:-1]).any()):
        raise ValueError(
            "the event stream is not in non-decreasing time order. TGN's "
            "temporal guarantee IS the update order -- there is no per-seed "
            "filter to fall back on, unlike Stage 2's sampler. Sort on "
            "event_seq (a dense rank over (unix_time, id)) and do not shuffle."
        )
