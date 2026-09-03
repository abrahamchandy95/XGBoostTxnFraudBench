from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

from tfgnn.stage2.fanout import NeighborFanout
from tfgnn.stage2.graph import ResidentGraph, TimedCSR
from tfgnn.stage2.id_map import BatchIdMap
from tfgnn.stage2.schema_spec import (
    EdgeTriple,
    assert_time_filtered,
    message_triple_of,
)


class SamplerError(RuntimeError):
    pass


class Strategy(Enum):
    MOST_RECENT = "most_recent"
    UNIFORM = "uniform"


@dataclass(slots=True)
class Batch:
    """One sampled batch: local edge lists plus the map back to global ids."""

    id_map: BatchIdMap
    edge_index: dict[EdgeTriple, Tensor]
    edge_time: dict[EdgeTriple, Tensor]
    global_ids: dict[str, Tensor]
    seed_global: Tensor
    seed_time: Tensor
    edges_sampled: int
    edges_dropped_by_time: int

    def node_counts(self) -> dict[str, int]:
        return {k: int(v.numel()) for k, v in self.global_ids.items()}

    def bytes_held(self) -> int:
        total = self.id_map.bytes_held()
        for mapping in (self.edge_index, self.edge_time):
            for tensor in mapping.values():
                total += tensor.numel() * tensor.element_size()
        return total

    def free(self) -> None:
        """Drop the batch's tensors. Fresh ids next batch, nothing retained."""
        self.id_map.reset()
        self.edge_index.clear()
        self.edge_time.clear()
        self.global_ids.clear()


def _slice_bounds(
    csr: TimedCSR,
    nodes: Tensor,
    cutoff: Tensor,
    cap: int,
    strategy: Strategy,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, int]:
    """Per node, the chosen edge positions. Returns (positions, owner, dropped).

    ``owner[i]`` is the index into ``nodes`` that position ``positions[i]``
    belongs to, so the caller can pair each sampled destination with its source
    without a second lookup.
    """
    start = csr.indptr[nodes]
    end = csr.admissible_end(nodes, cutoff)
    full_end = csr.indptr[nodes + 1]
    dropped = int((full_end - end).sum())

    available = (end - start).clamp(min=0)
    take = torch.minimum(available, torch.full_like(available, cap))
    total = int(take.sum())
    if total == 0:
        empty = torch.empty(0, dtype=torch.int64, device=csr.dst.device)
        return empty, empty, dropped

    owner = torch.repeat_interleave(
        torch.arange(nodes.numel(), device=take.device), take
    )
    cumulative = torch.cumsum(take, dim=0) - take
    within = torch.arange(total, device=take.device) - cumulative[owner]

    if strategy is Strategy.MOST_RECENT:
        # The last `take` admissible edges: a contiguous slice ending at `end`.
        positions = end[owner] - take[owner] + within
    else:
        span = available[owner]
        draw = torch.randint(
            high=1 << 62,
            size=(total,),
            device=take.device,
            dtype=torch.int64,
            generator=generator,
        )
        positions = start[owner] + (draw % span)

    return positions, owner, dropped


@dataclass(slots=True)
class NeighborSampler:
    """Two-hop time-filtered sampler over a resident graph."""

    graph: ResidentGraph
    fanout: NeighborFanout
    strategy: Strategy = Strategy.MOST_RECENT
    generator: torch.Generator | None = None
    _asserted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        assert_time_filtered(self.graph.time_filtered_relations())
        self._asserted = True

    def sample(
        self,
        seed_global: Tensor,
        seed_time: Tensor,
        extra_seeds: dict[str, tuple[Tensor, Tensor]] | None = None,
    ) -> Batch:
        """Expand a seed set into a batch.

        ``seed_global`` are Card global ids; ``seed_time`` their event_seq. Both
        [N] and aligned.

        *** extra_seeds IS REQUIRED FOR LINK PREDICTION, NOT OPTIONAL POLISH. ***
        A discovered bug, worth stating so it is not re-introduced: seeding only
        on the Card does NOT guarantee the positive edge's MERCHANT lands in the
        batch. The time filter excludes the target edge itself -- its event_seq
        equals the seed time, and admission is strictly ``<`` -- so the one edge
        that would have reached that merchant is precisely the one withheld. The
        merchant then has no embedding and the positive cannot be scored.

        That is the sampler behaving correctly. The fix belongs here: pass the
        positive (and negative) merchants as ``extra_seeds`` so they enter the
        batch as NODES, expanded from their own past, while the edge being
        predicted stays excluded. Seeding a node is not leakage; sampling the
        target edge would be.

        ``extra_seeds`` maps node type -> (global_ids, cutoffs), and the cutoffs
        should be the ORIGINATING SEED's time so the same outward-propagation
        rule applies to them.
        """
        if seed_global.numel() != seed_time.numel():
            raise SamplerError("seed ids and seed times must be the same length")
        if not self._asserted:  # pragma: no cover - guarded by __post_init__
            raise SamplerError("temporal guarantee was not asserted")

        device = self.graph.device
        seed_global = seed_global.to(device, torch.int64)
        seed_time = seed_time.to(device, torch.int64)

        caps = self.fanout.as_pyg()
        message_of = message_triple_of()
        reached: dict[str, Tensor] = {"Card": seed_global}
        cutoff: dict[str, Tensor] = {"Card": seed_time}
        for node_type, (ids, times) in (extra_seeds or {}).items():
            ids = ids.to(device, torch.int64)
            times = times.to(device, torch.int64)
            if ids.numel() != times.numel():
                raise SamplerError(
                    f"extra_seeds[{node_type!r}]: ids and cutoffs differ in length"
                )
            if node_type in reached:
                reached[node_type] = torch.cat([reached[node_type], ids])
                cutoff[node_type] = torch.cat([cutoff[node_type], times])
            else:
                reached[node_type] = ids
                cutoff[node_type] = times
        frontier: dict[str, Tensor] = dict(reached)
        frontier_cut: dict[str, Tensor] = dict(cutoff)
        collected: dict[EdgeTriple, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        sampled = 0
        dropped = 0

        for hop in range(self.fanout.hops):
            discovered: dict[str, list[Tensor]] = {}
            discovered_cut: dict[str, list[Tensor]] = {}

            for triple, csr in self.graph.relations.items():
                src_type, _, dst_type = triple
                nodes = frontier.get(src_type)
                if nodes is None or nodes.numel() == 0:
                    continue

                node_cut = frontier_cut[src_type]
                positions, owner, dropped_here = _slice_bounds(
                    csr,
                    nodes,
                    node_cut,
                    caps[triple][hop],
                    self.strategy,
                    self.generator,
                )
                dropped += dropped_here
                if positions.numel() == 0:
                    continue

                walk_from = nodes[owner]
                walk_to = csr.dst[positions]
                edge_t = csr.time[positions]
                inherited = node_cut[owner]
                sampled += int(positions.numel())

                target = message_of[triple]
                previous = collected.get(target)
                if previous is None:
                    collected[target] = (walk_to, walk_from, edge_t, inherited)
                else:
                    collected[target] = (
                        torch.cat([previous[0], walk_to]),
                        torch.cat([previous[1], walk_from]),
                        torch.cat([previous[2], edge_t]),
                        torch.cat([previous[3], inherited]),
                    )

                discovered.setdefault(dst_type, []).append(walk_to)
                discovered_cut.setdefault(dst_type, []).append(inherited)

            next_frontier: dict[str, Tensor] = {}
            next_cut: dict[str, Tensor] = {}
            for node_type, chunks in discovered.items():
                merged = torch.cat(chunks)
                merged_cut = torch.cat(discovered_cut[node_type])
                next_frontier[node_type] = merged
                next_cut[node_type] = merged_cut
                existing = reached.get(node_type)
                if existing is None:
                    reached[node_type] = merged
                    cutoff[node_type] = merged_cut
                else:
                    reached[node_type] = torch.cat([existing, merged])
                    cutoff[node_type] = torch.cat([cutoff[node_type], merged_cut])
            frontier = next_frontier
            frontier_cut = next_cut

        # ---- number the batch, fresh, and rewrite the edge lists ----
        id_map = BatchIdMap(device=device)
        global_ids: dict[str, Tensor] = {}
        for node_type, ids in reached.items():
            _ = id_map.register(node_type, ids)
            global_ids[node_type] = id_map.local_to_global(node_type)

        edge_index: dict[EdgeTriple, Tensor] = {}
        edge_time: dict[EdgeTriple, Tensor] = {}
        for triple, (src_global, dst_global, edge_t, inherited) in collected.items():
            src_type, _, dst_type = triple
            if bool((edge_t >= inherited).any()):
                offenders = int((edge_t >= inherited).sum())
                raise SamplerError(
                    f"{triple}: {offenders} sampled edges are at or after their "
                    "seed's cutoff. The temporal guarantee is broken -- check "
                    "TimedCSR.validate and that hop 2 inherited the SEED time."
                )
            edge_index[triple] = torch.stack(
                [
                    id_map.to_local(src_type, src_global),
                    id_map.to_local(dst_type, dst_global),
                ],
                dim=0,
            )
            edge_time[triple] = edge_t

        return Batch(
            id_map=id_map,
            edge_index=edge_index,
            edge_time=edge_time,
            global_ids=global_ids,
            seed_global=seed_global,
            seed_time=seed_time,
            edges_sampled=sampled,
            edges_dropped_by_time=dropped,
        )
