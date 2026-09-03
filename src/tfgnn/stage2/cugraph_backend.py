from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from tfgnn.stage2.fanout import NeighborFanout
from tfgnn.stage2.id_map import BatchIdMap
from tfgnn.stage2.schema_spec import (
    EdgeTriple,
    TARGET_EQUIVALENCE_GROUP,
    pyg_metadata,
)

STRICT_PER_SEED = {
    "tfgnn.stage2.sampler.NeighborSampler": "per-seed exact (edge_event_seq < seed time)",
    "tfgnn.stage2.cugraph_backend.CuGraphSampler": "per-FOLD exact (edge_event_seq < fold cutoff)",
}


class CuGraphError(RuntimeError):
    pass


def _import_cugraph() -> tuple[Any, Any]:
    try:
        import cudf
        import cugraph
    except ImportError as exc:
        raise CuGraphError(
            "cugraph and cudf are required for this backend and are Linux+CUDA "
            "only. Install on the GPU box with:\n"
            "  pip install --extra-index-url=https://pypi.nvidia.com "
            "cugraph-cu12 cugraph-pyg-cu12 cudf-cu12\n"
            "For CPU development use tfgnn.stage2.sampler.NeighborSampler, "
            "which is pure torch and per-seed exact."
        ) from exc
    return cugraph, cudf


@dataclass(slots=True)
class FoldGraph:
    """One fold's cuGraph graph: every edge strictly below `cutoff`.

    Relations are collapsed into ONE homogeneous cuGraph graph with an
    ``etype`` column, because cuGraph's sampler is homogeneous. Node ids are
    offset per type into a single space so a sampled vertex can be attributed
    back to its type; ``type_offsets`` is that mapping and it is what makes the
    heterogeneous reconstruction possible.
    """

    cutoff: int
    graph: Any
    type_offsets: dict[str, tuple[int, int]]
    etype_of: dict[int, EdgeTriple]
    num_edges: int

    def node_type_of(self, global_offset: Tensor) -> dict[str, Tensor]:
        """Split a mixed vertex tensor back into per-type LOCAL-to-type ids."""
        out: dict[str, Tensor] = {}
        for node_type, (lo, hi) in self.type_offsets.items():
            mask = (global_offset >= lo) & (global_offset < hi)
            if bool(mask.any()):
                out[node_type] = global_offset[mask] - lo
        return out


def build_fold_graph(
    edges: dict[EdgeTriple, tuple[Tensor, Tensor, Tensor]],
    node_counts: dict[str, int],
    cutoff: int,
) -> FoldGraph:
    """Filter every relation to `event_seq < cutoff` and build one cuGraph graph.

    *** THE FILTER HAPPENS HERE AND NOWHERE ELSE. *** That is the whole point:
    once an edge is in this graph it is admissible for every seed in the fold, so
    cuGraph's sampler needs no temporal awareness. A relation NOT filtered here
    is a leak that no downstream check will catch.
    """
    cugraph, cudf = _import_cugraph()

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for node_type in sorted(node_counts):
        offsets[node_type] = (cursor, cursor + node_counts[node_type])
        cursor += node_counts[node_type]

    src_all: list[Tensor] = []
    dst_all: list[Tensor] = []
    etype_all: list[Tensor] = []
    etype_of: dict[int, EdgeTriple] = {}

    _, expected = pyg_metadata()
    missing = sorted(set(expected) - set(edges))
    if missing:
        raise CuGraphError(
            f"no edges supplied for {missing}. A relation absent from the fold "
            "graph is silently absent from every batch."
        )

    for code, triple in enumerate(expected):
        src_type, relation, dst_type = triple
        src, dst, time = edges[triple]
        etype_of[code] = triple

        keep = time < cutoff
        if relation in TARGET_EQUIVALENCE_GROUP and not bool(keep.all()):
            pass
        src_all.append(dst[keep] + offsets[dst_type][0])
        dst_all.append(src[keep] + offsets[src_type][0])
        etype_all.append(torch.full((int(keep.sum()),), code, dtype=torch.int32))

    src_cat = torch.cat(src_all)
    dst_cat = torch.cat(dst_all)
    if src_cat.numel() == 0:
        raise CuGraphError(
            f"fold cutoff {cutoff} admitted ZERO edges. Every seed would have an "
            "empty neighborhood and the loss would be uninformative."
        )

    frame = cudf.DataFrame(
        {
            "src": cudf.Series(src_cat.cpu().numpy()),
            "dst": cudf.Series(dst_cat.cpu().numpy()),
            "etype": cudf.Series(torch.cat(etype_all).cpu().numpy()),
        }
    )
    graph = cugraph.MultiGraph(directed=True)
    graph.from_cudf_edgelist(
        frame, source="src", destination="dst", edge_type="etype", renumber=False
    )
    return FoldGraph(
        cutoff=cutoff,
        graph=graph,
        type_offsets=offsets,
        etype_of=etype_of,
        num_edges=int(src_cat.numel()),
    )


@dataclass(slots=True)
class CuGraphSampler:
    """Two-hop sampling with cuGraph, per fold.

    Returns the same shape ``sampler.Batch`` consumers expect -- local
    ``edge_index`` per triple plus a fresh ``BatchIdMap`` -- so the model,
    decoder and training loop are unchanged between backends. That symmetry is
    deliberate: it makes the two samplers A/B comparable on identical everything
    else, which is the only way to measure what the coarsening costs.
    """

    fold: FoldGraph
    fanout: NeighborFanout
    device: torch.device

    def sample(self, seed_global: Tensor, seed_type: str = "Card") -> Any:
        """Sample around seeds. NOTE: no per-seed time argument, by design.

        The absence of a time parameter is the honest signal that this backend
        gives per-FOLD rather than per-seed exactness. If a caller wants to pass
        a cutoff, they want ``sampler.NeighborSampler``.
        """
        cugraph, cudf = _import_cugraph()

        lo, _ = self.fold.type_offsets[seed_type]
        seeds = cudf.Series((seed_global + lo).cpu().numpy())

        per_hop = [
            min(counts[hop] for counts in self.fanout.per_relation.values())
            for hop in range(self.fanout.hops)
        ]
        result = cugraph.uniform_neighbor_sample(
            self.fold.graph,
            start_list=seeds,
            fanout_vals=per_hop,
            with_replacement=False,
        )

        src = torch.as_tensor(
            result["sources"].to_numpy(), dtype=torch.int64, device=self.device
        )
        dst = torch.as_tensor(
            result["destinations"].to_numpy(), dtype=torch.int64, device=self.device
        )
        etype = torch.as_tensor(
            result["edge_type"].to_numpy(), dtype=torch.int64, device=self.device
        )

        # Fresh per batch, freed after. Same contract as the torch backend: no
        # global id table is ever materialised.
        id_map = BatchIdMap(device=self.device)
        per_type: dict[str, list[Tensor]] = {}
        for offsets_tensor in (src, dst):
            for node_type, ids in self.fold.node_type_of(offsets_tensor).items():
                per_type.setdefault(node_type, []).append(ids)
        for node_type, chunks in per_type.items():
            _ = id_map.register(node_type, torch.cat(chunks))

        edge_index: dict[EdgeTriple, Tensor] = {}
        for code, triple in self.fold.etype_of.items():
            mask = etype == code
            if not bool(mask.any()):
                continue
            src_type, _, dst_type = triple
            s_lo = self.fold.type_offsets[src_type][0]
            d_lo = self.fold.type_offsets[dst_type][0]
            edge_index[triple] = torch.stack(
                [
                    id_map.to_local(src_type, dst[mask] - s_lo),
                    id_map.to_local(dst_type, src[mask] - d_lo),
                ],
                dim=0,
            )

        from tfgnn.stage2.sampler import Batch

        return Batch(
            id_map=id_map,
            edge_index=edge_index,
            edge_time={},  # per-fold filtering: no per-edge time is carried
            global_ids={t: id_map.local_to_global(t) for t in id_map.registered()},
            seed_global=seed_global,
            seed_time=torch.full_like(seed_global, self.fold.cutoff),
            edges_sampled=int(src.numel()),
            edges_dropped_by_time=0,  # filtered at graph construction, not here
        )


def describe_coarsening(cutoffs: list[int]) -> str:
    """Report the temporal resolution this backend actually provides.

    Print it next to any metric. The number that matters is the fold SPAN: that
    is how far into its own future a seed at the start of a fold can see.
    """
    lines = ["cuGraph backend temporal resolution (per-FOLD, not per-seed):"]
    for index, (lower, upper) in enumerate(zip(cutoffs, cutoffs[1:]), start=1):
        span = upper - lower
        lines.append(
            f"  fold {index}: graph < {lower:,}, serves [{lower:,}, {upper:,}) "
            f"-> a seed at the start can see up to {span:,} transactions ahead"
        )
    lines.append(
        "  Raise stage1_pipeline.train_folds to shrink this, or use "
        "sampler.NeighborSampler for per-seed exactness."
    )
    return "\n".join(lines)
