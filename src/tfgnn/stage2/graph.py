from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from tfgnn.stage2.schema_spec import (
    EdgeTriple,
    message_passing_specs,
    pyg_metadata,
    spec,
)


class GraphError(RuntimeError):
    pass


@dataclass(slots=True)
class TimedCSR:
    """One relation, as CSR with each neighbor segment sorted by time.

    ``indptr`` is [num_src + 1]; ``dst`` and ``time`` are [num_edges], both
    ordered so that ``dst[indptr[i]:indptr[i+1]]`` are node i's neighbors in
    ascending time order.
    """

    triple: EdgeTriple
    indptr: Tensor
    dst: Tensor
    time: Tensor
    num_src: int
    num_dst: int
    key: Tensor
    span: int

    @property
    def num_edges(self) -> int:
        return int(self.dst.numel())

    def bytes_held(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.indptr, self.dst, self.time, self.key)
        )

    def admissible_end(self, nodes: Tensor, cutoff: Tensor) -> Tensor:
        """Per node, the end of the prefix of neighbors strictly before cutoff.

        ``nodes`` and ``cutoff`` are aligned [N] tensors: seed i carries its OWN
        time, which is what makes this per-seed temporal sampling rather than one
        global cutoff. Returns [N] absolute indices into ``dst``, so
        ``[indptr[node], result[i])`` is the admissible slice.

        *** WHY THE COMPOSITE KEY. *** The obvious implementation searches
        ``self.time`` directly -- and it is WRONG, because ``time`` is sorted
        only WITHIN each node's segment, not globally. A global binary search
        over a piecewise-sorted array returns an arbitrary offset, and clamping
        it into the segment turns that arbitrary offset into a plausible-looking
        one. The result admits future edges on some nodes and hides past edges
        on others, with no error raised.

        ``key = src * span + time`` IS globally sorted (the loader sorts on it),
        so searching for ``node * span + cutoff`` lands exactly at the first
        edge of ``node`` whose time is >= cutoff. ``right=False`` gives strict
        ``<``, which is the leak rule: the cutoff is the first rank NOT
        included. The clamp is then a cheap guard, not the mechanism.
        """
        start = self.indptr[nodes]
        end = self.indptr[nodes + 1]
        probe = nodes.to(torch.int64) * self.span + cutoff.to(torch.int64)
        position = torch.searchsorted(self.key, probe, right=False)
        return torch.clamp(position, min=start, max=end)

    def validate(self, sample: int | None = 4096) -> None:
        """Assert the invariants the time filter depends on.

        ``sample`` bounds the per-segment monotonicity check, which is the
        expensive one. Pass None to check every segment -- worth doing once on a
        new export, not every run.
        """
        if self.indptr.numel() != self.num_src + 1:
            raise GraphError(
                f"{self.triple}: indptr has {self.indptr.numel()} entries for "
                f"{self.num_src} source nodes; expected {self.num_src + 1}"
            )
        if int(self.indptr[0]) != 0:
            raise GraphError(f"{self.triple}: indptr must start at 0")
        if int(self.indptr[-1]) != self.num_edges:
            raise GraphError(
                f"{self.triple}: indptr ends at {int(self.indptr[-1])} but there "
                f"are {self.num_edges} edges"
            )
        if bool((self.indptr[1:] < self.indptr[:-1]).any()):
            raise GraphError(f"{self.triple}: indptr is not non-decreasing")
        if self.dst.numel() != self.time.numel():
            raise GraphError(f"{self.triple}: dst and time have different lengths")
        # The composite key is the mechanism admissible_end relies on, so it is
        # checked as hard as the CSR itself. It must be GLOBALLY sorted -- that
        # is the property `time` does not have.
        if self.key.numel() != self.num_edges:
            raise GraphError(f"{self.triple}: key length != edge count")
        if self.num_edges and bool((self.key[1:] < self.key[:-1]).any()):
            raise GraphError(
                f"{self.triple}: the composite key is not globally sorted, so "
                "admissible_end's binary search is invalid and would admit "
                "future edges."
            )
        if self.num_edges and int(self.time.max()) >= self.span:
            raise GraphError(
                f"{self.triple}: span {self.span} does not exceed max time "
                f"{int(self.time.max())}, so the composite key no longer "
                "separates source nodes."
            )
        if self.num_edges and int(self.dst.max()) >= self.num_dst:
            raise GraphError(
                f"{self.triple}: dst id {int(self.dst.max())} is outside "
                f"[0, {self.num_dst})"
            )

        # THE PREFIX PROPERTY. Without this the whole time filter is a no-op
        # that looks like it works.
        if not self.num_edges:
            return
        boundaries = self.indptr
        n = self.num_src
        indices = (
            torch.arange(n, device=self.indptr.device)
            if sample is None or n <= sample
            else torch.linspace(
                0, n - 1, steps=sample, device=self.indptr.device
            ).long()
        )
        for i in indices.tolist():
            lo = int(boundaries[i])
            hi = int(boundaries[i + 1])
            if hi - lo > 1:
                segment = self.time[lo:hi]
                if bool((segment[1:] < segment[:-1]).any()):
                    raise GraphError(
                        f"{self.triple}: source node {i}'s neighbor segment is "
                        "not sorted ascending by time. The time filter reads a "
                        "PREFIX of each segment, so an unsorted segment makes it "
                        "admit future edges silently. Re-sort the export by "
                        "(src, time)."
                    )


def build_timed_csr(
    triple: EdgeTriple,
    src: Tensor,
    dst: Tensor,
    time: Tensor,
    num_src: int,
    num_dst: int,
    device: torch.device,
) -> TimedCSR:
    """Sort by (src, time) and build CSR. The sort IS the invariant."""
    if not (src.numel() == dst.numel() == time.numel()):
        raise GraphError(f"{triple}: src, dst and time must be the same length")

    src = src.to(device, torch.int64)
    dst = dst.to(device, torch.int64)
    time = time.to(device, torch.int64)

    if bool((time < 0).any()):
        raise GraphError(
            f"{triple}: negative event_seq. The composite sort key assumes "
            "non-negative times; a -1 sentinel here would sort before every "
            "real edge and be admitted by every cutoff."
        )

    span = (int(time.max()) + 1) if time.numel() else 1
    key = src * span + time
    if time.numel():
        order = torch.argsort(key)
        src, dst, time, key = src[order], dst[order], time[order], key[order]

    counts = torch.bincount(src, minlength=num_src)
    indptr = torch.zeros(num_src + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=indptr[1:])

    csr = TimedCSR(
        triple=triple,
        indptr=indptr,
        dst=dst,
        time=time,
        num_src=num_src,
        num_dst=num_dst,
        key=key,
        span=span,
    )
    csr.validate()
    return csr


@dataclass(slots=True)
class ResidentGraph:
    """Every message-passing relation, GPU-resident, time-sorted.

    Design B from the memory analysis: the graph stays on the device (~3-6 GB of
    24 GB) and the sampler reads it in place. Per-batch id maps are built fresh
    and freed each batch (see tfgnn.stage2.id_map); the base graph is NOT
    rebuilt per batch, because that would be the slow half of a REST design with
    none of its memory benefit.
    """

    relations: dict[EdgeTriple, TimedCSR]
    node_counts: dict[str, int]
    device: torch.device

    def bytes_held(self) -> int:
        return sum(csr.bytes_held() for csr in self.relations.values())

    def time_filtered_relations(self) -> set[str]:
        """Relation NAMES whose sampling is time-filtered by construction.

        Feed straight into ``schema_spec.assert_time_filtered``. It reports what
        this structure actually guarantees rather than what a config claims,
        which is the difference between an assertion and a comment.
        """
        return {triple[1] for triple in self.relations}

    def validate(self, sample: int | None = 4096) -> None:
        _, expected = pyg_metadata()
        missing = sorted(set(expected) - set(self.relations))
        if missing:
            raise GraphError(
                f"resident graph is missing message-passing relations: "
                f"{missing}. A relation absent from the graph is silently "
                "absent from every batch."
            )
        for csr in self.relations.values():
            csr.validate(sample=sample)

    def describe(self) -> str:
        lines = [
            f"resident graph on {self.device}: "
            + f"{self.bytes_held() / 1e9:.2f} GB across "
            + f"{len(self.relations)} relations"
        ]
        for triple, csr in sorted(self.relations.items()):
            lines.append(
                f"  {triple[0]:<20s} -{triple[1]:<30s}-> {triple[2]:<20s} "
                f"{csr.num_edges:>12,} edges  "
                f"{csr.bytes_held() / 1e6:>8.1f} MB"
            )
        for node_type, count in sorted(self.node_counts.items()):
            lines.append(f"  |{node_type}| = {count:,}")
        return "\n".join(lines)


def time_attr_for(relation: str) -> str:
    """The attribute the loader must read as this relation's time.

    Delegates to EdgeSpec so the answer cannot drift from the schema contract.
    Has_Interaction_With_Merchant returns first_event_seq, and the module
    docstring explains why that is not interchangeable with last_event_seq.
    """
    return spec(relation).time_attr


def expected_relations() -> tuple[EdgeTriple, ...]:
    """Triples the loader must produce, in metadata order."""
    _, triples = pyg_metadata()
    return tuple(triples)


def reverse_of(triple: EdgeTriple) -> EdgeTriple | None:
    """The database reverse of a triple, if it declares one.

    The loader needs this to build both directions from ONE exported edge list:
    a reverse relation is the same edges with src and dst swapped, so exporting
    it separately would double the transfer and risk the two copies disagreeing.
    """
    for item in message_passing_specs():
        if item.triple == triple:
            return item.reverse_triple
        if item.reverse_triple == triple:
            return item.triple
    return None


def load_from_parquet(
    directory: Path,
    device: torch.device,
    node_counts: dict[str, int],
) -> ResidentGraph:
    """Build the resident graph from one parquet file per relation.

    Expects ``<relation>.parquet`` with columns ``src``, ``dst``, ``time``,
    where src/dst are ALREADY dense integer ids per node type. Producing those
    ids is the exporter's job, not this function's: doing it here would mean
    holding a global string->int table, which is exactly the resident structure
    the design forbids.

    Parquet rather than REST on purpose. 54M edges through runInstalledQuery is
    minutes of JSON parsing per run; a parquet round trip is seconds and the
    file can be staged to the GPU box once.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GraphError("pyarrow is required to load the resident graph") from exc

    relations: dict[EdgeTriple, TimedCSR] = {}
    for triple in expected_relations():
        src_type, relation, dst_type = triple
        path = directory / f"{relation}__{src_type}__{dst_type}.parquet"
        if not path.is_file():
            reverse = reverse_of(triple)
            if reverse is None:
                raise GraphError(f"missing edge export: {path}")
            rev_path = directory / f"{reverse[1]}__{reverse[0]}__{reverse[2]}.parquet"
            if not rev_path.is_file():
                raise GraphError(f"missing edge export: {path} and {rev_path}")
            table = pq.read_table(rev_path, columns=["src", "dst", "time"])
            src_col, dst_col = "dst", "src"
        else:
            table = pq.read_table(path, columns=["src", "dst", "time"])
            src_col, dst_col = "src", "dst"

        frame = table.to_pandas()
        relations[triple] = build_timed_csr(
            triple=triple,
            src=torch.as_tensor(frame[src_col].to_numpy(), dtype=torch.int64),
            dst=torch.as_tensor(frame[dst_col].to_numpy(), dtype=torch.int64),
            time=torch.as_tensor(frame["time"].to_numpy(), dtype=torch.int64),
            num_src=node_counts[src_type],
            num_dst=node_counts[dst_type],
            device=device,
        )

    graph = ResidentGraph(
        relations=relations, node_counts=dict(node_counts), device=device
    )
    graph.validate()
    return graph
