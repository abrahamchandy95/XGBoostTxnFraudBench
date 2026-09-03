from dataclasses import dataclass, field

from tfgnn.stage2.schema_spec import EdgeTriple, message_passing_specs, pyg_metadata
from tfgnn.tuning_provenance import MEASURED_ON, OBSERVATIONS


class FanoutError(ValueError):
    pass


_DEFAULTS: dict[str, tuple[int, int]] = {
    "Card_Merchant_Transaction": (20, 10),
    "Merchant_Transaction_From_Card": (8, 4),
    "Card_Send_Transaction": (25, 10),
    "Transaction_From_Card": (1, 1),
    "Transaction_To_Merchant": (1, 1),
    "Merchant_Received_Transaction": (8, 4),
    "Has_Interaction_With_Merchant": (20, 10),
    "Party_Has_Card": (5, 3),
    "Card_Belongs_To_Party": (1, 1),
    "Party_Is_Merchant": (5, 3),
    "Merchant_Owned_By_Party": (1, 1),
    "Merchant_Assigned": (4, 2),
    "Party_Shares_Device": (10, 5),
    "Party_Shares_IP": (10, 5),
}

_TRIPLE_DEFAULTS: dict[EdgeTriple, tuple[int, int]] = {
    ("Merchant", "Has_Interaction_With_Merchant", "Card"): (8, 4),
}

HUB_DIRECTIONS: frozenset[str] = frozenset(
    {
        "Merchant_Transaction_From_Card",
        "Merchant_Received_Transaction",
        "Merchant_Assigned",
    }
)

HUB_TRIPLES: frozenset[EdgeTriple] = frozenset(
    {
        ("Merchant", "Has_Interaction_With_Merchant", "Card"),
    }
)

HUB_WARN_ABOVE = 16


@dataclass(frozen=True, slots=True)
class NeighborFanout:
    """Per-triple, per-hop neighbor counts.

    ``per_relation`` maps a RELATION NAME to ``(hop1, hop2)``. Keying on the
    name rather than the full triple keeps the literal readable; ``as_pyg()``
    expands it to the ``dict[EdgeType, list[int]]`` PyG wants, which needs the
    triple.

    ``hops`` is fixed at 2. A third hop from a hub reaches essentially every
    node no matter how tight the caps are, so the depth is a structural
    decision about this dataset, not a hyperparameter to sweep.
    """

    per_relation: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(_DEFAULTS)
    )
    # Overrides keyed by FULL TRIPLE, for orientations the name cannot
    # distinguish (an undirected relation's two directions share one name).
    per_triple: dict[EdgeTriple, tuple[int, int]] = field(
        default_factory=lambda: dict(_TRIPLE_DEFAULTS)
    )
    hops: int = 2
    strict: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        known = {item.name for item in message_passing_specs()}
        # Every reverse name is also a valid key, since fanout is per DIRECTION.
        for item in message_passing_specs():
            if item.reverse_name is not None:
                known.add(item.reverse_name)

        unknown = sorted(set(self.per_relation) - known)
        if unknown:
            raise FanoutError(
                f"fanout given for relations that are not in the "
                f"message-passing set: {unknown}. Known: {sorted(known)}"
            )

        _, triples = pyg_metadata()
        missing = sorted({name for _, name, _ in triples} - set(self.per_relation))
        if missing:
            raise FanoutError(
                f"no fanout for {missing}. An unset relation would be sampled "
                "with NO cap, which on this dataset means the batch is most of "
                "the graph -- see the module docstring."
            )

        _, triples = pyg_metadata()
        unknown_triples = sorted(str(t) for t in set(self.per_triple) - set(triples))
        if unknown_triples:
            raise FanoutError(
                f"per_triple fanout given for triples that are not in the "
                f"message-passing metadata: {unknown_triples}"
            )

        shaped: list[tuple[str, tuple[int, int]]] = [
            *sorted(self.per_relation.items()),
            *sorted((str(t), c) for t, c in self.per_triple.items()),
        ]
        for name, counts in shaped:
            if len(counts) != self.hops:
                raise FanoutError(
                    f"{name}: expected {self.hops} hop counts, got {len(counts)}"
                )
            for hop, value in enumerate(counts, start=1):
                if isinstance(value, bool) or value < 1:
                    raise FanoutError(
                        f"{name} hop {hop}: fanout must be an int >= 1, got {value!r}"
                    )
            if counts[1] > counts[0]:
                raise FanoutError(
                    f"{name}: hop-2 fanout {counts[1]} exceeds hop-1 "
                    f"{counts[0]}. Increasing fanout with depth multiplies the "
                    "batch instead of bounding it."
                )

        for name, counts in sorted(self.per_relation.items()):
            if self.strict and name in HUB_DIRECTIONS and counts[0] > HUB_WARN_ABOVE:
                raise FanoutError(
                    f"{name} hop-1 fanout is {counts[0]}, above "
                    f"{HUB_WARN_ABOVE}. This direction points AT a measured hub "
                    "(one merchant reaches 49% of all cards), so the batch "
                    "grows with it. Pass strict=False if this is deliberate."
                )

        for triple in sorted(HUB_TRIPLES):
            effective = self.per_triple.get(
                triple, self.per_relation.get(triple[1], (0, 0))
            )
            if self.strict and effective[0] > HUB_WARN_ABOVE:
                raise FanoutError(
                    f"{triple} hop-1 fanout is {effective[0]}, above "
                    f"{HUB_WARN_ABOVE}. This orientation points AT a measured "
                    "hub and shares its relation name with the narrow "
                    "orientation, so cap it via per_triple. Pass strict=False "
                    "if this is deliberate."
                )

    def as_pyg(self) -> dict[EdgeTriple, list[int]]:
        """The ``num_neighbors`` mapping PyG's loaders and samplers expect.

        Per-triple overrides win over the name-keyed entry, which is what lets
        an undirected relation's two orientations carry asymmetric caps.
        """
        _, triples = pyg_metadata()
        out: dict[EdgeTriple, list[int]] = {}
        for triple in triples:
            counts = self.per_triple.get(triple, self.per_relation[triple[1]])
            out[triple] = list(counts)
        return out

    def worst_case_nodes(self, seeds: int) -> int:
        """Upper bound on nodes touched by one batch. Print this before running.

        Deliberately pessimistic: it assumes every relation expands fully at
        every hop, which no real batch does. The point is that even the
        pessimistic bound must fit in memory -- if it does not, the run is one
        unlucky batch away from an out-of-memory kill, and finding that out at
        setup is much cheaper than finding it out 20 minutes in.

        Sums over TRIPLES (via as_pyg), not over name-keyed entries: a
        name-keyed sum counted an undirected relation once when it expands in
        both orientations, which made the bound optimistic -- the wrong
        direction for a bound to be wrong in.
        """
        caps = self.as_pyg()
        total = seeds
        frontier = seeds
        for hop in range(self.hops):
            per_seed = sum(counts[hop] for counts in caps.values())
            frontier = frontier * per_seed
            total += frontier
        return total


def describe(fanout: NeighborFanout, seeds: int = 1024) -> str:
    lines = [
        f"fanout: {fanout.hops} hops, {len(fanout.per_relation)} directed relations",
    ]
    for name, counts in sorted(fanout.per_relation.items()):
        hub = "  <-- HUB DIRECTION" if name in HUB_DIRECTIONS else ""
        lines.append(f"  {name:<32s} hop1={counts[0]:>3d} hop2={counts[1]:>3d}{hub}")
    for triple, counts in sorted(fanout.per_triple.items()):
        hub = "  <-- HUB DIRECTION" if triple in HUB_TRIPLES else ""
        label = f"{triple[0]}->{triple[1]}->{triple[2]}"
        lines.append(
            f"  {label:<32s} hop1={counts[0]:>3d} hop2={counts[1]:>3d}"
            f"{hub}  (per-triple override)"
        )
    lines.append(
        f"  worst-case nodes for {seeds:,} seeds: "
        f"{fanout.worst_case_nodes(seeds):,} (pessimistic bound)"
    )
    hub_s = OBSERVATIONS["hub_concentration_S"]
    reach = OBSERVATIONS["max_merchant_reach_share"]
    lines.append(
        f"  tuned against hub_concentration_S={hub_s.value:.2f} and "
        f"max_merchant_reach_share={reach.value:.3f}, measured {MEASURED_ON}."
    )
    lines.append(
        "  projection_density_check reports both per build; the Stage 1 "
        "pipeline flags drift."
    )
    return "\n".join(lines)
