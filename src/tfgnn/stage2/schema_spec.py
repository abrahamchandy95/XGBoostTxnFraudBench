from dataclasses import dataclass

NodeType = str
EdgeTriple = tuple[str, str, str]


class SchemaSpecError(ValueError):
    pass


# ======================================================================
# vertices
# ======================================================================
CARD = "Card"
MERCHANT = "Merchant"
TRANSACTION = "Payment_Transaction"
CATEGORY = "Merchant_Category"
PARTY = "Party"


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """One relation, with the reverse TigerGraph declares for it.

    ``reverse_name`` is the name in the DATABASE. It is not the same thing as
    PyG's ``rev_`` convention: PyG generates reverses itself when asked, and
    conflating the two is how an exclusion silently misses half a pair.
    ``time_attr`` is mandatory -- D3a: a relation exported without a time
    attribute is sampled with NO temporal constraint.
    """

    name: str
    src: NodeType
    dst: NodeType
    reverse_name: str | None
    undirected: bool
    time_attr: str

    def __post_init__(self) -> None:
        if self.undirected and self.reverse_name is not None:
            raise SchemaSpecError(
                f"{self.name}: an undirected edge is one object in TigerGraph "
                "and has no separate reverse to exclude."
            )

    @property
    def triple(self) -> EdgeTriple:
        return (self.src, self.name, self.dst)

    @property
    def reverse_triple(self) -> EdgeTriple | None:
        if self.reverse_name is None:
            return None
        return (self.dst, self.reverse_name, self.src)


# The five DIRECTED relations that declare a REVERSE_EDGE in schema r3, plus
# the undirected ones Stage 2 can use. Verified against gsql/schema/schema.gsql
# rather than remembered: reverse names are at :574, :579, :627, :976, :981.
_SPECS: tuple[EdgeSpec, ...] = (
    # ---- THE TARGET ----
    EdgeSpec(
        name="Card_Merchant_Transaction",
        src=CARD,
        dst=MERCHANT,
        reverse_name="Merchant_Transaction_From_Card",
        undirected=False,
        time_attr="edge_event_seq",
    ),
    # ---- the transaction-as-vertex path: RECONSTRUCTS THE TARGET ----
    EdgeSpec(
        name="Card_Send_Transaction",
        src=CARD,
        dst=TRANSACTION,
        reverse_name="Transaction_From_Card",
        undirected=False,
        time_attr="edge_event_seq",
    ),
    EdgeSpec(
        name="Transaction_To_Merchant",
        src=TRANSACTION,
        dst=MERCHANT,
        reverse_name="Merchant_Received_Transaction",
        undirected=False,
        time_attr="edge_event_seq",
    ),
    # ---- the deduplicated pair graph: ALSO RECONSTRUCTS THE TARGET ----
    # first_event_seq, NOT last_event_seq, and this is the one field on this
    # spec that must not be "corrected" back. The edge is one row per (card,
    # merchant) PAIR summarising many transactions, so the question it answers
    # is "had these two interacted before t" -- and the pair has existed since
    # its FIRST transaction. Filtering on last_event_seq deletes the whole pair
    # whenever its most recent transaction is in the future, taking its
    # legitimate past with it and biasing the graph toward dormant
    # relationships. This said last_event_seq until 2026-07-31 while the export
    # read first_event_seq; the export was right and this was inert, because
    # nothing calls time_attr_for.
    EdgeSpec(
        name="Has_Interaction_With_Merchant",
        src=CARD,
        dst=MERCHANT,
        reverse_name=None,
        undirected=True,
        time_attr="first_event_seq",
    ),
    # ---- derived from the above, so transitively reconstructs it ----
    # time_attr is EMPTY, and that is the honest declaration: schema.gsql gives
    # Card_Card exactly one attribute, `weight INT`. There is no time on this
    # edge to name -- its weight is a per-BUILD count of shared merchants, not
    # an event -- which is precisely why it is in NOT_TIME_FILTERABLE. Naming
    # `last_event_seq` here, as this did, invented a column.
    EdgeSpec(
        name="Card_Card",
        src=CARD,
        dst=CARD,
        reverse_name=None,
        undirected=True,
        time_attr="",
    ),
    # ---- SAFE to message-pass over: no card-merchant event in them ----
    EdgeSpec(
        name="Party_Has_Card",
        src=PARTY,
        dst=CARD,
        reverse_name="Card_Belongs_To_Party",
        undirected=False,
        time_attr="edge_event_seq",
    ),
    EdgeSpec(
        name="Party_Is_Merchant",
        src=PARTY,
        dst=MERCHANT,
        reverse_name="Merchant_Owned_By_Party",
        undirected=False,
        time_attr="edge_event_seq",
    ),
    # Merchant_Assigned carries NO edge attributes: a merchant is born with
    # its category, so the export stamps the MERCHANT's first_seen_event_seq
    # as the edge time -- the assignment is observable exactly when the
    # merchant is. time_attr names that source, not an edge attribute.
    EdgeSpec(
        name="Merchant_Assigned",
        src=MERCHANT,
        dst=CATEGORY,
        reverse_name=None,
        undirected=True,
        time_attr="first_seen_event_seq",
    ),
    # ---- derived co-use relations: ring structure, NOT database edge
    # types. export_relation_edges walks Party -(Party_Has_*)- endpoint
    # -(Party_Has_*)- Party at export time, so the schema is unchanged.
    # An edge exists only when two parties hold the same Device/IP, which
    # is what makes it discriminative: rings share endpoints at p=0.70 /
    # 0.60 against ~5% household noise. Time is the MAX of the two
    # attachment stamps -- sharing becomes observable when the SECOND
    # party attaches -- so the per-seed `< cutoff` rule applies cleanly.
    # The Device/IP vertices themselves are deliberately NOT node types
    # here: with featureless endpoints and mean aggregation, a bipartite
    # Party->Device edge can only ever deliver a constant "has a device"
    # bit; the party-to-party projection is what makes CO-USE readable
    # within two hops. (Device/IP were also removed from the identity
    # match surface for the same underlying reason: co-use is not
    # identity. See gsql/identity/03_match_parties.gsql.)
    EdgeSpec(
        name="Party_Shares_Device",
        src=PARTY,
        dst=PARTY,
        reverse_name=None,
        undirected=True,
        time_attr="edge_event_seq",
    ),
    EdgeSpec(
        name="Party_Shares_IP",
        src=PARTY,
        dst=PARTY,
        reverse_name=None,
        undirected=True,
        time_attr="edge_event_seq",
    ),
)

_BY_NAME: dict[str, EdgeSpec] = {spec.name: spec for spec in _SPECS}

#: The link-prediction target.
TARGET_RELATION = "Card_Merchant_Transaction"

#: Relations expressing the SAME EVENT as the target. NOT a delete list -- see
#: the module docstring. These are the relations for which the sampler's
#: ``edge_event_seq < seed_event_seq`` filter is LOAD-BEARING: sampled without
#: it, any one of them hands the model the answer. Everything here must be
#: asserted time-filtered before training starts.
#:
#: A model that scores ~1.0 with a flat loss curve is the signature of this
#: filter being absent on one of these relations.
TARGET_EQUIVALENCE_GROUP: frozenset[str] = frozenset(
    {
        "Card_Merchant_Transaction",
        "Card_Send_Transaction",
        "Transaction_To_Merchant",
        "Has_Interaction_With_Merchant",
    }
)

#: Excluded from message passing on UTILITY grounds, not leakage grounds, and
#: the difference matters. Card_Card is derived per-BUILD rather than per-event,
#: so its weights cannot be time-filtered at all -- but on this dataset it is
#: also measured degenerate (92% dense, 628 of 629 components are isolated
#: pairs), so it carries nothing worth the risk. If the projection is fixed
#: upstream (merchant churn would do it: S = 2.72 -> ~0.10), revisit this as a
#: leakage question, because the utility argument will no longer hold.
NOT_TIME_FILTERABLE: frozenset[str] = frozenset({"Card_Card"})


def spec(name: str) -> EdgeSpec:
    found = _BY_NAME.get(name)
    if found is None:
        raise SchemaSpecError(f"unknown relation {name!r}; known: {sorted(_BY_NAME)}")
    return found


def all_specs() -> tuple[EdgeSpec, ...]:
    return _SPECS


def exclusion_pairs() -> tuple[tuple[EdgeTriple, EdgeTriple | None], ...]:
    """The ``(type, reverse_type)`` pairs to drop, per D3c.

    Returned as PAIRS rather than a flat list on purpose: the pairing is the
    invariant. Excluding a directed edge without its reverse leaks the edge
    straight back in, and a flat list makes that mistake unreviewable.
    """
    return tuple(
        (item.triple, item.reverse_triple)
        for item in _SPECS
        if item.name in TARGET_EQUIVALENCE_GROUP
    )


def message_passing_specs() -> tuple[EdgeSpec, ...]:
    """Relations the R-GCN may pass messages over. The one sanctioned source.

    Keeps the transaction-as-vertex path -- Card -> Payment_Transaction ->
    Merchant -- because that is where the structure AND the features are:
    Payment_Transaction carries amount, error and is_online, and Stage 1
    measured is_online alone at 25% of total gain. Card and Merchant have no
    intrinsic features at all, so without transaction nodes the graph is nearly
    featureless and D3d's no-ID-embedding rule leaves nothing to initialise
    from.

    Only ``NOT_TIME_FILTERABLE`` relations are withheld, and on utility grounds.
    The target's own edges are excluded PER BATCH by the sampler's time filter,
    not here -- see the module docstring.

    *** THE SAMPLER MUST ENFORCE edge_event_seq < seed_event_seq ON EVERY
    RELATION IN TARGET_EQUIVALENCE_GROUP. *** Without it this set hands the
    model the answer, and the failure is silent in the loss curve.
    ``assert_time_filtered`` exists to be called before training.
    """
    return tuple(item for item in _SPECS if item.name not in NOT_TIME_FILTERABLE)


def assert_time_filtered(filtered: set[str]) -> None:
    """Fail before training if a leak-critical relation is sampled untimed.

    ``filtered`` is the set of relation names the sampler actually applies
    ``edge_event_seq < seed_event_seq`` to. Call this from the training entry
    point with the sampler's own record of what it filtered -- not with a
    hand-written list, which would assert only that someone typed the names
    twice.
    """
    passing = {item.name for item in message_passing_specs()}
    unguarded = sorted((TARGET_EQUIVALENCE_GROUP & passing) - filtered)
    if unguarded:
        raise SchemaSpecError(
            f"these relations are in the message-passing set but are NOT "
            f"time-filtered: {unguarded}. Each one expresses the same event as "
            f"{TARGET_RELATION}, so an untimed sample hands the model the "
            "answer and the loss curve will look excellent (D3a, D3c)."
        )


def spec_for_triple(triple: EdgeTriple) -> tuple[EdgeSpec, bool]:
    """The spec that owns a metadata triple, and whether it is the FORWARD one.

    ``pyg_metadata`` emits three kinds of triple and only one of them carries
    the exported relation's own name:

      forward            (src, name, dst)          name == spec.name
      directed reverse   (dst, reverse_name, src)   name != spec.name
      undirected mirror  (dst, name, src)           name == spec.name

    A loader that keys on ``triple[1]`` alone therefore resolves the forward and
    mirrored triples and SILENTLY SKIPS every directed reverse -- which is
    exactly the bug this function exists to make unrepresentable. ``load_edges``
    dropped Card_Belongs_To_Party, Transaction_From_Card,
    Merchant_Transaction_From_Card, Merchant_Received_Transaction and
    Merchant_Owned_By_Party that way: 11 of 16 triples reached the graph, and
    both backends then refused to build one.

    The bool is what the caller needs to know whether to swap the exported
    (src, dst) columns, since only the forward direction is ever exported.
    """
    for item in message_passing_specs():
        if item.triple == triple:
            return item, True
        if item.reverse_triple == triple:
            return item, False
        if item.undirected and (item.dst, item.name, item.src) == triple:
            # A self-relation's mirror IS its forward triple, and its export
            # already carries both ordered orientations, so it must not swap.
            return item, item.src == item.dst
    raise SchemaSpecError(
        f"{triple} is not a metadata triple of any message-passing spec; "
        f"known relations: {sorted(_BY_NAME)}"
    )


def message_triple_of() -> dict[EdgeTriple, EdgeTriple]:
    """Map each SAMPLING triple to the triple its edges message under.

    Expanding a frontier of type A over the triple (A, rel, B) walks A's
    out-neighbors -- but a GNN layer aggregates messages INTO a node, so
    the sampled edges must enter the batch pointing back at the frontier:
    (b -> a) under the PAIRED triple. For a directed relation that pair is
    the database reverse; for an undirected relation it is the mirrored
    triple; for an undirected self-relation it is the triple itself with
    the endpoints swapped at storage time.

    Without this flip the batch's messages flow AWAY from the seeds:
    everything discovered at hop 2 becomes dead weight whose
    representation cannot reach a seed embedding, and the model's "two
    layers matching the sampler's two hops" contract is silently false.
    Both samplers apply it -- sampler.NeighborSampler when storing sampled
    edges, cugraph_backend when building the fold graph reversed.
    """
    out: dict[EdgeTriple, EdgeTriple] = {}
    for item in message_passing_specs():
        forward = item.triple
        reverse = item.reverse_triple
        if reverse is not None:
            out[forward] = reverse
            out[reverse] = forward
        elif item.undirected and item.src != item.dst:
            mirror = (item.dst, item.name, item.src)
            out[forward] = mirror
            out[mirror] = forward
        else:
            # Undirected self-relation: one triple, one CSR holding both
            # ordered orientations; the swap happens within the same key.
            out[forward] = forward
    return out


def pyg_metadata() -> tuple[list[NodeType], list[EdgeTriple]]:
    """PyG's ``metadata`` tuple: node types, then edge triples.

    There is no ``include_target`` flag. An earlier version had one, because the
    target was deleted from message passing and the loss needed it added back.
    Now the target relation is present by construction -- excluded per batch by
    the time filter, not by relation surgery -- so the loss and the convolutions
    read the SAME metadata. One source, no flag to get wrong.
    """
    specs = list(message_passing_specs())

    triples: list[EdgeTriple] = []
    for item in specs:
        triples.append(item.triple)
        reverse = item.reverse_triple
        if reverse is not None:
            triples.append(reverse)
        elif item.undirected and item.src != item.dst:
            # An undirected TigerGraph edge is ONE object, but PyG needs both
            # directions present to pass messages both ways. Emitting the
            # mirrored triple here is not the same as a database reverse and
            # must not be fed to exclusion_pairs().
            triples.append((item.dst, item.name, item.src))

    node_types = sorted({t for item in specs for t in (item.src, item.dst)})
    return node_types, triples


def _assert_contract() -> None:
    """Run at import. A schema drift should fail here, not mid-training."""
    for name in TARGET_EQUIVALENCE_GROUP:
        _ = spec(name)
    if TARGET_RELATION not in TARGET_EQUIVALENCE_GROUP:
        raise SchemaSpecError(
            "the target relation must be in its own equivalence group"
        )
    surviving = {item.name for item in message_passing_specs()}
    # Target-equivalent relations ARE expected here: they are time-filtered per
    # batch, not deleted. PAST edges of the target type are legitimate history
    # and standard in temporal link prediction; only the edges being SCORED are
    # withheld, by the time filter. What must not survive is a relation whose
    # time filter cannot be applied at all.
    if surviving & NOT_TIME_FILTERABLE:
        raise SchemaSpecError(
            f"message_passing_specs includes relations that cannot be "
            f"time-filtered: {sorted(surviving & NOT_TIME_FILTERABLE)}. Their "
            "weights are per-build, not per-event, so no sampler filter can "
            "make them cutoff-correct."
        )
    if not TARGET_EQUIVALENCE_GROUP & surviving:
        raise SchemaSpecError(
            "no target-equivalent relation survives into message passing, so "
            "the R-GCN has no structure expressing card-merchant activity. "
            "That was a real bug once; do not reintroduce it by widening "
            "NOT_TIME_FILTERABLE."
        )
    # TIME ATTRIBUTES, BOTH DIRECTIONS. A relation that carries messages must
    # name the attribute its time comes from (D3a: exported untimed means
    # sampled with no temporal constraint). A relation in NOT_TIME_FILTERABLE
    # must name NOTHING, because it has no time -- and inventing a plausible
    # attribute name to satisfy a mandatory field is how Card_Card came to
    # declare `last_event_seq`, which schema.gsql does not give it.
    for item in _SPECS:
        untimeable = item.name in NOT_TIME_FILTERABLE
        if not untimeable and not item.time_attr:
            raise SchemaSpecError(
                f"{item.name} carries messages but names no time attribute, so "
                "the sampler would apply no temporal constraint to it "
                "(DECISIONS_LEAK_CONTROL D3a)."
            )
        if untimeable and item.time_attr:
            raise SchemaSpecError(
                f"{item.name} is in NOT_TIME_FILTERABLE -- it has no per-event "
                f"time -- yet declares time_attr={item.time_attr!r}. Naming an "
                "attribute here claims a filter that cannot be applied."
            )
    # Every directed spec must name a reverse, or D3c's pairing is impossible.
    for item in _SPECS:
        if not item.undirected and item.reverse_name is None:
            raise SchemaSpecError(
                f"{item.name} is directed but declares no reverse; D3c's "
                "(type, reverse_type) exclusion cannot be formed"
            )
    _, triples = pyg_metadata()
    if len(triples) != len(set(triples)):
        duplicates = sorted({t for t in triples if triples.count(t) > 1})
        raise SchemaSpecError(f"duplicate edge triples in metadata: {duplicates}")


_assert_contract()
