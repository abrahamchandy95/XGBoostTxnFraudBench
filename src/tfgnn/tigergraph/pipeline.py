import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

from pydantic import TypeAdapter

from common.config import load_raw_config, project_root
from tfgnn.tigergraph import gates as gate_module
from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.export import (
    EXPORT_PLAN_ADAPTER,
    ExportPlan,
    build_dir,
    export_dimensions,
    prune_stale_builds,
    resolve as resolve_output,
)
from tfgnn.tigergraph.plan import (
    FOLD_SOURCES,
    Build,
    LoadFacts,
    build_plan,
    deduplicate,
    describe,
    load_facts,
    offline_plan,
)
from tfgnn.tigergraph.settings import Settings
from tfgnn.tuning_provenance import drift_notes


# ======================================================================
# configuration
# ======================================================================


class BuildSpec(TypedDict):
    build_id: str
    cutoff: str | int
    serves_splits: list[int]


class IdentityPlan(TypedDict):
    enabled: bool
    threshold: float
    pii_low_connections_limit: int
    pii_high_connections_limit: int
    birthdate_weight: float
    street_address_weight: float
    email_weight: float
    name_weight: float
    phone_weight: float
    identity_document_weight: float
    num_of_source_batches: int
    max_member_count: int
    max_component_fraction: float
    drop_conflicts: bool


class TopologyPlan(TypedDict):
    interaction_batches: int


class ProjectionPlan(TypedDict):
    min_edge_weight: int
    max_merchant_degree: int
    max_card_degree: int
    min_interaction_txn_count: int
    num_of_source_batches: int
    density_top_k: int
    # The candidate-pair budget the configured max_merchant_degree must price
    # under (sum over merchants with deg <= cap of deg*(deg-1)/2). Armed only
    # in projection mode: density_check prints
    # ABORT_capped_candidate_pairs_over_budget and the gate machinery stops
    # the build BEFORE the builders run, instead of wedging the engine
    # mid-pass. 0 disarms.
    pair_budget: float


class GraphFeaturesPlan(TypedDict):
    # Which substrate the structural features (pagerank, c_id/c_size/cc_degree,
    # louvain_*) are computed on:
    #   "projection"  materialize Card_Card / Merchant_Merchant, run the
    #                 per-side readers on them. Cost is sum-over-hubs of
    #                 degree squared; exists only for the A/B against
    #                 "bipartite" and dies with it.
    #   "bipartite"   run wcc_bipartite / pagerank_bipartite /
    #                 louvain_bipartite directly on
    #                 Has_Interaction_With_Merchant. Linear in interaction
    #                 edges; no materialization.
    source: str


GRAPH_FEATURE_SOURCES = ("projection", "bipartite")


class StructurePlan(TypedDict):
    # min_weight is shared by the projection readers (wcc_*, pagerank_*,
    # louvain_*, kcore_*); min_txn_count is shared by the bipartite readers
    # (wcc_bipartite, pagerank_bipartite, louvain_bipartite). ONE value per
    # mode, deliberately: different thresholds would make c_size and pagerank
    # describe different graphs and no reader could tell. See the module note.
    min_weight: float
    min_txn_count: int


class PagerankPlan(TypedDict):
    max_change: float
    maximum_iteration: int
    damping: float


class OptionalPlan(TypedDict):
    louvain: bool
    louvain_max_iteration: int
    kcore: bool
    kcore_max_core: int
    card_interval_max: bool
    card_interval_minutes: int
    card_interval_heap_capacity: int
    card_interval_batches: int


class PipelinePlan(TypedDict):
    builds: list[BuildSpec]
    identity: IdentityPlan
    topology: TopologyPlan
    graph_features: GraphFeaturesPlan
    projection: ProjectionPlan
    structure: StructurePlan
    pagerank: PagerankPlan
    optional: OptionalPlan
    enforce_gates: bool
    export_dimensions: bool
    merge_identical_cutoffs: bool
    # 1 keeps one training snapshot. >1 replaces any build serving only split 0
    # with that many forward-chaining snapshots, so every training row reads
    # features fitted strictly below its own fold instead of over a window
    # containing itself. See tfgnn.tigergraph.plan.Build for the measurement
    # that motivates it. Costs `train_folds - 1` builds and drops fold 0 as
    # warm-up.
    train_folds: int
    # How many bounded transactions reset_build is split into (one sync call
    # per source batch, everything partitioned by getvid residue). A
    # one-transaction reset of a materialized projection is what wedged the
    # live engine for 17+ hours on 2026-07-31.
    reset_batches: int
    # Which authority places the fold boundaries: auto | causal_fold |
    # event_seq. See tfgnn.tigergraph.plan.fold_boundaries. Only read when
    # train_folds > 1.
    fold_source: str


PIPELINE_ADAPTER = TypeAdapter(PipelinePlan)


# ======================================================================
# call plan
# ======================================================================


@dataclass(frozen=True)
class QueryCall:
    name: str
    params: dict[str, object]
    phase: str
    build_id: str
    detached: bool = True
    # Extra keys that must be zero but are not named *_MUST_BE_ZERO in the
    # GSQL. reset_identity's verification pass is the only current user.
    expect_zero: tuple[str, ...] = field(default_factory=tuple)
    # Keys that must be PRESENT in the output, whatever their value. This is
    # the stale-endpoint tripwire: TigerGraph silently ignores unknown query
    # parameters (systemprompt.md section 0a), so an endpoint installed before
    # a query grew a parameter accepts the call and runs the OLD logic -- for
    # reset_build that is the monolithic single-transaction delete the
    # batching exists to prevent, and for an armed density gate it is a gate
    # that never prints and therefore never enforces (gates.py only evaluates
    # keys that appear). A missing key fails the call loudly instead.
    expect_present: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


def _identity_calls(plan: PipelinePlan, build: Build) -> list[QueryCall]:
    """The eight-step identity chain. Order is load-bearing; see the module note."""
    identity = plan["identity"]
    cutoff = build.cutoff_event_seq
    bid = build.build_id

    return [
        # Read-only. Tells you what the two degree caps should be on YOUR
        # data rather than the kit's 100 and 25000.
        QueryCall(
            "measure_pii_degrees",
            {
                "cutoff_event_seq": cutoff,
                "candidate_low_limit": identity["pii_low_connections_limit"],
                "candidate_high_limit": identity["pii_high_connections_limit"],
            },
            "identity",
            bid,
            detached=False,
            note="read degree_one_per_relation: a relation that is almost all "
            "degree 1 has a weight parameter that does nothing",
        ),
        QueryCall(
            "match_parties",
            {
                "cutoff_event_seq": cutoff,
                "birthdate_weight": identity["birthdate_weight"],
                "street_address_weight": identity["street_address_weight"],
                "email_weight": identity["email_weight"],
                "name_weight": identity["name_weight"],
                "phone_weight": identity["phone_weight"],
                "identity_document_weight": identity["identity_document_weight"],
                "num_of_source_batches": identity["num_of_source_batches"],
                "threshold": identity["threshold"],
                "pii_low_connections_limit": identity["pii_low_connections_limit"],
                "pii_high_connections_limit": identity["pii_high_connections_limit"],
            },
            "identity",
            bid,
        ),
        # MUST run between match_parties and unify_parties.
        QueryCall(
            "stamp_same_as_provenance",
            {
                "build_id": bid,
                "cutoff_event_seq": cutoff,
                "pii_high_connections_limit": identity["pii_high_connections_limit"],
            },
            "identity",
            bid,
        ),
        QueryCall(
            "unify_parties",
            {"build_id": bid, "cutoff_event_seq": cutoff, "print_results": False},
            "identity",
            bid,
        ),
        # Abort gate. MUST precede resolved_entity_stats.
        QueryCall(
            "assert_resolved_entity",
            {
                "cutoff_event_seq": cutoff,
                "max_member_count": identity["max_member_count"],
                "max_component_fraction": identity["max_component_fraction"],
            },
            "identity",
            bid,
            detached=False,
        ),
        QueryCall(
            "resolved_entity_stats",
            {"build_id": bid, "cutoff_event_seq": cutoff},
            "identity",
            bid,
        ),
        # The only reason the identity chain can reach a model.
        QueryCall(
            "stamp_resolved_entity_keys",
            {
                "build_id": bid,
                "cutoff_event_seq": cutoff,
                "drop_conflicts": identity["drop_conflicts"],
            },
            "identity",
            bid,
        ),
    ]


def build_calls(plan: PipelinePlan, build: Build) -> list[QueryCall]:
    """The whole call sequence for one snapshot."""
    cutoff = build.cutoff_event_seq
    bid = build.build_id
    projection = plan["projection"]
    structure = plan["structure"]
    pagerank = plan["pagerank"]
    optional = plan["optional"]
    min_weight = structure["min_weight"]
    min_txn_count = structure["min_txn_count"]
    bipartite = plan["graph_features"]["source"] == "bipartite"
    projection_batches = projection["num_of_source_batches"]
    reset_batches = plan["reset_batches"]
    interaction_batches = plan["topology"]["interaction_batches"]

    if projection_batches < 1:
        raise ValueError(
            "stage1_pipeline.projection.num_of_source_batches must be at least 1"
        )
    if reset_batches < 1:
        raise ValueError("stage1_pipeline.reset_batches must be at least 1")
    if interaction_batches < 1:
        raise ValueError(
            "stage1_pipeline.topology.interaction_batches must be at least 1"
        )

    calls: list[QueryCall] = [
        # ---- phase 1: clear derived state ----
        # One bounded transaction per source batch. Clearing everything --
        # a materialized projection included -- inside ONE transaction is
        # what wedged the live engine for 17+ hours on 2026-07-31 (mass
        # DELETEs never reach an abort checkpoint, and an expired query
        # commits nothing, so the time was pure loss). Every reset phase
        # partitions by its source vertex's getvid residue; per-batch
        # counts telescope to the old single-call print. Re-running a
        # batch is idempotent SEMANTICALLY, but a changed reset_batches
        # re-partitions the residues AND re-bases every --from-index, so
        # resume_from_manifest refuses a resume past a build's first
        # reset call when the recorded count differs -- same rule as the
        # projection batches. expect_present is the stale-endpoint
        # tripwire: an old endpoint ignores both batch parameters and
        # runs the monolithic single-transaction delete this batching
        # exists to prevent.
        *[
            QueryCall(
                "reset_build",
                {
                    "source_batch": source_batch,
                    "num_of_source_batches": reset_batches,
                },
                "reset",
                bid,
                expect_present=("source_batch_count",),
                note=f"bounded reset transaction {source_batch + 1}/{reset_batches}",
            )
            for source_batch in range(reset_batches)
        ],
        QueryCall("reset_identity", {"dry_run": False}, "reset", bid, detached=False),
        # Deletes commit at the end of a query, so reset_identity cannot
        # verify its own work. The second call is the only proof.
        QueryCall(
            "reset_identity",
            {"dry_run": True},
            "reset",
            bid,
            detached=False,
            expect_zero=(
                "same_as_edges",
                "resolved_entity_vertices",
                "parties_with_stale_re_id",
                "cards_with_stale_re_id",
                "merchants_with_stale_re_id",
            ),
            note="verification pass; every count must be 0",
        ),
    ]

    # ---- phase 2: entity resolution ----
    if plan["identity"]["enabled"]:
        calls.extend(_identity_calls(plan, build))
    else:
        print(
            "  identity disabled: Card.re_id and Merchant.re_id stay at -1, so "
            "every cre_* / mre_* column exports as NaN"
        )

    # ---- phase 3: topology. The degree stats set `seen`. ----
    # build_interaction_edges is the query whose WRITE SET grows with every
    # fold (a later cutoff admits more transactions, hence more pairs), so it
    # commits one bounded transaction per source batch -- same idiom as
    # card_card_with_weights and reset_build. Its per-batch counters are
    # slices, so the build-level gates (cutoff bound; sum of txn_count over
    # the build's edges == transactions under the cutoff, which catches a
    # residue-partition hole) moved to the read-only verifier that follows.
    calls.extend(
        [
            *[
                QueryCall(
                    "build_interaction_edges",
                    {
                        "build_id": bid,
                        "cutoff_event_seq": cutoff,
                        "source_batch": source_batch,
                        "num_of_source_batches": interaction_batches,
                    },
                    "topology",
                    bid,
                    expect_present=("source_batch_count",),
                    note=(
                        "bounded insert transaction "
                        f"{source_batch + 1}/{interaction_batches}"
                    ),
                )
                for source_batch in range(interaction_batches)
            ],
            QueryCall(
                "verify_interaction_build",
                {"build_id": bid, "cutoff_event_seq": cutoff},
                "topology",
                bid,
                # The comparand must be present too: gates.py treats a
                # MISSING comparand for a *_MUST_EQUAL_* key as passed
                # ("cannot check"), so without this a renamed or dropped
                # comparand would turn the residue-hole gate permanently
                # vacuous with no failure anywhere.
                expect_present=(
                    "interaction_txn_sum_MUST_EQUAL_transactions_under_cutoff",
                    "transactions_under_cutoff",
                ),
                note="independent recount; holds the gates the batched "
                "writer can no longer print",
            ),
            QueryCall(
                "card_merchant_degree_stats",
                {"build_id": bid},
                "topology",
                bid,
                note="the only query that sets Card.seen / Merchant.seen",
            ),
            # Geography onto the interaction edge. It writes only edges
            # build_interaction_edges created, which is what makes it
            # cutoff-correct while taking no cutoff of its own. Batched by
            # card residue like the builder it follows: its write set is one
            # attribute on every interaction edge of the build, which grows
            # with every fold; the merchant census gate stays whole per call.
            *[
                QueryCall(
                    "card_home_distance",
                    {
                        "build_id": bid,
                        "cutoff_event_seq": cutoff,
                        "source_batch": source_batch,
                        "num_of_source_batches": interaction_batches,
                    },
                    "topology",
                    bid,
                    expect_present=("source_batch_count",),
                    note=(
                        "fills home_distance_km; the home is the tenure live "
                        f"at this rank. Bounded write transaction "
                        f"{source_batch + 1}/{interaction_batches}"
                    ),
                )
                for source_batch in range(interaction_batches)
            ],
            # Read-only. In projection mode it can cancel the whole community
            # stage; in bipartite mode its hub stats still feed the Stage-2
            # fan-out drift check, which is why it runs in BOTH modes.
            QueryCall(
                "projection_density_check",
                {
                    "build_id": bid,
                    "top_k": projection["density_top_k"],
                    # Price the configured cap exactly (the builder SKIPS
                    # merchants above it, so the capped sum is the real
                    # pairing cost) and arm the ABORT_* gate only where the
                    # price is owed: in bipartite mode nothing builds the
                    # projection, so the gate is disarmed with a 0 budget.
                    "probe_cap": 0 if bipartite else projection["max_merchant_degree"],
                    "pair_budget": 0.0 if bipartite else projection["pair_budget"],
                },
                "topology",
                bid,
                detached=False,
                # Armed gates must be seen to exist: gates.py only evaluates
                # keys that APPEAR, so a stale endpoint that ignores the
                # pair_budget parameter never prints the ABORT_* key and the
                # budget silently stops being enforced. Only required when
                # armed -- a stale endpoint in bipartite mode is harmless
                # (the projection is never built there).
                expect_present=(
                    () if bipartite else ("ABORT_capped_candidate_pairs_over_budget",)
                ),
                note=(
                    "hub stats feed the Stage-2 fan-out drift check; nothing "
                    "here gates the bipartite readers"
                    if bipartite
                    else "set max_merchant_degree / max_card_degree from "
                    "top_merchants_by_degree, not from a guess; the ABORT_* "
                    "gate stops the build if the capped pair count prices "
                    "above projection.pair_budget"
                ),
            ),
        ]
    )

    # ---- phase 3b: per-entity aggregates. THEY RUN BEFORE THE COMMUNITY
    # AND CENTRALITY PHASES, and that order is load-bearing. community_stats
    # and merchant_category_stats no longer walk transactions -- they COMPOSE
    # their aggregates from Card.txn_count / Merchant.txn_count and the amount
    # attributes these three queries write. Run them after the readers and
    # every entity reads its -1 sentinel: the readers complete, report no
    # history, and say nothing. validate_stage1_static asserts this order in
    # both graph_features modes, and community_stats carries
    # member_cards_at_sentinel_MUST_BE_ZERO as the runtime witness.
    calls.extend(
        [
            QueryCall(
                "card_transaction_stats",
                {"build_id": bid, "cutoff_event_seq": cutoff},
                "aggregates",
                bid,
            ),
            QueryCall(
                "merchant_transaction_stats",
                {"build_id": bid, "cutoff_event_seq": cutoff},
                "aggregates",
                bid,
            ),
            QueryCall(
                "merchant_category_stats",
                {"build_id": bid, "cutoff_event_seq": cutoff},
                "aggregates",
                bid,
                note="composed from Merchant.txn_count; the 27.4M-edge "
                "transaction walk is gone",
            ),
        ]
    )

    if bipartite:
        calls.extend(
            [
                # Read-only. Prices the structure.min_txn_count sweep BEFORE
                # every reader that depends on the answer -- the histogram's
                # one advantage over its projection-era predecessor, which had
                # to wait for the builders. probe_e carries the CONFIGURED
                # threshold so the attachment column has a row for the value
                # actually in force.
                QueryCall(
                    "interaction_strength_histogram",
                    {"build_id": bid, "probe_e": min_txn_count},
                    "topology",
                    bid,
                    detached=False,
                    note="attachment and surviving edges at the configured "
                    "min_txn_count; the stale-tuning check reads both",
                ),
                # ---- phase 5: community. ONE WCC: components span both types. ----
                QueryCall(
                    "wcc_bipartite",
                    {
                        "build_id": bid,
                        "min_txn_count": min_txn_count,
                        "print_results": False,
                    },
                    "community",
                    bid,
                ),
                QueryCall(
                    "community_stats",
                    {"build_id": bid, "cutoff_event_seq": cutoff},
                    "community",
                    bid,
                    note="composes from the per-entity aggregates written "
                    "in phase 3b; both transaction walks are gone",
                ),
                # ---- phase 6: centrality. ONE PageRank: mass alternates sides. ----
                QueryCall(
                    "pagerank_bipartite",
                    {
                        "build_id": bid,
                        "min_txn_count": min_txn_count,
                        "max_change": pagerank["max_change"],
                        "maximum_iteration": pagerank["maximum_iteration"],
                        "damping": pagerank["damping"],
                        "print_results": False,
                    },
                    "centrality",
                    bid,
                ),
            ]
        )
        if optional["louvain"]:
            calls.append(
                QueryCall(
                    "louvain_bipartite",
                    {
                        "build_id": bid,
                        "min_txn_count": min_txn_count,
                        "max_iteration": optional["louvain_max_iteration"],
                    },
                    "community",
                    bid,
                )
            )
        # optional["kcore"] in bipartite mode is rejected at startup in main():
        # there is no bipartite k-core query, and a silently skipped family
        # would export core_number as NaN while config.yaml claims otherwise.
    else:
        calls.extend(
            [
                # ---- phase 4: projections ----
                # One separate REST invocation per source-card batch, each a
                # bounded graph-update transaction: deletes/inserts commit at
                # the end of a QUERY, so the old in-query FOREACH committed
                # every batch's edges in one enormous transaction, and the
                # only way to bound the commit is to bound the query. A crash
                # mid-projection now resumes at --from-index instead of
                # redoing the whole pass.
                #
                # DETACHED, not sync, and the reason sync was ever chosen is
                # gone: the ~16 s that detached polling used to add per small
                # batch was a FLAT poll interval, now fixed by the backoff in
                # Client.run_installed_detached. Sync is actively worse here
                # because read_timeout_s is 86,520 s -- a query that dies or
                # is aborted server-side leaves the client blocked for 24
                # HOURS with no status, which is exactly how a run appeared
                # "stuck on build_interaction_edges" while the instance was
                # running nothing at all. Detached polls checkQueryStatus and
                # raises on aborted/failed.
                #
                # If a single batch times out, raise
                # projection.num_of_source_batches -- it changes transaction
                # boundaries, never the produced graph -- BUT re-run the
                # build's whole projection phase from its FIRST batch, never
                # --from-index past completed batches: batches partition
                # cards by getvid % N, so batches committed under N_old plus
                # batches resumed under N_new leave cards in neither residue
                # set with no pairs, and nothing downstream detects the hole.
                # check_projection_partition refuses that resume rather than
                # leaving it to whoever reads this comment.
                *[
                    QueryCall(
                        "card_card_with_weights",
                        {
                            "build_id": bid,
                            "min_edge_weight": projection["min_edge_weight"],
                            "max_merchant_degree": projection["max_merchant_degree"],
                            "min_interaction_txn_count": projection[
                                "min_interaction_txn_count"
                            ],
                            "source_batch": source_batch,
                            "num_of_source_batches": projection_batches,
                        },
                        "projection",
                        bid,
                        detached=False,
                        note=(
                            "independent graph-update transaction "
                            f"{source_batch + 1}/{projection_batches}"
                        ),
                    )
                    for source_batch in range(projection_batches)
                ],
                QueryCall(
                    "merchant_merchant_with_weights",
                    {
                        "build_id": bid,
                        "min_edge_weight": projection["min_edge_weight"],
                        "max_card_degree": projection["max_card_degree"],
                        "min_interaction_txn_count": projection[
                            "min_interaction_txn_count"
                        ],
                        "num_of_source_batches": projection["num_of_source_batches"],
                    },
                    "projection",
                    bid,
                ),
                # Read-only, and it runs HERE because it is the only query that
                # can tell whether structure.min_weight still means what
                # config.yaml says it means. It used to be installed but never
                # called, so re-measuring after a regeneration was a manual step
                # someone had to remember -- the failure mode §0b TODO-3 was
                # written about. probe_e carries the CONFIGURED threshold so the
                # attachment column has a row for the value actually in force.
                QueryCall(
                    "projection_weight_histogram",
                    {"probe_e": min_weight},
                    "projection",
                    bid,
                    detached=False,
                    note="attachment and surviving edges at the configured "
                    "min_weight; the stale-tuning check reads both",
                ),
                # ---- phase 5: community. community_stats needs both WCCs. ----
                QueryCall(
                    "wcc_card",
                    {"build_id": bid, "min_weight": min_weight, "print_results": False},
                    "community",
                    bid,
                ),
                QueryCall(
                    "wcc_merchant",
                    {"build_id": bid, "min_weight": min_weight, "print_results": False},
                    "community",
                    bid,
                ),
                QueryCall(
                    "community_stats",
                    {"build_id": bid, "cutoff_event_seq": cutoff},
                    "community",
                    bid,
                    note="composes from the per-entity aggregates written "
                    "in phase 3b; both transaction walks are gone",
                ),
                # ---- phase 6: centrality ----
                QueryCall(
                    "pagerank_card",
                    {
                        "build_id": bid,
                        "min_weight": min_weight,
                        "max_change": pagerank["max_change"],
                        "maximum_iteration": pagerank["maximum_iteration"],
                        "damping": pagerank["damping"],
                        "print_results": False,
                    },
                    "centrality",
                    bid,
                ),
                QueryCall(
                    "pagerank_merchant",
                    {
                        "build_id": bid,
                        "min_weight": min_weight,
                        "max_change": pagerank["max_change"],
                        "maximum_iteration": pagerank["maximum_iteration"],
                        "damping": pagerank["damping"],
                        "print_results": False,
                    },
                    "centrality",
                    bid,
                ),
            ]
        )

        if optional["louvain"]:
            for side in ("card", "merchant"):
                calls.append(
                    QueryCall(
                        f"louvain_{side}",
                        {
                            "build_id": bid,
                            "min_weight": int(min_weight),
                            "max_iteration": optional["louvain_max_iteration"],
                        },
                        "community",
                        bid,
                    )
                )
        if optional["kcore"]:
            for side in ("card", "merchant"):
                calls.append(
                    QueryCall(
                        f"kcore_{side}",
                        {
                            "build_id": bid,
                            "min_weight": int(min_weight),
                            "max_core": optional["kcore_max_core"],
                        },
                        "structure",
                        bid,
                    )
                )

    # ---- phase 6b: one-hop aggregates. NOT graph features. ----

    if optional["card_interval_max"]:
        calls.append(
            QueryCall(
                "card_interval_max",
                {
                    "build_id": bid,
                    "cutoff_event_seq": cutoff,
                    "interval_minutes": optional["card_interval_minutes"],
                    "heap_capacity": optional["card_interval_heap_capacity"],
                    "num_of_source_batches": optional["card_interval_batches"],
                },
                "aggregates",
                bid,
                note="the heaviest query in the repo; its own header "
                "recommends skipping it for a first Stage-1 number",
            )
        )

    return calls


# ======================================================================
# advisories: numbers whose headers say READ THIS, but which are not gates
# ======================================================================


def _scalar(value: object) -> float | None:
    """One numeric value out of a query result, or None if it is not one.

    Split out of ``_num`` because MapAccum outputs arrive as nested mappings,
    and their VALUES need the same coercion without a key to look up.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _num(record: Mapping[str, object], key: str) -> float | None:
    return _scalar(record.get(key))


def tuning_observations(
    query: str,
    record: Mapping[str, object],
    *,
    min_weight: float | None = None,
    min_txn_count: float | None = None,
) -> dict[str, float]:
    """The live numbers ``tuning_provenance``'s recorded Observations score.

    Returned rather than only compared because two of these keys are DERIVED
    and appear in no query output: an attachment SHARE is a count out of
    ``cards_seen``, and the query prints only the count. §0f names
    ``cards_attached_share_at_min_txn_count`` and ``interaction_max_txn_count``
    as the two inputs the projection A/B has to cite when it picks
    ``structure.min_txn_count``, so ``advise`` scores them against the record
    and ``Runner.run_call`` writes them into the manifest -- citing them is
    then a read rather than a hand recomputation from two other keys.

    Attachment is read at the CONFIGURED threshold, never at whichever probe
    happens to be nearest it: both histograms take the threshold in force as a
    probe for exactly this reason.
    """
    observed: dict[str, float] = {}

    def share_at(bucket_key: str, threshold: float | None) -> float | None:
        bucket = record.get(bucket_key)
        seen = _num(record, "cards_seen")
        if not isinstance(bucket, Mapping) or not seen or threshold is None:
            return None
        counts = cast(Mapping[object, object], bucket)
        attached = _num({str(k): v for k, v in counts.items()}, f"{threshold:g}")
        return None if attached is None else attached / seen

    if query in {"projection_density_check", "projection_weight_histogram"}:
        for key in (
            "hub_concentration_S",
            "max_merchant_reach_share",
            "card_card_max_weight",
        ):
            live = _num(record, key)
            if live is not None:
                observed[key] = live
        share = share_at("cards_with_degree_by_threshold", min_weight)
        if share is not None:
            observed["cards_attached_share_at_min_weight"] = share

    if query == "interaction_strength_histogram":
        live = _num(record, "interaction_max_txn_count")
        if live is not None:
            observed["interaction_max_txn_count"] = live
        share = share_at("cards_attached_by_threshold", min_txn_count)
        if share is not None:
            observed["cards_attached_share_at_min_txn_count"] = share

    return observed


def advise(
    query: str,
    record: Mapping[str, object],
    build: Build | None = None,
    min_weight: float | None = None,
    min_txn_count: float | None = None,
    bipartite: bool = False,
) -> list[str]:
    """Surface the findings the query headers tell you to read first.

    ``min_weight`` / ``min_txn_count`` are the CONFIGURED structure thresholds
    for the projection and bipartite reader families respectively. They are
    passed in rather than read from a module constant because the stale-tuning
    check has to score the threshold actually in use, not the one this file was
    written against. ``bipartite`` says which family this run uses, because
    projection_density_check runs in BOTH modes and its projection-cost
    advisories predict work the bipartite mode never does.
    """
    notes: list[str] = []

    def value(key: str) -> float | None:
        return _num(record, key)

    if query == "match_parties":
        matched = value("parties_with_at_least_one_match")
        if matched is not None and matched == 0:
            notes.append(
                "parties_with_at_least_one_match is 0. Every Party becomes a "
                "singleton Resolved_Entity and every identity feature is a "
                "constant. That is an answer, not a failure: it says the "
                "identity branch does not earn its keep on this data."
            )
        # A WEIGHT WITH NO EVIDENCE BEHIND IT DOES NOTHING, whatever it is set
        # to, and match_parties' own comment says to cross-read these two maps
        # against measure_pii_degrees' degree_one_per_relation before tuning.
        #
        # This replaces a check on parties_with_multiple_email / _name / _phone,
        # which match_parties STOPPED PRINTING when the SumAccum<STRING> that
        # concatenated raw values was deleted. `value()` returned None for all
        # three, the guard never fired, and the advice it would have given named
        # an accumulator that no longer exists -- a check that is structurally
        # always silent, which is the failure pagerank_card's header is about.
        weights = record.get("weights_used")
        evidence = record.get("scored_relations")
        if isinstance(weights, Mapping) and isinstance(evidence, Mapping):
            scored = {
                str(name): _scalar(count)
                for name, count in cast(Mapping[object, object], evidence).items()
            }
            inert = sorted(
                str(name)
                for name, weight in cast(Mapping[object, object], weights).items()
                if (_scalar(weight) or 0.0) > 0.0 and not (scored.get(str(name)) or 0.0)
            )
            if inert:
                notes.append(
                    f"{len(inert)} PII relation(s) carry a non-zero weight but "
                    f"contributed no evidence: {inert}. Those weights are inert "
                    "-- setting them higher changes nothing. Read "
                    "measure_pii_degrees' degree_one_per_relation: a relation "
                    "that is almost all degree 1 can never produce a match."
                )

    if query == "unify_parties":
        resolved = value("parties_resolved")
        singles = value("singleton_resolved_entities")
        if resolved and singles is not None and resolved == singles:
            notes.append(
                "every resolved entity is a singleton: no Same_As edge was in "
                "scope at this cutoff, so identity features are constant for "
                "this build"
            )
        largest = value("largest_component_size")
        if resolved and largest is not None and largest > 0.05 * resolved:
            notes.append(
                f"largest resolved entity holds {largest:,.0f} of "
                f"{resolved:,.0f} parties. Check assert_resolved_entity's "
                "worst_hub_per_relation before trusting identity features."
            )

    if query == "card_merchant_degree_stats":
        unseen = value("cards_unseen")
        if unseen is not None and unseen == 0:
            source = build.cutoff_source if build is not None else "the cutoff"
            notes.append(
                "cards_unseen is 0. On a cutoff that is not the end of the "
                "data this is the failure, not the success: it means "
                f"{source} did not bind."
            )

    if query in {"projection_density_check", "projection_weight_histogram"}:
        # The stale-tuning check. structure.min_weight and the Stage 2 fan-out
        # caps are hand-picked against a degree distribution these two queries
        # measure, and until now nothing compared the two: the provenance was a
        # docstring and the measurement was a number a human was supposed to
        # re-read. tuning_provenance holds the measured facts with a date, and
        # these notes say which constants no longer rest on a current one.
        notes.extend(
            drift_notes(tuning_observations(query, record, min_weight=min_weight))
        )

    if query == "projection_density_check" and not bipartite:
        for side in ("card_card", "merchant_merchant"):
            hint = value(f"{side}_saturation_hint")
            if hint is not None and hint > 0.3:
                notes.append(
                    f"{side}_saturation_hint is {hint:.3f}. Above 0.3 the "
                    "projection may be near-complete, which would make c_size "
                    "a constant and the whole community family a global mean "
                    "on every row. Measure the real edge count before "
                    "committing to these columns."
                )
        candidates = value("card_card_candidate_pairs")
        if candidates is not None and candidates > 1e9:
            notes.append(
                f"card_card_candidate_pairs is {candidates:,.0f}. Do not run "
                "card_card_with_weights with max_merchant_degree unset."
            )

    if query == "projection_density_check" and bipartite:
        # The bipartite readers never run the pairing loop, so the
        # saturation / candidate-pair advisories above would warn about work
        # this mode does not do. The number that DOES predict this mode's
        # degeneracy is the hub's reach: one merchant touching a large share
        # of cards glues both populations into one component at low
        # min_txn_count, the same way the saturated projection did.
        reach = value("max_merchant_reach_share")
        if reach is not None and reach > 0.3:
            notes.append(
                f"max_merchant_reach_share is {reach:.3f}. Expect one giant "
                "bipartite component at low min_txn_count: c_size will be "
                "near-constant (the trainer drops it, as before) and louvain "
                "remains the partitioner that still works. Read "
                "wcc_bipartite's largest_component_size, not component_count."
            )

    if query == "interaction_strength_histogram":
        # The stale-tuning check, bipartite side: structure.min_txn_count is
        # picked against this histogram the way min_weight was picked against
        # weight_histogram. Record the observations in tuning_provenance IN
        # THE SAME COMMIT that picks the threshold; until then these keys
        # have no Observation and contribute nothing HERE -- but both land in
        # the manifest as tuning_inputs either way, which is where the A/B
        # cites them from.
        notes.extend(
            drift_notes(tuning_observations(query, record, min_txn_count=min_txn_count))
        )

        # The bipartite analog of card_card_edges_inserted == 0: nothing
        # survives the configured threshold, so every structural feature will
        # be a sentinel. An absent probe key IS zero -- the accumulator only
        # gains the key when an edge qualifies.
        surviving = record.get("surviving_edges_by_threshold")
        if isinstance(surviving, Mapping) and min_txn_count is not None:
            bucket = cast(Mapping[object, object], surviving)
            probe = f"{min_txn_count:g}"
            kept = _num({str(k): v for k, v in bucket.items()}, probe)
            if not kept:
                notes.append(
                    f"surviving_edges_by_threshold has no edges at the "
                    f"configured min_txn_count={probe}. Every downstream "
                    "structural feature will be a sentinel. Lower "
                    "structure.min_txn_count."
                )

    if query in {"card_card_with_weights", "merchant_merchant_with_weights"}:
        key = (
            "card_card_edges_inserted"
            if query.startswith("card")
            else "merchant_merchant_edges_inserted"
        )
        inserted = value(key)
        if inserted is not None and inserted == 0:
            notes.append(
                f"{key} is 0. Every downstream structural feature will be a "
                "sentinel. Check the degree caps and min_interaction_txn_count."
            )

    if query in {"wcc_card", "wcc_merchant"}:
        members = value("cards_in_components") or value("merchants_in_components")
        largest = value("largest_component_size")
        if members and largest is not None and largest > 0.9 * members:
            notes.append(
                f"one component holds {largest:,.0f} of {members:,.0f} members. "
                "The projection saturated, c_size is effectively a constant, "
                "and a constant feature is a column of noise that absorbs "
                "regularisation budget. Raise structure.min_weight or "
                "projection.min_interaction_txn_count, or enable louvain."
            )
        isolated = value("cards_isolated_in_projection") or value(
            "merchants_isolated_in_projection"
        )
        if members and isolated is not None and isolated > members:
            notes.append(
                f"{isolated:,.0f} members are isolated in the projection "
                f"against {members:,.0f} in components. The degree caps are "
                "cutting deeper than intended."
            )

    if query == "wcc_bipartite":
        members = (value("cards_in_components") or 0) + (
            value("merchants_in_components") or 0
        )
        largest = value("largest_component_size")
        if members and largest is not None and largest > 0.9 * members:
            notes.append(
                f"one component holds {largest:,.0f} of {members:,.0f} members "
                "(cards + merchants). Expected when one hub merchant reaches a "
                "large share of cards: c_size is effectively a constant and "
                "the trainer drops it, as it did under the saturated "
                "projection. louvain_bipartite is the partitioner that still "
                "works; raising structure.min_txn_count is the retreat."
            )
        isolated = (value("cards_isolated_at_threshold") or 0) + (
            value("merchants_isolated_at_threshold") or 0
        )
        if members and isolated > members:
            notes.append(
                f"{isolated:,.0f} vertices are isolated at this threshold "
                f"against {members:,.0f} in components. structure.min_txn_count "
                "is cutting deeper than intended; every isolated vertex "
                "exports c_id/c_size as NaN."
            )

    if query in {"pagerank_card", "pagerank_merchant", "pagerank_bipartite"}:
        ran = value("iterations_run")
        limit = value("iteration_limit")
        if ran is not None and limit is not None and ran >= limit:
            notes.append(
                f"PageRank did not converge ({ran:.0f} of {limit:.0f} "
                "iterations). Scores are whatever that iteration produced and "
                "are not comparable across builds. Raise "
                "pagerank.maximum_iteration or report it next to the metric."
            )

    if query in {"louvain_card", "louvain_merchant", "louvain_bipartite"}:
        modularity = value("modularity_Q")
        if modularity is not None and modularity < 0.3:
            notes.append(
                f"modularity Q = {modularity:.3f}. Below ~0.3 there is no real "
                "community structure to find and louvain_size is close to noise."
            )
        ran = value("iterations_run")
        limit = value("iteration_limit")
        if ran is not None and limit is not None and ran >= limit:
            notes.append("Louvain did not settle; the partition is not comparable")

    if query in {"kcore_card", "kcore_merchant"}:
        at_cap = value("vertices_at_cap")
        if at_cap is not None and at_cap > 0:
            notes.append(
                f"{at_cap:,.0f} vertices sit at max_core, so core_number is "
                "censored for them. Raise optional.kcore_max_core."
            )

    if query == "community_stats":
        empty = value("communities_with_no_txns_under_cutoff")
        if empty is not None and empty > 0:
            notes.append(
                f"{empty:,.0f} communities have no transactions under the "
                "cutoff. Expected: they keep -1 and export as NaN."
            )

    if query == "stamp_resolved_entity_keys":
        conflicting = value("cards_with_conflicting_re_id")
        if conflicting is not None and conflicting > 0:
            notes.append(
                f"{conflicting:,.0f} cards are held by parties resolved to "
                "different entities. With drop_conflicts false the lower re_id "
                "wins arbitrarily. If this number is large, set "
                "identity.drop_conflicts true so those rows export as NaN "
                "instead of a coin flip."
            )

    return notes


# ======================================================================
# runner
# ======================================================================


# Config keys that change HOW the work is executed but not WHAT it produces.
# Excluded from the fingerprint so that raising a batch count after a timeout
# does not invalidate finished builds -- every one of these is documented in
# its own query header as "changes transaction boundaries, never the produced
# graph". Everything else in stage1_pipeline is included, so a threshold, a
# cutoff, a mode switch or an optional family all force a rebuild.
_FINGERPRINT_IGNORED: tuple[tuple[str, ...], ...] = (
    ("reset_batches",),
    ("enforce_gates",),
    ("export_dimensions",),
    ("topology", "interaction_batches"),
    ("projection", "num_of_source_batches"),
    ("projection", "density_top_k"),
    ("identity", "num_of_source_batches"),
    ("optional", "card_interval_batches"),
)


def build_fingerprint(plan: PipelinePlan, build: Build) -> str:
    """What a build's dimension tables depend on, as one hash.

    Two runs whose fingerprints agree for a build would compute byte-identical
    features for it, so the second can reuse the first's exported tables. The
    inputs are the build's own cutoff and window plus every feature-affecting
    config key; execution-only keys are excluded (see _FINGERPRINT_IGNORED).

    Deliberately CONSERVATIVE: anything not explicitly ignored is included, so
    a new config key invalidates prior builds until someone decides otherwise.
    Reusing a stale build silently serves features fitted under a different
    threshold, which is the kind of failure this repo has no way to detect
    downstream -- the columns are all present and all plausible.
    """
    payload = {
        key: value
        for key, value in sorted(plan.items())
        if (key,) not in _FINGERPRINT_IGNORED
    }
    for path in _FINGERPRINT_IGNORED:
        if len(path) != 2:
            continue
        section = payload.get(path[0])
        if isinstance(section, Mapping):
            payload[path[0]] = {
                k: v
                for k, v in cast(Mapping[str, object], section).items()
                if k != path[1]
            }
    # `builds` in the config is the whole plan; the build's OWN identity is
    # what matters here, and it already carries the resolved cutoff.
    payload.pop("builds", None)
    payload["_build"] = {
        "build_id": build.build_id,
        "cutoff_event_seq": build.cutoff_event_seq,
        "serves_splits": list(build.serves_splits),
        "serves_from": build.serves_from,
        "serves_until": build.serves_until,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def reusable_builds(
    plan: PipelinePlan,
    builds: Sequence[Build],
    output_root: Path,
    resident: tuple[str, int] | None,
) -> dict[str, str]:
    """Which builds already have valid exports, and why each one qualifies.

    THE PROBLEM THIS SOLVES. Every build opens with ``reset_build``, so the
    feature build is idempotent in its RESULT but not in its WORK: re-running
    after a completed Stage 1 destroys a known-good graph and redoes hours of
    compute. Worse, it destroys it BEFORE producing the replacement, so an
    interruption leaves neither -- which is exactly how a run that only needed
    Stage 2 ended up with b_test wiped and Stage 2 unable to start.

    WHAT MAKES A BUILD REUSABLE. Its durable output is the dimension export on
    disk; the graph state is transient and only the LAST build's survives. So:

      * any build whose manifest records the current fingerprint is reusable,
        because assemble reads those tables and nothing else needs its graph
        state;
      * EXCEPT the last build, which Stage 2 exports its edges from. That one
        is only reusable if the graph still holds it COMPLETELY.

    "Completely" is the load-bearing word and it is why ``resident`` carries a
    COUNT. ``export_pair_features`` walks the same relation under the same
    build_id and emits one row per edge, so the pair-row count in that build's
    manifest and the resident edge count are the same quantity measured at two
    times. Equal means finished; short means a run died mid-build, which is
    observed behaviour -- a crash left HIWM at 941,270 edges stamped with a
    build that should have had millions. An id-only check would have called
    that resident and handed Stage 2 half a graph.

    Returns {build_id: reason}. A build absent from the mapping must run.
    """
    reusable: dict[str, str] = {}
    if not builds:
        return reusable
    last_id = builds[-1].build_id
    for build in builds:
        manifest = build_dir(output_root, build.build_id) / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            payload = cast(
                dict[str, object], json.loads(manifest.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            continue
        recorded = str(payload.get("fingerprint", ""))
        if not recorded or recorded != build_fingerprint(plan, build):
            continue
        if build.build_id == last_id:
            # Stage 2 reads THIS build's edges out of the graph, so the tables
            # being right is not enough -- the graph has to hold it, and hold
            # all of it.
            if resident is None or resident[0] != build.build_id:
                continue
            rows = cast(Mapping[str, object], payload.get("rows") or {})
            expected = _num(rows, "export_pair_features")
            if expected is None:
                # An older manifest that predates the pair table, or an export
                # that did not run it. Nothing to compare against, so do not
                # claim completeness.
                continue
            if resident[1] != int(expected):
                print(
                    f"  build {build.build_id} is resident but INCOMPLETE: "
                    f"{resident[1]:,} interaction edges in the graph against "
                    f"{int(expected):,} in its export. A run died mid-build; "
                    "rebuilding it."
                )
                continue
            reusable[build.build_id] = (
                f"dimension tables present with matching fingerprint "
                f"{recorded}, and all {resident[1]:,} of its interaction "
                "edges are resident"
            )
            continue
        reusable[build.build_id] = (
            f"dimension tables present with matching fingerprint {recorded}"
        )
    return reusable


def resident_build(client: Client) -> tuple[str, int] | None:
    """(build_id, edge_count) the graph currently holds, or None.

    Read from the edges themselves rather than remembered, because the thing
    that invalidates a record -- a crashed or partial run -- is precisely the
    case where no record was written.

    Returns None when nothing is resident OR when MORE THAN ONE build_id is
    present. A mixture means a reset did not finish, which is the cross-build
    contamination reset_build exists to prevent, so nothing may be reused.

    The COUNT is what makes the answer trustworthy: an id alone cannot
    distinguish a finished build from one that crashed halfway, and a
    half-built graph reported as resident would be skipped and then exported
    by Stage 2. The caller checks it against the build's recorded pair-row
    count.
    """
    try:
        result = client.run_installed_with_timeout(
            "resident_build_id", {}, timeout_s=1800.0
        )
    except Exception:  # noqa: BLE001 - a probe must not fail the run
        return None
    record = gate_module.flatten(result)
    text = str(record.get("resident_build_id") or "").strip()
    if not text:
        return None
    distinct = record.get("distinct_build_ids")
    if (
        isinstance(distinct, (list, tuple, set))
        and len(cast(Sequence[object], distinct)) > 1
    ):
        print(
            "  the graph holds interaction edges from MORE THAN ONE build "
            f"({sorted(str(x) for x in cast(Sequence[object], distinct))}); a "
            "reset did not finish, so no build is reusable"
        )
        return None
    count = _num(record, "resident_edge_count")
    return (text, int(count)) if count is not None else None


def _resolve_manifest(path: Path) -> Path:
    """Where a manifest path actually lands. Shared with the resume reader.

    A relative --manifest is relative to the project, not the shell's cwd, so
    the resume path has to resolve it the same way write_manifest does or it
    would read a different file than the crashed run wrote.
    """
    return path if path.is_absolute() else project_root() / path


@dataclass
class Runner:
    client: Client
    plan: PipelinePlan
    export_plan: ExportPlan
    facts: LoadFacts
    output_root: Path
    records: list[dict[str, object]] = field(default_factory=list)
    manifest_path: Path = Path("artifacts/stage1/feature_pipeline_manifest.json")

    def write_manifest(self) -> None:
        path = _resolve_manifest(self.manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": self.client.graphname,
            "total_transactions": self.facts.total_transactions,
            "split_rows": {str(k): v for k, v in sorted(self.facts.split_rows.items())},
            "split_frauds": {
                str(k): v for k, v in sorted(self.facts.split_frauds.items())
            },
            "split_first_event_seq": {
                str(k): v for k, v in sorted(self.facts.split_first_event_seq.items())
            },
            "records": self.records,
        }
        _ = path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def run_call(self, index: int, call: QueryCall, build: Build | None) -> None:
        started = time.time()
        mode = "async" if call.detached else "sync "
        print(f"[{index:03d}] {mode} {call.phase:11s} {call.name}")
        if call.params:
            rendered = ", ".join(f"{k}={v}" for k, v in call.params.items())
            print(f"      params  {rendered}")
        if call.note:
            print(f"      note    {call.note}")

        try:
            if call.detached:
                result = self.client.run_installed_detached(call.name, call.params)
            else:
                result = self.client.run_installed_with_timeout(call.name, call.params)
        except Exception as exc:
            self.records.append(
                {
                    "index": index,
                    "build_id": call.build_id,
                    "phase": call.phase,
                    "query": call.name,
                    "params": call.params,
                    "elapsed_s": time.time() - started,
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            self.write_manifest()
            raise

        record = gate_module.flatten(result)
        strict = self.plan["enforce_gates"]
        checked = gate_module.enforce(
            call.name,
            result,
            {"total_transactions": self.facts.total_transactions},
            strict=strict,
        )

        # Explicit zero expectations that the GSQL does not name as gates.
        explicit_failures: list[str] = []
        for key in call.expect_zero:
            observed = _num(record, key)
            if observed is None:
                explicit_failures.append(f"{key} is missing or non-numeric")
            elif observed != 0:
                explicit_failures.append(f"{key}={observed:,.0f}, expected 0")
        # Presence expectations: the stale-endpoint tripwire. See QueryCall.
        for key in call.expect_present:
            if key not in record:
                explicit_failures.append(
                    f"{key} is missing from the output entirely. The installed "
                    "endpoint predates this query's current contract -- "
                    "TigerGraph silently ignores unknown parameters, so the "
                    "call 'succeeded' running the OLD logic. Run install_gsql "
                    "and retry."
                )
        for message in explicit_failures:
            print(f"    check FAIL {message}")
        if explicit_failures and strict:
            raise gate_module.GateFailure(
                f"{call.name} failed its explicit checks:\n  "
                + "\n  ".join(explicit_failures)
            )

        for message in advise(
            call.name,
            record,
            build,
            min_weight=self.plan["structure"]["min_weight"],
            min_txn_count=self.plan["structure"]["min_txn_count"],
            bipartite=self.plan["graph_features"]["source"] == "bipartite",
        ):
            print(f"    ADVISORY {message}")

        # The tuning-provenance inputs, printed and recorded rather than only
        # scored: the attachment shares are derived from two other keys, so
        # without this the commit that picks a threshold has to recompute them
        # by hand -- the transcription step tuning_provenance exists to remove.
        tuning_inputs = tuning_observations(
            call.name,
            record,
            min_weight=self.plan["structure"]["min_weight"],
            min_txn_count=self.plan["structure"]["min_txn_count"],
        )
        for key, live in tuning_inputs.items():
            print(f"      tuning  {key}={live:g}")

        entry: dict[str, object] = {
            "index": index,
            "build_id": call.build_id,
            "phase": call.phase,
            "query": call.name,
            "params": call.params,
            "elapsed_s": time.time() - started,
            "status": "ok",
            "gates": [
                {"key": g.key, "passed": g.passed, "detail": g.detail} for g in checked
            ],
            "output": record,
        }
        if tuning_inputs:
            entry["tuning_inputs"] = tuning_inputs
        self.records.append(entry)
        self.write_manifest()
        print(f"      done in {time.time() - started:,.1f}s")


# ======================================================================
# resume
# ======================================================================
#
# --from-index used to be a bounds check and nothing else: it never read the
# manifest the crashed run left behind. Two things depend on that manifest.
# The projection batch plan has to be the SAME one the committed batches were
# produced under, because batches partition cards by a modulus (see
# check_projection_partition); and the pre-crash gate records are the evidence
# the A/B acceptance cites, which write_manifest would otherwise overwrite on
# the first post-resume call.


def _load_manifest(path: Path) -> Mapping[str, object] | None:
    """The manifest a previous run left at this path, if it is readable."""
    resolved = _resolve_manifest(path)
    if not resolved.is_file():
        return None
    try:
        payload = cast(object, json.loads(resolved.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        print(f"  WARNING {resolved} is unreadable ({exc}); resuming without it")
        return None
    if not isinstance(payload, Mapping):
        print(f"  WARNING {resolved} holds no manifest object; resuming without it")
        return None
    return cast(Mapping[str, object], payload)


def _manifest_records(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """The record list out of a manifest, skipping anything malformed."""
    records = payload.get("records")
    if not isinstance(records, list):
        return []
    return [
        cast(dict[str, object], item)
        for item in cast(list[object], records)
        if isinstance(item, dict)
    ]


def _archive_manifest(path: Path) -> Path:
    """Move a manifest aside under a numbered suffix, never overwriting one."""
    resolved = _resolve_manifest(path)
    for attempt in range(1, 1000):
        candidate = resolved.with_name(
            f"{resolved.stem}.superseded-{attempt}{resolved.suffix}"
        )
        if not candidate.exists():
            _ = resolved.rename(candidate)
            return candidate
    raise SystemExit(
        f"too many superseded manifests beside {resolved}; move them out of "
        "the way before resuming"
    )


def _recorded_batch_count(entry: Mapping[str, object]) -> float | None:
    """What ``num_of_source_batches`` was when this call committed.

    ``card_card_with_weights`` PRINTs it as ``source_batch_count``; the params
    are the fallback for a manifest written before it did.
    """
    output = entry.get("output")
    if isinstance(output, Mapping):
        recorded = _num(cast(Mapping[str, object], output), "source_batch_count")
        if recorded is not None:
            return recorded
    params = entry.get("params")
    if isinstance(params, Mapping):
        return _num(cast(Mapping[str, object], params), "num_of_source_batches")
    return None


def check_projection_partition(
    prior: Sequence[Mapping[str, object]],
    flat: Sequence[tuple[QueryCall, Build | None]],
    *,
    from_index: int,
    projection_batches: int,
    reset_batches: int | None = None,
    interaction_batches: int | None = None,
) -> dict[str, object]:
    """Refuse a resume that would leave vertices in no source batch at all.

    FOUR queries partition their work by ``getvid % num_of_source_batches``:
    ``card_card_with_weights`` (pairs), ``reset_build`` (deletes and attribute
    resets), ``build_interaction_edges`` (interaction-edge inserts) and
    ``card_home_distance`` (edge-attribute writes).
    Batches committed under N_old plus batches resumed under N_new leave
    every vertex whose residue is in neither set uncovered, and for three of
    the four nothing downstream notices: ``card_card_edges_inserted`` is
    per-invocation and ``projection_weight_histogram`` measures whatever
    exists; an un-reset vertex carries the PREVIOUS build's state into
    exports that filter on the new build_id (the cross-build contamination
    reset_build's own header exists to prevent). Only the interaction edges
    have an independent recount (``verify_interaction_build``'s equality
    gate), and a refused resume is still cheaper than a failed gate after N
    batches. This is not a hypothetical modulus change -- raising the batch
    count is the documented recovery from a sync batch timing out.

    ``card_home_distance`` was MISSING from this set until 2026-08-02, and it
    is the worst of the four to leave out. It shares ``interaction_batches``
    with ``build_interaction_edges`` (see build_calls), so it inherits every
    retune of that count -- including the 10 -> 41 raise after the System
    Memory abort -- and it writes ``home_distance_km`` ON THE EDGE. An
    uncovered residue therefore leaves that attribute at its -1 sentinel,
    which exports as NaN rather than as an error, on the feature systemprompt
    records as the strongest single pair feature in the matrix. Nothing
    recounts it: ``verify_interaction_build`` audits ``txn_count`` only.

    Resuming at or before a build's FIRST call of the query is allowed: that
    build's whole phase re-runs under the new modulus, and the batches
    committed under the old one are a subset of what the full sweep redoes
    idempotently. Anything past it is refused, a multiple of the old count
    included -- the batch list grows with N, so every later index shifts and
    the manifest's recorded failure index no longer names the call it did.

    ``reset_batches`` / ``interaction_batches`` default to None so callers
    that predate the generalization (and probes that only exercise the
    projection case) keep their behaviour; None skips that query's guard.

    Returns the evidence to record, so the manifest says the check ran.
    """
    guarded: dict[str, int] = {"card_card_with_weights": projection_batches}
    if reset_batches is not None:
        guarded["reset_build"] = reset_batches
    if interaction_batches is not None:
        guarded["build_interaction_edges"] = interaction_batches
        # Same count, same residues, same hole -- see the docstring.
        guarded["card_home_distance"] = interaction_batches

    first_batch: dict[tuple[str, str], int] = {}
    for index, (call, _owner) in enumerate(flat):
        if call.name in guarded:
            _ = first_batch.setdefault((call.name, call.build_id), index)

    committed: list[tuple[str, str, int]] = []
    for entry in prior:
        query = str(entry.get("query"))
        if query not in guarded:
            continue
        if entry.get("status") != "ok":
            continue
        recorded = _recorded_batch_count(entry)
        if recorded is not None:
            committed.append((query, str(entry.get("build_id")), int(recorded)))

    stale: list[tuple[str, str, int, int]] = []
    for query, build_id, recorded in sorted(set(committed)):
        if recorded == guarded[query]:
            continue
        first = first_batch.get((query, build_id))
        # No call of that query for that build in THIS plan (bipartite mode,
        # or the build is not selected), so this run cannot widen a hole.
        if first is None or from_index <= first:
            continue
        stale.append((query, build_id, recorded, first))

    if stale:
        detail = "\n".join(
            f"  {query} for build {build_id}: committed batches under "
            f"num_of_source_batches={recorded} (now {guarded[query]}), and "
            f"this build's first {query} call is index {first}"
            for query, build_id, recorded, first in stale
        )
        raise SystemExit(
            f"refusing to resume at --from-index {from_index}: the existing "
            "manifest records completed batched calls under a different "
            "num_of_source_batches than the one now configured.\n"
            f"{detail}\n"
            "Batches partition their source vertices by getvid % "
            "num_of_source_batches, so batches committed under the recorded "
            "count plus batches resumed under the configured one leave every "
            "vertex whose residue is in neither set uncovered -- no pairs, or "
            "the previous build's un-reset state -- and no per-batch counter "
            "detects the hole.\n"
            "Either re-run each affected build's whole phase from the first "
            "index above, or restore the batch count to the recorded value "
            "and resume where the crash left off. Pass a different "
            "--manifest if the recorded run is not the one being resumed."
        )

    projection_committed = [c for c in committed if c[0] == "card_card_with_weights"]
    return {
        "from_index": from_index,
        "configured_source_batch_count": projection_batches,
        "configured_batch_counts": dict(sorted(guarded.items())),
        "committed_card_card_calls": len(projection_committed),
        "committed_source_batch_counts": sorted(
            {count for _, _, count in projection_committed}
        ),
        "committed_batched_calls": len(committed),
    }


def check_plan_alignment(
    prior: Sequence[Mapping[str, object]],
    flat: Sequence[tuple[QueryCall, Build | None]],
    *,
    from_index: int,
) -> dict[str, object]:
    """Refuse a resume whose carried indices name different calls now.

    ``--from-index`` is a position in the FLAT CALL LIST, and the list's
    shape is a function of the config: reset_batches, topology.
    interaction_batches, projection.num_of_source_batches, identity.enabled,
    the optional families and graph_features.source all add or remove calls.
    Change any of them between a crash and its resume and every later index
    shifts -- the resume then starts at a call N positions away from the
    crash point, silently skipping (or re-running) a block of work, with the
    batch-count guard blind to it whenever the counts themselves match.

    The recorded (index, query, build_id) triples are the crashed plan's
    shape, so requiring each carried record to name the SAME call in the new
    plan catches any re-basing at all. Records whose query build_calls never
    emits (assert_cardinality, the exporter's calls) are skipped: they are
    recorded by other stages and do not live at flat indices.
    """
    plan_names = {call.name for call, _owner in flat}
    mismatches: list[str] = []
    checked = 0
    for entry in prior:
        index = entry.get("index")
        if not isinstance(index, int) or index >= from_index:
            continue
        query = entry.get("query")
        if not isinstance(query, str) or query not in plan_names:
            continue
        checked += 1
        if index >= len(flat):
            mismatches.append(
                f"  index {index}: manifest says {query}, but this plan has "
                f"only {len(flat)} calls"
            )
        else:
            call, _owner = flat[index]
            recorded_build = entry.get("build_id")
            build_mismatch = (
                isinstance(recorded_build, str) and recorded_build != call.build_id
            )
            if call.name != query or build_mismatch:
                mismatches.append(
                    f"  index {index}: manifest says {query} for build "
                    f"{recorded_build}, this plan has {call.name} for build "
                    f"{call.build_id}"
                )
        if len(mismatches) >= 3:
            break

    if mismatches:
        raise SystemExit(
            f"refusing to resume at --from-index {from_index}: the plan's "
            "shape changed since the crashed run, so its recorded indices no "
            "longer name the same calls.\n" + "\n".join(mismatches) + "\n"
            "Any of reset_batches, topology.interaction_batches, "
            "projection.num_of_source_batches, identity.enabled, the "
            "optional families or graph_features.source re-bases every "
            "--from-index. Recompute the index against --list (or --dry-run) "
            "under the current config, or restore the config the crashed run "
            "used. Pass a different --manifest if the recorded run is not "
            "the one being resumed."
        )

    return {"plan_alignment_checked": checked}


def resume_from_manifest(
    manifest: Path,
    flat: Sequence[tuple[QueryCall, Build | None]],
    *,
    from_index: int,
    projection_batches: int,
    reset_batches: int | None = None,
    interaction_batches: int | None = None,
    graphname: str,
    total_transactions: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate the resume against the previous manifest and carry it forward.

    ``write_manifest`` rewrites the whole file after every call, so without the
    carry-forward a resume replaces the manifest with post-resume records only:
    assert_cardinality, the reset verification, the identity gates and
    projection_density_check's hub stats disappear from the artifact of exactly
    the runs -- a crashed projection -- most likely to be audited. Records from
    a DIFFERENT load are archived rather than merged: blending two graphs' gate
    evidence into one artifact reads as a passing check for a run that never
    happened, which is worse than losing it.
    """
    payload = _load_manifest(manifest)
    if payload is None:
        print(
            f"  WARNING no readable manifest at {_resolve_manifest(manifest)}: "
            "resuming without checking the batch plan against the crashed run, "
            "and its gate records are already gone"
        )
        return [], {"from_index": from_index, "prior_manifest": "absent"}

    records = _manifest_records(payload)

    # THE LOAD CHECK COMES FIRST, BEFORE EITHER GUARD (moved 2026-08-02).
    # It used to sit between them, so check_projection_partition graded a
    # manifest that might describe an entirely different corpus -- and its
    # refusal says "restore the batch count to the recorded value", which is
    # unfollowable advice when the recorded value was chosen for data that is
    # no longer loaded. That turned a reload, which should be a clean start,
    # into a dead end. The reasoning was already written down one branch
    # below for check_plan_alignment ("a different load's indices constrain
    # nothing here"); it applies to the batch counts for exactly the same
    # reason, and only one of the two checks was placed to honour it.
    same_load = payload.get("graph") == graphname and _num(
        payload, "total_transactions"
    ) == float(total_transactions)
    if not same_load:
        archived = _archive_manifest(manifest)
        print(
            f"  the manifest at {_resolve_manifest(manifest)} describes a "
            f"different load; archived it as {archived.name} rather than "
            "merging its gate records into this run's"
        )
        return [], {
            "from_index": from_index,
            "superseded_manifest": archived.name,
            # Say the checks were SKIPPED rather than let their absence read
            # as a pass. Neither had anything to grade.
            "partition_check": "skipped: manifest describes a different load",
            "alignment_check": "skipped: manifest describes a different load",
        }

    evidence = check_projection_partition(
        records,
        flat,
        from_index=from_index,
        projection_batches=projection_batches,
        reset_batches=reset_batches,
        interaction_batches=interaction_batches,
    )

    # Same load, so the recorded indices claim to be positions in THIS plan:
    # hold them to it.
    evidence.update(check_plan_alignment(records, flat, from_index=from_index))

    carried = [
        {**entry, "carried_over": True}
        for entry in records
        if isinstance(entry.get("index"), int)
        and cast(int, entry["index"]) < from_index
    ]
    print(
        f"  carrying {len(carried)} pre-resume record(s) forward from "
        f"{_resolve_manifest(manifest).name}"
    )
    evidence["carried_over_records"] = len(carried)
    return carried, evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage-1 TigerGraph feature build"
    )
    _ = parser.add_argument(
        "--list", action="store_true", dest="list_calls", help="print the plan and exit"
    )
    _ = parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "with --list, print the call order without connecting. Cutoffs show "
            "as their config tokens rather than resolved ranks, because the "
            "ranks come from derive_build_plan."
        ),
    )
    _ = parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="skip assert_cardinality. Only after it has passed once on this load.",
    )
    _ = parser.add_argument(
        "--builds",
        default="",
        help="comma-separated build_ids to run; defaults to all",
    )
    _ = parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="recompute every build even if its exports are already valid "
        "for this config. Without it, a build whose dimension tables carry "
        "the current fingerprint is skipped -- and the last build is only "
        "skipped if its edges are still resident, since Stage 2 reads them "
        "from the graph.",
    )
    _ = parser.add_argument(
        "--from-index",
        type=int,
        default=0,
        help=(
            "resume within the selected builds. Above 0 the existing --manifest "
            "is read: its pre-crash records are carried forward instead of "
            "overwritten, and a resume past a build's first projection batch is "
            "refused if projection.num_of_source_batches has changed."
        ),
    )
    _ = parser.add_argument("--through-index", type=int)
    _ = parser.add_argument(
        "--no-export", action="store_true", help="skip the per-build dimension export"
    )
    _ = parser.add_argument(
        "--manifest", default="artifacts/stage1/feature_pipeline_manifest.json"
    )
    return parser.parse_args()


# The ordering constraints that are documented silent failures rather than
# preferences. Printed next to the call so the plan explains itself.
_WHY_HERE: dict[str, str] = {
    "assert_cardinality": "gate on the load; community_stats and "
    "merchant_category_stats depend on exactly-one-Card / one-Merchant",
    "assert_edge_stamp": "gate on the load; Stage 2 filters its two largest "
    "relations on edge_event_seq and nothing else checks it equals the "
    "transaction's own event_seq",
    "reset_identity": "unconditional: a partial reset is how stale "
    "provenance stamps survive into the next build",
    "stamp_same_as_provenance": "<-- MUST be between match_parties and "
    "unify_parties, or every Same_As is silently discarded",
    "unify_parties": "collapses Same_As groups into Resolved_Entity",
    "assert_resolved_entity": "<-- MUST precede resolved_entity_stats, which "
    "meets a collapsed component as an OOM rather than a diagnosis",
    "stamp_resolved_entity_keys": "writes Card.re_id: the only path from the "
    "identity chain to a model",
    "build_interaction_edges": "the cutoff guard; both projections derive "
    "from this edge",
    "card_merchant_degree_stats": "<-- MUST be immediately after; the only "
    "query that sets seen, which 14 queries filter on",
    "card_home_distance": "the only geographic feature; writes onto the edge "
    "build_interaction_edges just created, so it must follow it",
    "projection_density_check": "read-only, and it can cancel the community "
    "stage; set the degree caps from its output",
    "projection_weight_histogram": "<-- MUST be after both projections, since "
    "it reads their weights; the only check that structure.min_weight still "
    "means what config.yaml says it means",
    "interaction_strength_histogram": "read-only; the only check that "
    "structure.min_txn_count still means what config.yaml says it means, and "
    "it runs BEFORE every reader that depends on the answer",
    "wcc_card": "creates the Community vertices",
    "wcc_merchant": "creates the Community vertices",
    "wcc_bipartite": "creates the Community vertices; ONE query because a "
    "bipartite component spans both types",
    "community_stats": "<-- MUST be after the WCC pass; inserted "
    "vertices are not visible to a later SELECT in the same query",
}


def _print_plan(
    flat: Sequence[tuple[QueryCall, Build | None]],
    builds: Sequence[Build],
    plan: PipelinePlan,
    *,
    no_export: bool,
) -> None:
    """Print the call order, grouped by build and phase."""
    current_build = "\0"
    current_phase = "\0"

    for index, (call, build) in enumerate(flat):
        if call.build_id != current_build:
            current_build = call.build_id
            current_phase = "\0"
            if build is None:
                print("\n" + "=" * 78)
                print("ONCE PER LOAD")
                print("=" * 78)
            else:
                print("\n" + "=" * 78)
                print(
                    f"BUILD {build.build_id}   cutoff={build.cutoff_source}   "
                    f"serves split_id {list(build.serves_splits)}"
                )
                print("=" * 78)
        if call.phase != current_phase:
            current_phase = call.phase
            print(f"\n  -- {call.phase} --")

        mode = "async" if call.detached else "sync "
        reason = _WHY_HERE.get(call.name, "")
        print(f"  {index:03d} {mode} {call.name}")
        if reason:
            print(f"          {reason}")

    if builds and plan["export_dimensions"] and not no_export:
        print("\n  -- export (at the end of EACH build) --")
        print("      export_card_features, export_merchant_features,")
        print("      export_category_features, export_community_features,")
        print("      export_resolved_entity_features")
        print(
            "          <-- MUST run inside the build: the next build's "
            "reset_build\n              sets every attribute back to -1 and "
            "deletes the Communities"
        )

    print("\n" + "=" * 78)
    print(f"{len(flat)} queries across {len(builds)} build(s)")
    print("=" * 78)
    print(
        "\nAfter the build, in this order and outside it:\n"
        "  python -m tfgnn.tigergraph.export     export_transaction_rows, paged.\n"
        "                                        No build dependency: "
        "Payment_Transaction\n"
        "                                        is read-only after load.\n"
        "  python -m tfgnn.tigergraph.assemble   join the star schema in pandas\n"
        "  python -m baseline.train_xgboost      three arms"
    )


def _resolve_plan(client: Client, plan: PipelinePlan) -> tuple[LoadFacts, list[Build]]:
    print("reading the build plan from derive_build_plan (read-only)")
    facts = load_facts(client.run_installed_with_timeout("derive_build_plan", {}))
    builds = build_plan(
        cast(Sequence[Mapping[str, object]], plan["builds"]),
        facts,
        train_folds=plan["train_folds"],
        fold_source=plan["fold_source"],
    )
    if plan["merge_identical_cutoffs"]:
        builds = deduplicate(builds)
    print(describe(facts, builds))
    return facts, builds


def main() -> None:
    args = _parse_args()
    config = load_raw_config()
    if "stage1_pipeline" not in config:
        raise KeyError("config.yaml is missing the stage1_pipeline section")
    plan = PIPELINE_ADAPTER.validate_python(config["stage1_pipeline"])
    export_plan = EXPORT_PLAN_ADAPTER.validate_python(config["tigergraph_export"])

    if plan["fold_source"] not in FOLD_SOURCES:
        raise ValueError(
            f"stage1_pipeline.fold_source={plan['fold_source']!r} is not one of "
            f"{list(FOLD_SOURCES)}. A typo here would silently fall back to a "
            "different set of fold boundaries than the one asked for."
        )

    source = plan["graph_features"]["source"]
    if source not in GRAPH_FEATURE_SOURCES:
        raise ValueError(
            f"stage1_pipeline.graph_features.source={source!r} is not one of "
            f"{list(GRAPH_FEATURE_SOURCES)}. A typo here would silently run a "
            "different feature substrate than the one asked for."
        )

    structure = plan["structure"]
    if source == "bipartite":
        if plan["optional"]["kcore"]:
            raise ValueError(
                "optional.kcore is true but graph_features.source is "
                "'bipartite', and no bipartite k-core query exists. Skipping "
                "it silently would export core_number as NaN while config.yaml "
                "claims otherwise. Disable kcore or use source: projection."
            )
    elif plan["optional"]["louvain"] or plan["optional"]["kcore"]:
        if float(structure["min_weight"]) != int(structure["min_weight"]):
            raise ValueError(
                f"structure.min_weight={structure['min_weight']} is fractional, "
                "but louvain_* and kcore_* declare min_weight as INT while "
                "wcc_* and pagerank_* declare it FLOAT. A fractional value "
                "would give them different edge sets, so c_size, pagerank, "
                "louvain_size and core_number would describe different graphs. "
                "Use an integer or disable louvain and kcore."
            )

    offline = bool(args.offline)
    if offline and not args.list_calls:
        raise SystemExit("--offline only makes sense with --list")

    client: Client | None = None
    if offline:
        print(
            "OFFLINE: showing the call ORDER only. Cutoffs are config tokens, "
            "not resolved ranks -- those come from derive_build_plan."
        )
        if plan["merge_identical_cutoffs"]:
            print(
                "Every configured build is listed separately. At run time, "
                "builds whose cutoffs\nresolve to the SAME rank are merged "
                "into one pass, so the real run is usually\nshorter than this "
                "listing. Connect and drop --offline to see the merge.\n"
            )
            folds = plan["train_folds"]
            if folds > 1:
                # Fold boundaries are event_seq quantiles, and offline there is
                # no derive_build_plan to supply the ranks, so the expansion
                # CANNOT be shown here. Say so rather than print a build count
                # the real run will not match.
                print(
                    f"train_folds={folds}: the build serving split 0 will "
                    f"expand into {folds - 1} forward-chaining\nsnapshots at "
                    "run time, so the real run has MORE builds and more calls "
                    "than this\nlisting shows. The boundaries are event_seq "
                    "quantiles and need a connection to\nresolve. Fold 0 "
                    "becomes warm-up and is withheld from training.\n"
                )
        facts = LoadFacts(0, {}, {}, {}, {}, {}, {}, {})
        builds = offline_plan(cast(Sequence[Mapping[str, object]], plan["builds"]))
    else:
        settings = Settings()
        client = Client(settings)
        facts, builds = _resolve_plan(client, plan)

    # THE WHOLE PLAN'S IDS, captured before ANY filtering. prune_stale_builds
    # moves aside every directory not in the list it is given, so handing it a
    # filtered list deletes the exports of builds this run is deliberately not
    # running -- both the --builds subset and the reuse gate's remainder.
    # assemble needs all of them, and Stage 2's embedding export reads
    # dimensions/<build_id>/ for EVERY build, so losing them breaks both.
    planned_ids = [item.build_id for item in builds]

    selected_ids = {name.strip() for name in args.builds.split(",") if name.strip()}
    if selected_ids:
        unknown = selected_ids - {item.build_id for item in builds}
        if unknown:
            raise SystemExit(f"unknown build_ids: {sorted(unknown)}")
        builds = [item for item in builds if item.build_id in selected_ids]

    output_root = resolve_output(export_plan["output_dir"])

    # ---- reuse gate: do not redo builds that are already correct ----
    # Every build opens with reset_build, so without this the pipeline is
    # idempotent in RESULT but not in WORK: a re-run destroys a known-good
    # graph and recomputes hours of features, and because the destruction
    # happens FIRST, an interruption leaves neither the old state nor the new.
    # Skipping is keyed on a fingerprint of the feature-affecting config, so a
    # changed threshold still forces the rebuild it must.
    skipped: dict[str, str] = {}
    if client is not None and not args.force_rebuild:
        skipped = reusable_builds(plan, builds, output_root, resident_build(client))
        if skipped:
            for build_id, reason in sorted(skipped.items()):
                print(f"  reusing build {build_id}: {reason}")
            print(
                f"  {len(skipped)} of {len(builds)} builds reused. Pass "
                "--force-rebuild to recompute them."
            )
        builds = [item for item in builds if item.build_id not in skipped]
        if not builds:
            print(
                "\nevery build in the plan is already built and exported; "
                "nothing to do. The graph still holds the last build's edges, "
                "so Stage 2 can run."
            )
            return

    # Prune build exports from a PREVIOUS plan before writing this one's.
    # assemble globs dimensions/ and cannot tell a leftover from a current
    # build, so a stale directory silently collides -- and it is only caught
    # after the whole feature build has run. Done here because the pipeline
    # owns dimensions/; the exports are moved to _stale/, not deleted.
    if plan["export_dimensions"] and not args.no_export:
        _ = prune_stale_builds(output_root, planned_ids)

    validate_calls: list[QueryCall] = []
    if not args.skip_validate:
        validate_calls.append(
            QueryCall(
                "assert_cardinality",
                {},
                "validate",
                "-",
                detached=False,
                note="run once per load. Any failing gate stops the pipeline.",
            )
        )
    # NOT under --skip-validate, deliberately. Three queries now evaluate their
    # cutoff on edge_event_seq instead of the transaction's own event_seq
    # (build_interaction_edges, card_transaction_stats,
    # merchant_transaction_stats), and this gate is the ONLY thing establishing
    # that those are the same number. Skipping it would not merely skip a check,
    # it would let the whole feature build run on an unverified premise -- and
    # the failure is silent: every column is present and every value plausible.
    # Two passes over the transaction edges, once per RUN, against a run
    # measured in hours.
    validate_calls.append(
        QueryCall(
            "assert_edge_stamp",
            {},
            "validate",
            "-",
            detached=False,
            note="run once per load, NOT skippable: does edge_event_seq equal "
            "the transaction's event_seq? Three cutoff predicates and Stage 2's "
            "D3a filter all depend on the answer",
        )
    )

    # (call, owning build). The validate call belongs to no build, so its
    # slot is None; that is also what tells the loop below not to treat it as
    # the last call of a build and trigger an export.
    flat: list[tuple[QueryCall, Build | None]] = [
        (call, None) for call in validate_calls
    ]
    for item in builds:
        flat.extend((call, item) for call in build_calls(plan, item))

    if args.list_calls:
        _print_plan(flat, builds, plan, no_export=bool(args.no_export))
        return

    stop = args.through_index + 1 if args.through_index is not None else len(flat)
    if args.from_index < 0 or args.from_index >= len(flat):
        raise ValueError(f"--from-index must be between 0 and {len(flat) - 1}")

    if client is None:
        raise SystemExit("no connection: --offline is only valid with --list")

    # A resume is only as good as the plan it resumes into, and the manifest
    # left by the crashed run is the only record of what already committed.
    carried: list[dict[str, object]] = []
    resume_evidence: dict[str, object] | None = None
    if args.from_index > 0:
        carried, resume_evidence = resume_from_manifest(
            Path(args.manifest),
            flat,
            from_index=args.from_index,
            projection_batches=plan["projection"]["num_of_source_batches"],
            reset_batches=plan["reset_batches"],
            interaction_batches=plan["topology"]["interaction_batches"],
            graphname=client.graphname,
            total_transactions=facts.total_transactions,
        )

    runner = Runner(
        client=client,
        plan=plan,
        export_plan=export_plan,
        facts=facts,
        output_root=output_root,
        records=carried,
        manifest_path=Path(args.manifest),
    )

    if resume_evidence is not None:
        # Written before the first call, so the carried-forward records survive
        # even if this run dies on the call the last one died on.
        runner.records.append(
            {
                "index": args.from_index,
                "build_id": "-",
                "phase": "resume",
                "query": "resume_plan_check",
                "status": "ok",
                "output": resume_evidence,
            }
        )
        runner.write_manifest()

    exported: set[str] = set()
    last_index = min(stop, len(flat)) - 1
    for index in range(args.from_index, min(stop, len(flat))):
        call, build = flat[index]
        runner.run_call(index, call, build)

        if build is None:
            continue

        # Export this build's dimension tables as soon as its last compute
        # call lands. The next build opens with reset_build, which sets every
        # build-scoped attribute back to -1 and deletes the Community
        # vertices, so there is no later chance to read them.
        is_last_for_build = index == last_index or flat[index + 1][1] is not build
        if (
            is_last_for_build
            and plan["export_dimensions"]
            and not args.no_export
            and build.build_id not in exported
        ):
            print(f"  exporting dimension tables for build {build.build_id}")
            counts = export_dimensions(
                client,
                build,
                output_root,
                export_plan,
                fingerprint=build_fingerprint(plan, build),
            )
            exported.add(build.build_id)
            runner.records.append(
                {
                    "index": index,
                    "build_id": build.build_id,
                    "phase": "export",
                    "query": "export_dimensions",
                    "status": "ok",
                    "output": counts,
                }
            )
            runner.write_manifest()

    print(f"\ncompleted {min(stop, len(flat)) - args.from_index} calls")
    # The RESOLVED path: this line is how the operator finds the artifact, and
    # printing the relative form names a file that only exists if you happen
    # to be standing in the project root.
    print(f"manifest: {_resolve_manifest(runner.manifest_path)}")
    if exported:
        print(f"dimension tables exported for: {', '.join(sorted(exported))}")
        print(
            "next: python -m tfgnn.tigergraph.export   (fact table)\n"
            "then: python -m tfgnn.tigergraph.assemble (join into the matrix)"
        )


if __name__ == "__main__":
    main()
