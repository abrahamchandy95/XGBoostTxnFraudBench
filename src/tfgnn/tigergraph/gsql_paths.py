from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


type Group = str

SCHEMA: Group = "schema"
CORE: Group = "core"
OPTIONAL: Group = "optional"
STAGE2: Group = "stage2"


@dataclass(frozen=True)
class ScriptSpec:
    name: str
    relpath: str
    queries: tuple[str, ...]
    group: Group


# Run order. Phase comments match tfgnn.tigergraph.pipeline.
_SCRIPTS: tuple[ScriptSpec, ...] = (
    # ---- schema: opt-in only ----
    ScriptSpec("schema", "schema/schema.gsql", (), SCHEMA),
    # ---- phase 0: validate the load, then read the build plan ----
    ScriptSpec(
        "assert_cardinality",
        "validate/assert_cardinality.gsql",
        ("assert_cardinality",),
        CORE,
    ),
    # Same lifetime and purpose as assert_cardinality: a load gate, run once,
    # answering a question worth answering before hours are spent. It is the
    # only thing that checks the column Stage 2's D3a guarantee rests on for
    # its two largest relations.
    ScriptSpec(
        "assert_edge_stamp",
        "validate/assert_edge_stamp.gsql",
        ("assert_edge_stamp",),
        CORE,
    ),
    ScriptSpec(
        "derive_build_plan",
        "prepare/derive_build_plan.gsql",
        ("derive_build_plan",),
        CORE,
    ),
    # ---- phase 1: clear derived state ----
    # reset_build is per-invocation batched (stage1_pipeline.reset_batches):
    # a one-transaction reset of a materialized projection is what wedged the
    # live engine for 17+ hours on 2026-07-31.
    ScriptSpec("reset_build", "lifecycle/reset_build.gsql", ("reset_build",), CORE),
    # Read-only probe: which build's edges the graph holds. The build-reuse
    # gate needs it to decide whether the LAST build can be skipped, since
    # Stage 2 exports that build's edges out of the graph.
    ScriptSpec(
        "resident_build_id",
        "lifecycle/resident_build_id.gsql",
        ("resident_build_id",),
        CORE,
    ),
    # gsql/identity/ and gsql/projection/ carry NUMERIC FILENAME PREFIXES that
    # mirror this registry's order (identity 01..08 is the ER chain the module
    # docstring calls load-bearing; projection 01..04 is density_check ->
    # builders -> histogram). The prefixes are a reading aid only: the run
    # order is THIS tuple, and nothing parses a filename for a number.
    ScriptSpec(
        "reset_identity", "identity/01_reset_identity.gsql", ("reset_identity",), CORE
    ),
    # ---- phase 2: entity resolution ----
    ScriptSpec(
        "measure_pii_degrees",
        "identity/02_measure_pii_degrees.gsql",
        ("measure_pii_degrees",),
        CORE,
    ),
    ScriptSpec(
        "match_parties", "identity/03_match_parties.gsql", ("match_parties",), CORE
    ),
    ScriptSpec(
        "stamp_same_as_provenance",
        "identity/04_stamp_same_as_provenance.gsql",
        ("stamp_same_as_provenance",),
        CORE,
    ),
    ScriptSpec(
        "unify_parties", "identity/05_unify_parties.gsql", ("unify_parties",), CORE
    ),
    ScriptSpec(
        "assert_resolved_entity",
        "identity/06_assert_resolved_entity.gsql",
        ("assert_resolved_entity",),
        CORE,
    ),
    ScriptSpec(
        "resolved_entity_stats",
        "identity/07_resolved_entity_stats.gsql",
        ("resolved_entity_stats",),
        CORE,
    ),
    ScriptSpec(
        "stamp_resolved_entity_keys",
        "identity/08_stamp_resolved_entity_keys.gsql",
        ("stamp_resolved_entity_keys",),
        CORE,
    ),
    # ---- phase 3: topology. degree stats set `seen`, so they come second ----
    ScriptSpec(
        "build_interaction_edges",
        "topology/build_interaction_edges.gsql",
        ("build_interaction_edges",),
        CORE,
    ),
    # Read-only. Holds the build-level gates the batched writer can no longer
    # print (per-batch counters are slices): the cutoff-bound check, and
    # sum(txn_count) over this build's edges == transactions under the cutoff,
    # which catches a residue-partition hole no batch can see from inside.
    ScriptSpec(
        "verify_interaction_build",
        "topology/verify_interaction_build.gsql",
        ("verify_interaction_build",),
        CORE,
    ),
    ScriptSpec(
        "card_merchant_degree_stats",
        "topology/card_merchant_degree_stats.gsql",
        ("card_merchant_degree_stats",),
        CORE,
    ),
    # Geography onto the interaction edge. MUST follow build_interaction_edges:
    # it only writes edges that query created, which is also what keeps it
    # cutoff-correct without taking a cutoff of its own.
    ScriptSpec(
        "card_home_distance",
        "aggregates/card_home_distance.gsql",
        ("card_home_distance",),
        CORE,
    ),
    ScriptSpec(
        "density_check",
        "projection/01_density_check.gsql",
        ("projection_density_check",),
        CORE,
    ),
    # Prices the structure.min_txn_count sweep for the bipartite feature path
    # (graph_features.source: bipartite). Reads only Has_Interaction_With_
    # Merchant, so unlike weight_histogram it runs BEFORE every reader that
    # depends on the answer. CORE for the same reason weight_histogram is:
    # the pipeline calls it, and an OPTIONAL script is not installed without
    # --include-optional.
    ScriptSpec(
        "interaction_histogram",
        "topology/interaction_histogram.gsql",
        ("interaction_strength_histogram",),
        CORE,
    ),
    # STAGE 3. Per-row temporal features, run ONCE for the whole run rather
    # than once per build -- it takes no build_id and no cutoff, because a
    # window anchored at the row's own unix_time is causal by construction and
    # therefore valid for every build at once (see the query header). CORE
    # because it is unconditional: nothing in config.yaml turns Stage 3 off,
    # and an OPTIONAL script is not installed without --include-optional.
    # Driven by tfgnn.tigergraph.temporal, NOT by pipeline.build_calls.
    ScriptSpec(
        "card_row_temporal",
        "aggregates/card_row_temporal.gsql",
        ("card_row_temporal",),
        CORE,
    ),
    # STAGE 4. Last-N adjacency for the TGN's attention module, fetched ONCE
    # PER WINDOW at the forward-chaining cutoffs -- not per batch, which was
    # costed at 1.05 full relation passes per step. Driven by
    # tfgnn.stage4.window, not by pipeline.build_calls.
    ScriptSpec(
        "tgn_window_neighbors",
        "topology/tgn_window_neighbors.gsql",
        ("tgn_window_neighbors",),
        CORE,
    ),
    # ---- phase 4: projections (graph_features.source: projection only) ----
    ScriptSpec(
        "card_card_with_weights",
        "projection/02_card_card_with_weights.gsql",
        ("card_card_with_weights",),
        CORE,
    ),
    ScriptSpec(
        "merchant_merchant_with_weights",
        "projection/03_merchant_merchant_with_weights.gsql",
        ("merchant_merchant_with_weights",),
        CORE,
    ),
    # ---- phase 5: community ----
    # wcc_card/wcc_merchant read the projections; wcc_bipartite reads the
    # interaction edge directly. pipeline.build_calls picks ONE side per
    # graph_features.source; community_stats serves either (it keys on
    # whatever Community vertices exist).
    ScriptSpec("wcc_card", "community/wcc_card.gsql", ("wcc_card",), CORE),
    ScriptSpec("wcc_merchant", "community/wcc_merchant.gsql", ("wcc_merchant",), CORE),
    ScriptSpec(
        "wcc_bipartite", "community/wcc_bipartite.gsql", ("wcc_bipartite",), CORE
    ),
    ScriptSpec(
        "community_stats", "community/community_stats.gsql", ("community_stats",), CORE
    ),
    # ---- phase 6: features ----
    ScriptSpec(
        "pagerank_card", "centrality/pagerank_card.gsql", ("pagerank_card",), CORE
    ),
    ScriptSpec(
        "pagerank_merchant",
        "centrality/pagerank_merchant.gsql",
        ("pagerank_merchant",),
        CORE,
    ),
    ScriptSpec(
        "pagerank_bipartite",
        "centrality/pagerank_bipartite.gsql",
        ("pagerank_bipartite",),
        CORE,
    ),
    ScriptSpec(
        "card_transaction_stats",
        "aggregates/card_transaction_stats.gsql",
        ("card_transaction_stats",),
        CORE,
    ),
    ScriptSpec(
        "merchant_transaction_stats",
        "aggregates/merchant_transaction_stats.gsql",
        ("merchant_transaction_stats",),
        CORE,
    ),
    ScriptSpec(
        "merchant_category_stats",
        "aggregates/merchant_category_stats.gsql",
        ("merchant_category_stats",),
        CORE,
    ),
    # ---- phase 7: export ----
    ScriptSpec(
        "export_transaction_rows",
        "export/export_transaction_rows.gsql",
        ("export_transaction_rows",),
        CORE,
    ),
    ScriptSpec(
        "export_card_and_merchant_features",
        "export/export_card_and_merchant_features.gsql",
        ("export_card_features", "export_merchant_features"),
        CORE,
    ),
    # The per-PAIR distance. Exported instead of the four raw coordinate columns
    # the two entity dimensions used to carry, so pair_home_distance_km is
    # computed once -- by the query that has the cutoff -- rather than twice.
    ScriptSpec(
        "export_pair_features",
        "export/export_pair_features.gsql",
        ("export_pair_features",),
        CORE,
    ),
    ScriptSpec(
        "export_category_features",
        "export/export_category_features.gsql",
        ("export_category_features",),
        CORE,
    ),
    ScriptSpec(
        "export_community_features",
        "export/export_community_features.gsql",
        ("export_community_features",),
        CORE,
    ),
    ScriptSpec(
        "export_resolved_entity_features",
        "export/export_resolved_entity_features.gsql",
        ("export_resolved_entity_features",),
        CORE,
    ),
    # ---- optional ----
    ScriptSpec(
        "louvain_card", "community/louvain_card.gsql", ("louvain_card",), OPTIONAL
    ),
    ScriptSpec(
        "louvain_merchant",
        "community/louvain_merchant.gsql",
        ("louvain_merchant",),
        OPTIONAL,
    ),
    ScriptSpec(
        "louvain_bipartite",
        "community/louvain_bipartite.gsql",
        ("louvain_bipartite",),
        OPTIONAL,
    ),
    ScriptSpec("kcore_card", "structure/kcore_card.gsql", ("kcore_card",), OPTIONAL),
    ScriptSpec(
        "kcore_merchant", "structure/kcore_merchant.gsql", ("kcore_merchant",), OPTIONAL
    ),
    ScriptSpec(
        "card_interval_max",
        "aggregates/card_interval_max.gsql",
        ("card_interval_max",),
        OPTIONAL,
    ),
    ScriptSpec(
        "wcc_card_hierarchical",
        "community/wcc_card_hierarchical.gsql",
        ("wcc_card_hierarchical",),
        OPTIONAL,
    ),
    ScriptSpec(
        "wcc_merchant_hierarchical",
        "community/wcc_merchant_heirarchical.gsql",
        ("wcc_merchant_hierarchical",),
        OPTIONAL,
    ),
    # CORE, not OPTIONAL, since the pipeline calls it: it is the only query that
    # can tell whether structure.min_weight still means what config.yaml says it
    # means. It was OPTIONAL while it was a query someone ran by hand, which is
    # exactly the arrangement that made re-measuring after a regeneration a step
    # to forget. An OPTIONAL script is not installed without --include-optional,
    # so leaving it here would make the default install path fail on a missing
    # endpoint partway into the first build.
    ScriptSpec(
        "weight_histogram",
        "projection/04_weight_histogram.gsql",
        ("projection_weight_histogram",),
        CORE,
    ),
    ScriptSpec(
        "permutation_control",
        "validate/permutation_control.gsql",
        ("permutation_control",),
        OPTIONAL,
    ),
    # ---- stage 2 ----
    # Materialises the link-prediction target relation from data already in
    # the graph. DELIBERATELY NOT IN `core`: the core set runs once per build,
    # and this is a one-time 27M-edge write against a LOADED relation that
    # reset_build does not clear. Running it per build would triple the work
    # for no effect, since re-inserting upserts the same rows.
    ScriptSpec(
        "build_card_merchant_transaction_edges",
        "topology/build_card_merchant_transaction_edges.gsql",
        ("build_card_merchant_transaction_edges",),
        STAGE2,
    ),
    # Edge lists for the cuGraph-resident graph, plus the two vertex maps the
    # embedding join needs. getvid-based, so no 27M-row host-side id dict.
    ScriptSpec(
        "export_graph_edges",
        "export/export_graph_edges.gsql",
        (
            "export_relation_edges",
            "export_vertex_key_map",
            "export_vertex_id_range",
        ),
        STAGE2,
    ),
    ScriptSpec("fastrp_card", "embedding/fastrp_card.gsql", ("fastrp_card",), STAGE2),
    ScriptSpec(
        "fastrp_merchant",
        "embedding/fastrp_merchant.gsql",
        ("fastrp_merchant",),
        STAGE2,
    ),
    ScriptSpec(
        "neighborhood_aggregates_card",
        "embedding/neighborhood_aggregates_card.gsql",
        ("neighborhood_aggregates_card",),
        STAGE2,
    ),
    ScriptSpec(
        "neighborhood_aggregates_merchant",
        "embedding/neighborhood_aggregates_merchant.gsql",
        ("neighborhood_aggregates_merchant",),
        STAGE2,
    ),
    ScriptSpec(
        "export_embedding_features",
        "embedding/export_embedding_features.gsql",
        ("export_embedding_features",),
        STAGE2,
    ),
)

_BY_NAME: dict[str, ScriptSpec] = {spec.name: spec for spec in _SCRIPTS}

#: query name -> the script that CREATEs it. A query is created by exactly one
#: script, so this inverts cleanly; the installer needs it to answer "which
#: files must I run so that this call has an endpoint".
_BY_QUERY: dict[str, ScriptSpec] = {
    query: spec for spec in _SCRIPTS for query in spec.queries
}


class GsqlPathError(KeyError):
    pass


def gsql_root() -> Path:
    return Path(__file__).resolve().parents[3] / "gsql"


def spec(script_name: str) -> ScriptSpec:
    found = _BY_NAME.get(script_name)
    if found is None:
        known = ", ".join(sorted(_BY_NAME))
        raise GsqlPathError(f"unknown GSQL script {script_name!r}; known: {known}")
    return found


def gsql_path(script_name: str) -> Path:
    return gsql_root() / spec(script_name).relpath


def script_names(groups: tuple[Group, ...] = (CORE,)) -> list[str]:
    return [item.name for item in _SCRIPTS if item.group in groups]


def query_names(groups: tuple[Group, ...] = (CORE,)) -> list[str]:
    names: list[str] = []
    for item in _SCRIPTS:
        if item.group in groups:
            names.extend(item.queries)
    return names


def script_for_query(query_name: str) -> str:
    """The script that CREATEs ``query_name``.

    The installer works in scripts and the pipeline works in queries, and the
    two are not 1:1 -- ``export_card_and_merchant_features`` creates two. Going
    from a call back to the file that must run is the only way to answer
    "would `make stage1-install` give this call an endpoint" without a
    hand-maintained second list to fall out of date.
    """
    found = _BY_QUERY.get(query_name)
    if found is None:
        known = ", ".join(sorted(_BY_QUERY))
        raise GsqlPathError(
            f"no GSQL script creates query {query_name!r}; known: {known}"
        )
    return found.name


def scripts_for_queries(query_names_wanted: Iterable[str]) -> list[str]:
    """Deduplicated scripts covering every named query, in REGISTRY order.

    Registry order is RUN order (see the module header), so preserving it keeps
    one list documenting both the pipeline and the installer.
    """
    wanted = {script_for_query(name) for name in query_names_wanted}
    return [item.name for item in _SCRIPTS if item.name in wanted]


def all_specs() -> tuple[ScriptSpec, ...]:
    return _SCRIPTS
