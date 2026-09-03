FACT_ID = "transaction_id"
FACT_JOIN_KEYS = ["card_number", "merchant_id", "mer_cat"]
FACT_ORDERING = ["event_seq", "unix_time", "split_id", "causal_fold"]
FACT_NUMERIC = ["amount"]
FACT_CATEGORICAL = ["use_chip", "error"]
FACT_BOOLEAN = ["is_online"]
LABEL = "is_fraud"
TIME = "unix_time"
SPLIT = "split_id"

FACT_COLUMNS = (
    [FACT_ID]
    + FACT_JOIN_KEYS
    + FACT_ORDERING
    + FACT_NUMERIC
    + FACT_CATEGORICAL
    + FACT_BOOLEAN
    + [LABEL]
)

# split_id domain, from schema.gsql: 0 train, 1 val, 2 test, -1 unset.
SPLIT_IDS: dict[str, int] = {"train": 0, "val": 1, "test": 2}


# ======================================================================
# dimension tables, as the export queries project them
# ======================================================================
# Order matches the PRINT projection in each file so a column-count drift
# shows up as a name mismatch here rather than as a silent reindex.

CARD_DIM_COLUMNS = [
    "card_number",
    "build_id",
    # NO GEOGRAPHY COLUMNS. home_lat / home_lon used to ride in here so
    # assemble could haversine them against the merchant's pair. The
    # cardholder home is now a function of TIME -- relocation -- so the
    # distance is computed once, in card_home_distance, which has the
    # cutoff, and arrives via PAIR_DIM_COLUMNS instead.
    "pagerank",
    "c_size",
    "cc_degree",
    "louvain_size",
    "core_number",
    "distinct_merchant_count",
    "repeated_merchant_count",
    "txn_count",
    "total_amount",
    "max_txn_amount",
    "min_txn_amount",
    "avg_txn_amount",
    "max_amount_in_interval",
    "max_txn_count_in_interval",
    "c_id",
    "louvain_id",
    "re_id",
]

MERCHANT_DIM_COLUMNS = [
    "id",
    "build_id",
    # No geography columns. See CARD_DIM_COLUMNS.
    "pagerank",
    "c_size",
    "cc_degree",
    "louvain_size",
    "core_number",
    "distinct_card_count",
    "repeated_card_count",
    "txn_count",
    "total_amount",
    "max_txn_amount",
    "min_txn_amount",
    "avg_txn_amount",
    "c_id",
    "louvain_id",
    "re_id",
]

CATEGORY_DIM_COLUMNS = [
    "category",
    "build_id",
    "distinct_merchant_count",
    "txn_count",
    "total_amount",
    "max_txn_amount",
    "min_txn_amount",
    "avg_txn_amount",
]

COMMUNITY_DIM_COLUMNS = [
    "cid",
    "build_id",
    "entity_family",
    "member_count",
    "txn_count",
    "total_amount",
    "max_amount",
    "min_amount",
    "avg_amount",
]

RESOLVED_ENTITY_DIM_COLUMNS = [
    "reid",
    "build_id",
    "member_count",
    "distinct_pii_count",
    "connected_card_count",
    "connected_merchant_count",
    "max_pair_score",
]

# query name -> (expected columns, primary key)
DIMENSION_TABLES: dict[str, tuple[list[str], str]] = {
    "export_card_features": (CARD_DIM_COLUMNS, "card_number"),
    "export_merchant_features": (MERCHANT_DIM_COLUMNS, "id"),
    "export_category_features": (CATEGORY_DIM_COLUMNS, "category"),
    "export_community_features": (COMMUNITY_DIM_COLUMNS, "cid"),
    "export_resolved_entity_features": (RESOLVED_ENTITY_DIM_COLUMNS, "reid"),
}

# ----------------------------------------------------------------------
# the pair table -- gsql/export/export_pair_features.gsql
# ----------------------------------------------------------------------
# NOT a dimension: its key is the (card, merchant) PAIR, so it is kept out of
# DIMENSION_TABLES, whose contract is one table per vertex type with a single
# primary key that export_dimensions asserts is unique.
#
# It exists because pair_home_distance_km stopped being derivable from
# per-entity columns. The cardholder home is now a function of time, so the
# card dimension carries a different home per build and the flat join cannot
# express "combine THIS card's home-at-this-cutoff with that merchant".
# card_home_distance has the cutoff and does it once; this exports the answer.
PAIR_DIM_COLUMNS = [
    "card_number",
    "merchant_id",
    "home_distance_km",
]

#: query name -> (expected columns, composite key)
PAIR_TABLES: dict[str, tuple[list[str], list[str]]] = {
    "export_pair_features": (PAIR_DIM_COLUMNS, ["card_number", "merchant_id"]),
}


# ======================================================================
# prefixes in the assembled matrix
# ======================================================================
# Community and Resolved_Entity are each joined TWICE -- once through the
# card and once through the merchant -- so a prefix is the only thing
# keeping the two copies apart.

CARD_PREFIX = "card_"
MERCHANT_PREFIX = "mer_"
CATEGORY_PREFIX = "cat_"
CARD_COMMUNITY_PREFIX = "ccom_"
MERCHANT_COMMUNITY_PREFIX = "mcom_"
CARD_ENTITY_PREFIX = "cre_"
MERCHANT_ENTITY_PREFIX = "mre_"


def _pref(prefix: str, names: list[str]) -> list[str]:
    return [f"{prefix}{name}" for name in names]


# ======================================================================
# feature families
# ======================================================================

# Fact-table attributes. Identical across all four project stages.
FACT_FEATURES = FACT_NUMERIC + FACT_CATEGORICAL + FACT_BOOLEAN

# One-hop group-bys. Shared byte for byte with any no-graph arm.
AGGREGATE_FEATURES = (
    _pref(
        CARD_PREFIX,
        [
            "txn_count",
            "total_amount",
            "max_txn_amount",
            "min_txn_amount",
            "avg_txn_amount",
            "max_amount_in_interval",
            "max_txn_count_in_interval",
        ],
    )
    + _pref(
        MERCHANT_PREFIX,
        [
            "txn_count",
            "total_amount",
            "max_txn_amount",
            "min_txn_amount",
            "avg_txn_amount",
        ],
    )
    + _pref(
        CATEGORY_PREFIX,
        [
            "distinct_merchant_count",
            "txn_count",
            "total_amount",
            "max_txn_amount",
            "min_txn_amount",
            "avg_txn_amount",
        ],
    )
)

# Everything that required a traversal. This is the only set whose
# contribution may be reported as graph lift.
GRAPH_FEATURES = (
    _pref(
        CARD_PREFIX,
        [
            "pagerank",
            "c_size",
            "cc_degree",
            "louvain_size",
            "core_number",
            "distinct_merchant_count",
            "repeated_merchant_count",
        ],
    )
    + _pref(
        MERCHANT_PREFIX,
        [
            "pagerank",
            "c_size",
            "cc_degree",
            "louvain_size",
            "core_number",
            "distinct_card_count",
            "repeated_card_count",
        ],
    )
    + _pref(
        CARD_COMMUNITY_PREFIX,
        [
            "member_count",
            "txn_count",
            "total_amount",
            "max_amount",
            "min_amount",
            "avg_amount",
        ],
    )
    + _pref(
        MERCHANT_COMMUNITY_PREFIX,
        [
            "member_count",
            "txn_count",
            "total_amount",
            "max_amount",
            "min_amount",
            "avg_amount",
        ],
    )
    + _pref(
        CARD_ENTITY_PREFIX,
        [
            "member_count",
            "distinct_pii_count",
            "connected_card_count",
            "connected_merchant_count",
            "max_pair_score",
        ],
    )
    + _pref(
        MERCHANT_ENTITY_PREFIX,
        [
            "member_count",
            "distinct_pii_count",
            "connected_card_count",
            "connected_merchant_count",
            "max_pair_score",
        ],
    )
    # ------------------------------------------------------------------
    # GEOGRAPHY. The only PAIR-level feature: it is a property of
    # (card, merchant), not of either one alone, so it is the one column
    # here that cannot fingerprint a single entity.
    # ------------------------------------------------------------------
    # In GRAPH_FEATURES because it requires traversal -- Card -> Party ->
    # Street_Address and Merchant -> Merchant_Location -- which is the
    # ablation's own definition of the graph arm. It is not an aggregate:
    # no group-by produces it.
    #
    # WHY IT IS HERE, measured rather than assumed: physical-merchant
    # fraud rate separates 7.6x between top and bottom quintile AND
    # TRANSFERS across the temporal split at r = 0.91, over 85.6% of
    # rows. PhantomLedger's card-present fraud picks its merchant with
    # weight popularity * exp(-miles / decayScale(home)), so distance
    # from home IS the generating mechanism. Before this column the
    # feature set contained no geography at all, so the only route to
    # that signal was memorising merchant identity -- part of the
    # measured entity fingerprinting was the model routing around this
    # absence.
    #
    # EXCLUDED FROM THE RANK+BIN TRANSFORM in tfgnn.entity_scale: it is
    # per-pair rather than per-entity, so it needs neither the rank (it
    # does not drift: geography does not move) nor the bins (it cannot
    # key a dimension).
    + ["pair_home_distance_km"]
)

#: Per-pair features. Excluded from entity_scale's transform because they
#: are not per-entity quantities and neither of its two justifications
#: applies to them.
PAIR_FEATURES = frozenset({"pair_home_distance_km"})


# ======================================================================
# sentinel handling
# ======================================================================
# Safe to test against -1: the real domain excludes it.

SENTINEL_AT_MINUS_ONE = (
    _pref(
        CARD_PREFIX,
        [
            "pagerank",
            "c_size",
            "cc_degree",
            "louvain_size",
            "core_number",
            "distinct_merchant_count",
            "repeated_merchant_count",
            "txn_count",
            "max_txn_count_in_interval",
        ],
    )
    + _pref(
        MERCHANT_PREFIX,
        [
            "pagerank",
            "c_size",
            "cc_degree",
            "louvain_size",
            "core_number",
            "distinct_card_count",
            "repeated_card_count",
            "txn_count",
        ],
    )
    + _pref(CATEGORY_PREFIX, ["distinct_merchant_count", "txn_count"])
    + _pref(CARD_COMMUNITY_PREFIX, ["member_count", "txn_count"])
    + _pref(MERCHANT_COMMUNITY_PREFIX, ["member_count", "txn_count"])
    + _pref(
        CARD_ENTITY_PREFIX,
        [
            "member_count",
            "distinct_pii_count",
            "connected_card_count",
            "connected_merchant_count",
            "max_pair_score",
        ],
    )
    + _pref(
        MERCHANT_ENTITY_PREFIX,
        [
            "member_count",
            "distinct_pii_count",
            "connected_card_count",
            "connected_merchant_count",
            "max_pair_score",
        ],
    )
    # STAGE 3, and ONLY THE TWO GAPS. A card's first transaction has no
    # previous one, so both gaps are UNDEFINED and -1 is correct. The counts,
    # the amount and the tenure are 0 on a first transaction, and 0 is the
    # TRUE, KNOWABLE answer -- zero prior transactions, zero prior spend, zero
    # elapsed tenure. Coding those as -1 would discard a real observation and
    # make "no history" indistinguishable from "not computed", which is the
    # distinction wcc_bipartite's cc_degree header draws for exactly this
    # reason. Listed as literals because TEMPORAL_FEATURES is defined below.
    + [
        "row_card_gap_seconds",
        "row_pair_gap_seconds",
        # gap_ratio needs a MEAN of prior gaps, so it is undefined until two
        # priors exist. row_amount_z is NOT here: a z-score of -1.0 is a
        # legitimate value, so it cannot be tested against the sentinel --
        # it is guarded in SENTINEL_GROUPS instead, exactly as amounts are.
        "row_card_gap_ratio",
    ]
)

# ======================================================================
# STAGE 3. Per-row temporal features, from gsql/aggregates/card_row_temporal.
# ======================================================================
# PREFIX IS ``row_``, NOT ``t_``. baseline/features.py already emits t_month,
# t_day, t_hour and t_dayofweek into EVERY arm including raw, so a t_ prefix
# here would make the double-counting audit unreadable -- and the distinction
# is the whole point of Stage 3: raw already has CALENDAR time, and these add
# INTER-ARRIVAL and VELOCITY.
#
# Unlike every other family in this file these are PER-ROW, not per-entity.
# They are keyed on the transaction, joined by `id`, and carry no build_id:
# each window is anchored at the row's own unix_time, which makes it causal by
# construction and identical under every build. See the query header for why a
# cutoff-anchored version would encode which snapshot served the row.
TEMPORAL_FEATURES = [
    "row_card_gap_seconds",
    "row_pair_gap_seconds",
    "row_card_txn_count_1h",
    "row_card_txn_count_24h",
    "row_card_amount_24h",
    "row_card_tenure_seconds",
    # Added 2026-08-03, all from the SAME scan -- no new traversal. Aimed at
    # the mechanism that actually worked (+0.3115) rather than at another GNN.
    "row_card_gap_ratio",
    "row_amount_z",
    "row_pair_txn_count",
]

# (guard, dependents). An amount of -1.0 is legal for a refund, so amounts
# are never tested directly. Each producing query writes its amount block
# in the same ``IF count > 0`` branch as the guard count, so a guard still
# at -1 means the whole block was never computed.
SENTINEL_GROUPS: list[tuple[str, list[str]]] = [
    # STAGE 3. A z-score of -1.0 is legal, so row_amount_z cannot be tested
    # directly. row_card_gap_seconds is -1 exactly when there is no prior
    # transaction, which is also exactly when no mean or variance exists, so
    # it is the correct guard.
    ("row_card_gap_seconds", ["row_amount_z"]),
    (
        f"{CARD_PREFIX}txn_count",
        _pref(
            CARD_PREFIX,
            ["total_amount", "max_txn_amount", "min_txn_amount", "avg_txn_amount"],
        ),
    ),
    (
        f"{CARD_PREFIX}max_txn_count_in_interval",
        _pref(CARD_PREFIX, ["max_amount_in_interval"]),
    ),
    (
        f"{MERCHANT_PREFIX}txn_count",
        _pref(
            MERCHANT_PREFIX,
            ["total_amount", "max_txn_amount", "min_txn_amount", "avg_txn_amount"],
        ),
    ),
    (
        f"{CATEGORY_PREFIX}txn_count",
        _pref(
            CATEGORY_PREFIX,
            ["total_amount", "max_txn_amount", "min_txn_amount", "avg_txn_amount"],
        ),
    ),
    (
        f"{CARD_COMMUNITY_PREFIX}txn_count",
        _pref(
            CARD_COMMUNITY_PREFIX,
            ["total_amount", "max_amount", "min_amount", "avg_amount"],
        ),
    ),
    (
        f"{MERCHANT_COMMUNITY_PREFIX}txn_count",
        _pref(
            MERCHANT_COMMUNITY_PREFIX,
            ["total_amount", "max_amount", "min_amount", "avg_amount"],
        ),
    ),
]


# ======================================================================
# the assembled matrix
# ======================================================================

METADATA_COLUMNS = [FACT_ID] + FACT_JOIN_KEYS + FACT_ORDERING + ["build_id"]

ALL_FEATURE_COLUMNS = (
    FACT_FEATURES + AGGREGATE_FEATURES + GRAPH_FEATURES + TEMPORAL_FEATURES
)

ASSEMBLED_COLUMNS = METADATA_COLUMNS + ALL_FEATURE_COLUMNS + [LABEL]

# Join keys that must never reach the model. All three of c_id, louvain_id
# and re_id are getvid propagated to a group minimum, so the same group
# carries a different number after any reload.
FORBIDDEN_AS_FEATURES = frozenset(
    _pref(CARD_PREFIX, ["c_id", "louvain_id", "re_id"])
    + _pref(MERCHANT_PREFIX, ["c_id", "louvain_id", "re_id"])
    + ["cid", "reid", "entity_family", "build_id"]
    # The raw coordinate columns are GONE from the exports, not merely
    # forbidden: a latitude is a near-unique per-entity value and would be one
    # more fingerprint, and the transferable quantity is the DISTANCE, which is
    # a property of the pair rather than of either entity. Kept listed so a
    # future edit that reintroduces them cannot make them features by accident.
    + _pref(CARD_PREFIX, ["home_lat", "home_lon"])
    + _pref(MERCHANT_PREFIX, ["loc_lat", "loc_lon"])
    + FACT_JOIN_KEYS
    + FACT_ORDERING
)


#: Set by the trainer once the Stage 2 embedding tables are read, because the
#: dimension is a run-time property of the GNN (--out-dim) rather than a
#: contract constant. Empty means the embedding arm has nothing to add and the
#: trainer skips it rather than training a duplicate of raw_plus_graph.
EMBEDDING_FEATURES: list[str] = []


def set_embedding_features(columns: list[str]) -> None:
    """Register the embedding columns present in the matrix.

    Called by the trainer after joining the Stage 2 tables. A module-level
    mutable is not elegant, but the alternative is threading the width through
    every variant_columns caller for one arm, and this keeps the contract check
    able to see the columns.
    """
    global EMBEDDING_FEATURES
    EMBEDDING_FEATURES = list(columns)


#: Stage 4's TGN memory read-out (`temb_*`), kept in a SEPARATE registry from
#: Stage 2's R-GCN embeddings (`cemb_`/`memb_`).
#:
#: THIS SPLIT IS WHY `+ TGN embeddings` WAS NEVER A TGN NUMBER. There was one
#: registry, and `_attach_embeddings` populated it from
#: `artifacts/stage2/embeddings` -- the only directory it knew -- using Stage 2's
#: prefixes. So `raw_plus_temporal_embeddings` and `raw_plus_embeddings` read the
#: SAME R-GCN vectors, and the arm reported as the TGN's was
#: `raw_plus_temporal + R-GCN`. The two measured deltas agree with that reading
#: exactly: -0.0231 for R-GCN on the graph base and -0.0496 for the same encoder
#: on the temporal base, both negative, same direction.
#:
#: With two registries the arms name what they contain, and both can be measured
#: in one run.
TEMPORAL_EMBEDDING_FEATURES: list[str] = []

#: Pair-interaction columns derived from an embedding family: dot product,
#: cosine and L2 distance between a row's card and merchant vectors.
#:
#: THREE COLUMNS INSTEAD OF 128 OR 200, and the reason is structural rather than
#: economical. A link-prediction GNN encodes what it knows about a pair in the
#: INTERACTION of the two vectors; a tree splits one dimension at a time and
#: cannot form that interaction. Stage 4a's FraudHead already acts on this --
#: it takes the elementwise product explicitly -- and these arms carry the same
#: reasoning into the XGBoost comparison.
RGCN_PAIR_FEATURES: list[str] = []
TGN_PAIR_FEATURES: list[str] = []

RGCN_PAIR_PREFIX = "rgcn_pair"
TGN_PAIR_PREFIX = "tgn_pair"


def set_rgcn_pair_features(columns: list[str]) -> None:
    """Register the Stage 2 pair-score columns present in the matrix."""
    global RGCN_PAIR_FEATURES
    RGCN_PAIR_FEATURES = list(columns)


def set_tgn_pair_features(columns: list[str]) -> None:
    """Register the Stage 4 pair-score columns present in the matrix."""
    global TGN_PAIR_FEATURES
    TGN_PAIR_FEATURES = list(columns)


#: Column prefixes for Stage 4's export. Declared HERE rather than in
#: tfgnn.stage4.run so that baseline.dataset can read them without importing
#: torch -- the offline validators and the smoke run without it.
TEMPORAL_CARD_EMBED_PREFIX = "temb_c"
TEMPORAL_MERCHANT_EMBED_PREFIX = "temb_m"


def set_temporal_embedding_features(columns: list[str]) -> None:
    """Register the Stage 4 TGN embedding columns present in the matrix."""
    global TEMPORAL_EMBEDDING_FEATURES
    TEMPORAL_EMBEDDING_FEATURES = list(columns)


def variant_columns(variant: str) -> list[str]:
    """Feature columns for one ablation arm."""
    if variant == "raw":
        return list(FACT_FEATURES)
    if variant == "raw_plus_aggregates":
        return FACT_FEATURES + AGGREGATE_FEATURES
    if variant == "raw_plus_graph":
        return FACT_FEATURES + AGGREGATE_FEATURES + GRAPH_FEATURES
    if variant == "raw_plus_temporal_embeddings":
        # STAGE 4b. Stage 3's columns PLUS the TGN memory read-out. Built on
        # raw_plus_temporal, so its lift isolates what the temporal GNN adds
        # over the hand-built temporal features it was fed -- the same
        # discipline that makes the GNN lift measure the R-GCN rather than the
        # graph features. The embedding columns are registered at runtime by
        # set_temporal_embedding_features -- the TGN family (`temb_*`), NOT
        # Stage 2's. Reading EMBEDDING_FEATURES here is what made this arm
        # report the R-GCN under the TGN's name.
        if not TEMPORAL_EMBEDDING_FEATURES:
            raise ValueError(
                "raw_plus_temporal_embeddings needs the Stage 4 TGN embedding "
                "columns (temb_*). Run "
                "`python -m tfgnn.stage4.run --objective link` first -- the "
                "`fraud` objective writes to embeddings_fraud/, which the lift "
                "table deliberately does not read, because its vectors come "
                "from label-trained weights."
            )
        return (
            FACT_FEATURES
            + AGGREGATE_FEATURES
            + GRAPH_FEATURES
            + TEMPORAL_FEATURES
            + TEMPORAL_EMBEDDING_FEATURES
        )
    if variant == "raw_plus_temporal":
        # STAGE 3. Built on raw_plus_graph, NOT on raw -- "no graph" in the
        # architecture doc means no GNN, and Stage 3 is specified as "same
        # data and schema as Stage 1, plus temporal". So this is a SIBLING of
        # raw_plus_embeddings off the same base, and the two together are the
        # 2x2's temporal and GNN cells. Stage 4 is the interaction of both.
        return FACT_FEATURES + AGGREGATE_FEATURES + GRAPH_FEATURES + TEMPORAL_FEATURES
    if variant == "raw_plus_aggregates_embeddings":
        # THE REPLACEMENT QUESTION, and no other arm asks it. Every embedding arm
        # carries the hand-built GRAPH block alongside the embeddings, so nothing
        # tests whether the GNN can STAND IN for PageRank, entity resolution and
        # pair geography rather than pile on top of them. This arm drops the graph
        # block and keeps the embeddings, so its comparison against
        # raw_plus_graph is "learned structure versus engineered structure" at the
        # same level of investment.
        #
        # ARCHITECTURE.md's design implies this question -- the whole point of a
        # GNN over a graph database is that it learns what you would otherwise
        # hand-build -- and never states it, so it was never measured.
        if not EMBEDDING_FEATURES:
            raise ValueError(
                "raw_plus_aggregates_embeddings needs the Stage 2 embedding "
                "columns. Run `python -m tfgnn.stage2.run` first."
            )
        return FACT_FEATURES + AGGREGATE_FEATURES + EMBEDDING_FEATURES

    if variant == "raw_plus_pair_scores":
        # Sibling of raw_plus_embeddings off the SAME base, differing only in how
        # the R-GCN is presented: three pair statistics rather than 128 raw
        # dimensions. The pair of arms isolates dilution from information.
        if not RGCN_PAIR_FEATURES:
            raise ValueError(
                "raw_plus_pair_scores needs the Stage 2 pair-score columns. Run "
                "`python -m tfgnn.stage2.run` first; they are derived from its "
                "embedding tables at load time."
            )
        return FACT_FEATURES + AGGREGATE_FEATURES + GRAPH_FEATURES + RGCN_PAIR_FEATURES

    if variant == "raw_plus_temporal_pair_scores":
        # Sibling of raw_plus_temporal_embeddings off raw_plus_temporal, same
        # contrast for the TGN: three pair statistics rather than 200 dimensions.
        if not TGN_PAIR_FEATURES:
            raise ValueError(
                "raw_plus_temporal_pair_scores needs the Stage 4 pair-score "
                "columns. Run `python -m tfgnn.stage4.run --objective link` "
                "first; they are derived from its embedding tables at load time."
            )
        return (
            FACT_FEATURES
            + AGGREGATE_FEATURES
            + GRAPH_FEATURES
            + TEMPORAL_FEATURES
            + TGN_PAIR_FEATURES
        )

    if variant == "raw_plus_embeddings":
        # Stage 1's full column set PLUS the GNN. Built on raw_plus_graph, not
        # on raw, so the lift measures what the GNN adds to the graph features
        # rather than what the graph and the GNN add together.
        if not EMBEDDING_FEATURES:
            raise ValueError(
                "raw_plus_embeddings needs embedding columns. Run "
                "`python -m tfgnn.stage2.run` first, then re-run the trainer; "
                "it registers them via set_embedding_features()."
            )
        return FACT_FEATURES + AGGREGATE_FEATURES + GRAPH_FEATURES + EMBEDDING_FEATURES
    raise ValueError(f"unknown variant: {variant!r}")


def _assert_contract() -> None:
    """Fail at import if the contract contradicts itself."""
    overlap = set(AGGREGATE_FEATURES) & set(GRAPH_FEATURES)
    if overlap:
        raise RuntimeError(
            "a column cannot be both a group-by and graph lift: " + f"{sorted(overlap)}"
        )
    leaked = set(ALL_FEATURE_COLUMNS) & FORBIDDEN_AS_FEATURES
    if leaked:
        raise RuntimeError(f"join keys present in the feature set: {sorted(leaked)}")
    duplicates = sorted(
        {name for name in ASSEMBLED_COLUMNS if ASSEMBLED_COLUMNS.count(name) > 1}
    )
    if duplicates:
        raise RuntimeError(f"duplicate columns in the assembled matrix: {duplicates}")
    known = set(ALL_FEATURE_COLUMNS)
    unknown = sorted(set(SENTINEL_AT_MINUS_ONE) - known)
    if unknown:
        raise RuntimeError(f"sentinel list names non-feature columns: {unknown}")
    for guard, dependents in SENTINEL_GROUPS:
        if guard not in known:
            raise RuntimeError(f"sentinel guard is not a feature column: {guard}")
        missing = sorted(set(dependents) - known)
        if missing:
            raise RuntimeError(
                f"sentinel group {guard} names unknown columns: {missing}"
            )


_assert_contract()
