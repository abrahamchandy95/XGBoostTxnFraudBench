import argparse
from collections.abc import Mapping
from typing import Any, cast

from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.gsql_paths import CORE, query_names
from tfgnn.tigergraph.settings import Settings


_REQUIRED_VERTICES: frozenset[str] = frozenset(
    {
        "Card",
        "Merchant",
        "Payment_Transaction",
        "Merchant_Category",
        "Community",
        "Party",
        "Resolved_Entity",
        "Birthdate",
        "Street_Address",
        "Identity_Document",
        "Device",
        "IP",
        "Email_Address",
        "Email_Address_Hash",
        "Full_Name",
        "Full_Name_Hash",
        "Phone",
        "Phone_Hash",
        "Merchant_Location",
    }
)

_REQUIRED_EDGES: frozenset[str] = frozenset(
    {
        # raw transaction edges and the reverses the queries actually traverse
        "Card_Send_Transaction",
        "Transaction_From_Card",
        "Transaction_To_Merchant",
        "Merchant_Received_Transaction",
        # derived, cleared per build
        "Has_Interaction_With_Merchant",
        "Card_Card",
        "Merchant_Merchant",
        "Has_Community",
        # category and geography
        "Merchant_Assigned",
        "Merchant_Has_Location",
        # identity
        "Party_Has_Card",
        "Party_Is_Merchant",
        "Party_Resolved_As",
        "Same_As",
    }
)

# The six-entry match surface, byte-identical in five identity queries.
# Party_Has_Device / Party_Has_IP are deliberately absent: co-use is not
# identity. They are Stage 2 message-passing inputs (Party_Shares_*), not
# match evidence -- see match_parties.gsql's header.
_MATCH_SURFACE: tuple[str, ...] = (
    "Party_Has_Birthdate",
    "Party_Has_Street_Address",
    "Party_Has_Identity_Document",
    "Party_Has_Email_Address_Hash",
    "Party_Has_Name_Hash",
    "Party_Has_Phone_Hash",
)

# The raw-value edges match_parties reads for the fuzzy comparison.
_FUZZY_SOURCE_EDGES: tuple[str, ...] = (
    "Party_Has_Email_Address",
    "Party_Has_Name",
    "Party_Has_Phone",
)

_REQUIRED_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "Payment_Transaction": (
        # The r3 change that unblocked Stage 1. Without these two, every
        # dimension table has nothing to join onto.
        "card_number",
        "merchant_id",
        # The ordering and split axis. Every cutoff is read off event_seq.
        "event_seq",
        "unix_time",
        "split_id",
        "causal_fold",
        # Fact-table features and the label.
        "amount",
        "is_fraud",
        "mer_cat",
        "use_chip",
        "error",
        "is_online",
    ),
    "Card": (
        "card_number",
        "first_seen_event_seq",
        "build_id",
        "seen",
        "pagerank",
        "c_id",
        "c_size",
        "cc_degree",
        "louvain_size",
        "core_number",
        "distinct_merchant_count",
        "repeated_merchant_count",
        "re_id",
        "txn_count",
        "total_amount",
        "max_txn_amount",
        "min_txn_amount",
        "avg_txn_amount",
        "max_amount_in_interval",
        "max_txn_count_in_interval",
    ),
    "Merchant": (
        "id",
        "build_id",
        "seen",
        "pagerank",
        "c_id",
        "c_size",
        "cc_degree",
        "louvain_size",
        "core_number",
        "distinct_card_count",
        "repeated_card_count",
        "re_id",
        "txn_count",
        "total_amount",
        "max_txn_amount",
        "min_txn_amount",
        "avg_txn_amount",
    ),
    "Merchant_Category": (
        "category",
        "build_id",
        "distinct_merchant_count",
        "txn_count",
        "total_amount",
        "max_txn_amount",
        "min_txn_amount",
        "avg_txn_amount",
    ),
    "Community": (
        "cid",
        "entity_family",
        "member_count",
        "build_id",
        "txn_count",
        "total_amount",
        "max_amount",
        "min_amount",
        "avg_amount",
    ),
    "Resolved_Entity": (
        "reid",
        "member_count",
        "distinct_pii_count",
        "max_pair_score",
        "connected_card_count",
        "connected_merchant_count",
        "build_id",
    ),
    "Party": ("id", "first_seen_event_seq", "re_id"),
}

# Attributes the queries would use if they existed, but whose absence is
# handled rather than fatal.
_KNOWN_ABSENT: dict[str, str] = {
    "Merchant.distinct_location_count": (
        "DECLARED in schema r3 and in tf_gnn_loader_v2's schema -- the two are "
        "byte-identical on all 22 vertices and 31 edges -- but populated by "
        "NEITHER: the loader's Merchant job skips the column and "
        "card_merchant_degree_stats prints max_locations_per_merchant rather "
        "than storing it. So it sits at its -1 default. Nothing projects it, so "
        "this is expected. It stays unwritten deliberately: PhantomLedger gives "
        "each merchant exactly one location, so the column would be a constant, "
        "and a constant is a column of noise that absorbs regularisation budget."
    )
}


class PreflightError(RuntimeError):
    pass


def _entries(schema: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    values = schema.get(key, [])
    if not isinstance(values, list):
        return []
    return [item for item in cast(list[Any], values) if isinstance(item, dict)]


def _name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("Name") or entry.get("name") or "")


def _attribute_names(entry: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    primary = entry.get("PrimaryId")
    if isinstance(primary, Mapping):
        candidate = cast(Mapping[str, Any], primary).get("AttributeName")
        if candidate:
            names.add(str(candidate))
    attributes = entry.get("Attributes", [])
    if isinstance(attributes, list):
        for attribute in cast(list[Any], attributes):
            if isinstance(attribute, Mapping):
                label = cast(Mapping[str, Any], attribute).get("AttributeName")
                if label:
                    names.add(str(label))
    return names


def _has_outdegree_stats(entry: Mapping[str, Any]) -> bool:
    config = entry.get("Config")
    if not isinstance(config, Mapping):
        return False
    stats = cast(Mapping[str, Any], config).get("STATS")
    return str(stats or "").strip().upper() == "OUTDEGREE_BY_EDGETYPE"


def _installed_query_params(client: Client) -> dict[str, set[str]] | None:
    """Per installed query, the parameter names its REST endpoint accepts.

    This is the stale-endpoint check's raw material. TigerGraph SILENTLY
    IGNORES unknown query parameters (systemprompt.md section 0a), so an
    endpoint installed before a query grew a parameter accepts the call and
    runs the OLD logic -- for reset_build that is the monolithic
    single-transaction delete that wedged the engine on 2026-07-31, and for
    an armed density gate it is an ABORT_* key that never prints and
    therefore never enforces. Name-presence checking cannot see any of this;
    only the parameter signature can.

    Returns None when this pyTigerGraph version or instance exposes no
    parameter metadata -- the caller downgrades to a warning rather than
    inventing a verdict.
    """
    function = getattr(client.conn, "getEndpoints", None)
    if function is None:
        return None
    try:
        payload = function(dynamic=True)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    out: dict[str, set[str]] = {}
    for key, entry in cast(Mapping[str, Any], payload).items():
        tail = str(key).rstrip("/").rsplit("/", 1)[-1]
        if not tail or not isinstance(entry, Mapping):
            continue
        parameters = cast(Mapping[str, Any], entry).get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        names = {str(name) for name in cast(Mapping[str, Any], parameters)}
        previous = out.get(tail)
        out[tail] = (previous | names) if previous else names
    return out or None


def _plan_call_params() -> dict[str, set[str]]:
    """Per query, the union of parameter names the configured plan passes.

    Derived from ``pipeline.build_calls`` -- the same source install_gsql's
    plan_queries uses for names -- so the checked set cannot disagree with
    what the run will actually send.
    """
    from common.config import load_raw_config
    from tfgnn.tigergraph.pipeline import PIPELINE_ADAPTER, build_calls
    from tfgnn.tigergraph.plan import Build

    config = load_raw_config()
    plan = PIPELINE_ADAPTER.validate_python(config["stage1_pipeline"])
    probe = Build("b_probe", 1, "literal:1", (0,))
    out: dict[str, set[str]] = {}
    for call in build_calls(plan, probe):
        out.setdefault(call.name, set()).update(call.params)
    return out


def _installed_queries(client: Client) -> set[str] | None:
    """Best effort: pyTigerGraph exposes this differently across versions."""
    for method in ("getInstalledQueries", "getEndpoints"):
        function = getattr(client.conn, method, None)
        if function is None:
            continue
        try:
            payload = (
                function()
                if method == "getInstalledQueries"
                else function(dynamic=True)
            )
        except Exception:
            continue
        if isinstance(payload, Mapping):
            names: set[str] = set()
            for key in cast(Mapping[str, Any], payload):
                # endpoints look like "GET /query/TF_GNN/wcc_card"
                tail = str(key).rstrip("/").rsplit("/", 1)[-1]
                if tail:
                    names.add(tail)
            if names:
                return names
    return None


def run(*, strict: bool) -> None:
    settings = Settings()
    client = Client(settings)

    print(f"graph        {settings.graphname}")
    print(f"REST++ echo  {client.conn.echo()}")

    schema = cast(dict[str, Any], client.conn.getSchema(force=True))
    vertices = {_name(item): item for item in _entries(schema, "VertexTypes")}
    edges = {_name(item): item for item in _entries(schema, "EdgeTypes")}

    # REVERSE EDGES ARE QUERYABLE TYPES BUT ARE NOT LISTED AS EdgeTypes.
    # getSchema reports them only as the REVERSE_EDGE config of their forward
    # edge, so checking EdgeTypes names alone reported Transaction_From_Card
    # and Merchant_Received_Transaction as missing on a perfectly healthy
    # instance. schema.gsql warns about exactly this population: "reverse
    # edges are queryable edge types, they are exactly the relations nobody
    # thinks to declare, and exactly the ones that leak."
    reverses: dict[str, dict[str, Any]] = {}
    for entry in edges.values():
        config = entry.get("Config")
        if isinstance(config, Mapping):
            name = str(cast(Mapping[str, Any], config).get("REVERSE_EDGE") or "")
            if name:
                reverses[name] = entry
    edges.update(reverses)
    print(
        f"schema       {len(vertices)} vertex types, "
        f"{len(edges) - len(reverses)} edge types + {len(reverses)} reverses"
    )

    problems: list[str] = []
    warnings: list[str] = []

    missing_vertices = sorted(_REQUIRED_VERTICES - set(vertices))
    if missing_vertices:
        problems.append(f"missing vertex types: {missing_vertices}")

    required_edges = set(_REQUIRED_EDGES) | set(_FUZZY_SOURCE_EDGES)
    missing_edges = sorted(required_edges - set(edges))
    if missing_edges:
        problems.append(f"missing edge types: {missing_edges}")

    # The match surface is a set of runtime string literals, so a missing type
    # is silent: no error, no match, no score contribution.
    absent_match = [name for name in _MATCH_SURFACE if name not in edges]
    if absent_match:
        problems.append(
            f"missing PII match edge types: {absent_match}. match_parties "
            "resolves these as string literals at run time, so it would "
            "install, run, and score nothing on those relations without an "
            "error."
        )

    for vertex, attributes in _REQUIRED_ATTRIBUTES.items():
        entry = vertices.get(vertex)
        if entry is None:
            continue
        present = _attribute_names(entry)
        absent = sorted(set(attributes) - present)
        if absent:
            problems.append(f"{vertex} is missing attributes: {absent}")

    for qualified, reason in _KNOWN_ABSENT.items():
        vertex, attribute = qualified.split(".", 1)
        entry = vertices.get(vertex)
        if entry is not None and attribute in _attribute_names(entry):
            warnings.append(
                f"{qualified} EXISTS on this instance, but the repo's queries "
                f"no longer write or project it ({reason}) Restore the three "
                "sites if you want the column."
            )

    for vertex in ("Card", "Merchant"):
        entry = vertices.get(vertex)
        if entry is not None and not _has_outdegree_stats(entry):
            problems.append(
                f'{vertex} does not declare STATS="OUTDEGREE_BY_EDGETYPE". '
                "card_card_with_weights and merchant_merchant_with_weights "
                'call outdegree("Has_Interaction_With_Merchant") for the hub '
                "degree cap, which is their only lever against a hub. Without "
                "it the projection may not return at all."
            )

    installed = _installed_queries(client)
    # CORE is not the set the run needs: `optional.*` config flags pull OPTIONAL
    # scripts into the call set, and checking CORE alone would report "all
    # queries installed" while the very query the first build calls is missing --
    # which is this check's entire job. install_gsql derives the required set
    # from pipeline.build_calls, so ask it rather than re-deriving here.
    required_queries = set(query_names((CORE,)))
    try:
        from tfgnn.tigergraph.gsql_paths import spec
        from tfgnn.tigergraph.install_gsql import default_scripts

        required_queries = {
            query
            for name in default_scripts(include_optional=False, include_stage2=False)
            for query in spec(name).queries
        }
    except Exception as error:  # noqa: BLE001 - a preflight must not be the thing that fails
        warnings.append(
            f"could not read the configured call set ({error!r}); falling back "
            "to the CORE group. An enabled optional family may be uninstalled "
            "without this check noticing."
        )

    if installed is None:
        warnings.append(
            "could not read the installed-query list from this pyTigerGraph "
            "version; skipping that check"
        )
    else:
        absent_queries = sorted(required_queries - installed)
        if absent_queries:
            warnings.append(
                f"{len(absent_queries)} queries the configured plan calls are "
                f"not installed: {absent_queries}. Run "
                "`python -m tfgnn.tigergraph.install_gsql`; CREATE alone does "
                "not produce a REST endpoint."
            )
        else:
            print(
                f"queries      all {len(required_queries)} queries the "
                "configured plan calls are installed"
            )

        # Name presence is not enough: TigerGraph silently ignores unknown
        # query parameters (section 0a), so an endpoint installed before a
        # query grew a parameter runs the OLD logic with no error anywhere.
        # For reset_build that old logic is the monolithic single-transaction
        # delete that wedged the engine on 2026-07-31, and for an armed
        # density gate it is an ABORT_* key that never prints and therefore
        # never enforces. A signature mismatch is a PROBLEM, not a warning.
        endpoint_params = _installed_query_params(client)
        if endpoint_params is None:
            warnings.append(
                "could not read endpoint parameter metadata from this "
                "pyTigerGraph version; skipping the stale-endpoint signature "
                "check. A stale endpoint silently ignores parameters it "
                "predates -- re-run install_gsql if in any doubt."
            )
        else:
            try:
                plan_params = _plan_call_params()
            except Exception as error:  # noqa: BLE001 - preflight must not be the thing that fails
                plan_params = {}
                warnings.append(
                    f"could not derive the plan's call parameters ({error!r}); "
                    "skipping the stale-endpoint signature check"
                )
            stale_endpoints: list[str] = []
            for query, wanted in sorted(plan_params.items()):
                accepted = endpoint_params.get(query)
                if accepted is None:
                    continue  # not installed at all; reported above
                unknown = sorted(wanted - accepted)
                if unknown:
                    stale_endpoints.append(f"{query} ignores {unknown}")
            if stale_endpoints:
                problems.append(
                    "stale installed endpoints -- these queries' endpoints "
                    "predate parameters the plan passes, and TigerGraph "
                    "silently drops unknown parameters, so each call would "
                    "'succeed' running the OLD logic: "
                    + "; ".join(stale_endpoints)
                    + ". Run `python -m tfgnn.tigergraph.install_gsql`."
                )
            elif plan_params:
                print(
                    "signatures   every parameter the plan passes is accepted "
                    "by its installed endpoint"
                )

    for message in warnings:
        print(f"  ! {message}")

    if problems:
        detail = "\n".join(f"  - {message}" for message in problems)
        message = f"schema preflight found {len(problems)} problem(s):\n{detail}"
        if strict:
            raise PreflightError(message)
        print(message)
        return

    print("preflight passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check auth, the r3 schema and the installed query set"
    )
    _ = parser.add_argument(
        "--report-only",
        action="store_true",
        help="print problems instead of exiting non-zero",
    )
    args = parser.parse_args()
    run(strict=not bool(args.report_only))


if __name__ == "__main__":
    main()
