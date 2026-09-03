#!/usr/bin/env python
import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfgnn.tigergraph import gsql_paths  # noqa: E402
from tfgnn.tigergraph.plan import Build  # noqa: E402


_QUERY = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:DISTRIBUTED\s+)?QUERY\s+"
    r"([A-Za-z_]\w*)\s*\(",
    re.IGNORECASE,
)
_ACCUM_DECL = re.compile(
    r"\b(?:Sum|Min|Max|Avg|Or|And|Set|List|Map|Heap|Array|Bitwise\w*|Group\w*)"
    r"Accum\s*(?:<[^;]*?>)?\s*(?:\([^)]*\))?\s*([^;]*);",
    re.S,
)
_ACCUM_NAME = re.compile(r"(@@?[A-Za-z_]\w*)")
_TYPES = "UINT|INT|STRING|DOUBLE|FLOAT|BOOL|DATETIME"

# TUPLE field access and GSQL methods look like attribute access but are not.
_BUILTIN_MEMBERS = frozenset(
    {
        "outdegree",
        "indegree",
        "size",
        "clear",
        "get",
        "containsKey",
        "contains",
        "pop",
        "top",
        "type",
        "keys",
        "remove",
        "add",
        "update",
        "setAttr",
        "getAttr",
        "filter",
        "neighbors",
        "neighborAttribute",
        "ver",
        "str",
    }
)


# Names that cannot be a vertex-set variable, ESTABLISHED EMPIRICALLY by
# feeding `<name> = SELECT c FROM Card:c LIMIT 1;` to the parser -- 43 of 76
# plausible candidates. Guessing from the parser's expected-token list does
# NOT work: `all`, `init`, `path` and `union`-adjacent words like `on` and
# `by` appear in that list and are perfectly legal here, while `filter`,
# `order` and `collect` are not. assert_cardinality's `all = SELECT ...`
# installs cleanly, so a guessed list produces false failures on working
# queries. Re-derive this list against a new TigerGraph version rather than
# extending it by intuition.
_RESERVED = frozenset(
    {
        "accum",
        "and",
        "any",
        "avg",
        "batch",
        "break",
        "case",
        "coalesce",
        "collect",
        "continue",
        "count",
        "delete",
        "edge",
        "false",
        "filter",
        "foreach",
        "having",
        "intersect",
        "isempty",
        "limit",
        "max",
        "min",
        "minus",
        "not",
        "null",
        "or",
        "order",
        "print",
        "raise",
        "return",
        "sample",
        "select",
        "set",
        "static",
        "stdev",
        "sum",
        "trim",
        "true",
        "try",
        "union",
        "vertex",
        "where",
        "while",
    }
)

# `name = SELECT ...` at the start of a statement.
_ASSIGNMENT = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*SELECT\b", re.IGNORECASE)

# The target side of an edge pattern: -( ... )- Name:alias
_TRAVERSAL_TARGET = re.compile(r"-\s*\([^)]*\)\s*-\s*(?:>\s*)?([A-Za-z_]\w*)\s*:")

_SYNTAX_V1 = re.compile(r"\bSYNTAX\s+V1\b", re.IGNORECASE)


class Failure(RuntimeError):
    pass


@dataclass(frozen=True)
class Query:
    name: str
    parameters: set[str]
    path: Path


def strip_comments(text: str) -> str:
    """Blank out comments while PRESERVING offsets and line breaks.

    Replacing a block comment with a single space shifts every offset after
    it, so reported line numbers pointed into the wrong part of the file --
    which is worse than no line number, because it sends you to a line that
    looks fine. Each comment character becomes a space and each newline
    survives, so offsets and line numbers stay exact.
    """

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    return re.sub(r"//[^\n]*", blank, text)


def _balanced(text: str, opening: int, open_char: str, close_char: str) -> int:
    depth = 0
    index = opening
    while index < len(text):
        if text[index] == open_char:
            depth += 1
        elif text[index] == close_char:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise Failure(f"unbalanced {open_char}{close_char} from offset {opening}")


def schema_attributes() -> set[str]:
    text = strip_comments(gsql_paths.gsql_path("schema").read_text(encoding="utf-8"))
    names: set[str] = set()
    for match in re.finditer(
        r"ADD\s+VERTEX\s+\w+\s*\((.*?)\)\s*WITH", text, re.S | re.I
    ):
        for line in match.group(1).split(","):
            found = re.match(r"\s*(?:PRIMARY_ID\s+)?([A-Za-z_]\w*)\s+", line)
            if found:
                names.add(found.group(1))
    for match in re.finditer(
        r"ADD\s+(?:DIRECTED|UNDIRECTED)\s+EDGE\s+\w+\s*\((.*?)\)\s*(?:WITH|;)",
        text,
        re.S | re.I,
    ):
        for line in match.group(1).split(","):
            found = re.match(rf"\s*([A-Za-z_]\w*)\s+(?:{_TYPES})", line)
            if found:
                names.add(found.group(1))
        for extra in re.findall(
            rf"DISCRIMINATOR\(\s*(\w+)\s+(?:{_TYPES})", match.group(1)
        ):
            names.add(extra)
    if not names:
        raise Failure("could not parse any attributes out of schema.gsql")
    return names


def parse_queries(path: Path) -> list[Query]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    queries: list[Query] = []
    for match in _QUERY.finditer(text):
        open_paren = text.index("(", match.end() - 1)
        close_paren = _balanced(text, open_paren, "(", ")")
        signature = text[open_paren + 1 : close_paren]

        parameters: set[str] = set()
        for chunk in signature.split(","):
            found = re.match(rf"\s*(?:{_TYPES})\s+([A-Za-z_]\w*)", chunk.strip())
            if found:
                parameters.add(found.group(1))
        queries.append(Query(match.group(1), parameters, path))
    return queries


def check_registry() -> dict[str, Query]:
    problems: list[str] = []
    catalogue: dict[str, Query] = {}

    for spec in gsql_paths.all_specs():
        path = gsql_paths.gsql_root() / spec.relpath
        if not path.is_file():
            problems.append(f"{spec.name}: registry points at a missing file {path}")
            continue
        if spec.group == gsql_paths.SCHEMA:
            continue

        found = parse_queries(path)
        names = {query.name for query in found}
        declared = set(spec.queries)

        for missing in sorted(declared - names):
            problems.append(
                f"{spec.name}: registry claims query {missing!r} but "
                f"{spec.relpath} does not create it (creates: {sorted(names)})"
            )
        for extra in sorted(names - declared):
            problems.append(
                f"{spec.name}: {spec.relpath} creates {extra!r}, which the "
                "registry does not list, so the installer will never INSTALL it"
            )
        for query in found:
            catalogue[query.name] = query

    if problems:
        raise Failure("\n".join(problems))
    print(
        f"  registry: {len(gsql_paths.all_specs())} scripts, {len(catalogue)} queries"
    )
    return catalogue


def check_accumulators() -> None:
    problems: list[str] = []
    for spec in gsql_paths.all_specs():
        if spec.group == gsql_paths.SCHEMA:
            continue
        path = gsql_paths.gsql_root() / spec.relpath
        text = strip_comments(path.read_text(encoding="utf-8"))

        for match in _QUERY.finditer(text):
            brace = text.index(
                "{", _balanced(text, text.index("(", match.end() - 1), "(", ")")
            )
            body = text[brace : _balanced(text, brace, "{", "}") + 1]

            declared: set[str] = set()
            for declaration in _ACCUM_DECL.finditer(body):
                declared.update(_ACCUM_NAME.findall(declaration.group(1)))
            used = set(_ACCUM_NAME.findall(body))
            for name in sorted(used - declared):
                problems.append(
                    f"{spec.relpath}:{match.group(1)}: accumulator {name} is "
                    "used but never declared, so the query will not install"
                )
    if problems:
        raise Failure("\n".join(problems))
    print("  accumulators: every @ and @@ name is declared in its own query")


def check_attributes(known: set[str]) -> None:
    problems: list[str] = []
    for spec in gsql_paths.all_specs():
        if spec.group == gsql_paths.SCHEMA:
            continue
        path = gsql_paths.gsql_root() / spec.relpath
        text = strip_comments(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\b([A-Za-z]\w{0,14})\.([A-Za-z_]\w*)\b", text):
            alias, attribute = match.group(1), match.group(2)
            if attribute in known or attribute in _BUILTIN_MEMBERS:
                continue
            if alias in {"e", "s", "t", "v"} or len(alias) <= 12:
                problems.append(
                    f"{spec.relpath}: {alias}.{attribute} -- {attribute!r} is "
                    "not an attribute of any vertex or edge type in schema.gsql"
                )
    if problems:
        raise Failure("\n".join(sorted(set(problems))))
    print(f"  attributes: every reference resolves against {len(known)} schema names")


def check_pipeline(catalogue: dict[str, Query]) -> None:
    from tfgnn.tigergraph.pipeline import PIPELINE_ADAPTER, build_calls
    from common.config import load_raw_config

    config = load_raw_config()
    plan = PIPELINE_ADAPTER.validate_python(config["stage1_pipeline"])

    # Build probe plans that exercise every optional branch AND both
    # graph_features sources, so a param typo in a disabled family or in the
    # mode not currently configured is still caught. TigerGraph silently
    # ignores unknown parameters, which is the whole reason this check exists;
    # a branch the config does not select is exactly where such a typo hides.
    probe = Build("b_probe", 1234, "literal:1234", (0,))
    problems: list[str] = []
    total_calls = 0
    for source in ("projection", "bipartite"):
        exhaustive = dict(plan)
        exhaustive["graph_features"] = {"source": source}
        exhaustive["optional"] = {
            **plan["optional"],
            "louvain": True,
            # build_calls never emits kcore in bipartite mode (main() rejects
            # the combination before a run); forcing it on here still probes
            # the projection branch.
            "kcore": True,
            "card_interval_max": True,
        }
        exhaustive["identity"] = {**plan["identity"], "enabled": True}

        calls = build_calls(exhaustive, probe)  # type: ignore[arg-type]
        total_calls += len(calls)

        for call in calls:
            query = catalogue.get(call.name)
            if query is None:
                problems.append(
                    f"pipeline ({source} mode) calls {call.name!r}, which no "
                    "GSQL file creates"
                )
                continue
            unknown = sorted(set(call.params) - query.parameters)
            if unknown:
                problems.append(
                    f"{call.name} ({source} mode): pipeline passes parameters "
                    f"{unknown} that the query does not declare. TigerGraph "
                    f"ignores unknown parameters, so these would silently take "
                    f"their defaults. Declared: {sorted(query.parameters)}"
                )

    # EVERY QUERY THE CONFIGURED PLAN CALLS MUST BE INSTALLED BY THE COMMAND THE
    # RUNBOOK TELLS PEOPLE TO USE. Two real bugs of this shape:
    #
    #   projection_weight_histogram stayed OPTIONAL from its days as a by-hand
    #   diagnostic after the pipeline started calling it unconditionally.
    #
    #   config.yaml enables optional.louvain, so build_calls emits louvain_card /
    #   louvain_merchant, but both are OPTIONAL scripts and `make stage1-install`
    #   installed the CORE group only. the runner would have died on a missing
    #   REST endpoint partway into the first build, AFTER reset_build destroyed
    #   the previous one. It never bit because the archived runs went through
    #   run_all.sh, which passes --include-optional.
    #
    # Neither is visible to the catalogue check above: that is built from every
    # spec regardless of group, so the query exists -- it is just not installed.
    #
    # This asserts the property that actually matters, against the CONFIGURED
    # plan rather than a minimal one: install_gsql's default set covers every
    # call. install_gsql derives that set from build_calls, so this passes by
    # construction while the derivation is right -- and fails the moment the
    # query-to-script mapping or the group assignment breaks it.
    from tfgnn.tigergraph.install_gsql import default_scripts

    installed = {
        query
        for name in default_scripts(include_optional=False, include_stage2=False)
        for query in gsql_paths.spec(name).queries
    }
    for call in build_calls(plan, probe):  # type: ignore[arg-type]
        if call.name not in installed:
            problems.append(
                f"the pipeline calls {call.name!r} with the config on disk, but "
                "install_gsql's default set does not install it. "
                "`make stage1-install` would leave it without a REST endpoint "
                "and the build would fail partway in, after reset_build. Either "
                "move its ScriptSpec to CORE, or make sure install_gsql."
                "plan_queries() reaches the call."
            )

    # The same guarantee for the queries build_calls does NOT emit: the load gate
    # and everything the exporter calls by name.
    for name in (
        "assert_cardinality",
        "assert_edge_stamp",
        "derive_build_plan",
        "export_transaction_rows",
    ):
        if name not in installed:
            problems.append(
                f"{name!r} is called by the exporter or the load gate but is not "
                "in install_gsql's default set"
            )

    # The export and validate queries the other stages call by name.
    from tfgnn.features_meta import DIMENSION_TABLES, PAIR_TABLES

    for name in (
        "assert_cardinality",
        "derive_build_plan",
        "export_transaction_rows",
        *DIMENSION_TABLES,
        # The pair table is exported by the same per-build path but is not a
        # DIMENSION_TABLE, so it needs naming separately or nothing checks that
        # the exporter's own call has a query behind it.
        *PAIR_TABLES,
    ):
        if name not in catalogue:
            problems.append(f"{name!r} is called by the exporter but never created")

    for name, (_, _) in DIMENSION_TABLES.items():
        query = catalogue.get(name)
        if query is not None and "build_id" not in query.parameters:
            problems.append(
                f"{name} does not take build_id, but the exporter passes it"
            )

    # ---- READ-AFTER-WRITE ORDER BETWEEN QUERIES ----
    # These pairs are ordering dependencies created by composing an aggregate
    # from a STORED ATTRIBUTE another query writes, rather than by re-walking
    # the transactions. Composition is what makes them cheap -- and it is also
    # what makes the order load-bearing in a way no gate inside either query
    # can see: run the reader first and every attribute reads its -1 sentinel,
    # so the reader completes, writes plausible zeros-and-sentinels, and says
    # nothing. Checked HERE because the order lives in build_calls, which is
    # exactly what this file already reconstructs.
    for mode in ("bipartite", "projection"):
        mode_plan = dict(plan)
        mode_plan["graph_features"] = {"source": mode}
        names = [
            call.name
            for call in build_calls(mode_plan, probe)  # type: ignore[arg-type]
        ]
        for writer, reader, what in (
            (
                "merchant_transaction_stats",
                "merchant_category_stats",
                "category aggregates are summed from Merchant.txn_count and "
                "the four amount attributes",
            ),
            (
                "card_transaction_stats",
                "community_stats",
                "community aggregates are summed from Card.txn_count and the "
                "four amount attributes",
            ),
            (
                "merchant_transaction_stats",
                "community_stats",
                "merchant-only communities sum their aggregates from "
                "Merchant.txn_count and the four amount attributes",
            ),
        ):
            if writer not in names or reader not in names:
                continue
            if names.index(writer) > names.index(reader):
                problems.append(
                    f"[{mode}] {reader} runs BEFORE {writer}, but {what}. "
                    "Every merchant would read -1 and every category would "
                    "fall to its no-transactions sentinel, silently."
                )

    if problems:
        raise Failure("\n".join(problems))
    print(
        f"  pipeline: {total_calls} calls across both graph_features sources "
        "(all optional families forced on), every query and parameter name "
        "resolves"
    )


def check_reserved_words() -> None:
    problems: list[str] = []
    for spec in gsql_paths.all_specs():
        if spec.group == gsql_paths.SCHEMA:
            continue
        path = gsql_paths.gsql_root() / spec.relpath
        text = strip_comments(path.read_text(encoding="utf-8"))
        for match in _ASSIGNMENT.finditer(text):
            name = match.group(1)
            if name.lower() in _RESERVED:
                problems.append(
                    f"{spec.relpath}: `{name} = SELECT ...` -- {name!r} is a "
                    "GSQL keyword, so the query will not parse. Rename it."
                )
    if problems:
        raise Failure("\n".join(problems))
    print("  reserved words: no vertex-set variable shadows a GSQL keyword")


def check_v1_traversal_targets(vertex_types: set[str]) -> None:
    """A V1 query may only traverse INTO a vertex type, never a set variable.

    Core failures block. Optional and Stage-2 failures are reported and do not
    block, because they are not installed by the default run -- but they are
    real, and every one of them must be fixed before that family is enabled.
    louvain in particular is the documented remedy when wcc_card reports one
    giant component, so a broken louvain is a broken escape hatch.
    """
    problems: list[str] = []
    deferred: list[str] = []
    for spec in gsql_paths.all_specs():
        if spec.group == gsql_paths.SCHEMA:
            continue
        path = gsql_paths.gsql_root() / spec.relpath
        text = strip_comments(path.read_text(encoding="utf-8"))
        if not _SYNTAX_V1.search(text):
            continue
        for match in _TRAVERSAL_TARGET.finditer(text):
            target = match.group(1)
            if target in vertex_types:
                continue
            line = text[: match.start()].count("\n") + 1
            bucket = problems if spec.group == gsql_paths.CORE else deferred
            bucket.append(
                f"{spec.relpath}:{line}: SYNTAX V1 query traverses into "
                f"{target!r}, which is not a vertex type. A vertex-set "
                "variable as a traversal target is V2 syntax and TigerGraph "
                'rejects the query with "The query specifies V1 syntax but '
                'uses V2 here." Target the vertex type and restate the set '
                "membership as a WHERE predicate."
            )
    if problems:
        raise Failure("\n".join(problems))
    if deferred:
        names = sorted({item.split(":")[0] for item in deferred})
        print(
            f"  V1 traversals: core is clean; {len(deferred)} violations remain "
            f"in {len(names)} NON-CORE files, which are not installed by the "
            "default run:"
        )
        for name in names:
            print(f"      {name}")
        print(
            "      Each needs the same fix as wcc_card: target the vertex type "
            "and\n      restate set membership as a predicate. kcore and the "
            "hierarchical\n      sweeps additionally need an OrAccum flag, "
            "because their target set\n      SHRINKS and cannot be written as "
            "a static predicate."
        )
    else:
        print("  V1 traversals: every SYNTAX V1 target is a vertex type")


def schema_vertex_types() -> set[str]:
    text = strip_comments(gsql_paths.gsql_path("schema").read_text(encoding="utf-8"))
    return set(re.findall(r"ADD\s+VERTEX\s+(\w+)", text, re.I))


def check_contract() -> None:
    from tfgnn.features_meta import (
        AGGREGATE_FEATURES,
        ALL_FEATURE_COLUMNS,
        ASSEMBLED_COLUMNS,
        GRAPH_FEATURES,
        TEMPORAL_FEATURES,
        variant_columns,
    )

    # THE REFERENCE ARM IS THE WIDEST NON-GNN ONE, and that is now
    # raw_plus_temporal (Stage 3), not raw_plus_graph. The check is "some arm
    # accounts for every assembled feature column", which catches a family
    # added to ALL_FEATURE_COLUMNS and then never routed to a variant -- it
    # would be exported, assembled, and read by nothing.
    #
    # raw_plus_embeddings is deliberately NOT the reference: its extra columns
    # are registered at RUNTIME by set_embedding_features, so offline it is
    # exactly raw_plus_graph and would make this vacuous.
    total = len(variant_columns("raw_plus_temporal"))
    if total != len(ALL_FEATURE_COLUMNS):
        raise Failure(
            "raw_plus_temporal does not cover every feature column: "
            f"{total} in the arm against {len(ALL_FEATURE_COLUMNS)} in "
            "ALL_FEATURE_COLUMNS. A feature family was added to the contract "
            "without being routed to a variant."
        )
    print(
        f"  contract: {len(ASSEMBLED_COLUMNS)} columns, "
        f"{len(AGGREGATE_FEATURES)} aggregate + {len(GRAPH_FEATURES)} graph "
        f"+ {len(TEMPORAL_FEATURES)} temporal = {total} features"
    )


def check_fold_boundaries() -> None:
    """fold_boundaries against every fold_source, with no connection.

    This path is otherwise untested offline: the smoke test drives build_plan
    with the default train_folds=1, so the whole forward-chaining expansion --
    the change that produced the +0.0286 graph lift -- never runs there.

    The cases below are the ones where being wrong is silent. A boundary off by
    one fold withholds the wrong rows from training, and nothing downstream
    reports it as anything but a slightly different row count.
    """
    from tfgnn.tigergraph.plan import (
        BuildPlanError,
        LoadFacts,
        build_plan,
        fold_boundaries,
    )

    def facts(folds: dict[int, int], fold_rows: dict[int, int]) -> LoadFacts:
        # 1,000 training rows at ranks 1..1000, val at 1001, test at 1501.
        return LoadFacts(
            total_transactions=2000,
            split_first_event_seq={0: 1, 1: 1001, 2: 1501},
            split_last_event_seq={0: 1000, 1: 1500, 2: 2000},
            split_rows={0: 1000, 1: 500, 2: 500},
            split_frauds={0: 10, 1: 5, 2: 5},
            fold_first_event_seq=folds,
            fold_rows=fold_rows,
            fold_frauds={},
        )

    problems: list[str] = []

    def expect(label: str, got: object, want: object) -> None:
        if got != want:
            problems.append(f"{label}: expected {want!r}, got {got!r}")

    # -- 1. no folds stamped: derive from event_seq. Today's live behaviour. --
    bare = facts({}, {})
    edges, origin = fold_boundaries(bare, 5, "auto")
    expect("auto with no folds -> source", origin, "event_seq")
    expect("auto with no folds -> edges", edges, [1, 201, 401, 601, 801, 1001])

    # ONE fold is the ABSENCE of a partition, not a partition of one. The smoke
    # test's fake graph reports exactly this, so a regression here would change
    # what the smoke test exercises without failing it.
    one = facts({0: 1}, {0: 1000})
    _, origin = fold_boundaries(one, 5, "auto")
    expect("auto with one fold -> source", origin, "event_seq")

    # -- 2. real folds, count matching train_folds: prefer them. --
    real = facts(
        {0: 1, 1: 150, 2: 400, 3: 615, 4: 900},
        {0: 149, 1: 250, 2: 215, 3: 285, 4: 101},
    )
    edges, origin = fold_boundaries(real, 5, "auto")
    expect("auto with five folds -> source", origin, "causal_fold")
    expect("auto with five folds -> edges", edges, [1, 150, 400, 615, 900, 1001])

    # event_seq stays available and unchanged, so an archived run reproduces
    # even against data that now carries real folds.
    edges, origin = fold_boundaries(real, 5, "event_seq")
    expect("forced event_seq -> source", origin, "event_seq")
    expect("forced event_seq -> edges", edges, [1, 201, 401, 601, 801, 1001])

    # -- 3. fold 0 is warm-up: 5 folds -> 4 builds, windows abutting exactly.
    #       Driven through build_plan rather than the expansion helper, so the
    #       dispatch that decides a spec is expandable is covered too. All three
    #       splits are specified because build_plan refuses a plan that leaves
    #       any split unserved -- which is itself the behaviour being relied on.
    spec = [
        {"build_id": "b_train", "cutoff": "val_boundary", "serves_splits": [0]},
        {"build_id": "b_val", "cutoff": "val_boundary", "serves_splits": [1]},
        {"build_id": "b_test", "cutoff": "test_boundary", "serves_splits": [2]},
    ]
    planned = build_plan(spec, real, train_folds=5, fold_source="auto")
    builds = [item for item in planned if item.is_windowed]
    expect("five folds -> build count", len(builds), 4)
    expect("val and test still served", len(planned) - len(builds), 2)
    expect(
        "windows",
        [(b.serves_from, b.serves_until) for b in builds],
        [(150, 400), (400, 615), (615, 900), (900, 1001)],
    )
    expect(
        "cutoff equals the window start",
        [b.cutoff_event_seq for b in builds],
        [150, 400, 615, 900],
    )
    if "causal_fold" not in builds[0].cutoff_source:
        problems.append(
            "cutoff_source does not record the authority that placed the "
            f"boundary: {builds[0].cutoff_source!r}"
        )

    # -- 4. every way the loader can disagree must RAISE, never pick. --
    def must_raise(label: str, facts_in: LoadFacts, folds: int, source: str) -> None:
        try:
            _ = fold_boundaries(facts_in, folds, source)
        except BuildPlanError:
            return
        problems.append(f"{label}: expected BuildPlanError, got a plan")

    must_raise(
        "count mismatch",
        facts({0: 1, 1: 400, 2: 800}, {0: 399, 1: 400, 2: 201}),
        5,
        "auto",
    )
    must_raise(
        "non-contiguous fold indices",
        facts({0: 1, 2: 400}, {0: 399, 2: 601}),
        2,
        "auto",
    )
    must_raise(
        "folds not strictly increasing",
        facts({0: 400, 1: 400}, {0: 500, 1: 500}),
        2,
        "auto",
    )
    must_raise(
        "a fold outside the training split",
        facts({0: 1, 1: 1200}, {0: 500, 1: 500}),
        2,
        "auto",
    )
    must_raise(
        "folds do not cover split 0",
        facts({0: 1, 1: 400}, {0: 399, 1: 100}),
        2,
        "auto",
    )
    must_raise("causal_fold demanded but absent", bare, 5, "causal_fold")
    must_raise("unknown fold_source", real, 5, "fold:0")

    if problems:
        raise Failure("\n".join(problems))
    print(
        "  fold boundaries: event_seq fallback, causal_fold preference and "
        "7 disagreement cases all behave"
    )


def check_tuning_provenance() -> None:
    """The stale-tuning advisory fires on drift and stays quiet in band.

    structure.min_weight and the Stage 2 fan-out caps are hand-picked against a
    degree distribution the pipeline now measures. An advisory that has only
    ever been seen not to fire is the same failure as pagerank_card's
    structurally-zero counter, so both branches are exercised here rather than
    waiting for a regenerated dataset to try them.
    """
    from tfgnn.tigergraph.pipeline import advise
    from tfgnn.tuning_provenance import OBSERVATIONS, drift_notes

    problems: list[str] = []

    # -- 1. in band: silent. --
    at_rest = {key: record.value for key, record in OBSERVATIONS.items()}
    if drift_notes(at_rest):
        problems.append(
            "drift_notes fired on the exact values it records as measured: "
            f"{drift_notes(at_rest)}"
        )

    # -- 2. the predicted churn case must be caught. S is 3.507 as measured on
    #       b_test; merchant churn is predicted to take it to about 0.10. --
    if not drift_notes({"hub_concentration_S": 0.10}):
        problems.append(
            "drift_notes missed hub_concentration_S collapsing to 0.10, which is "
            "the change merchant churn is predicted to make and the reason this "
            "check exists"
        )

    # -- 3. an unknown key is not an alarm, and a missing key is not either. --
    if drift_notes({"some_key_nobody_recorded": 1e9}) or drift_notes({}):
        problems.append("drift_notes invents findings for keys it has no record of")

    # -- 4. end to end through advise(), including the attachment share, which
    #       is derived from two keys rather than read from one. --
    quiet = advise(
        "projection_weight_histogram",
        {
            "cards_seen": 1000,
            "card_card_max_weight": 39,
            "cards_with_degree_by_threshold": {"10": 918},
        },
        min_weight=10,
    )
    if quiet:
        problems.append(f"advise fired at the tuned attachment share: {quiet}")

    loud = advise(
        "projection_weight_histogram",
        {
            "cards_seen": 1000,
            "card_card_max_weight": 39,
            # min_weight now isolates all but 43% of cards, which is what
            # config.yaml records happening at a threshold of 20.
            "cards_with_degree_by_threshold": {"10": 432},
        },
        min_weight=10,
    )
    if not any("min_weight" in message for message in loud):
        problems.append(
            "advise did not flag min_weight when attachment fell from 91.8% to "
            f"43.2%: {loud}"
        )

    # A threshold the histogram has no row for must not be scored against
    # whichever probe happens to be nearest.
    if advise(
        "projection_weight_histogram",
        {"cards_seen": 1000, "cards_with_degree_by_threshold": {"10": 918}},
        min_weight=7,
    ):
        problems.append(
            "advise scored min_weight=7 against the row for a different threshold"
        )

    if problems:
        raise Failure("\n".join(problems))
    print(
        "  tuning provenance: stale-tuning advisory fires on the predicted "
        "churn and is quiet in band"
    )


def check_resume_guards() -> None:
    """--from-index checks the plan it resumes into and keeps the old evidence.

    Both halves guard failures that are invisible in a run which succeeds.
    Batches partition cards by ``getvid % num_of_source_batches``, so the
    recovery the projection comment recommends -- raise the batch count after a
    sync batch times out, then resume -- leaves cards covered by neither the
    committed batches nor the resumed ones, producing no Card_Card pairs with
    no gate that notices. And ``write_manifest`` rewrites the whole file, so
    the first post-resume call used to erase every pre-crash gate record, on
    exactly the runs the A/B acceptance needs to read.

    The third case is §0f's pair of bipartite tuning inputs: the attachment
    share is DERIVED and appears in no query output, so the manifest has to
    carry it or the commit that picks structure.min_txn_count recomputes it by
    hand.
    """
    from common.config import load_raw_config
    from tfgnn.tigergraph.pipeline import (
        PIPELINE_ADAPTER,
        resume_from_manifest,
        build_calls,
        check_plan_alignment,
        check_projection_partition,
        tuning_observations,
    )

    problems: list[str] = []
    config = load_raw_config()
    plan = PIPELINE_ADAPTER.validate_python(config["stage1_pipeline"])

    # The configured default is bipartite, which emits no projection batches at
    # all; the batch plan this check is about only exists in projection mode.
    # 16 batches so the "resumed under a different modulus" case has room.
    probe = dict(plan)
    probe["graph_features"] = {"source": "projection"}
    probe["projection"] = {**plan["projection"], "num_of_source_batches": 16}
    build = Build("b_probe", 1234, "literal:1234", (0,))
    flat = [(call, build) for call in build_calls(probe, build)]  # type: ignore[arg-type]
    first = next(
        index
        for index, (call, _owner) in enumerate(flat)
        if call.name == "card_card_with_weights"
    )

    # What the crashed run's manifest holds: five batches committed under 10.
    committed: list[dict[str, object]] = [
        {
            "index": first + offset,
            "build_id": "b_probe",
            "query": "card_card_with_weights",
            "status": "ok",
            "params": {"source_batch": offset, "num_of_source_batches": 10},
            "output": {"card_card_edges_inserted": 4000, "source_batch_count": 10},
        }
        for offset in range(5)
    ]

    # -- 1. resuming PAST a committed batch under a different count is the hole,
    #       and the refusal has to name both counts or it is not actionable. --
    try:
        _ = check_projection_partition(
            committed, flat, from_index=first + 5, projection_batches=16
        )
        problems.append(
            "check_projection_partition allowed a resume at a batch count of 16 "
            "over batches committed under 10, which leaves cards in neither "
            "residue set with no Card_Card pairs"
        )
    except SystemExit as refusal:
        message = str(refusal)
        if "16" not in message or "10" not in message:
            problems.append(
                "the resume refusal names neither the configured nor the "
                f"recorded batch count, so it cannot be acted on: {message}"
            )
        if str(first) not in message:
            problems.append(
                "the resume refusal does not name the build's first batch index "
                f"{first}, which is the only safe place to restart: {message}"
            )

    # -- 2. resuming AT the build's first batch re-runs the whole projection
    #       phase under the new count, which is the documented recovery. --
    try:
        _ = check_projection_partition(
            committed, flat, from_index=first, projection_batches=16
        )
    except SystemExit as refusal:
        problems.append(
            "check_projection_partition refused a resume at the build's FIRST "
            f"batch, which is the recovery the comments prescribe: {refusal}"
        )

    # -- 3. an unchanged count resumes anywhere, and a batch that never
    #       committed constrains nothing. --
    unchanged = [
        {
            **entry,
            "params": {"source_batch": 0, "num_of_source_batches": 16},
            "output": {"source_batch_count": 16},
        }
        for entry in committed
    ]
    crashed = [{**entry, "status": "failed", "output": {}} for entry in committed]
    for label, records in (("unchanged", unchanged), ("failed", crashed)):
        try:
            _ = check_projection_partition(
                records, flat, from_index=first + 5, projection_batches=16
            )
        except SystemExit as refusal:
            problems.append(
                f"check_projection_partition refused a resume over {label} "
                f"batch records: {refusal}"
            )

    # -- 4. bipartite mode materializes no projection, so a manifest from a
    #       projection run cannot block a resume in it. --
    bipartite = dict(plan)
    bipartite["graph_features"] = {"source": "bipartite"}
    bipartite_flat = [
        (call, build)
        for call in build_calls(bipartite, build)  # type: ignore[arg-type]
    ]
    try:
        _ = check_projection_partition(
            committed, bipartite_flat, from_index=2, projection_batches=16
        )
    except SystemExit as refusal:
        problems.append(
            f"check_projection_partition refused a bipartite-mode resume, which "
            f"runs no card_card_with_weights at all: {refusal}"
        )

    # -- 4b. the SAME residue hole through reset_build: batches committed
    #        under one count, resumed under another, past the build's first
    #        reset call -- un-reset vertices carry the previous build's state
    #        and no per-batch counter notices. None must SKIP the guard, so
    #        pre-generalization callers keep their meaning. --
    reset_committed: list[dict[str, object]] = [
        {
            "index": offset,
            "build_id": "b_probe",
            "query": "reset_build",
            "status": "ok",
            "params": {"source_batch": offset, "num_of_source_batches": 10},
            "output": {"cards_reset": 5, "source_batch_count": 10},
        }
        for offset in range(5)
    ]
    try:
        _ = check_projection_partition(
            reset_committed,
            flat,
            from_index=5,
            projection_batches=16,
            reset_batches=16,
        )
        problems.append(
            "check_projection_partition allowed a resume past reset_build "
            "batches committed under a different count, which leaves vertices "
            "in neither residue set carrying the previous build's state"
        )
    except SystemExit as refusal:
        if "reset_build" not in str(refusal):
            problems.append(
                f"the reset_build resume refusal does not name the query: {refusal}"
            )
    try:
        _ = check_projection_partition(
            reset_committed, flat, from_index=5, projection_batches=16
        )
    except SystemExit as refusal:
        problems.append(
            f"reset_batches=None must skip the reset_build guard: {refusal}"
        )

    # -- 4c. a changed plan SHAPE re-bases every --from-index; the alignment
    #        check refuses a carried record that names a different call now,
    #        and skips records build_calls never emits. --
    aligned = check_plan_alignment(
        [
            {
                "index": 0,
                "build_id": "b_probe",
                "query": flat[0][0].name,
                "status": "ok",
            },
            {"index": 1, "query": "assert_cardinality", "status": "ok"},
        ],
        flat,
        from_index=2,
    )
    if aligned.get("plan_alignment_checked") != 1:
        problems.append(
            "check_plan_alignment should check exactly the one plan-emitted "
            f"record and skip assert_cardinality, got {aligned}"
        )
    try:
        _ = check_plan_alignment(
            [
                {
                    "index": 0,
                    "build_id": "b_probe",
                    "query": "card_card_with_weights",
                    "status": "ok",
                }
            ],
            flat,
            from_index=2,
        )
        problems.append(
            "check_plan_alignment allowed a resume whose recorded index 0 "
            f"names card_card_with_weights where this plan has {flat[0][0].name}"
        )
    except SystemExit as refusal:
        if "card_card_with_weights" not in str(refusal):
            problems.append(
                f"the alignment refusal does not name the mismatch: {refusal}"
            )

    # -- 5. the pre-crash gate records survive the resume, and records from a
    #       DIFFERENT load are archived rather than blended into this one's. --
    with tempfile.TemporaryDirectory() as workspace:
        manifest = Path(workspace) / "manifest.json"
        _ = manifest.write_text(
            json.dumps(
                {
                    "graph": "ProbeGraph",
                    "total_transactions": 999,
                    "records": [
                        {
                            "index": 0,
                            "query": "assert_cardinality",
                            "status": "ok",
                            "gates": [{"key": "cards_MUST_BE_ONE", "passed": True}],
                        },
                        {
                            "index": first + 5,
                            "query": "card_card_with_weights",
                            "status": "failed",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        # Its own progress lines belong in a run, not in this listing.
        noise = io.StringIO()
        with contextlib.redirect_stdout(noise):
            carried, evidence = resume_from_manifest(
                manifest,
                flat,
                from_index=first + 5,
                projection_batches=16,
                graphname="ProbeGraph",
                total_transactions=999,
            )
        queries = [entry.get("query") for entry in carried]
        if queries != ["assert_cardinality"]:
            problems.append(
                "the resume carried the wrong records forward: expected the "
                f"pre-crash assert_cardinality alone, got {queries}"
            )
        elif carried[0].get("carried_over") is not True:
            problems.append(
                "carried-forward records are not marked, so the manifest cannot "
                "say which gates ran before the crash"
            )
        elif carried[0].get("gates") != [{"key": "cards_MUST_BE_ONE", "passed": True}]:
            problems.append("the carried-forward record lost its gate results")
        if evidence.get("carried_over_records") != 1:
            problems.append(
                f"the resume evidence miscounts what it carried: {evidence}"
            )

        with contextlib.redirect_stdout(noise):
            other, other_evidence = resume_from_manifest(
                manifest,
                flat,
                from_index=first + 5,
                projection_batches=16,
                graphname="ADifferentGraph",
                total_transactions=999,
            )
        if other:
            problems.append(
                "the resume merged gate records from a different load into this "
                "run's manifest, which reads as a passing check for a run that "
                "never happened"
            )
        if "superseded_manifest" not in other_evidence:
            problems.append(f"a superseded manifest was not archived: {other_evidence}")
        if manifest.exists() or not list(Path(workspace).glob("*.superseded-*.json")):
            problems.append(
                "the other load's manifest was overwritten instead of archived"
            )

    # -- 6. §0f's two bipartite tuning inputs, read at the CONFIGURED
    #       threshold, are what the manifest records. --
    observed = tuning_observations(
        "interaction_strength_histogram",
        {
            "interaction_max_txn_count": 34,
            "cards_seen": 1000,
            "cards_attached_by_threshold": {"2": 927},
        },
        min_txn_count=2,
    )
    wanted = {"interaction_max_txn_count", "cards_attached_share_at_min_txn_count"}
    missing = sorted(wanted - set(observed))
    if missing:
        problems.append(
            f"tuning_observations drops {missing}, so the manifest cannot carry "
            "the inputs §0f's threshold pick has to cite"
        )
    elif abs(observed["cards_attached_share_at_min_txn_count"] - 0.927) > 1e-9:
        problems.append(
            "the recorded attachment share is not attached/seen: "
            f"{observed['cards_attached_share_at_min_txn_count']}"
        )
    if "cards_attached_share_at_min_txn_count" in tuning_observations(
        "interaction_strength_histogram",
        {"cards_seen": 1000, "cards_attached_by_threshold": {"2": 927}},
        min_txn_count=3,
    ):
        problems.append(
            "tuning_observations recorded a share for min_txn_count=3 off the "
            "row for a different threshold"
        )

    if problems:
        raise Failure("\n".join(problems))
    print(
        "  resume: a changed batch count (projection or reset) refuses the "
        "resume, a re-based plan fails alignment, and pre-crash gate records "
        "survive it"
    )


def check_identity_advice() -> None:
    """The inert-weight advisory fires, and is quiet when it should be.

    It replaced a check on match_parties' parties_with_multiple_email / _name /
    _phone, which the query stopped printing when the SumAccum<STRING> that
    concatenated raw PII values was deleted. `value()` returned None for all
    three, so the guard never fired and the advice it would have printed named
    an accumulator that no longer existed. An advisory nobody has ever seen fire
    is the same failure as pagerank_card's structurally-zero counter, so both
    branches are exercised here.
    """
    from tfgnn.tigergraph.pipeline import advise

    problems: list[str] = []
    # The advisory is generic over names, but the fixtures use relations that
    # can actually appear in match_parties' output. They used to name
    # Party_Has_IP / Party_Has_Device, which are off the match surface now --
    # co-use is not identity -- so a fixture naming them would quietly assert
    # behaviour on names the query can no longer print.
    weights = {
        "Party_Has_Birthdate": 0.5,
        "Party_Has_Phone_Hash": 0.5,
        "Party_Has_Identity_Document": 0.6,
    }

    # Every weighted relation carried evidence: silent.
    quiet = advise(
        "match_parties",
        {
            "parties_with_at_least_one_match": 46,
            "weights_used": weights,
            "scored_relations": {name: 4 for name in weights},
        },
    )
    if quiet:
        problems.append(f"inert-weight advisory fired with full evidence: {quiet}")

    # Two weighted relations with no evidence: must name both.
    loud = advise(
        "match_parties",
        {
            "parties_with_at_least_one_match": 46,
            "weights_used": weights,
            "scored_relations": {"Party_Has_Birthdate": 18},
        },
    )
    if not any(
        "Party_Has_Phone_Hash" in m and "Party_Has_Identity_Document" in m for m in loud
    ):
        problems.append(f"inert-weight advisory missed two dead weights: {loud}")

    # A weight deliberately set to 0 is DISABLED, not inert. config.yaml says
    # "0 disables a signal", so flagging it would flag the documented way to
    # turn a signal off.
    if advise(
        "match_parties",
        {
            "parties_with_at_least_one_match": 46,
            "weights_used": {"Party_Has_Identity_Document": 0.0},
            "scored_relations": {},
        },
    ):
        problems.append("inert-weight advisory flagged a deliberately zero weight")

    # Neither map present: no crash, no false alarm. A query that failed before
    # printing them must not also produce a misleading advisory.
    if advise("match_parties", {"parties_with_at_least_one_match": 46}):
        problems.append("inert-weight advisory fired with no maps in the result")

    if problems:
        raise Failure("\n".join(problems))
    print(
        "  identity advice: inert-weight advisory fires on dead weights and is "
        "quiet otherwise"
    )


#: The five files carrying the match surface, as REGISTRY SCRIPT NAMES. A name
#: is the stable identifier; the path is derived from it. Listing relpaths here
#: instead duplicates the layout gsql_paths already owns, and the duplicate is
#: only discovered when someone moves a file -- see _export_gsql.
_MATCH_SURFACE_SCRIPTS = (
    "match_parties",
    "measure_pii_degrees",
    "stamp_same_as_provenance",
    "resolved_entity_stats",
    "assert_resolved_entity",
)

_MATCH_SURFACE_EXPECTED = (
    "Party_Has_Birthdate",
    "Party_Has_Street_Address",
    "Party_Has_Identity_Document",
    "Party_Has_Email_Address_Hash",
    "Party_Has_Name_Hash",
    "Party_Has_Phone_Hash",
)


def check_match_surface() -> None:
    """The six-entry @@edge_type_set is byte-identical in all five files.

    The five queries promise this in a comment, and divergence is only caught
    at runtime (stamp_same_as_provenance's no-shared-pii gate) after a full
    match pass. Device/IP must NOT reappear: co-use is not identity, and a
    single file drifting back would quietly resume merging fraud rings into
    single entities while the other four report nothing unusual.
    """
    pattern = re.compile(r"@@edge_type_set = \((.*?)\);", re.S)
    problems: list[str] = []
    sets: dict[str, tuple[str, ...]] = {}
    for name in _MATCH_SURFACE_SCRIPTS:
        text = gsql_paths.gsql_path(name).read_text(encoding="utf-8")
        found = pattern.search(text)
        if not found:
            problems.append(f"{name}: no @@edge_type_set literal found")
            continue
        entries = tuple(re.findall(r'"([^"]+)"', found.group(1)))
        sets[name] = entries
        if entries != _MATCH_SURFACE_EXPECTED:
            problems.append(
                f"{name}: match surface is {entries}, expected "
                f"{_MATCH_SURFACE_EXPECTED}"
            )
    if len(set(sets.values())) > 1:
        problems.append(f"match surfaces diverge across files: {sets}")

    # Private on purpose: this check IS the sanctioned external reader.
    from tfgnn.tigergraph.preflight import (
        _MATCH_SURFACE,  # pyright: ignore[reportPrivateUsage]
    )

    if tuple(_MATCH_SURFACE) != _MATCH_SURFACE_EXPECTED:
        problems.append(
            f"preflight._MATCH_SURFACE is {_MATCH_SURFACE}, expected "
            f"{_MATCH_SURFACE_EXPECTED}"
        )
    if problems:
        raise Failure("\n".join(problems))
    print("  match surface: six entries, byte-identical in all five files")


#: The query whose branches this check reads. A QUERY NAME, because that is
#: what the loader calls and what the registry indexes -- the file it lives in
#: is gsql_paths' business, not this file's.
_EDGE_EXPORT_QUERY = "export_relation_edges"


def _export_gsql() -> Path:
    """Resolve the edge-export query's file THROUGH THE REGISTRY.

    This used to be the literal ``ROOT / "gsql/export/export_stage2_edges.gsql"``,
    and it is the reason renaming that file was a two-place edit: the rename
    updated gsql_paths, and this string had to be found by hand. Every other
    check in this module already goes through ``gsql_paths``; these two did not,
    and a hand-maintained copy of someone else's layout only announces itself
    when the layout moves.
    """
    return gsql_paths.gsql_path(gsql_paths.script_for_query(_EDGE_EXPORT_QUERY))


def _export_branch(gsql: str, relation: str) -> str | None:
    """The body of one `relation == "X" THEN` branch in the export query."""
    marker = f'relation == "{relation}"'
    start = gsql.find(marker)
    if start < 0:
        return None
    nxt = gsql.find('relation == "', start + len(marker))
    end = gsql.find("PRINT @@rows", start) if nxt < 0 else nxt
    return gsql[start : end if end > start else len(gsql)]


def _edgerow_time_expression(branch: str) -> str | None:
    """The THIRD argument of the branch's ``EdgeRow(...)`` -- its time value."""
    at = branch.find("EdgeRow(")
    if at < 0:
        return None
    depth, current, args = 0, "", []
    for char in branch[at + len("EdgeRow(") :]:
        if char == "(":
            depth += 1
        elif char == ")" and depth == 0:
            args.append(current)
            break
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            args.append(current)
            current = ""
            continue
        current += char
    return args[2].strip() if len(args) >= 3 else None


def check_stage2_wiring() -> None:
    """Every Stage 2 relation is exportable, loadable, and capped.

    The three registries -- schema_spec._SPECS, load.ALL_RELATIONS, and the
    branch chain in export_graph_edges.gsql -- name the same relations, or a
    relation is silently absent from every batch (spec without export), or the
    export dies at runtime (export without a GSQL branch). This is the same
    failure class as the weight_histogram install-group regression: wiring
    that drifts without any check firing.
    """
    from tfgnn.stage2.fanout import NeighborFanout
    from tfgnn.stage2.load import ALL_RELATIONS, PAIR_ONLY
    from tfgnn.stage2.schema_spec import (
        message_passing_specs,
        message_triple_of,
        pyg_metadata,
    )
    from tfgnn.stage2.schema_spec import spec as edge_spec

    problems: list[str] = []
    passing = {item.name for item in message_passing_specs()}
    exported = set(ALL_RELATIONS)
    if exported != passing:
        problems.append(
            f"load.ALL_RELATIONS {sorted(exported)} != message-passing specs "
            f"{sorted(passing)}: a spec without an export is silently absent "
            "from every batch; an export without a spec cannot load"
        )
    if not set(PAIR_ONLY) <= exported:
        problems.append(
            f"PAIR_ONLY names relations outside ALL_RELATIONS: "
            f"{sorted(set(PAIR_ONLY) - exported)}"
        )

    export_path = _export_gsql()
    gsql = export_path.read_text(encoding="utf-8")
    unbranched = sorted(
        name for name in exported if f'relation == "{name}"' not in gsql
    )
    if unbranched:
        problems.append(
            f"{export_path.name} has no branch for {unbranched}; the "
            "export aborts with unknown-relation at runtime"
        )

    # *** time_attr MUST NAME THE ATTRIBUTE THE EXPORT ACTUALLY READS. ***
    # These are two independent statements of the same fact -- a Python
    # declaration and a GSQL expression -- and nothing forced them to agree.
    # Has_Interaction_With_Merchant declared last_event_seq while the export
    # read first_event_seq, which are not interchangeable: filtering a
    # deduplicated pair on its LAST transaction deletes the pair whenever that
    # transaction is in the future, discarding its legitimate past. It went
    # unnoticed because time_attr's only reader, graph.time_attr_for, is called
    # by nothing -- so the declaration was free to be wrong.
    for relation in sorted(exported):
        branch = _export_branch(gsql, relation)
        if branch is None:
            continue  # already reported as unbranched above
        declared = edge_spec(relation).time_attr
        read = _edgerow_time_expression(branch)
        if read is None:
            problems.append(
                f"{relation}: export branch builds no EdgeRow, so nothing "
                "carries a time into the graph"
            )
        elif "." in read:
            attribute = read.split(".", 1)[1].strip()
            if attribute != declared:
                problems.append(
                    f"{relation}: schema_spec declares time_attr="
                    f"{declared!r} but the export branch emits {read!r}. "
                    "One of the two is wrong about which column is the "
                    "temporal guarantee."
                )
        elif declared not in branch:
            # A computed local (the derived co-use relations build `seq` from
            # two stamps). Then the branch must at least READ the declared
            # attribute somewhere, or the declaration is unmoored.
            problems.append(
                f"{relation}: export emits computed {read!r} and its branch "
                f"never reads {declared!r}, which schema_spec declares as its "
                "time attribute"
            )

    # Every page-SOURCE type needs a vid-range branch: getvid does not start
    # at zero (Party begins at ~69M), so a source type the range query cannot
    # answer for would page over [0, 0) and export nothing, silently.
    from tfgnn.stage2.schema_spec import spec as edge_spec

    sources = {edge_spec(name).src for name in exported}
    unranged = sorted(t for t in sources if f'node_type == "{t}"' not in gsql)
    if unranged:
        problems.append(
            f"export_vertex_id_range has no branch for page-source types "
            f"{unranged}; their relations would export zero rows"
        )

    # NeighborFanout() validates that every metadata triple has caps; building
    # it here turns a missing entry into a static failure instead of a
    # sampler-construction crash on the GPU box.
    try:
        NeighborFanout()
    except Exception as exc:  # noqa: BLE001 -- any construction failure counts
        problems.append(f"NeighborFanout() no longer constructs: {exc}")

    # EVERY metadata triple must resolve to an exported relation. load_edges
    # keyed on triple[1] once, which is the exported name for forward and
    # mirrored triples but NOT for a directed reverse -- so all five reverses
    # were silently skipped, 11 of 16 triples reached the graph, and both
    # backends refused to build one. Resolution is by triple now; this asserts
    # it stays that way.
    from tfgnn.stage2.schema_spec import spec_for_triple

    _, triples = pyg_metadata()
    unresolved: list[str] = []
    for triple in triples:
        try:
            item, _ = spec_for_triple(triple)
        except Exception as exc:  # noqa: BLE001 -- any resolution failure counts
            unresolved.append(f"{triple}: {exc}")
            continue
        if item.name not in exported:
            unresolved.append(
                f"{triple} resolves to {item.name}, which ALL_RELATIONS does not export"
            )
    if unresolved:
        problems.append(
            "metadata triples that would be dropped by load_edges: "
            + "; ".join(unresolved)
        )

    # The message-orientation map must cover every triple and be an
    # involution, or the sampler drops/misfiles sampled edges.
    mapping = message_triple_of()
    uncovered = sorted(set(triples) - set(mapping))
    if uncovered:
        problems.append(f"message_triple_of misses triples: {uncovered}")
    broken = sorted(str(t) for t in mapping if mapping.get(mapping[t]) != t)
    if broken:
        problems.append(f"message_triple_of is not an involution at: {broken}")

    if problems:
        raise Failure("\n".join(problems))
    print(
        "  stage 2 wiring: specs, export branches, fanout caps and "
        "orientation map agree"
    )


#: Registered queries that NOTHING calls, on purpose, each with the reason.
#: A query absent from both the pipeline and this map is an ORPHAN: installed
#: on every run, compiled, and invoked by nothing.
#:
#: build_card_merchant_transaction_edges was exactly that until 2026-07-30 --
#: and it materialises Stage 2's link-prediction TARGET, so the whole GNN arm
#: had no positives to train on. The failure surfaced 40 minutes into a run,
#: as an export that returned zero edges.
_INTENTIONALLY_UNCALLED: dict[str, str] = {
    "wcc_card_hierarchical": (
        "manual alternative to wcc_card for investigating a collapsed "
        "projection; not part of any automated build"
    ),
    "wcc_merchant_hierarchical": "as wcc_card_hierarchical",
    "fastrp_card": (
        "deliberately excluded from Stage 2 node features: computed on "
        "Card_Card, which is measured degenerate (schema_spec docstring)"
    ),
    "fastrp_merchant": "as fastrp_card",
    "neighborhood_aggregates_card": (
        "alternative hand-rolled embedding path, superseded by the R-GCN"
    ),
    "neighborhood_aggregates_merchant": "as neighborhood_aggregates_card",
    "export_embedding_features": (
        "exports the fastrp/neighborhood block, which Stage 2 does not consume"
    ),
    "permutation_control": (
        "D7's ground-truth leak test -- a degree-stratified reshuffle that "
        "preserves marginals and destroys structure -- run BY HAND when a "
        "graph lift needs challenging, never as part of a build. Reachable "
        "only via --include-optional; no config flag selects it. Read its "
        "header before trusting a null from it: it takes donor accumulators "
        "off a bare VERTEX variable, which may not be legal here, and a "
        "control arm that silently does nothing returns a reassuring null"
    ),
}


def check_no_orphan_queries(catalogue: dict[str, Query]) -> None:
    """Every installed query is called by something, or listed as deliberate.

    Installing a query proves it COMPILES, not that anything runs it. This is
    the check that was missing when the Stage 2 target relation stayed empty.
    """
    from tfgnn.tigergraph import gsql_paths

    registered = {q for item in gsql_paths.all_specs() for q in item.queries}
    problems: list[str] = []
    for name in sorted(registered):
        # THE QUOTED FORM, not the bare name. Every real call site names its
        # query as a string literal -- QueryCall("x", ...),
        # run_installed_with_timeout("x", ...), the DIMENSION_TABLES lists --
        # whereas a bare-substring search also matches the name written in
        # PROSE. install_gsql's module docstring mentions
        # ``permutation_control`` while explaining --include-optional, and
        # that single sentence was enough to make a query nothing calls look
        # called: the check silently passed on the one query systemprompt's
        # open items list as never run. A comment is not a caller.
        hits = subprocess.run(
            [
                "grep",
                "-rln",
                "--include=*.py",
                "--include=*.sh",
                f'"{name}"',
                "src",
                "scripts",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.split()
        # gsql_paths only REGISTERS a query; the validator itself only names it.
        real = [
            h
            for h in hits
            if "gsql_paths" not in h and "validate_stage1_static" not in h
        ]
        if not real and name not in _INTENTIONALLY_UNCALLED:
            problems.append(
                f"{name} is installed but called by nothing. Either wire it up "
                "or add it to _INTENTIONALLY_UNCALLED with a reason."
            )
        if real and name in _INTENTIONALLY_UNCALLED:
            problems.append(
                f"{name} is listed as intentionally uncalled but is referenced "
                f"by {real}. Remove it from _INTENTIONALLY_UNCALLED."
            )

    if problems:
        raise Failure("\n".join(problems))
    print(
        f"  no orphan queries: {len(registered)} registered, "
        f"{len(_INTENTIONALLY_UNCALLED)} deliberately uncalled"
    )


def main() -> None:
    print("static Stage-1 validation")
    catalogue = check_registry()
    check_accumulators()
    check_attributes(schema_attributes())
    check_reserved_words()
    check_v1_traversal_targets(schema_vertex_types())
    check_pipeline(catalogue)
    check_fold_boundaries()
    check_tuning_provenance()
    check_resume_guards()
    check_identity_advice()
    check_match_surface()
    check_stage2_wiring()
    check_no_orphan_queries(catalogue)
    check_contract()
    print("all static checks passed")


if __name__ == "__main__":
    try:
        main()
    except Failure as failure:
        print(f"\nFAILED\n{failure}", file=sys.stderr)
        raise SystemExit(1) from failure
