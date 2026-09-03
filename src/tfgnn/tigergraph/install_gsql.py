import argparse
import sys

from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.gsql_paths import (
    CORE,
    OPTIONAL,
    STAGE2,
    Group,
    gsql_path,
    script_names,
    scripts_for_queries,
    spec,
)
from tfgnn.tigergraph.settings import Settings


_FAILURE_MARKERS: tuple[str, ...] = (
    "semantic check fails",
    "semantic check error",
    "saved as draft query",
    "type/semantic error",
    "syntax error",
    "parsing error",
    "failed to create",
    "failed to install",
    "install failed",
    "does not exist",
    "is not defined",
    "cannot resolve",
    "unknown attribute",
    "not a valid",
    "error occurred",
    "exception",
)

_SUCCESS_MARKERS: tuple[str, ...] = (
    "successfully created",
    "has been added",
    "query installation finished",
    "successfully installed",
    "install query successfully",
)


class GsqlInstallError(RuntimeError):
    """The GSQL shell rejected a statement."""


def _check(label: str, response: object) -> str:
    text = response if isinstance(response, str) else repr(response)
    lowered = text.lower()

    hits = [marker for marker in _FAILURE_MARKERS if marker in lowered]
    if hits:
        raise GsqlInstallError(f"{label} failed (matched {hits}):\n{text.strip()}")
    if not any(marker in lowered for marker in _SUCCESS_MARKERS):
        print(
            f"  ! {label}: no success marker in the response; "
            "check the text above against your TigerGraph version",
            file=sys.stderr,
        )
    return text


def _groups(include_optional: bool, include_stage2: bool) -> tuple[Group, ...]:
    groups: list[Group] = [CORE]
    if include_optional:
        groups.append(OPTIONAL)
    if include_stage2:
        groups.append(STAGE2)
    return tuple(groups)


def plan_queries() -> list[str]:
    """Every query the pipeline will call for the config currently on disk.

    Read off ``pipeline.build_calls`` -- the same function the run uses -- so
    the install set cannot disagree with the call set. A flag-to-script table
    maintained beside ``build_calls`` would be a second source of truth, and the
    bug this exists to prevent is exactly the two of them drifting apart.

    The cutoff is a placeholder: which queries get called depends on the config
    flags, never on the rank.
    """
    # Imported here rather than at module scope: the installer is the first
    # thing a fresh checkout runs, and a config or pydantic problem should
    # surface as a plan error rather than as an import error on --list.
    from common.config import load_raw_config
    from tfgnn.features_meta import DIMENSION_TABLES, PAIR_TABLES
    from tfgnn.tigergraph.pipeline import PIPELINE_ADAPTER, build_calls
    from tfgnn.tigergraph.plan import Build

    config = load_raw_config()
    if "stage1_pipeline" not in config:
        raise KeyError("config.yaml is missing the stage1_pipeline section")
    plan = PIPELINE_ADAPTER.validate_python(config["stage1_pipeline"])
    probe = Build("b_probe", 1, "literal:1", (0,))

    names = [call.name for call in build_calls(plan, probe)]
    # The queries the other stages call by name. build_calls does not emit
    # these -- the exporter and the load gate do -- and every one of them is
    # CORE today, so they add nothing. Listed anyway so that moving one out of
    # CORE cannot quietly drop it from the install set.
    names.extend(
        [
            "assert_cardinality",
            "assert_edge_stamp",
            "derive_build_plan",
            "export_transaction_rows",
        ]
    )
    names.extend(DIMENSION_TABLES)
    names.extend(PAIR_TABLES)
    return names


def default_scripts(include_optional: bool, include_stage2: bool) -> list[str]:
    """CORE, plus whatever the configured plan needs, in registry order.

    Never a SUBSET of CORE: the config can only ADD to the default set. An
    explicit ``--include-optional`` or ``--include-stage2`` widens it further.
    """
    groups = _groups(include_optional, include_stage2)
    selected = set(script_names(groups))
    selected.update(scripts_for_queries(plan_queries()))
    return [item for item in script_names((CORE, OPTIONAL, STAGE2)) if item in selected]


def config_required_extras(include_optional: bool, include_stage2: bool) -> list[str]:
    """Scripts the config pulls in that the plain group selection would miss.

    Empty when every configured call is already in the selected groups, which is
    the state `validate_stage1_static` asserts for the default command.
    """
    groups = _groups(include_optional, include_stage2)
    inside = set(script_names(groups))
    return [
        name
        for name in default_scripts(include_optional, include_stage2)
        if name not in inside
    ]


def install(
    client: Client,
    scripts: list[str],
    *,
    with_schema: bool,
    force: bool,
    create_only: bool,
) -> None:
    if with_schema:
        path = gsql_path("schema")
        print(f"[schema] {path}")
        print(
            "  ! CREATE GRAPH plus a schema-change job. Only correct on a "
            "fresh instance."
        )
        _ = _check("schema", client.gsql(path.read_text(encoding="utf-8")))

    created: list[str] = []
    for name in scripts:
        path = gsql_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"missing GSQL file: {path}")
        print(f"[create] {name:34s} {path.relative_to(path.parents[2])}")
        _ = _check(f"create {name}", client.gsql(path.read_text(encoding="utf-8")))
        created.extend(spec(name).queries)

    if create_only:
        print(f"created {len(created)} queries; INSTALL skipped (--create-only)")
        return

    if not created:
        print("nothing to install")
        return

    # One batched INSTALL. -force reinstalls queries whose text did not
    # change, which is what you want after a schema change: an unchanged
    # query compiled against the old catalogue is still stale.
    flag = "-force " if force else ""
    statement = f"USE GRAPH {client.graphname}\nINSTALL QUERY {flag}" + ", ".join(
        created
    )
    print(f"[install] {len(created)} queries (this compiles; expect minutes)")
    _ = _check("install", client.gsql(statement))
    print(f"installed {len(created)} queries on {client.graphname}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and install the ordered TF_GNN GSQL scripts"
    )
    _ = parser.add_argument(
        "scripts",
        nargs="*",
        help="script names; defaults to the whole selected group set",
    )
    _ = parser.add_argument("--list", action="store_true", dest="list_scripts")
    _ = parser.add_argument(
        "--with-schema",
        action="store_true",
        help="run gsql/schema/schema.gsql first. Fresh instances only.",
    )
    _ = parser.add_argument("--include-optional", action="store_true")
    _ = parser.add_argument("--include-stage2", action="store_true")
    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="INSTALL QUERY -force, reinstalling unchanged queries too",
    )
    _ = parser.add_argument(
        "--create-only",
        action="store_true",
        help="CREATE without INSTALL. The queries will not be callable.",
    )
    args = parser.parse_args()

    groups = _groups(bool(args.include_optional), bool(args.include_stage2))
    available = default_scripts(bool(args.include_optional), bool(args.include_stage2))
    extras = config_required_extras(
        bool(args.include_optional), bool(args.include_stage2)
    )
    if extras:
        print(
            f"config.yaml enables optional families, so {len(extras)} OPTIONAL "
            f"script(s) join the install set:\n  {', '.join(extras)}\n"
            "  The pipeline WILL call these. Installing the CORE group alone "
            "would fail on a missing\n  endpoint partway into the first build, "
            "after reset_build had already run.\n"
        )

    if args.list_scripts:
        core_only = set(script_names(groups))
        for name in available:
            path = gsql_path(name)
            queries = ", ".join(spec(name).queries) or "(DDL)"
            why = "" if name in core_only else "   <-- required by config.yaml"
            print(
                f"{name:34s} {path.relative_to(path.parents[2])!s:52s} {queries}{why}"
            )
        installed_queries = [
            query for name in available for query in spec(name).queries
        ]
        print(f"\n{len(available)} scripts, {len(installed_queries)} queries")
        if not args.include_optional:
            skipped = sorted(set(script_names((OPTIONAL,))) - set(available))
            if skipped:
                print(
                    f"not installed, and the configured plan calls none of them: "
                    f"{', '.join(skipped)}\n"
                    "  Pass --include-optional if you want them anyway."
                )
        return

    requested: list[str] = list(args.scripts) if args.scripts else available
    unknown = [name for name in requested if name not in {s for s in available}]
    if unknown:
        raise SystemExit(
            f"unknown or out-of-group scripts: {unknown}. "
            f"Available: {', '.join(available)}"
        )

    settings = Settings()
    if settings.graphname != "TF_GNN":
        raise ValueError(
            "these canonical GSQL files target TF_GNN, but "
            f"GRAPHNAME={settings.graphname!r}"
        )

    install(
        Client(settings),
        requested,
        with_schema=bool(args.with_schema),
        force=bool(args.force),
        create_only=bool(args.create_only),
    )


if __name__ == "__main__":
    main()
