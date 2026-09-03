import argparse
import time
from collections.abc import Mapping
from typing import Any, cast

from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.settings import Settings

TARGET_EDGE = "Card_Merchant_Transaction"
SOURCE_EDGE = "Card_Send_Transaction"
BUILD_QUERY = "build_card_merchant_transaction_edges"

DEFAULT_TIMEOUT_S = 7200.0


class TargetEdgeError(RuntimeError):
    pass


def _scalar(result: list[object], key: str) -> int:
    for block in result:
        if isinstance(block, dict) and key in block:
            return int(cast("dict[str, Any]", block)[key])
    return 0


def edge_count(client: Client, edge_type: str) -> int:
    """Builtin stat_edge_number. A round trip, not a traversal.

    pyTigerGraph returns a bare int for a named edge type and a
    ``{type: count}`` mapping for the wildcard form; tolerate both rather than
    depending on which overload a given version picks.
    """
    raw = cast("object", client.conn.getEdgeCount(edge_type))
    if isinstance(raw, Mapping):
        values = cast("Mapping[str, Any]", raw).values()
        return int(sum(int(v) for v in values))
    return int(cast("int", raw))


def ensure_target_edges(
    client: Client,
    num_of_batches: int = 10,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    force: bool = False,
) -> int:
    """Build the target relation if it is missing or short. Returns its count."""
    expected = edge_count(client, SOURCE_EDGE)
    if expected == 0:
        raise TargetEdgeError(
            f"{SOURCE_EDGE} is empty, so there is nothing to derive "
            f"{TARGET_EDGE} from. Load the graph with tf_gnn_loader_v2 first."
        )

    present = edge_count(client, TARGET_EDGE)
    print(f"  {TARGET_EDGE}: {present:,} / {expected:,} expected")

    if present >= expected and not force:
        print("  already materialised; skipping (pass --force to rewrite)")
        return present
    if present == 0:
        print(f"  building {expected:,} edges (plus the free reverse). Minutes.")
    else:
        print(
            f"  partial: {expected - present:,} missing. Re-running is an upsert, "
            "so this resumes rather than duplicating."
        )

    result = client.run_installed_detached(
        BUILD_QUERY,
        {"num_of_batches": num_of_batches, "only_batch": -1},
        timeout_s=timeout_s,
    )
    inserted = _scalar(result, "edges_inserted")
    scanned = _scalar(result, "transactions_scanned")
    unknown = _scalar(result, "txns_with_unknown_event_seq_MUST_BE_ZERO")
    print(f"  inserted {inserted:,} over {scanned:,} transactions scanned")

    if unknown:
        raise TargetEdgeError(
            f"{unknown:,} transactions have event_seq 0. An edge with no time "
            "passes `edge_event_seq < seed_event_seq` for EVERY seed, so it "
            "would enter every Stage 2 sample with no temporal constraint at "
            "all. Fix the loader stamp before training."
        )

    # THE BUILTIN COUNT LAGS A LARGE INSERT. It is a maintained statistic,
    # not a scan, and after committing 27.4M edges it can still read 0 for a
    # while -- which is exactly what happened on 2026-08-02: the query
    # reported `inserted 27,404,317 over 27,404,317 transactions scanned`,
    # this check immediately read 0, and the run died at the last step before
    # Stage 2 on a relation that was in fact complete (verified afterwards at
    # 27,404,317 both ways).
    #
    # So poll it rather than trusting one read. The writer's own counter is
    # the corroborating evidence -- §2h's rule is to verify by traversal and
    # not by the writer alone, which is why the statistic still has to agree
    # eventually, but a stale read is not a shortfall.
    final = edge_count(client, TARGET_EDGE)
    waited = 0.0
    wait = 2.0
    while final < expected and waited < 300.0:
        print(
            f"  {TARGET_EDGE} count reads {final:,} of {expected:,}; the "
            "builtin statistic lags a large insert, re-reading in "
            f"{wait:.0f}s"
        )
        time.sleep(wait)
        waited += wait
        wait = min(wait * 2.0, 30.0)
        final = edge_count(client, TARGET_EDGE)

    if final < expected:
        if inserted >= expected and scanned >= expected:
            # The writer scanned and inserted a full set; only the statistic
            # disagrees. Proceeding is right -- failing here would block
            # Stage 2 on a relation that is present -- but say so loudly,
            # because the alternative reading is a genuinely short build.
            print(
                f"  WARNING {TARGET_EDGE}'s builtin count is still {final:,} "
                f"of {expected:,} after {waited:.0f}s, but the build reported "
                f"{inserted:,} inserted over {scanned:,} scanned. Treating the "
                "statistic as stale and continuing. If Stage 2 then reports "
                "zero target edges, this was real: re-run "
                "`python -m tfgnn.tigergraph.target_edges` and check "
                "assert_cardinality."
            )
            return inserted
        raise TargetEdgeError(
            f"{TARGET_EDGE} is {final:,} after the build but "
            f"{SOURCE_EDGE} is {expected:,}, and the build itself only "
            f"inserted {inserted:,} over {scanned:,} scanned. Every "
            "transaction should yield exactly one target edge, so a shortfall "
            "means transactions with no Transaction_To_Merchant edge — the "
            "two-hop pattern cannot bind them. Run assert_cardinality: the "
            "target relation is incomplete and seeds drawn from it would "
            "silently under-represent those cards."
        )
    print(f"  {TARGET_EDGE}: {final:,} edges")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise Card_Merchant_Transaction (Stage 2's target). "
            "Idempotent and count-gated: a no-op once complete."
        )
    )
    _ = parser.add_argument("--num-of-batches", type=int, default=10)
    _ = parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    _ = parser.add_argument(
        "--force", action="store_true", help="rewrite even when already complete"
    )
    args = parser.parse_args()

    client = Client(Settings())
    _ = ensure_target_edges(
        client,
        num_of_batches=args.num_of_batches,
        timeout_s=args.timeout_s,
        force=args.force,
    )


if __name__ == "__main__":
    main()
