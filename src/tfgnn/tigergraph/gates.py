from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast


class GateFailure(RuntimeError):
    """A query reported that one of its own invariants does not hold."""


@dataclass(frozen=True)
class GateResult:
    query: str
    key: str
    detail: str
    passed: bool


_ZERO = "MUST_BE_ZERO"
_ONE = "MUST_BE_1"
_EQUAL = "MUST_EQUAL_"
_NO_KEY_1 = "MUST_HAVE_NO_KEY_1"
_BELOW_TOTAL = "MUST_BE_BELOW_TOTAL"
_ABORT = "ABORT_"


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _truthy(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes"}:
            return True
        if lowered in {"false", "f", "0", "no"}:
            return False
    return None


def flatten(result: Sequence[object]) -> dict[str, object]:
    """Merge every PRINT block in one query result into a single mapping.

    Row projections (``PRINT Rows[...] AS transaction_rows``) come back as a
    list and are skipped: a gate is always a scalar or a map.
    """
    merged: dict[str, object] = {}
    for item in result:
        if not isinstance(item, Mapping):
            continue
        for key, value in cast(Mapping[object, object], item).items():
            if isinstance(key, str) and not isinstance(value, list):
                merged[key] = value
    return merged


def evaluate(
    query: str,
    result: Sequence[object],
    context: Mapping[str, object] | None = None,
) -> list[GateResult]:
    """Check every gate this query named in its own output."""
    record = flatten(result)
    extra = dict(context or {})
    gates: list[GateResult] = []

    for key, value in record.items():
        if key.startswith(_ABORT):
            flag = _truthy(value)
            if flag is None:
                gates.append(GateResult(query, key, f"not a boolean: {value!r}", False))
            else:
                gates.append(GateResult(query, key, f"{key}={flag}", not flag))
            continue

        if _NO_KEY_1 in key:
            if isinstance(value, Mapping):
                keys = {str(k).strip() for k in cast(Mapping[object, object], value)}
                present = "1" in keys
                gates.append(
                    GateResult(
                        query,
                        key,
                        "component of size 1 present; impossible by "
                        "construction, so this is a bug in the query"
                        if present
                        else "no size-1 component",
                        not present,
                    )
                )
            else:
                gates.append(GateResult(query, key, f"not a map: {value!r}", False))
            continue

        if _BELOW_TOTAL in key:
            observed = _number(value)
            total = _number(extra.get("total_transactions"))
            if observed is None:
                gates.append(GateResult(query, key, f"not numeric: {value!r}", False))
            elif total is None:
                gates.append(
                    GateResult(
                        query,
                        key,
                        f"{key}={observed:,.0f} but total_transactions is "
                        "unknown, so the cutoff cannot be checked",
                        False,
                    )
                )
            else:
                gates.append(
                    GateResult(
                        query,
                        key,
                        f"{observed:,.0f} of {total:,.0f} transactions under "
                        "the cutoff"
                        + (
                            "; equality means the cutoff did not bind and "
                            "features were fitted on the evaluation set"
                            if observed >= total
                            else ""
                        ),
                        observed < total,
                    )
                )
            continue

        if _EQUAL in key:
            other = key.split(_EQUAL, 1)[1]
            observed = _number(value)
            expected = _number(record.get(other, extra.get(other)))
            if observed is None:
                gates.append(GateResult(query, key, f"not numeric: {value!r}", False))
            elif expected is None:
                gates.append(
                    GateResult(
                        query,
                        key,
                        f"cannot check: {other!r} is not in this result",
                        True,
                    )
                )
            else:
                gates.append(
                    GateResult(
                        query,
                        key,
                        f"{observed:,.0f} vs {other}={expected:,.0f}",
                        observed == expected,
                    )
                )
            continue

        if key.endswith(_ONE) or f"{_ONE}_" in key:
            observed = _number(value)
            if observed is None:
                gates.append(GateResult(query, key, f"not numeric: {value!r}", False))
            else:
                gates.append(
                    GateResult(query, key, f"{key}={observed:,.0f}", observed == 1)
                )
            continue

        if _ZERO in key:
            observed = _number(value)
            if observed is None:
                gates.append(GateResult(query, key, f"not numeric: {value!r}", False))
            else:
                gates.append(
                    GateResult(query, key, f"{key}={observed:,.0f}", observed == 0)
                )
            continue

    return gates


def enforce(
    query: str,
    result: Sequence[object],
    context: Mapping[str, object] | None = None,
    *,
    strict: bool = True,
) -> list[GateResult]:
    """Evaluate and, when strict, raise on the first failing gate."""
    gates = evaluate(query, result, context)
    failures = [gate for gate in gates if not gate.passed]

    for gate in gates:
        mark = "ok  " if gate.passed else "FAIL"
        print(f"    gate {mark} {gate.key}: {gate.detail}")

    if failures and strict:
        detail = "\n".join(f"  {g.key}: {g.detail}" for g in failures)
        raise GateFailure(
            f"{query} failed {len(failures)} of its own invariants:\n{detail}\n"
            "These are the query authors' own stop conditions. Fix the cause "
            "rather than continuing: violations of fitting and traversal "
            "exclusion are silent and they improve the reported metric."
        )
    return gates
