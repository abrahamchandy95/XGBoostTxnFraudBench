"""Optional cuGraph-backed neighbour sampling for Stage 4.

======================================================================
OPTIONAL AND FAIL-SOFT, DELIBERATELY
======================================================================
cugraph / cugraph-pyg are CUDA-only and are commented out of
requirements.txt (line 42) so the file stays installable on a Mac. This
module therefore probes for them and reports what it found rather than
importing at module scope -- an unconditional import would make every
offline check on a non-CUDA machine fail, including the static validator
and the smoke test.

`available()` is the switch. `describe()` is what run.py prints, so a run
log SAYS whether cuGraph was used instead of leaving it to be assumed --
which is how this project came to believe cuGraph was in use when it was
PyG throughout.

======================================================================
WHAT IT ACCELERATES, AND WHAT IT DOES NOT
======================================================================
It replaces the last-N neighbour LOOKUP, not the memory. TGN's memory is
already an O(1) gather of two rows per event and there is nothing for a
graph library to do there. The neighbourhood is what cuGraph is for.

It also does NOT change accuracy. Same last-N semantics, same edges, same
attention. If the number moves, something is wrong -- that is the check
worth running, not a hoped-for improvement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    cugraph: bool
    cugraph_pyg: bool
    cudf: bool

    @property
    def usable(self) -> bool:
        return self.cugraph and self.cugraph_pyg


def probe() -> Probe:
    import importlib.util as util

    return Probe(
        cugraph=util.find_spec("cugraph") is not None,
        cugraph_pyg=util.find_spec("cugraph_pyg") is not None,
        cudf=util.find_spec("cudf") is not None,
    )


def available() -> bool:
    return probe().usable


def describe() -> str:
    found = probe()
    if found.usable:
        return "cuGraph: cugraph + cugraph_pyg present, GPU sampling ENABLED"
    missing = [
        name
        for name, present in (
            ("cugraph", found.cugraph),
            ("cugraph_pyg", found.cugraph_pyg),
            ("cudf", found.cudf),
        )
        if not present
    ]
    return (
        f"cuGraph: NOT used (missing {', '.join(missing)}). Falling back to "
        "PyG LastNeighborLoader. Install with: pip install cugraph-cu12 "
        "cugraph-pyg-cu12 cudf-cu12"
    )


def build_store(edge_index: object, num_nodes: int) -> object:
    """cugraph_pyg GraphStore over one window's edges.

    Imported INSIDE the function: at module scope this would break every
    offline check on a machine without CUDA.
    """
    if not available():
        raise RuntimeError(describe())
    from cugraph_pyg.data import GraphStore  # type: ignore[import-not-found]

    store = GraphStore()
    store[("node", "to", "node"), "coo", False, (num_nodes, num_nodes)] = edge_index
    return store
