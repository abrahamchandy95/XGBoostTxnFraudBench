import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from tfgnn.stage2.schema_spec import (
    EdgeTriple,
    pyg_metadata,
    spec,
    spec_for_triple,
)
from tfgnn.tigergraph.client import Client
from tfgnn.tigergraph.settings import Settings


class LoadError(RuntimeError):
    pass


_FORWARD: tuple[str, ...] = (
    "Card_Merchant_Transaction",
    "Card_Send_Transaction",
    "Transaction_To_Merchant",
    "Has_Interaction_With_Merchant",
    "Party_Has_Card",
    "Party_Is_Merchant",
    "Merchant_Assigned",
    "Party_Shares_Device",
    "Party_Shares_IP",
)

PAIR_ONLY: tuple[str, ...] = (
    "Has_Interaction_With_Merchant",
    "Party_Has_Card",
    "Party_Is_Merchant",
    "Merchant_Assigned",
    "Party_Shares_Device",
    "Party_Shares_IP",
)

ALL_RELATIONS: tuple[str, ...] = _FORWARD

_PAGE = 2_000_000


def _vid_range(
    client: Client,
    node_type: str,
    timeout_s: float,
    cache: dict[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    """(min_vid, max_vid, vertex_count) for a page-source node type.

    getvid is SEGMENT-based and does not start at zero: Party begins at
    69,206,016 on the measured instance. The first version of this pager
    swept up from 0, stopped on the first empty page beyond the first, and
    hard-capped at 60M -- BELOW Party's minimum vid -- so every
    Party-sourced relation exported zero rows, silently, and the failure
    surfaced as a misleading "relation is empty" error. Asking the graph
    for the range is what makes the sweep exact instead of heuristic.
    """
    if node_type not in cache:
        result = client.run_installed_with_timeout(
            "export_vertex_id_range",
            {"node_type": node_type},
            timeout_s=timeout_s,
        )
        cache[node_type] = (
            _scalar(result, "min_vid"),
            _scalar(result, "max_vid"),
            _scalar(result, "vertex_count"),
        )
    return cache[node_type]


def _rows(result: list[object], key: str) -> list[dict[str, Any]]:
    for block in result:
        if isinstance(block, dict) and key in block:
            payload = cast(dict[str, Any], block)[key]
            if isinstance(payload, list):
                return cast(list[dict[str, Any]], payload)
    return []


def _scalar(result: list[object], key: str) -> int:
    for block in result:
        if isinstance(block, dict) and key in block:
            return int(cast(dict[str, Any], block)[key])
    return 0


def export_relation(
    client: Client,
    relation: str,
    build_id: str,
    directory: Path,
    timeout_s: float,
    overwrite: bool = False,
    vid_cache: dict[str, tuple[int, int, int]] | None = None,
) -> Path:
    """Page one relation to parquet. Skips the transfer if the file exists."""
    import pandas as pd

    path = directory / f"{relation}.parquet"
    if path.is_file() and not overwrite:
        print(f"    {relation:<32s} cached -> {path.name}")
        return path

    # Page over the SOURCE type's real vid range -- see _vid_range for why
    # sweeping up from zero silently exported nothing for Party.
    source_type = spec(relation).src
    lo, hi, count = _vid_range(
        client, source_type, timeout_s, vid_cache if vid_cache is not None else {}
    )
    if count == 0:
        raise LoadError(
            f"{relation}: source type {source_type} has no vertices. Load the "
            "graph before exporting."
        )

    frames: list[Any] = []
    page = lo
    total = 0
    while page <= hi:
        result = client.run_installed_with_timeout(
            "export_relation_edges",
            {
                "relation": relation,
                "build_id": build_id,
                "page_start": page,
                "page_size": _PAGE,
            },
            timeout_s=timeout_s,
        )
        skipped = _scalar(result, "skipped_unknown_event_seq_MUST_BE_ZERO")
        if skipped:
            raise LoadError(
                f"{relation}: {skipped} edges have event_seq 0. An edge with no "
                "time is admitted by EVERY cutoff, so it would leak into every "
                "fold. Fix the loader stamp before continuing."
            )
        rows = _rows(result, "edges")
        if rows:
            frames.append(pd.DataFrame(rows))
            total += len(rows)
        # Empty pages INSIDE [lo, hi] are normal -- segments are sparse -- so
        # the only stop condition is passing the type's real maximum vid.
        page += _PAGE

    if not frames:
        raise LoadError(
            f"{relation}: exported zero edges. Either the relation is empty (run "
            "the pipeline first) or build_id does not match the resident build."
        )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(path, index=False)
    print(f"    {relation:<32s} {total:>12,} edges -> {path.name}")
    return path


def export_vertex_maps(
    client: Client, directory: Path, timeout_s: float, overwrite: bool = False
) -> dict[str, Path]:
    """(getvid, primary_id) for Card and Merchant. Needed for the embedding join."""
    import pandas as pd

    out: dict[str, Path] = {}
    for node_type in ("Card", "Merchant"):
        path = directory / f"vertexmap_{node_type}.parquet"
        if not path.is_file() or overwrite:
            result = client.run_installed_with_timeout(
                "export_vertex_key_map",
                {"node_type": node_type},
                timeout_s=timeout_s,
            )
            rows = _rows(result, "vertices")
            if not rows:
                raise LoadError(f"vertex map for {node_type} came back empty")
            pd.DataFrame(rows).to_parquet(path, index=False)
            print(f"    vertexmap {node_type:<22s} {len(rows):>12,} rows")
        out[node_type] = path
    return out


def export_graph(
    directory: Path,
    build_id: str,
    cutoff_event_seq: int = 0,
    relations: tuple[str, ...] = _FORWARD,
    overwrite: bool = False,
) -> Path:
    """Export every needed relation plus the vertex maps."""
    directory.mkdir(parents=True, exist_ok=True)
    # A cached parquet is only reusable if it was exported for the SAME build
    # OF THE SAME LOAD: export_relation skips any existing file without
    # reading it, and the build NAME is a fixed config string ('b_test' on
    # every standard run), so a name-only check passes straight across a data
    # regeneration and trains on the previous load's edges. The cutoff is
    # data-derived -- it moves whenever the load does -- so name + cutoff
    # together are the cache identity. Refuse rather than silently train.
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        cached = cast(dict[str, Any], json.loads(manifest_path.read_text()))
        cached_build = str(cached.get("build_id", ""))
        cached_cutoff = int(cached.get("cutoff_event_seq", 0))
        if cached_build != build_id or cached_cutoff != cutoff_event_seq:
            raise LoadError(
                f"graph dir {directory} holds an export for build "
                f"'{cached_build}' at cutoff {cached_cutoff:,}, but this run "
                f"wants '{build_id}' at cutoff {cutoff_event_seq:,}. A "
                "matching name with a different cutoff means the DATA was "
                "regenerated since the cache was written. Re-run with "
                "--overwrite-export or point --graph-dir at a fresh "
                "directory; reusing the cache would train on a different "
                "load's edges."
            )
    settings = Settings()
    client = Client(settings)
    print(f"  exporting Stage 2 graph to {directory}")
    vid_cache: dict[str, tuple[int, int, int]] = {}
    for relation in relations:
        export_relation(
            client,
            relation,
            build_id,
            directory,
            timeout_s=settings.query_timeout_s,
            overwrite=overwrite,
            vid_cache=vid_cache,
        )
    export_vertex_maps(
        client, directory, timeout_s=settings.query_timeout_s, overwrite=overwrite
    )
    manifest = {
        "build_id": build_id,
        "cutoff_event_seq": cutoff_event_seq,
        "relations": list(relations),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def load_edges(
    directory: Path, device: torch.device
) -> tuple[
    dict[EdgeTriple, tuple[Tensor, Tensor, Tensor]],
    dict[str, int],
    dict[str, Tensor],
]:
    """Parquet -> per-triple tensors, with getvid DENSIFIED per node type.

    Returns (edges, node_counts, dense_ids). ``dense_ids[type][i]`` is the
    getvid of dense node i -- returned rather than kept private because it is
    the ONLY thing that can align the vertex map and the Stage-1 feature
    tables to this graph's node ids; deriving it a second time from the
    vertex map is what silently mislabelled them.

    Densification is the step that lets the export use getvid at all: getvid is
    dense per segment but not globally, so the ids are remapped to [0, N) per
    type here, with one torch.unique -- no host-side dict, and the 27M
    transaction ids never become Python objects.

    Reverses are generated by swapping src/dst rather than exported, so the two
    directions cannot disagree.
    """
    import pandas as pd

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise LoadError(
            f"no manifest at {manifest_path}. Run the exporter first "
            "(scripts/run_all.sh does this)."
        )
    present = set(
        cast(dict[str, Any], json.loads(manifest_path.read_text()))["relations"]
    )

    raw: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    observed: dict[str, list[Tensor]] = {}
    for relation in sorted(present):
        frame = pd.read_parquet(directory / f"{relation}.parquet")
        item = spec(relation)
        src = torch.as_tensor(frame["src"].to_numpy(), dtype=torch.int64)
        dst = torch.as_tensor(frame["dst"].to_numpy(), dtype=torch.int64)
        time = torch.as_tensor(frame["time"].to_numpy(), dtype=torch.int64)
        raw[relation] = (src, dst, time)
        observed.setdefault(item.src, []).append(src)
        observed.setdefault(item.dst, []).append(dst)

    # One dense id space per node type, from every getvid that appears in it.
    tables: dict[str, Tensor] = {
        node_type: torch.unique(torch.cat(chunks), sorted=True)
        for node_type, chunks in observed.items()
    }
    node_counts = {t: int(v.numel()) for t, v in tables.items()}

    def densify(node_type: str, ids: Tensor) -> Tensor:
        table = tables[node_type]
        return torch.searchsorted(table, ids).to(device)

    edges: dict[EdgeTriple, tuple[Tensor, Tensor, Tensor]] = {}
    _, expected = pyg_metadata()
    for triple in expected:
        # Resolve by TRIPLE, not by triple[1]. A directed reverse triple is
        # named after the reverse edge, which is never an export key, so keying
        # on the name skipped all five of them and left the graph unbuildable.
        # See schema_spec.spec_for_triple.
        item, forward = spec_for_triple(triple)
        if item.name not in raw:
            continue
        src, dst, time = raw[item.name]
        a, b = (src, dst) if forward else (dst, src)
        a_type, b_type = (item.src, item.dst) if forward else (item.dst, item.src)
        edges[triple] = (
            densify(a_type, a),
            densify(b_type, b),
            time.to(device),
        )

    missing = sorted({spec_for_triple(t)[0].name for t in expected} - present)
    if missing:
        print(
            f"  NOTE relations not exported and therefore absent from the graph: "
            f"{missing}"
        )
    return edges, node_counts, tables


def load_vertex_maps(
    directory: Path, dense_ids: Mapping[str, Tensor]
) -> dict[str, Any]:
    """Card/Merchant primary ids in DENSE id order, for the embedding join.

    *** THE ORDER IS DEFINED BY ``dense_ids``, NOT BY THE VERTEX MAP. ***
    The dense id space is built from the getvids that actually APPEAR IN
    EXPORTED EDGES, while the vertex map holds every vertex of the type. The
    two are equal only when no vertex is edgeless -- and they are routinely
    NOT equal, most obviously under ``--relations pair``, where only cards
    holding an interaction edge for the build enter the graph at all.

    Reading the vertex map positionally therefore assigned each embedding the
    key of whichever vertex sat at that ROW, shifting every key past the
    first absent vertex. Silent, and it mislabels the embeddings joined onto
    27M transaction rows -- the same class of failure as a mis-keyed join,
    with nothing to compare against afterwards.

    Selecting by vid keeps the two in step by construction: dense id i is
    ``dense_ids[type][i]``, so its key is the vertex map's row for that vid.
    """
    import pandas as pd

    out: dict[str, Any] = {}
    for node_type in ("Card", "Merchant"):
        path = directory / f"vertexmap_{node_type}.parquet"
        if not path.is_file():
            raise LoadError(f"missing vertex map {path}")
        frame = pd.read_parquet(path)
        wanted = dense_ids.get(node_type)
        if wanted is None:
            # No edge of this run mentions the type, so it has no dense space
            # and nothing will index into these keys.
            out[node_type] = pd.Series([], dtype="string")
            continue
        keyed = frame.set_index("vid")["key"].astype("string")
        ordered = keyed.reindex(wanted.cpu().numpy())
        unknown = int(ordered.isna().sum())
        if unknown:
            raise LoadError(
                f"{node_type}: {unknown} getvid(s) appear in exported edges "
                f"but not in {path.name}, so their embeddings cannot be "
                "keyed to a primary id. The vertex map and the edge export "
                "came from different loads -- re-export with "
                "--overwrite-export."
            )
        out[node_type] = ordered.reset_index(drop=True)
    return out


def load_node_features(
    stage1_export: Path,
    graph_dir: Path,
    build_id: str,
    node_counts: dict[str, int],
    device: "torch.device",
    dense_ids: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, int]]:
    """Stage 1's dimension features, aligned to DENSE Stage 2 node ids.

    Returns (features, raw_dims) ready for RGCN. Empty for a node type whose
    dimension table is absent, which the model handles with a type-level vector.

    *** THE SAME COLUMNS STAGE 1 USED, BY CONSTRUCTION. *** The lists come from
    features_meta rather than being retyped here. D5 wants the GNN's node inputs
    shared with Stage 1 so the comparison isolates the GNN; a divergence would
    make `raw_plus_graph -> raw_plus_embeddings` meaningless because the two arms
    would differ in two ways at once.

    ALIGNMENT IS THE WHOLE JOB. The dimension tables are keyed by card_number /
    merchant id; the graph is keyed by dense integers derived from getvid. The
    vertex map is what connects them, and a row order mismatch here would give
    every node ANOTHER node's features -- which trains and scores without any
    error, producing a quietly meaningless model. Hence the reindex against the
    map rather than a positional assumption.

    Missing entities become NaN, and the model's missingness indicator carries
    that through. NOT zero: an entity the build never saw has no history, and
    zero would assert its pagerank is zero.
    """
    import numpy as np
    import pandas as pd

    from tfgnn.features_meta import (
        CARD_DIM_COLUMNS,
        MERCHANT_DIM_COLUMNS,
    )

    dimensions = stage1_export / "dimensions" / build_id
    maps = load_vertex_maps(graph_dir, dense_ids)

    # Columns to use: the numeric feature columns of each dimension table, minus
    # join keys and the geography inputs assemble consumes and drops.
    # The four coordinate names are gone from the dimension exports -- the
    # distance is a pair-table column now -- so they are no longer listed here.
    # Stage 2 reads home_distance_km off Has_Interaction_With_Merchant directly,
    # which is where card_home_distance writes it.
    skip = {
        "card_number",
        "id",
        "build_id",
        "c_id",
        "louvain_id",
        "re_id",
    }
    spec_map = {
        "Card": ("export_card_features", CARD_DIM_COLUMNS, "card_number"),
        "Merchant": ("export_merchant_features", MERCHANT_DIM_COLUMNS, "id"),
    }

    features: dict[str, Tensor] = {}
    raw_dims: dict[str, int] = {}
    for node_type, (query, columns, key) in spec_map.items():
        path = dimensions / f"{query}.parquet"
        if not path.is_file():
            print(
                f"    NOTE no {query}.parquet for {build_id}; {node_type} "
                "falls back to a type-level vector"
            )
            continue
        frame = pd.read_parquet(path)
        use = [c for c in columns if c not in skip and c in frame.columns]
        frame[key] = frame[key].astype("string")
        frame = frame.set_index(key)

        # Reindex onto the DENSE id order the graph uses. Absent -> NaN.
        # maps[node_type] is now dense-ordered BY CONSTRUCTION (it is selected
        # by vid from the dense id table), so row i of this matrix is dense
        # node i. It used to be the vertex map read positionally, and the
        # pad/truncate below then silently handed the first `count` rows'
        # features to whichever nodes held those dense ids -- a mislabelling
        # whenever any vertex was absent from the exported edges, which is
        # the NORMAL case under --relations pair.
        ordered = frame.reindex(maps[node_type].to_numpy())[use]
        matrix = ordered.to_numpy(dtype="float32")
        # -1 is Stage 1's "not computed" sentinel; it must not reach a linear
        # layer as a number, for the same reason it must not reach a tree.
        matrix = np.where(matrix <= -1, np.nan, matrix)

        count = node_counts.get(node_type, matrix.shape[0])
        if matrix.shape[0] != count:
            raise LoadError(
                f"{node_type}: built {matrix.shape[0]} feature rows for "
                f"{count} dense nodes. The vertex map and the dense id space "
                "disagree, so every row past the first gap would be assigned "
                "to the wrong node. Re-export with --overwrite-export."
            )

        features[node_type] = torch.as_tensor(matrix, device=device)
        raw_dims[node_type] = int(matrix.shape[1])
        missing = float(np.isnan(matrix).mean())
        print(
            f"    {node_type:<10s} {matrix.shape[0]:>8,} x {matrix.shape[1]} "
            f"features ({missing:.1%} NaN)"
        )

    return features, raw_dims
