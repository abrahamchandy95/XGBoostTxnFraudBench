from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
import numpy.typing as npt


type Part = Literal["train", "val", "test"]

# Three arms, not two. schema.gsql is explicit that the entity aggregate
# block "must be shared BYTE FOR BYTE with the no-graph Stage 1 arm" and that
# only centrality, community and identity structure "can be claimed as graph
# lift". Reporting raw -> raw_plus_graph as one number folds a group-by into
# the graph's contribution.
# FOUR arms now. The fourth is Stage 2: the same rows and the same Stage 1
# columns, PLUS the R-GCN embeddings. Its comparison base is raw_plus_graph, so
# `raw_plus_graph -> raw_plus_embeddings` is the number that may be called the
# GNN's contribution -- NOT raw -> raw_plus_embeddings, which would fold the
# whole graph feature block into the GNN's credit.
# raw_plus_temporal (Stage 3) and raw_plus_embeddings (Stage 2) are SIBLINGS,
# both measured against raw_plus_graph. The chain stops being linear here and
# becomes a tree, which is what the four-stage 2x2 actually is: {no-GNN, GNN}
# x {no-temporal, temporal}. The lift list below is explicit pairs rather than
# a chain walk, so a tree needs no restructuring.
type Variant = Literal[
    "raw",
    "raw_plus_aggregates",
    "raw_plus_graph",
    "raw_plus_temporal",
    "raw_plus_embeddings",
    "raw_plus_temporal_embeddings",
    "raw_plus_aggregates_embeddings",
    "raw_plus_pair_scores",
    "raw_plus_temporal_pair_scores",
]

type FloatArray = npt.NDArray[np.float64]
type IndexArray = npt.NDArray[np.intp]

T = TypeVar("T")

PARTS: tuple[Part, Part, Part] = ("train", "val", "test")
VARIANTS: tuple[Variant, ...] = (
    "raw",
    "raw_plus_aggregates",
    "raw_plus_graph",
    "raw_plus_temporal",
    "raw_plus_embeddings",
    "raw_plus_temporal_embeddings",
    "raw_plus_aggregates_embeddings",
    "raw_plus_pair_scores",
    "raw_plus_temporal_pair_scores",
)


@dataclass(frozen=True)
class Partitions(Generic[T]):
    train: T
    val: T
    test: T

    def __getitem__(self, part: Part) -> T:
        if part == "train":
            return self.train
        if part == "val":
            return self.val
        return self.test
