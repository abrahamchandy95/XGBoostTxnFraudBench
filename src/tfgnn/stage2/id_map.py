from dataclasses import dataclass, field

import torch
from torch import Tensor

NodeType = str


class IdMapError(RuntimeError):
    pass


@dataclass(slots=True)
class BatchIdMap:
    """One batch's global<->local id mapping, per node type, on device.

    Not reusable across batches. Call ``reset()`` (or build a new instance)
    between batches; ``register`` on an already-registered type raises rather
    than silently extending, because extending would renumber ids that tensors
    elsewhere in the batch already refer to.
    """

    device: torch.device
    _local_to_global: dict[NodeType, Tensor] = field(default_factory=dict)

    def register(self, node_type: NodeType, global_ids: Tensor) -> Tensor:
        """Assign contiguous local ids to a node type's global ids.

        Returns the local ids for ``global_ids``, positionally aligned with the
        input, so an edge_index expressed in global ids can be rewritten by
        indexing with the returned tensor.

        Duplicates in the input are fine and expected -- an edge list mentions
        a hub node many times -- and collapse to one local id.
        """
        if node_type in self._local_to_global:
            raise IdMapError(
                f"{node_type!r} is already registered in this batch. Extending "
                "would renumber local ids that this batch's tensors already "
                "reference. Build a new map per batch."
            )
        if global_ids.dtype != torch.int64:
            raise IdMapError(
                f"{node_type!r}: global ids must be int64, got {global_ids.dtype}"
            )

        flat = global_ids.reshape(-1).to(self.device, non_blocking=True)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        self._local_to_global[node_type] = unique
        return inverse.reshape(global_ids.shape)

    def to_local(self, node_type: NodeType, global_ids: Tensor) -> Tensor:
        """Local ids for globals already registered in THIS batch.

        Uses searchsorted, which is valid only because ``register`` stored the
        SORTED uniques. Any global not present is an error, not a -1: a silent
        sentinel here would index the feature tensor at a wrong row and produce
        a plausible embedding for the wrong node.
        """
        table = self._table(node_type)
        flat = global_ids.reshape(-1).to(self.device, non_blocking=True)
        position = torch.searchsorted(table, flat)
        position = position.clamp(max=table.numel() - 1)
        if not bool(torch.all(table[position] == flat)):
            missing = int((table[position] != flat).sum())
            raise IdMapError(
                f"{node_type!r}: {missing} global ids are not in this batch's "
                "map. They were never registered, or a local id leaked in from "
                "another batch."
            )
        return position.reshape(global_ids.shape)

    def to_global(self, node_type: NodeType, local_ids: Tensor) -> Tensor:
        """Recover global ids. The only sanctioned way out of local space."""
        table = self._table(node_type)
        flat = local_ids.reshape(-1)
        if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= table.numel()):
            raise IdMapError(
                f"{node_type!r}: local id out of range [0, {table.numel()})"
            )
        return table[flat].reshape(local_ids.shape)

    def local_to_global(self, node_type: NodeType) -> Tensor:
        """The table itself, for gathering features by global id."""
        return self._table(node_type)

    def size(self, node_type: NodeType) -> int:
        return int(self._table(node_type).numel())

    def sizes(self) -> dict[NodeType, int]:
        return {k: int(v.numel()) for k, v in self._local_to_global.items()}

    def registered(self) -> tuple[NodeType, ...]:
        return tuple(self._local_to_global)

    def bytes_held(self) -> int:
        """Transient bytes this map holds. Log it to prove batches stay bounded."""
        return sum(t.numel() * t.element_size() for t in self._local_to_global.values())

    def reset(self) -> None:
        """Drop every table so the batch's id memory is released.

        Clearing the dict removes the last references; the caching allocator
        reuses the blocks for the next batch. That reuse is the desired
        behaviour and is why this does NOT call empty_cache(): returning the
        blocks to the driver every batch would make the next allocation slower
        for no benefit, since the next batch needs the same size again.
        """
        self._local_to_global.clear()

    def _table(self, node_type: NodeType) -> Tensor:
        table = self._local_to_global.get(node_type)
        if table is None:
            raise IdMapError(
                f"{node_type!r} is not registered in this batch; registered: "
                f"{list(self._local_to_global)}"
            )
        return table


def remap_edge_index(
    id_map: BatchIdMap,
    src_type: NodeType,
    dst_type: NodeType,
    src_global: Tensor,
    dst_global: Tensor,
) -> Tensor:
    """A global-id edge list as a PyG local ``edge_index`` [2, E].

    Both endpoint types must already be registered, because an edge_index is
    only meaningful once both node sets are numbered -- registering lazily here
    would mean the first relation processed decides the numbering and later ones
    silently disagree.
    """
    src_local = id_map.to_local(src_type, src_global)
    dst_local = id_map.to_local(dst_type, dst_global)
    return torch.stack([src_local, dst_local], dim=0)
