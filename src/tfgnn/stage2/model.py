import torch
from torch import Tensor, nn
from torch_geometric.nn import HeteroConv, SAGEConv

from tfgnn.stage2.schema_spec import EdgeTriple, pyg_metadata


class ModelError(RuntimeError):
    pass


class TypeLevelEmbedding(nn.Module):
    """One learnable vector per node type, broadcast to every node of it.

    D3d-safe: no per-node row, so nothing to index and nothing to memorise. Used
    for node types with no usable features (Party, Merchant_Category).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(1, dim))
        nn.init.normal_(self.vector, std=0.02)

    def forward(self, num_nodes: int) -> Tensor:
        return self.vector.expand(num_nodes, -1)


def encode_with_missingness(x: Tensor) -> Tensor:
    """[N, F] possibly-NaN features -> [N, 2F]: filled values, then indicators.

    Kept a free function so the caller can compute the expected input width
    without instantiating the model.
    """
    missing = torch.isnan(x)
    filled = torch.where(missing, torch.zeros_like(x), x)
    return torch.cat([filled, missing.to(x.dtype)], dim=-1)


class RGCN(nn.Module):
    """Two-layer relational GCN producing an embedding per node type.

    Two layers, matching the sampler's two hops: a third layer would consume
    neighbors the sampler never fetched, so the depths must agree. The sampler
    caps fan-out at 2 hops for a structural reason (one merchant reaches 49% of
    cards), which makes this a property of the dataset rather than a
    hyperparameter.
    """

    def __init__(
        self,
        raw_dims: dict[str, int],
        hidden: int = 128,
        out: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_types, triples = pyg_metadata()

        unknown = sorted(set(raw_dims) - set(node_types))
        if unknown:
            raise ModelError(
                f"raw_dims names node types not in the metadata: {unknown}"
            )

        self.featureless = nn.ModuleDict()
        in_dims: dict[str, int] = {}
        for node_type in node_types:
            width = raw_dims.get(node_type, 0)
            if width == 0:
                self.featureless[node_type] = TypeLevelEmbedding(hidden)
                in_dims[node_type] = hidden
            else:
                in_dims[node_type] = 2 * width

        self.project = nn.ModuleDict(
            {t: nn.Linear(in_dims[t], hidden) for t in node_types}
        )
        self.conv1 = self._make_conv(triples, hidden, hidden)
        self.conv2 = self._make_conv(triples, hidden, out)
        self.norm = nn.ModuleDict({t: nn.LayerNorm(hidden) for t in node_types})
        self.dropout = nn.Dropout(dropout)
        self._raw_dims = dict(raw_dims)
        self._node_types = tuple(node_types)
        self._out = out
        self._hidden = hidden

    @staticmethod
    def _make_conv(triples: list[EdgeTriple], in_dim: int, out_dim: int) -> HeteroConv:
        """One message function per relation, summed at the destination.

        ``aggr="sum"`` over relations is what makes this R-GCN rather than a
        generic hetero GNN: each relation gets its own weight matrix and the
        destination sums them. SAGEConv supplies the per-relation
        mean-aggregate-then-linear; the relational part is the per-triple
        instance plus the sum.
        """
        return HeteroConv(
            {triple: SAGEConv((in_dim, in_dim), out_dim) for triple in triples},
            aggr="sum",
        )

    @property
    def out_dim(self) -> int:
        return self._out

    @property
    def hidden_dim(self) -> int:
        """Exposed so the training loop can PRICE a batch's activations.

        ``--max-batch-bytes`` used to measure only the sampler's tensors,
        which are a small fraction of a step's real footprint; the width of
        the intermediate representations is what makes the rest.
        """
        return self._hidden

    def _initial_x(
        self,
        features: dict[str, Tensor],
        num_nodes: dict[str, int],
    ) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for node_type in self._node_types:
            count = num_nodes.get(node_type, 0)
            expected = self._raw_dims.get(node_type, 0)

            if expected == 0:
                out[node_type] = self.featureless[node_type](count)
                continue

            x = features.get(node_type)
            if x is None:
                raise ModelError(
                    f"{node_type}: raw_dims declares {expected} features but "
                    "none were supplied. A silently zero-filled node type would "
                    "train and score without ever using its features."
                )
            if x.shape[0] != count:
                raise ModelError(
                    f"{node_type}: {x.shape[0]} feature rows for {count} nodes"
                )
            if x.shape[1] != expected:
                raise ModelError(
                    f"{node_type}: expected {expected} features, got {x.shape[1]}"
                )
            out[node_type] = encode_with_missingness(x)
        return out

    def forward(
        self,
        features: dict[str, Tensor],
        edge_index: dict[EdgeTriple, Tensor],
        num_nodes: dict[str, int],
    ) -> dict[str, Tensor]:
        x = self._initial_x(features, num_nodes)
        h = {t: self.dropout(torch.relu(self.project[t](v))) for t, v in x.items()}

        h1 = self.conv1(h, edge_index)
        h = {
            t: self.norm[t](self.dropout(torch.relu(h1[t]))) if t in h1 else h[t]
            for t in h
        }
        h2 = self.conv2(h, edge_index)
        return {
            t: h2.get(
                t,
                torch.zeros(h[t].shape[0], self._out, device=h[t].device),
            )
            for t in h
        }
