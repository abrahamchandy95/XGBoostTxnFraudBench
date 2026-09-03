from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


class LinkPredError(RuntimeError):
    pass


class NegativeRegime(Enum):
    RANDOM = "random"
    HISTORICAL = "historical"


def pair_key(card: Tensor, merchant: Tensor, num_merchants: int) -> Tensor:
    """Bijective int64 key for a (card, merchant) pair.

    Requires ``merchant < num_merchants``; otherwise two distinct pairs collide
    onto one key and the collision check silently starts passing.
    """
    if merchant.numel() and int(merchant.max()) >= num_merchants:
        raise LinkPredError(
            f"merchant id {int(merchant.max())} >= num_merchants "
            f"{num_merchants}; the pair key would not be injective"
        )
    return card.to(torch.int64) * num_merchants + merchant.to(torch.int64)


@dataclass(slots=True)
class PairIndex:
    """Sorted keys of every REAL pair, for O(log n) membership on device."""

    keys: Tensor
    num_merchants: int

    @classmethod
    def build(cls, card: Tensor, merchant: Tensor, num_merchants: int) -> "PairIndex":
        keys = torch.unique(pair_key(card, merchant, num_merchants), sorted=True)
        return cls(keys=keys, num_merchants=num_merchants)

    def contains(self, card: Tensor, merchant: Tensor) -> Tensor:
        probe = pair_key(card, merchant, self.num_merchants)
        if self.keys.numel() == 0:
            return torch.zeros_like(probe, dtype=torch.bool)
        position = torch.searchsorted(self.keys, probe).clamp(max=self.keys.numel() - 1)
        return self.keys[position] == probe


def sample_negatives(
    positive_card: Tensor,
    num_cards: int,
    num_merchants: int,
    real_pairs: PairIndex,
    regime: NegativeRegime = NegativeRegime.RANDOM,
    historical_pool: PairIndex | None = None,
    max_rounds: int = 12,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """One negative per positive. Returns (card, merchant), both [N].

    Cards are REUSED from the positives rather than resampled, so each negative
    is a corruption of a real positive's head. That keeps the card marginal
    identical between positives and negatives -- otherwise the model can
    separate the two classes on card degree alone and never look at the
    merchant, which is a trivial solution that looks like learning.

    Rejection is iterative and bounded. A round redraws only the still-colliding
    entries, so cost falls geometrically: at a 3.9% collision rate two rounds
    leave ~0.15%. ``max_rounds`` exists because a card that has transacted with
    EVERY merchant has no valid negative at all, and looping forever on it would
    hang the run rather than report it.
    """
    device = positive_card.device
    card = positive_card.to(torch.int64)
    n = card.numel()
    if n == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty

    if regime is NegativeRegime.HISTORICAL:
        if historical_pool is None:
            raise LinkPredError(
                "HISTORICAL regime needs a historical_pool of pairs that "
                "interacted before the cutoff. Without it this silently "
                "degrades to RANDOM, which is the trivially separable setting "
                "D5 warns about."
            )
        return _sample_historical(card, historical_pool, real_pairs, generator)

    merchant = torch.randint(
        0, num_merchants, (n,), device=device, dtype=torch.int64, generator=generator
    )
    colliding = real_pairs.contains(card, merchant)
    for _ in range(max_rounds):
        count = int(colliding.sum())
        if count == 0:
            break
        redraw = torch.randint(
            0,
            num_merchants,
            (count,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
        merchant = merchant.clone()
        merchant[colliding] = redraw
        colliding = real_pairs.contains(card, merchant)

    if bool(colliding.any()):
        stuck = int(colliding.sum())
        raise LinkPredError(
            f"{stuck} of {n} negatives still collide with real edges after "
            f"{max_rounds} rounds. Those cards have transacted with nearly "
            "every merchant, so no negative exists for them. Drop them from "
            "the seed set rather than training on false negatives."
        )
    return card, merchant


def _sample_historical(
    card: Tensor,
    pool: PairIndex,
    real_pairs: PairIndex,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    """Draw from pairs seen BEFORE the cutoff but not active at it.

    The pool is built by the caller from pre-cutoff edges, so this function
    never sees the graph and cannot accidentally admit a future edge. It filters
    the pool against the CURRENT positives, which is what makes a negative
    "previously active, inactive now" rather than merely "previously active".
    """
    if pool.keys.numel() == 0:
        raise LinkPredError("historical pool is empty")
    n = card.numel()

    keys = torch.empty(0, dtype=torch.int64, device=card.device)
    remaining = n
    for _ in range(12):
        choice = torch.randint(
            0,
            pool.keys.numel(),
            (remaining,),
            device=card.device,
            dtype=torch.int64,
            generator=generator,
        )
        candidate = pool.keys[choice]
        cand_card = candidate // pool.num_merchants
        cand_merchant = candidate % pool.num_merchants
        # A pool pair that is ALSO a current positive is not a negative.
        keep = ~real_pairs.contains(cand_card, cand_merchant)
        keys = torch.cat([keys, candidate[keep]])
        remaining = n - int(keys.numel())
        if remaining <= 0:
            break

    if int(keys.numel()) < n:
        raise LinkPredError(
            f"historical pool yielded only {int(keys.numel())} of {n} needed "
            "negatives: nearly every previously-active pair is also active at "
            "this cutoff. Either the pool was built from the wrong window (it "
            "must be pairs active BEFORE the cutoff, and it must not be a "
            "subset of the current positives), or this window is too dense for "
            "the historical regime -- report that rather than padding with "
            "random negatives, which would silently mix the two regimes D5 "
            "requires be reported separately."
        )

    keys = keys[:n]
    return keys // pool.num_merchants, keys % pool.num_merchants


def verify_no_collisions(card: Tensor, merchant: Tensor, real_pairs: PairIndex) -> None:
    """Assert a negative set contains no real edges. Call it every batch.

    Cheap -- one searchsorted -- and it is the only thing standing between a
    correct objective and one that trains the model to score 3.9% of real edges
    as absent.
    """
    hit = real_pairs.contains(card, merchant)
    if bool(hit.any()):
        raise LinkPredError(
            f"{int(hit.sum())} of {card.numel()} sampled negatives are REAL "
            "edges. The collision check did not run or the pair index is stale."
        )


class DistMultDecoder(nn.Module):
    """Score a (card, merchant) pair from their embeddings.

    DistMult: ``sum(h_card * r * h_merchant)`` with one learnable diagonal
    relation vector. Chosen over a plain dot product because a dot product ties
    the two node types into one shared space and cannot express an asymmetric
    relation; chosen over a full bilinear matrix because d^2 parameters on one
    relation overfits a graph this size. The diagonal is the standard middle.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.relation = nn.Parameter(torch.ones(dim))

    def forward(self, card: Tensor, merchant: Tensor) -> Tensor:
        return (card * self.relation * merchant).sum(dim=-1)


def bce_loss(positive_score: Tensor, negative_score: Tensor) -> Tensor:
    """Binary cross-entropy over positives vs sampled negatives.

    ``logits`` form, so no sigmoid is applied twice and no probability is
    clamped. The label vector is constructed here rather than passed in, because
    a caller-supplied label is one more place for positives and negatives to be
    swapped -- a mistake that trains a perfectly confident wrong model.
    """
    scores = torch.cat([positive_score, negative_score])
    labels = torch.cat(
        [torch.ones_like(positive_score), torch.zeros_like(negative_score)]
    )
    return nn.functional.binary_cross_entropy_with_logits(scores, labels)
