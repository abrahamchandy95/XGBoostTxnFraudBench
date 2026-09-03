from dataclasses import dataclass


MEASURED_ON = "2026-07-30"

MEASURED_TRANSACTIONS = 27_019_580

MEASURED_BUILD = "b_test"


@dataclass(frozen=True, slots=True)
class Observation:
    """One measured quantity, and how far it may move before it stops holding.

    ``tolerance`` is a RELATIVE band, so 0.5 means "within a factor of 1.5
    either way". It is deliberately wide: the point is to catch a regenerated
    dataset that changed the graph's shape, not to flag ordinary noise between
    two builds of the same load.
    """

    key: str
    value: float
    tolerance: float
    depends_on: str
    note: str

    def drifted(self, observed: float) -> bool:
        if self.value == 0.0:
            return observed != 0.0
        return abs(observed - self.value) > self.tolerance * abs(self.value)

    def describe(self, observed: float) -> str:
        return (
            f"{self.key} is {observed:,.4g}, but {self.depends_on} was tuned "
            f"against {self.value:,.4g} (measured {MEASURED_ON}). {self.note}"
        )


OBSERVATIONS: dict[str, Observation] = {
    "hub_concentration_S": Observation(
        key="hub_concentration_S",
        value=3.507,
        tolerance=0.5,
        depends_on="the Stage 2 fan-out caps in tfgnn/stage2/fanout.py",
        note=(
            "Every `Merchant ->` direction is capped at 8 because one merchant "
            "reaches over half of all cards. A much SMALLER S means the caps are "
            "needlessly tight and HUB_WARN_ABOVE=16 will refuse to raise them "
            "without strict=False; a LARGER one means a batch is bigger than "
            "the caps were sized for. Re-read the fanout docstring before "
            "trusting worst_case_nodes. NOTE that S tracks "
            "card_card_saturation_hint closely -- 3.5069 vs 3.5054 on b_test, "
            "0.04% apart -- because sum(deg^2)/n^2 and sum(deg(deg-1))/n(n-1) "
            "converge once degrees are large. They are not independent "
            "evidence; S is reported because it is the quantity the fan-out "
            "caps were named against."
        ),
    ),
    "max_merchant_reach_share": Observation(
        key="max_merchant_reach_share",
        value=0.5153,
        tolerance=0.5,
        depends_on="the Stage 2 hub fan-out caps and projection.max_merchant_degree",
        note=(
            "This is the single number the asymmetric caps exist for: on b_test "
            "one merchant touches 7,448 of 14,453 seen cards. It is also what "
            "makes the uncapped two-hop expansion reach half the graph from one "
            "seed."
        ),
    ),
    # MEASURED 2026-07-30: projection_weight_histogram on b_test attaches 13,262
    # of 14,453 seen cards at threshold 10. Confirms config.yaml's recorded
    # 91.8% to three figures, and its whole surviving-edge table exactly
    # (73,275,959 / 12,533,787 / 1,340,618 / 15,181 at 2 / 6 / 10 / 20).
    "cards_attached_share_at_min_weight": Observation(
        key="cards_attached_share_at_min_weight",
        value=0.9176,
        tolerance=0.1,
        depends_on="stage1_pipeline.structure.min_weight",
        note=(
            "min_weight=10 was chosen to remove 69x the edges while keeping "
            "91.8% of cards attached. A threshold that isolates most cards "
            "leaves them exporting c_size as NaN, which is not an improvement "
            "over a constant -- see weight_histogram's header. Read the "
            "surviving-edge and attachment columns together and re-pick."
        ),
    ),
    "card_card_max_weight": Observation(
        key="card_card_max_weight",
        value=39.0,
        tolerance=0.5,
        depends_on="stage1_pipeline.structure.min_weight",
        note=(
            "The weight range bounds what any threshold can do: at max 39 "
            "there was no room above 20, which is why no threshold could "
            "fragment the projection without isolating 57% of cards."
        ),
    ),
}


def drift_notes(observed: dict[str, float]) -> list[str]:
    """One note per constant that no longer rests on a current measurement.

    Only keys present in ``observed`` are checked, so a query that did not run
    contributes nothing rather than a false alarm.
    """
    notes: list[str] = []
    for key, value in observed.items():
        record = OBSERVATIONS.get(key)
        if record is not None and record.drifted(value):
            notes.append(record.describe(value))
    return notes
