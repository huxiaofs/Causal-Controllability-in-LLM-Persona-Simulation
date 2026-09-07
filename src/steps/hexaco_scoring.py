from typing import Dict, List, Union, Iterable, Tuple

# Item and scoring configuration (5-point Likert; True means reverse-scored).
DIM_FACETS = {
    "Honesty-Humility": [
        ("Sincerity", [(6, False), (30, True), (54, False)]),
        ("Fairness", [(12, True), (36, False), (60, True)]),
        ("Greed-Avoidance", [(18, False), (42, True)]),
        ("Modesty", [(24, True), (48, True)]),
    ],
    "Emotionality": [
        ("Fearfulness", [(5, False), (29, False), (53, True)]),
        ("Anxiety", [(11, False), (35, True)]),
        ("Dependence", [(17, False), (41, True)]),
        ("Sentimentality", [(23, False), (47, False), (59, True)]),
    ],
    "Extraversion": [
        ("Social Self-Esteem", [(4, False), (28, True), (52, True)]),
        ("Social Boldness", [(10, True), (34, False), (58, False)]),
        ("Sociability", [(16, False), (40, False)]),
        ("Liveliness", [(22, False), (46, True)]),
    ],
    "Agreeableness": [
        ("Forgiveness", [(3, False), (27, False)]),
        ("Gentleness", [(9, True), (33, False), (51, False)]),
        ("Flexibility", [(15, True), (39, False), (57, True)]),
        ("Patience", [(21, True), (45, False)]),
    ],
    "Conscientiousness": [
        ("Organization", [(2, False), (26, True)]),
        ("Diligence", [(8, False), (32, True)]),
        ("Perfectionism", [(14, True), (38, False), (50, False)]),
        ("Prudence", [(20, True), (44, True), (56, True)]),
    ],
    "Openness": [
        ("Aesthetic Appreciation", [(1, True), (25, False)]),
        ("Inquisitiveness", [(7, False), (31, True)]),
        ("Creativity", [(13, False), (37, False), (49, True)]),
        ("Unconventionality", [(19, True), (43, False), (55, True)]),
    ],
}

ALL_ITEMS = set(n for facets in DIM_FACETS.values() for _, items in facets for n, _ in items)
REVERSED_ITEMS = set(n for facets in DIM_FACETS.values() for _, items in facets for n, r in items if r)


def _reverse_5pt(x: int) -> int:
    if x is None:
        return None
    if not (1 <= x <= 5):
        raise ValueError(f"Response scores must be in 1--5; received {x}")
    return 6 - x


def _coerce_responses(
    responses: Union[Dict[int, int], Iterable[int], Tuple[int, ...], List[int]]
) -> Dict[int, int]:
    if isinstance(responses, dict):
        cleaned = {int(k): int(v) for k, v in responses.items() if int(k) in ALL_ITEMS}
    else:
        seq = list(responses)
        if len(seq) != 60:
            raise ValueError(
                f"Response sequences must contain 60 items; received {len(seq)}"
            )
        cleaned = {i + 1: int(seq[i]) for i in range(60) if (i + 1) in ALL_ITEMS}
    return cleaned


def score_hexaco(
    responses: Union[Dict[int, int], Iterable[int], Tuple[int, ...], List[int]],
    round_facets: int | None = 2,
    round_dimensions: int | None = 1,
    allow_missing: bool = False,
):
    resp = _coerce_responses(responses)
    missing = sorted([n for n in ALL_ITEMS if n not in resp])
    if missing and not allow_missing:
        raise ValueError(
            f"Missing questionnaire item IDs: {missing}. "
            "Set allow_missing=True to score using the available items."
        )

    scored = {}
    for n in ALL_ITEMS:
        if n not in resp:
            continue
        x = resp[n]
        if not (1 <= x <= 5):
            raise ValueError(f"Item {n} has an invalid score {x}; expected 1--5")
        scored[n] = _reverse_5pt(x) if n in REVERSED_ITEMS else x

    facet_scores_by_dim = {}
    for dim, facets in DIM_FACETS.items():
        facet_scores = {}
        for facet_name, items in facets:
            vals = [scored[n] for n, _ in items if n in scored]
            if not vals:
                facet_avg = None
            else:
                facet_avg = sum(vals) / len(vals)
                if round_facets is not None:
                    facet_avg = round(facet_avg, round_facets)
            facet_scores[facet_name] = facet_avg
        facet_scores_by_dim[dim] = facet_scores

    dim_scores = {}
    for dim, facets in DIM_FACETS.items():
        dim_items = [n for _, items in facets for n, _ in items if n in scored]
        if not dim_items:
            dim_avg = None
        else:
            dim_avg = sum(scored[n] for n in dim_items) / len(dim_items)
            if round_dimensions is not None:
                dim_avg = round(dim_avg, round_dimensions)
        dim_scores[dim] = dim_avg

    return {
        "hexaco": {
            "facets": facet_scores_by_dim,
            "dimensions": dim_scores,
        }
    }


__all__ = [
    "DIM_FACETS",
    "ALL_ITEMS",
    "REVERSED_ITEMS",
    "_reverse_5pt",
    "_coerce_responses",
    "score_hexaco",
]
