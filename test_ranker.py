"""Unit tests for the deterministic scoring in ranker.py.

Pure functions only — no network, no LLM. Run with:  .venv/bin/python -m pytest
(or plain `.venv/bin/python test_ranker.py` for the lightweight runner below).
"""

from __future__ import annotations

from profiles import KibbeType, Measurements, UserProfile
from ranker import (
    DEFAULT_WEIGHTS,
    Garment,
    GarmentDimensions,
    RankingWeights,
    SubScore,
    rank,
    score_budget,
    score_fit,
    score_garment,
    score_target_match,
    score_taste,
    weighted_merge,
)

PROFILE = UserProfile(
    taste=("sculptural", "androgynous", "editorial"),
    monthly_budget_usd=300.0,
    measurements=Measurements(bust_cm=86.0, waist_cm=68.0, hip_cm=94.0),
    kibbe=KibbeType.DRAMATIC,
)


def _g(**kw) -> Garment:
    return Garment(title=kw.pop("title", "x"), url=kw.pop("url", "u"), **kw)


# --- target match ---------------------------------------------------------- #
def test_target_match_true():
    s = score_target_match(_g(is_target=True, is_target_confidence=0.9))
    assert s.value == 1.0 and s.confidence == 0.9


def test_target_match_false():
    s = score_target_match(_g(is_target=False, is_target_confidence=0.8))
    assert s.value == 0.0 and s.confidence == 0.8


def test_target_match_unknown_is_missing():
    s = score_target_match(_g(is_target=None))
    assert s.value is None and s.confidence == 0.0 and not s.known


# --- budget ---------------------------------------------------------------- #
def test_budget_within_is_full():
    s = score_budget(_g(price_usd=120.0), PROFILE)
    assert s.value == 1.0 and s.confidence == 1.0


def test_budget_decays_above():
    # 1.5x budget -> halfway down the linear decay to 2x.
    s = score_budget(_g(price_usd=450.0), PROFILE)
    assert abs(s.value - 0.5) < 1e-9


def test_budget_far_above_is_zero():
    assert score_budget(_g(price_usd=900.0), PROFILE).value == 0.0


def test_budget_missing_price_is_missing():
    assert not score_budget(_g(price_usd=None), PROFILE).known


def test_budget_no_budget_is_missing():
    p = UserProfile(taste=("x",), monthly_budget_usd=None)
    assert not score_budget(_g(price_usd=100.0), p).known


# --- taste ----------------------------------------------------------------- #
def test_taste_full_overlap():
    s = score_taste(_g(aesthetic=("sculptural", "androgynous", "editorial"),
                       attribute_confidence=0.8), PROFILE)
    assert s.value == 1.0 and s.confidence == 0.8


def test_taste_partial_overlap():
    s = score_taste(_g(aesthetic=("editorial", "cottagecore"),
                       attribute_confidence=0.6), PROFILE)
    assert abs(s.value - 1 / 3) < 1e-9


def test_taste_no_tags_is_missing():
    assert not score_taste(_g(aesthetic=()), PROFILE).known


# --- fit (silhouette + dimensional) --------------------------------------- #
def test_fit_silhouette_favored():
    s = score_fit(_g(silhouette=("sharp", "angular", "structured"),
                     attribute_confidence=0.9), PROFILE)
    assert s.value == 1.0 and s.confidence > 0


def test_fit_silhouette_avoided():
    s = score_fit(_g(silhouette=("ruffled", "frilly", "cropped"),
                     attribute_confidence=0.9), PROFILE)
    assert s.value == 0.0


def test_fit_uses_dimensions_when_present():
    # Default ease = 7cm; bust 86 -> ideal chest ~93. Exact match -> high score.
    g = _g(silhouette=("structured",), attribute_confidence=0.9,
           dimensions=GarmentDimensions(chest_cm=93.0), dimensions_confidence=0.9)
    assert score_fit(g, PROFILE).value > 0.8


def test_fit_missing_everything_is_missing():
    assert not score_fit(_g(), PROFILE).known


# --- confidence pulls weight down ----------------------------------------- #
def test_low_confidence_subscore_loses_influence():
    # Same values, different confidence: high-confidence taste should dominate.
    high = SubScore(1.0, 1.0)
    low = SubScore(0.0, 0.05)
    value, _ = weighted_merge([(high, 0.5), (low, 0.5)])
    assert value > 0.9  # the near-zero-confidence 0.0 barely counts


def test_confidence_power_sharpens():
    a = SubScore(1.0, 1.0)
    b = SubScore(0.0, 0.5)
    linear, _ = weighted_merge([(a, 1.0), (b, 1.0)], confidence_power=1.0)
    sharp, _ = weighted_merge([(a, 1.0), (b, 1.0)], confidence_power=3.0)
    assert sharp > linear  # punishing uncertainty pushes toward the confident score


def test_all_missing_overall_is_none():
    assert score_garment(_g(), UserProfile(taste=())).overall is None


# --- ranking order --------------------------------------------------------- #
def test_rank_orders_by_overall_and_missing_last():
    good = _g(title="good", is_target=True, is_target_confidence=1.0,
              price_usd=100.0, aesthetic=("sculptural", "androgynous", "editorial"),
              attribute_confidence=1.0, silhouette=("sharp", "angular"))
    weak = _g(title="weak", is_target=False, is_target_confidence=1.0,
              price_usd=900.0, aesthetic=("cottagecore",), attribute_confidence=1.0)
    empty = _g(title="empty")
    ranked = rank([weak, empty, good], PROFILE)
    assert [r.garment.title for r in ranked][0] == "good"
    assert ranked[-1].garment.title == "empty"  # no signal sorts last
    assert ranked[-1].overall is None


def test_weights_are_configurable():
    g = _g(is_target=True, is_target_confidence=1.0, price_usd=9999.0)
    only_target = RankingWeights(target_match=1.0, fit=0.0, taste=0.0, budget=0.0)
    # With budget weighted out, the over-budget price shouldn't drag the score.
    assert score_garment(g, PROFILE, only_target).overall == 1.0


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    raise SystemExit(0 if passed == len(tests) else 1)
