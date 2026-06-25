"""Unit tests for the explanation module's grounding contract.

Pure functions only — no network, no LLM. Run with:  .venv/bin/python -m pytest
(or plain `.venv/bin/python test_explain.py` for the lightweight runner below).

The LLM call itself isn't tested here; what matters is that the evidence we
build is faithful and the guard catches invented claims.
"""

from __future__ import annotations

from explain import (
    CLAIM_VOCAB,
    _caveats,
    _lead_reason,
    build_evidence,
    check_grounding,
)
from profiles import KibbeType, Measurements, UserProfile
from ranker import Garment, GarmentDimensions, score_garment

PROFILE = UserProfile(
    taste=("sculptural", "androgynous", "editorial"),
    monthly_budget_usd=400.0,
    measurements=Measurements(bust_cm=86.0, waist_cm=68.0, hip_cm=94.0),
    kibbe=KibbeType.DRAMATIC,
)


def _item(**kw):
    g = Garment(title=kw.pop("title", "x"), url=kw.pop("url", "u"), **kw)
    return score_garment(g, PROFILE)


def _evidence(**kw):
    return build_evidence(_item(**kw), PROFILE, item_id="i1")


# --- lead reason ----------------------------------------------------------- #
def test_lead_reason_among_explained_dimensions():
    # Taste is the only strong known signal, so it leads. Target match is never
    # a lead reason even when it's the strongest score (it's assumed).
    item = _item(aesthetic=("sculptural", "androgynous", "editorial"),
                 attribute_confidence=0.9, is_target=True, is_target_confidence=0.99)
    assert _lead_reason(item) == "taste"


def test_lead_reason_never_target_match():
    # A confident target match with no fit/taste/budget signal -> no lead.
    assert _lead_reason(_item(is_target=True, is_target_confidence=1.0)) is None


def test_lead_reason_none_when_no_signal():
    assert _lead_reason(_item()) is None


# --- caveats are honest ---------------------------------------------------- #
def test_caveat_flags_estimated_fit_without_dims():
    item = _item(silhouette=("sharp",), attribute_confidence=0.3, is_target=True,
                 is_target_confidence=0.9)
    caveats = _caveats(item, PROFILE)
    assert any("silhouette only" in c for c in caveats)


def test_caveat_omits_target_match():
    # Target match is assumed, so it never produces a caveat anymore.
    caveats = _caveats(_item(price_usd=100.0), PROFILE)
    assert not any("target" in c for c in caveats)


def test_caveat_flags_budget_driven_pick():
    # Cheap, no fit/taste signal -> honestly budget-driven.
    item = _item(price_usd=50.0)
    assert any("budget/availability" in c for c in _caveats(item, PROFILE))


# --- evidence assembly ----------------------------------------------------- #
def test_evidence_includes_real_numbers_and_drops_nulls():
    ev = _evidence(is_target=True, is_target_confidence=0.9, price_usd=129.0,
                   currency="USD", aesthetic=("editorial",), attribute_confidence=0.8)
    assert ev["garment"]["price_usd"] == 129.0
    assert ev["shopper"]["remaining_budget_usd"] == 271.0
    # No material/colour were extracted -> they must not appear at all.
    assert "material" not in ev["garment"]
    assert "color" not in ev["garment"]


# --- the guard: numbers ---------------------------------------------------- #
def test_guard_passes_grounded_numbers():
    ev = _evidence(is_target=True, is_target_confidence=0.9, price_usd=129.0,
                   currency="USD")
    text = "It's the piece you wanted and it's $129, leaving $271 of your $400."
    assert check_grounding(text, ev) == []


def test_guard_flags_invented_number():
    ev = _evidence(is_target=True, is_target_confidence=0.9, price_usd=129.0)
    text = "Great pick at just $49 — practically a steal."
    flags = check_grounding(text, ev)
    assert any("49" in f for f in flags)


# --- the guard: attributes ------------------------------------------------- #
def test_guard_flags_invented_attribute():
    ev = _evidence(is_target=True, is_target_confidence=0.9,
                   aesthetic=("editorial",), attribute_confidence=0.8)
    text = "The silk drape is gorgeous and very editorial."  # silk never in evidence
    flags = check_grounding(text, ev)
    assert any("silk" in f for f in flags)


def test_guard_passes_grounded_attribute():
    ev = _evidence(is_target=True, is_target_confidence=0.9,
                   silhouette=("structured", "sharp"), attribute_confidence=0.9)
    text = "Structured and sharp — exactly your line."
    assert check_grounding(text, ev) == []


def test_guard_ignores_plain_english():
    ev = _evidence(is_target=True, is_target_confidence=0.9)
    # No domain-attribute or stray-number claims -> nothing to flag.
    text = "Honestly, this is the one you came for. I'd grab it."
    assert check_grounding(text, ev) == []


def test_claim_vocab_does_not_contain_generic_words():
    for w in ("the", "good", "great", "you", "this", "piece"):
        assert w not in CLAIM_VOCAB


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
