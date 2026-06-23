"""Deterministic scoring + ranking for candidate garments.

Pure functions, no global mutable state, no I/O — so every scorer is directly
unit-testable. The LLM extraction step lives in `extract.py`; this module only
consumes already-extracted `Garment` objects.

Each scorer returns a `SubScore(value, confidence, reason)`:
  - `value` in [0, 1], or `None` when the data needed simply isn't there.
  - `confidence` in [0, 1] — how much to trust this sub-score.

The overall score is a confidence-weighted sum of the sub-scores. A sub-score's
*effective* weight is `base_weight * confidence**power`, so low-confidence
signals automatically pull their own influence down (and `None` drops out
entirely). Missing data is therefore represented honestly end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from profiles import FitPreference, KibbeType, UserProfile


# --------------------------------------------------------------------------- #
# Garment model (produced by the LLM extraction step in extract.py)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GarmentDimensions:
    """Actual garment measurements in cm, when stated. `None` = not found."""

    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    length_cm: Optional[float] = None
    inseam_cm: Optional[float] = None


@dataclass(frozen=True)
class Garment:
    title: str
    url: str
    price_usd: Optional[float] = None
    currency: Optional[str] = None
    # Whether this candidate actually *is* the target item the shopper wants.
    is_target: Optional[bool] = None
    is_target_confidence: float = 0.0
    garment_type: Optional[str] = None
    silhouette: Tuple[str, ...] = ()          # fit/line descriptors
    aesthetic: Tuple[str, ...] = ()           # vibe/taste descriptors
    material: Optional[str] = None
    color: Optional[str] = None
    dimensions: GarmentDimensions = field(default_factory=GarmentDimensions)
    attribute_confidence: float = 0.0          # trust in silhouette/aesthetic/etc.
    dimensions_confidence: float = 0.0          # trust in the measurements


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubScore:
    """A score in [0,1] with a confidence in [0,1]. `value=None` => missing data."""

    value: Optional[float]
    confidence: float
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.value is not None and self.confidence > 0.0


MISSING = SubScore(value=None, confidence=0.0, reason="no data")


@dataclass(frozen=True)
class RankingWeights:
    """Base weights for the weighted sum — tune freely; they need not sum to 1.

    `confidence_power` controls how hard low confidence pulls a sub-score down:
    effective_weight = base_weight * confidence ** confidence_power.
      - 1.0: linear (default)
      - >1:  punishes uncertainty harder
      - 0.0: ignore confidence entirely
    """

    target_match: float = 0.40
    fit: float = 0.25
    taste: float = 0.20
    budget: float = 0.15
    confidence_power: float = 1.0

    def base(self, name: str) -> float:
        return float(getattr(self, name))


DEFAULT_WEIGHTS = RankingWeights()


@dataclass(frozen=True)
class RankedItem:
    garment: Garment
    subscores: Mapping[str, SubScore]
    overall: Optional[float]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def weighted_merge(
    weighted: Sequence[Tuple[SubScore, float]], confidence_power: float = 1.0
) -> Tuple[Optional[float], float]:
    """Confidence-weighted merge of sub-scores.

    Returns ``(value, total_effective_weight)``. Each contributes with effective
    weight ``base * confidence ** power``; unknown or zero-weight sub-scores drop
    out. Returns ``(None, 0.0)`` when nothing usable is present.
    """
    num = 0.0
    den = 0.0
    for sub, w in weighted:
        if not sub.known or w <= 0:
            continue
        eff = w * (sub.confidence ** confidence_power)
        if eff <= 0:
            continue
        num += eff * sub.value
        den += eff
    if den <= 0:
        return None, 0.0
    return num / den, den


def _norm_tags(tags: Sequence[str]) -> Set[str]:
    return {t.strip().lower() for t in tags if t and t.strip()}


# --------------------------------------------------------------------------- #
# Kibbe silhouette lines (demo heuristics — tune to taste)
# --------------------------------------------------------------------------- #
# Each type favours / avoids certain garment line descriptors. The extractor is
# told this vocabulary so its `silhouette` tags line up with these sets.
KIBBE_LINES: Dict[KibbeType, Dict[str, Set[str]]] = {
    KibbeType.DRAMATIC: {
        "favor": {"sharp", "angular", "structured", "tailored", "elongated",
                  "sleek", "geometric", "minimal", "monochrome", "straight", "long"},
        "avoid": {"ruffled", "frilly", "cropped", "rounded", "gathered", "delicate"},
    },
    KibbeType.SOFT_DRAMATIC: {
        "favor": {"draped", "structured", "bold", "elongated", "lush", "statement",
                  "sweeping", "vampy", "long"},
        "avoid": {"boxy", "cropped", "stiff", "plain", "prim"},
    },
    KibbeType.FLAMBOYANT_NATURAL: {
        "favor": {"oversized", "relaxed", "draped", "long", "unstructured",
                  "flowing", "layered", "slouchy", "wide"},
        "avoid": {"fitted", "stiff", "prim", "delicate", "cropped"},
    },
    KibbeType.NATURAL: {
        "favor": {"relaxed", "unstructured", "soft-tailored", "easy", "draped",
                  "casual", "moderate"},
        "avoid": {"stiff", "ornate", "frilly", "sharp", "clingy"},
    },
    KibbeType.SOFT_NATURAL: {
        "favor": {"soft", "draped", "relaxed", "flowing", "easy", "rounded"},
        "avoid": {"stiff", "sharp", "boxy", "severe"},
    },
    KibbeType.FLAMBOYANT_GAMINE: {
        "favor": {"cropped", "fitted", "sharp", "bold", "geometric", "contrast",
                  "snug", "structured"},
        "avoid": {"long", "flowing", "draped", "oversized", "romantic"},
    },
    KibbeType.GAMINE: {
        "favor": {"cropped", "fitted", "crisp", "contrast", "snug", "structured"},
        "avoid": {"long", "draped", "flowing", "oversized"},
    },
    KibbeType.SOFT_GAMINE: {
        "favor": {"fitted", "cropped", "soft", "rounded", "playful", "snug"},
        "avoid": {"long", "severe", "oversized", "draped"},
    },
    KibbeType.DRAMATIC_CLASSIC: {
        "favor": {"tailored", "sharp", "structured", "clean", "sleek", "refined",
                  "straight", "minimal"},
        "avoid": {"frilly", "oversized", "slouchy", "ruffled"},
    },
    KibbeType.CLASSIC: {
        "favor": {"tailored", "balanced", "clean", "refined", "moderate",
                  "symmetrical", "polished"},
        "avoid": {"extreme", "oversized", "severe", "frilly"},
    },
    KibbeType.SOFT_CLASSIC: {
        "favor": {"soft-tailored", "refined", "draped", "polished", "smooth"},
        "avoid": {"sharp", "boxy", "severe", "extreme"},
    },
    KibbeType.THEATRICAL_ROMANTIC: {
        "favor": {"fitted", "draped", "ornate", "detailed", "soft", "curved",
                  "vampy", "embellished"},
        "avoid": {"boxy", "oversized", "stiff", "minimal", "severe"},
    },
    KibbeType.ROMANTIC: {
        "favor": {"soft", "fitted", "draped", "curved", "lush", "delicate",
                  "embellished", "rounded"},
        "avoid": {"boxy", "sharp", "oversized", "stiff", "minimal", "structured"},
    },
}

# Vocabulary the extractor should prefer for `silhouette` tags (union of above).
SILHOUETTE_VOCAB: Tuple[str, ...] = tuple(
    sorted({t for lines in KIBBE_LINES.values() for t in lines["favor"] | lines["avoid"]})
)

# Ease (cm) we expect a garment to add over the body, per fit preference.
FIT_EASE_CM: Dict[FitPreference, float] = {
    FitPreference.FITTED: 2.0,
    FitPreference.REGULAR: 7.0,
    FitPreference.RELAXED: 14.0,
    FitPreference.OVERSIZED: 24.0,
}
# How far (cm) from the ideal eased measurement drives the dimensional score to 0.
FIT_TOLERANCE_CM = 16.0


# --------------------------------------------------------------------------- #
# Sub-scorers (each pure: Garment [+ profile] -> SubScore)
# --------------------------------------------------------------------------- #
def score_target_match(g: Garment) -> SubScore:
    """Is this candidate actually the item the shopper is looking for?"""
    if g.is_target is None:
        return SubScore(None, 0.0, "target match unknown")
    value = 1.0 if g.is_target else 0.0
    reason = "matches target item" if g.is_target else "not the target item"
    return SubScore(value, _clamp01(g.is_target_confidence), reason)


def score_budget(g: Garment, profile: UserProfile) -> SubScore:
    """Price vs the shopper's budget. Price is hard data => full confidence."""
    budget = profile.monthly_budget_usd
    if budget is None or budget <= 0:
        return SubScore(None, 0.0, "no budget set")
    if g.price_usd is None:
        return SubScore(None, 0.0, "price unknown")
    if g.price_usd <= budget:
        value = 1.0
    else:
        # Linear decay: 1.0 at budget, 0.0 at 2x budget and beyond.
        value = _clamp01(1.0 - (g.price_usd - budget) / budget)
    return SubScore(value, 1.0, f"${g.price_usd:.0f} vs ${budget:.0f} budget")


def score_taste(g: Garment, profile: UserProfile) -> SubScore:
    """Overlap between the garment's aesthetic tags and the taste spec."""
    if not profile.taste:
        return SubScore(None, 0.0, "no taste spec")
    if not g.aesthetic:
        return SubScore(None, 0.0, "no aesthetic tags extracted")
    tags = _norm_tags(g.aesthetic)
    taste = [t.strip().lower() for t in profile.taste if t.strip()]
    hits = [t for t in taste if any(t == tag or t in tag or tag in t for tag in tags)]
    value = len(hits) / len(taste)
    reason = f"{len(hits)}/{len(taste)} taste cues" + (f": {', '.join(hits)}" if hits else "")
    return SubScore(_clamp01(value), _clamp01(g.attribute_confidence), reason)


def _silhouette_fit(g: Garment, profile: UserProfile) -> SubScore:
    """How well the garment's line aligns with the Kibbe type's favoured lines."""
    if not g.silhouette:
        return SubScore(None, 0.0, "no silhouette tags")
    if profile.kibbe is None:
        return SubScore(None, 0.0, "no kibbe type")
    lines = KIBBE_LINES.get(profile.kibbe)
    if not lines:
        return SubScore(None, 0.0, "kibbe lines unknown")
    tags = _norm_tags(g.silhouette)
    favor = sum(1 for t in tags if t in lines["favor"])
    avoid = sum(1 for t in tags if t in lines["avoid"])
    n = len(tags)
    raw = (favor - avoid) / n            # in [-1, 1]
    value = _clamp01((raw + 1.0) / 2.0)  # map to [0, 1]
    # More tags + higher attribute confidence => more trust (cap at 3 tags).
    confidence = _clamp01(g.attribute_confidence * min(1.0, n / 3.0))
    return SubScore(value, confidence,
                    f"kibbe {profile.kibbe.value}: +{favor}/-{avoid} of {n} lines")


def _dimensional_fit(g: Garment, profile: UserProfile) -> SubScore:
    """Closeness of actual garment dims to the body + expected ease."""
    body = profile.measurements
    ease = FIT_EASE_CM.get(profile.fit_preference, 7.0)
    comparables = (
        (g.dimensions.chest_cm, body.bust_cm),
        (g.dimensions.waist_cm, body.waist_cm),
        (g.dimensions.hip_cm, body.hip_cm),
    )
    diffs = [abs(gd - (bd + ease)) for gd, bd in comparables if gd is not None and bd is not None]
    if not diffs:
        return SubScore(None, 0.0, "no comparable garment dimensions")
    avg = sum(diffs) / len(diffs)
    value = _clamp01(1.0 - avg / FIT_TOLERANCE_CM)
    return SubScore(value, _clamp01(g.dimensions_confidence),
                    f"avg {avg:.0f}cm from ideal ({profile.fit_preference.value})")


def score_fit(g: Garment, profile: UserProfile) -> SubScore:
    """Combine silhouette (Kibbe) and dimensional fit into one sub-score.

    Silhouette is weighted higher because real garment dimensions are usually
    absent; when dims *are* present they refine the score.
    """
    silhouette = _silhouette_fit(g, profile)
    dimensional = _dimensional_fit(g, profile)
    value, evidence = weighted_merge([(silhouette, 0.6), (dimensional, 0.4)])
    if value is None:
        return SubScore(None, 0.0, "no fit signal")
    reason = "; ".join(s.reason for s in (silhouette, dimensional) if s.known)
    # `evidence` is the summed effective weight (<= 1.0) => use as confidence.
    return SubScore(value, _clamp01(evidence), reason)


# --------------------------------------------------------------------------- #
# Overall score + ranking
# --------------------------------------------------------------------------- #
def score_garment(
    g: Garment, profile: UserProfile, weights: RankingWeights = DEFAULT_WEIGHTS
) -> RankedItem:
    """Score one garment across all dimensions into a single RankedItem."""
    subs: Dict[str, SubScore] = {
        "target_match": score_target_match(g),
        "fit": score_fit(g, profile),
        "taste": score_taste(g, profile),
        "budget": score_budget(g, profile),
    }
    overall, _ = weighted_merge(
        [(subs[name], weights.base(name)) for name in subs],
        confidence_power=weights.confidence_power,
    )
    return RankedItem(garment=g, subscores=subs, overall=overall)


def rank(
    garments: Sequence[Garment],
    profile: UserProfile,
    weights: RankingWeights = DEFAULT_WEIGHTS,
) -> list[RankedItem]:
    """Score and rank candidates, best first. Items with no usable signal
    (overall is None) sort last. Pure: same inputs => same output."""
    scored = [score_garment(g, profile, weights) for g in garments]
    scored.sort(
        key=lambda it: (it.overall is not None, it.overall if it.overall is not None else -1.0),
        reverse=True,
    )
    return scored
