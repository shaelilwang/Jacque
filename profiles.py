"""User profile model for ranking — pure data, no I/O, no globals.

Everything optional is honestly `None` when unknown; the ranker reflects missing
data as low confidence rather than guessing. Named `profiles` (not `profile`) to
avoid shadowing Python's stdlib `profile` module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class KibbeType(str, Enum):
    """The 13 Kibbe body types (yin/yang balance of bone, flesh, vertical line)."""

    DRAMATIC = "Dramatic"
    SOFT_DRAMATIC = "Soft Dramatic"
    FLAMBOYANT_NATURAL = "Flamboyant Natural"
    NATURAL = "Natural"
    SOFT_NATURAL = "Soft Natural"
    FLAMBOYANT_GAMINE = "Flamboyant Gamine"
    GAMINE = "Gamine"
    SOFT_GAMINE = "Soft Gamine"
    DRAMATIC_CLASSIC = "Dramatic Classic"
    CLASSIC = "Classic"
    SOFT_CLASSIC = "Soft Classic"
    THEATRICAL_ROMANTIC = "Theatrical Romantic"
    ROMANTIC = "Romantic"


class FitPreference(str, Enum):
    FITTED = "fitted"
    REGULAR = "regular"
    RELAXED = "relaxed"
    OVERSIZED = "oversized"


@dataclass(frozen=True)
class Measurements:
    """Body measurements in centimetres. `None` = not provided (never guessed)."""

    height_cm: Optional[float] = None
    bust_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    inseam_cm: Optional[float] = None
    shoulder_cm: Optional[float] = None


@dataclass(frozen=True)
class UserProfile:
    """Who we're shopping for. `taste` is a tuple of aesthetic descriptors."""

    taste: Tuple[str, ...]
    fit_preference: FitPreference = FitPreference.REGULAR
    monthly_budget_usd: Optional[float] = None
    usual_size: Optional[str] = None
    measurements: Measurements = field(default_factory=Measurements)
    kibbe: Optional[KibbeType] = None


# Hardcoded taste spec for now (per request).
DEFAULT_TASTE: Tuple[str, ...] = (
    "sculptural",
    "subversive",
    "androgynous",
    "editorial",
    "neo-romantic",
)

# EXAMPLE profile — the taste spec is the real (hardcoded) input; the
# measurements / size / budget / kibbe below are placeholders to make the demo
# runnable. Replace them with your real values, or set to None to keep honest.
DEFAULT_PROFILE = UserProfile(
    taste=DEFAULT_TASTE,
    fit_preference=FitPreference.RELAXED,
    monthly_budget_usd=400.0,
    usual_size="US 4 / S",
    measurements=Measurements(height_cm=170.0, bust_cm=86.0, waist_cm=68.0, hip_cm=94.0),
    kibbe=KibbeType.DRAMATIC,
)
