"""Explanation step: turn ranked garments into grounded 'why this is a good pick'.

The ranker (`ranker.py`) emits a `RankedItem` per garment carrying a set of
`SubScore`s — each a value/confidence plus a `reason` string that already holds
the *actual* numbers ("$129 vs $400 budget", "avg 5cm from ideal (relaxed)",
"2/3 taste cues: editorial, androgynous"). This module hands that structured
evidence to a cheap model and asks it to write a short, specific note in the
voice of a personal shopper who knows the client.

The point of the module is the guard, not the prose. After generation we verify
the explanation only cites things that were actually in the evidence we sent it:
every number must trace back to the evidence, and any garment-attribute word
("silk", "cropped", "structured") must appear in the evidence. Anything that
doesn't is flagged for review — a faithful explainer that admits "I'm not sure
about the fit here" is the goal; a fluent one that invents reasons is the bug.

`build_evidence`, `check_grounding`, `_lead_reason` and `_caveats` are pure (no
I/O), so the whole grounding contract is unit-testable without the network.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cost as cost_mod
from profiles import DEFAULT_PROFILE, UserProfile
from ranker import SILHOUETTE_VOCAB, RankedItem

HAIKU_MODEL = "claude-haiku-4-5"

# The item is assumed to already be what the shopper searched for, so target
# match is not a reason we surface — the only legal values for `lead_reason`
# are the dimensions worth explaining.
LEAD_REASONS: Tuple[str, ...] = ("fit", "taste", "budget")

# Haiku 4.5 caps temperature at 1.0 (also the default), so this is the ceiling.
# The interpretive "how it matches your vibe" read comes from the prompt, not
# from temperature — this just runs the call at maximum sampling variety.
EXPLAIN_TEMPERATURE = 1.0

# Roughly the prompt the task specified, with one line of voice on top. The
# rules are the contract: only the evidence, lead with the strongest real
# reason, be specific with the numbers, state caveats plainly, no hype.
SYSTEM_PROMPT = """\
You explain why a shopping item is a good pick, in the voice of a sharp,
knowledgeable personal shopper who knows this client and their taste well — a
little sassy, never gushing. You are given structured scoring evidence. Write
2-4 sentences.

The item is already known to be what the client is looking for. Do NOT say
"this is the item you wanted", "great find/match", or otherwise restate that it
matches the search — that's assumed. Spend your words on WHY it's a good version
of it: how it fits, how it reads on them, and what it costs.

Rules:
- Ground every concrete claim in the evidence: the actual measurements, the
  actual price and remaining budget, the silhouette/aesthetic tags you were
  given. Never invent a material, colour, measurement, or price not in the input.
- You MAY interpret — connect the garment's stated attributes to the client's
  taste and Kibbe vibe and say how it reads on them. That inference is the value
  you add; just don't pass off an invented attribute as fact.
- Lead with the strongest real reason (highest score with high confidence).
- Be specific with the numbers you were given — that specificity is the point.
- If a caveat exists (e.g. fit estimated, no size chart), state it plainly.
  Do not hide low confidence behind confident language.
- If it mostly ranked on price or availability rather than fit or taste, say
  that honestly rather than dressing it up.
- No hype. Sound like someone who actually checked, then formed an opinion."""

# JSON schema the model must fill. `item_id` is set by us, not the model, so it
# can't be invented or mismatched.
_OUT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
            "description": "The grounded 2-4 sentence 'why this is a good pick'.",
        },
        "confidence_note": {
            "type": ["string", "null"],
            "description": "The single most important caveat to surface, or null.",
        },
        "lead_reason": {
            "type": "string",
            "enum": list(LEAD_REASONS),
            "description": "Which dimension actually drove the recommendation.",
        },
    },
    "required": ["explanation", "confidence_note", "lead_reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Explanation:
    """One garment's grounded explanation plus any grounding violations.

    `flags` is empty when the explanation is faithful; a non-empty `flags`
    means a human (or a stricter pass) should look before this is shown.
    """

    item_id: str
    explanation: str
    confidence_note: Optional[str]
    lead_reason: str
    flags: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.flags


# --------------------------------------------------------------------------- #
# Evidence assembly (pure) — this dict IS what the model sees AND the single
# source of truth the guard checks against.
# --------------------------------------------------------------------------- #
def _present(d: dict) -> dict:
    """Drop None-valued entries so we never imply we have data we don't."""
    return {k: v for k, v in d.items() if v is not None}


def _lead_reason(item: RankedItem) -> Optional[str]:
    """The dimension that most drove the rank: highest value*confidence among
    the known sub-scores. None when nothing is known."""
    best, best_contrib = None, -1.0
    for name in LEAD_REASONS:
        s = item.subscores.get(name)
        if s is not None and s.known:
            contrib = s.value * s.confidence
            if contrib > best_contrib:
                best, best_contrib = name, contrib
    return best


def _caveats(item: RankedItem, profile: UserProfile) -> List[str]:
    """Honest caveats drawn straight from the sub-scores, for the model to
    surface rather than paper over."""
    subs = item.subscores
    fit, taste = subs.get("fit"), subs.get("taste")
    g = item.garment
    out: List[str] = []

    if fit is not None and fit.known:
        has_dims = any(
            v is not None
            for v in (g.dimensions.chest_cm, g.dimensions.waist_cm, g.dimensions.hip_cm)
        )
        if not has_dims:
            out.append("no garment measurements / size chart found; fit is from silhouette only")
        if fit.confidence < 0.5:
            out.append(f"fit is an estimate (confidence {fit.confidence:.0%})")

    if taste is not None and taste.known and taste.confidence < 0.5:
        out.append(f"taste read is low-confidence ({taste.confidence:.0%})")

    if g.price_usd is None:
        out.append("price unknown")

    lead = _lead_reason(item)
    weak_fit = fit is None or not fit.known or fit.value < 0.5
    weak_taste = taste is None or not taste.known or taste.value < 0.5
    if lead == "budget" and weak_fit and weak_taste:
        out.append("ranked high mainly on budget/availability, not fit or taste")

    return out


def build_evidence(item: RankedItem, profile: UserProfile, item_id: str) -> dict:
    """The structured evidence for one item: garment facts, the shopper's
    relevant profile, every sub-score (value/confidence/reason), the caveats and
    a deterministic lead-reason hint. Only present (non-null) facts are included.
    """
    g = item.garment
    remaining = None
    if profile.monthly_budget_usd and g.price_usd is not None:
        remaining = round(profile.monthly_budget_usd - g.price_usd, 2)

    # Only the explained dimensions — target match is assumed, not surfaced.
    subs: Dict[str, dict] = {}
    for name in LEAD_REASONS:
        s = item.subscores.get(name)
        if s is None:
            continue
        subs[name] = {
            "value": round(s.value, 3) if s.value is not None else None,
            "confidence": round(s.confidence, 3),
            "reason": s.reason,
            "known": s.known,
        }

    evidence = {
        "item_id": item_id,
        "garment": _present({
            "title": g.title,
            "garment_type": g.garment_type,
            "price_usd": g.price_usd,
            "currency": g.currency,
            "material": g.material,
            "color": g.color,
            "silhouette": list(g.silhouette) or None,
            "aesthetic": list(g.aesthetic) or None,
            "dimensions_cm": _present(asdict(g.dimensions)) or None,
        }),
        "shopper": _present({
            "taste_spec": list(profile.taste) or None,
            "kibbe": profile.kibbe.value if profile.kibbe else None,
            "body_measurements_cm": _present(asdict(profile.measurements)) or None,
            "budget_usd": profile.monthly_budget_usd,
            "remaining_budget_usd": remaining,
        }),
        "overall_score": round(item.overall, 3) if item.overall is not None else None,
        "subscores": subs,
        "caveats": _caveats(item, profile),
        "lead_reason_hint": _lead_reason(item),
    }
    return evidence


# --------------------------------------------------------------------------- #
# Grounding guard (pure) — the point of the module.
# --------------------------------------------------------------------------- #
# A broad universe of garment-attribute words the model *might* assert. A word
# from this set in the explanation is only allowed if it also appears in the
# evidence we sent; otherwise it's an invented attribute. (Generic English is
# deliberately not in here — we only police domain claims.)
_MATERIALS: Set[str] = {
    "cotton", "linen", "wool", "merino", "cashmere", "silk", "satin", "velvet",
    "leather", "suede", "denim", "polyester", "nylon", "viscose", "rayon",
    "tweed", "knit", "ribbed", "mesh", "lace", "chiffon", "tulle", "jersey",
}
_COLORS: Set[str] = {
    "black", "white", "ivory", "cream", "beige", "navy", "grey", "gray",
    "brown", "tan", "camel", "red", "burgundy", "maroon", "green", "olive",
    "blue", "pink", "purple", "lilac", "gold", "silver", "charcoal", "khaki",
}
_FIT_WORDS: Set[str] = {
    "fitted", "oversized", "relaxed", "cropped", "slim", "loose", "tailored",
    "boxy", "flowy", "sheer", "snug", "structured", "draped",
}
CLAIM_VOCAB: Set[str] = (
    set(SILHOUETTE_VOCAB) | _MATERIALS | _COLORS | _FIT_WORDS
)

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z][a-z\-]+")


def _numbers(text: str) -> List[float]:
    return [float(m) for m in _NUM_RE.findall(text)]


def _allowed_numbers(evidence_text: str) -> Set[float]:
    """Numbers the explanation may cite: every number in the evidence, plus its
    integer-rounded form and (for 0-1 scores) its percentage form, so citing a
    confidence as '80%' against an evidence '0.8' doesn't false-flag."""
    nums: Set[float] = set()
    for n in _numbers(evidence_text):
        nums.add(round(n, 2))
        nums.add(float(round(n)))
        if 0.0 <= n <= 1.0:
            nums.add(float(round(n * 100)))
    return nums


def check_grounding(explanation: str, evidence: Mapping) -> List[str]:
    """Cheap guard: flag any specific claim in `explanation` not present in the
    `evidence` we sent. Returns a (possibly empty) list of human-readable flags.

    Two checks:
      1. Numbers — every number must trace back to an evidence number.
      2. Attributes — any garment-attribute word (material/colour/silhouette/fit)
         must appear in the evidence text.
    """
    flags: List[str] = []
    evidence_text = json.dumps(evidence, ensure_ascii=False).lower()
    expl = explanation.lower()

    allowed_nums = _allowed_numbers(evidence_text)
    for n in _numbers(expl):
        if not any(abs(n - a) < 0.5 for a in allowed_nums):
            flags.append(f"cites number {n:g} not found in the evidence")

    expl_words = set(_WORD_RE.findall(expl))
    for w in sorted(CLAIM_VOCAB & expl_words):
        if w not in evidence_text:
            flags.append(f"claims attribute '{w}' not found in the evidence")

    return flags


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #
def _default_item_id(item: RankedItem, i: int) -> str:
    return item.garment.url or f"item-{i + 1}"


def _parse_output(response) -> dict:
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _explain_one(
    item: RankedItem,
    profile: UserProfile,
    item_id: str,
    model: str,
    client,
    usage: dict,
) -> Explanation:
    """One grounded explanation. Accumulates token usage into `usage`."""
    evidence = build_evidence(item, profile, item_id)
    user_msg = json.dumps(evidence, ensure_ascii=False, indent=2)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=EXPLAIN_TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        output_config={"format": {"type": "json_schema", "schema": _OUT_SCHEMA}},
    )
    cost_mod.add_response_usage(usage, response)

    data = _parse_output(response)
    explanation = (data.get("explanation") or "").strip()
    lead_reason = data.get("lead_reason")
    confidence_note = data.get("confidence_note") or None

    # Guard both fields — the caveat note is model-written prose too, and can
    # invent a specific just as easily as the explanation can.
    checked = " ".join(t for t in (explanation, confidence_note) if t)
    flags = list(check_grounding(checked, evidence)) if explanation else ["empty explanation"]

    # Lead-reason sanity: it must name a sub-score we actually know about.
    if lead_reason not in LEAD_REASONS:
        flags.append(f"lead_reason {lead_reason!r} is not a known dimension")
        lead_reason = _lead_reason(item) or "budget"
    else:
        sub = item.subscores.get(lead_reason)
        if sub is None or not sub.known:
            flags.append(f"lead_reason '{lead_reason}' claims a signal that has no data")

    return Explanation(
        item_id=item_id,
        explanation=explanation,
        confidence_note=confidence_note,
        lead_reason=lead_reason,
        flags=tuple(flags),
    )


def explain_items(
    items: Sequence[RankedItem],
    profile: UserProfile = DEFAULT_PROFILE,
    model: str = HAIKU_MODEL,
    top_n: int = 10,
    item_id: Optional[Callable[[RankedItem, int], str]] = None,
) -> Tuple[List[Explanation], dict]:
    """Explain the ranker's top items, one grounded LLM call each.

    Returns (explanations, cost). Each explanation carries `flags` listing any
    claim it makes that isn't backed by the evidence — empty means faithful.
    """
    chosen = list(items)[:top_n]
    if not chosen:
        return [], cost_mod.summarize(model, cost_mod.empty_usage())
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY must be set for explanation generation.")

    import anthropic

    client = anthropic.Anthropic()
    id_fn = item_id or _default_item_id
    usage = cost_mod.empty_usage()
    out = [
        _explain_one(item, profile, id_fn(item, i), model, client, usage)
        for i, item in enumerate(chosen)
    ]

    cost = cost_mod.summarize(model, usage)
    cost["backend"] = "explain (haiku)"
    cost["items"] = len(out)
    return out, cost
