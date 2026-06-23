"""LLM extraction step: turn raw search results into structured `Garment`s.

This is the only non-pure part of ranking — it calls a cheap model (Haiku) to
read each candidate's text and pull out garment type, silhouette/aesthetic tags,
material/colour, any stated dimensions, and whether it actually *is* the target
item, each with a confidence. Anything not stated comes back `None`/empty so the
deterministic scorer in `ranker.py` can represent missing data honestly.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Sequence, Tuple

import cost as cost_mod
from profiles import DEFAULT_PROFILE, UserProfile
from ranker import (
    DEFAULT_WEIGHTS,
    SILHOUETTE_VOCAB,
    Garment,
    GarmentDimensions,
    RankedItem,
    RankingWeights,
    rank,
)

HAIKU_MODEL = "claude-haiku-4-5"

_MONEY_RE = re.compile(r"[$£€]?\s?(\d[\d,]*(?:\.\d{1,2})?)")

_DIM_PROPS = {
    "chest_cm": {"type": ["number", "null"]},
    "waist_cm": {"type": ["number", "null"]},
    "hip_cm": {"type": ["number", "null"]},
    "length_cm": {"type": ["number", "null"]},
    "inseam_cm": {"type": ["number", "null"]},
}

_GARMENT_PROPS = {
    "index": {"type": "integer", "description": "0-based index of the input candidate"},
    "is_target": {
        "type": ["boolean", "null"],
        "description": "Whether this item IS the target garment the shopper wants.",
    },
    "is_target_confidence": {"type": "number"},
    "garment_type": {"type": ["string", "null"]},
    "silhouette": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Line/fit descriptors; prefer the provided vocabulary.",
    },
    "aesthetic": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Vibe descriptors; reuse the shopper's taste words when they apply.",
    },
    "material": {"type": ["string", "null"]},
    "color": {"type": ["string", "null"]},
    "dimensions": {
        "type": "object",
        "properties": _DIM_PROPS,
        "required": list(_DIM_PROPS),
        "additionalProperties": False,
    },
    "attribute_confidence": {"type": "number"},
    "dimensions_confidence": {"type": "number"},
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "garments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _GARMENT_PROPS,
                "required": list(_GARMENT_PROPS),
                "additionalProperties": False,
            },
        }
    },
    "required": ["garments"],
    "additionalProperties": False,
}


def parse_price_usd(price: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    """Parse a price string like '$129.00' into (amount, currency). ('' => None)."""
    if not price:
        return None, None
    currency = "USD" if "$" in price else "GBP" if "£" in price else "EUR" if "€" in price else None
    m = _MONEY_RE.search(price)
    if not m:
        return None, currency
    try:
        return float(m.group(1).replace(",", "")), currency
    except ValueError:
        return None, currency


def _build_prompt(target: str, profile: UserProfile, results: Sequence[dict]) -> str:
    lines = [
        "You are extracting structured attributes from fashion product search results "
        "so a downstream system can score them. Be precise and DO NOT invent data: if "
        "something isn't stated or clearly implied, use null (or an empty list).",
        "",
        f"TARGET ITEM the shopper wants: {target!r}",
        "",
        "For each candidate set:",
        "- is_target: true only if the candidate clearly IS that target garment type "
        "(not an accessory/adjacent item). Use is_target_confidence to express doubt.",
        "- silhouette: line/fit descriptors. Prefer these where they apply: "
        + ", ".join(SILHOUETTE_VOCAB),
        "- aesthetic: vibe descriptors. Reuse the shopper's taste words where they truly "
        "apply: " + ", ".join(profile.taste),
        "- dimensions: only fill a measurement if an actual number is present in the text; "
        "otherwise null. Set dimensions_confidence accordingly (usually low/0).",
        "- attribute_confidence: how confident you are in the silhouette/aesthetic/type tags.",
        "",
        "CANDIDATES:",
    ]
    for i, r in enumerate(results):
        lines.append(
            f"[{i}] title={r.get('title','')!r} site={r.get('site','')!r} "
            f"price={r.get('price','') or 'unknown'!r} snippet={r.get('snippet','')!r}"
        )
    return "\n".join(lines)


def _to_garment(result: dict, attrs: dict) -> Garment:
    price_usd, currency = parse_price_usd(result.get("price"))
    dims = attrs.get("dimensions") or {}
    return Garment(
        title=result.get("title", ""),
        url=result.get("url", ""),
        price_usd=price_usd,
        currency=currency,
        is_target=attrs.get("is_target"),
        is_target_confidence=float(attrs.get("is_target_confidence") or 0.0),
        garment_type=attrs.get("garment_type"),
        silhouette=tuple(attrs.get("silhouette") or ()),
        aesthetic=tuple(attrs.get("aesthetic") or ()),
        material=attrs.get("material"),
        color=attrs.get("color"),
        dimensions=GarmentDimensions(
            chest_cm=dims.get("chest_cm"),
            waist_cm=dims.get("waist_cm"),
            hip_cm=dims.get("hip_cm"),
            length_cm=dims.get("length_cm"),
            inseam_cm=dims.get("inseam_cm"),
        ),
        attribute_confidence=float(attrs.get("attribute_confidence") or 0.0),
        dimensions_confidence=float(attrs.get("dimensions_confidence") or 0.0),
    )


def extract_garments(
    results: Sequence[dict],
    target: str,
    profile: UserProfile = DEFAULT_PROFILE,
    model: str = HAIKU_MODEL,
) -> Tuple[List[Garment], dict]:
    """Extract `Garment`s from search-result dicts via one batched Haiku call.

    Returns (garments, cost). Garments without LLM attributes still carry the
    deterministic fields (title/url/price) with everything else honestly empty.
    """
    if not results:
        return [], cost_mod.summarize(model, cost_mod.empty_usage())
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY must be set for garment extraction.")

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": _build_prompt(target, profile, results)}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )

    usage = cost_mod.empty_usage()
    cost_mod.add_response_usage(usage, response)
    cost = cost_mod.summarize(model, usage)
    cost["backend"] = "extract (haiku)"

    text = next((b.text for b in response.content if b.type == "text"), "")
    by_index = {}
    try:
        for item in (json.loads(text).get("garments") or []):
            if isinstance(item.get("index"), int):
                by_index[item["index"]] = item
    except (json.JSONDecodeError, AttributeError):
        by_index = {}

    garments = [_to_garment(r, by_index.get(i, {})) for i, r in enumerate(results)]
    return garments, cost


def rank_candidates(
    results: Sequence[dict],
    target: str,
    profile: UserProfile = DEFAULT_PROFILE,
    weights: RankingWeights = DEFAULT_WEIGHTS,
    model: str = HAIKU_MODEL,
) -> Tuple[List[RankedItem], dict]:
    """Full pipeline: extract garment attributes, then rank deterministically."""
    garments, cost = extract_garments(results, target, profile, model)
    return rank(garments, profile, weights), cost
