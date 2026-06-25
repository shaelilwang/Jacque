"""Reranker: score Google Shopping candidates against the reference vibe.

One Sonnet vision call sees the REFERENCE (the analysis JSON describing the item
the user wants to match) and every candidate image — each tagged with its index
and price. It returns two per-dimension scores (vibe_match, versatility) kept
SEPARATE so the UI can re-weight them, never collapsed into one number, plus a
few glanceable verdict notes (❤️ win / 💔 miss / ❓ unsure).

vibe_match is the dominant signal: matching the reference FEEL is what the user
asked for. The judgment is visual, so only candidates with an image are scored;
results carry the candidate's original-list `index` for mapping back.

Thumbnails are fetched server-side and passed as base64 image blocks rather than
URLs: our fetch isn't bound by the API fetcher's URL-access policy, and the model
receives the actual image bytes instead of a link to chase. Candidates whose
image can't be downloaded are dropped.
"""

import base64
import concurrent.futures
import json
import os

import requests

import cost as cost_mod

SONNET_MODEL = "claude-sonnet-4-6"
# Cap images per rerank call to bound cost/latency.
MAX_CANDIDATES = 24

RERANK_SYSTEM = (
    "You are a sharp, slightly sassy personal stylist re-ranking shopping "
    "candidates against a REFERENCE item the user wants to match the VIBE of. "
    "Score every candidate on separate dimensions so the UI can re-weight them — "
    "never collapse them into one number.\n\n"
    "Rules:\n"
    "- vibe_match (0-100) is the DOMINANT signal: similarity of FEEL to the "
    "reference, not literal sameness. The user asked for THIS vibe.\n"
    "- versatility (0-100): how many outfits/occasions the item slots into.\n"
    "- Do NOT reward or penalize price; budget is already satisfied. Mention price "
    "in a note only if it's a genuine standout value.\n"
    "- Penalize near-duplicates: if several candidates are nearly identical, do not "
    "give them all top vibe_match — spread the scores to preserve variety.\n"
    "- notes: 2-3 glanceable bullets, each tagged with a verdict — \"heart\" (a win: "
    "why it lands), \"heartbreak\" (a miss or real downside), or \"question\" "
    "(genuine uncertainty). Keep each to a short, clear, a-little-sassy phrase. Be "
    "honest, including when the vibe and the practicalities pull in opposite "
    "directions.\n\n"
    "Score each candidate using the `index` shown beside its image."
)

# Wrapped in an object because structured outputs key on an object root; the
# caller unwraps `scores` into the bare array.
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "the candidate's shown index"},
                    "vibe_match": {
                        "type": "integer",
                        "description": "0-100, similarity of feel to the reference (primary signal)",
                    },
                    "versatility": {
                        "type": "integer",
                        "description": "0-100, how many outfits/occasions it slots into",
                    },
                    "notes": {
                        "type": "array",
                        "description": "2-3 glanceable verdict bullets",
                        "items": {
                            "type": "object",
                            "properties": {
                                "verdict": {
                                    "type": "string",
                                    "enum": ["heart", "heartbreak", "question"],
                                    "description": "heart=win, heartbreak=miss/downside, question=unsure",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "short, clear, a-little-sassy phrase",
                                },
                            },
                            "required": ["verdict", "text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["index", "vibe_match", "versatility", "notes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


# Some retailer/CDN hosts serve a different (or no) response to non-browser
# user agents, so present as a browser when fetching thumbnails.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_EXT_TO_TYPE = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "gif": "image/gif", "webp": "image/webp",
}


def _media_type(content_type, url):
    """Pick an Anthropic-supported media type from the response header, falling
    back to the URL extension, then to JPEG."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _ALLOWED_IMAGE_TYPES:
        return ct
    ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
    return _EXT_TO_TYPE.get(ext, "image/jpeg")


def _fetch_image_b64(url, timeout=8, max_bytes=5_000_000):
    """Download a thumbnail and return (media_type, base64_data), or None on any
    failure (bad status, empty body, network error)."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _BROWSER_UA, "Accept": "image/*"},
            timeout=timeout,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.content:
        return None
    data = resp.content[:max_bytes]
    media_type = _media_type(resp.headers.get("Content-Type"), url)
    return media_type, base64.standard_b64encode(data).decode("ascii")


def _reference_block(reference):
    """The reference shown to the model: the item attributes the user is matching."""
    if isinstance(reference, dict) and "item" in reference:
        return reference["item"]
    return reference


def rerank(reference, candidates, model=SONNET_MODEL, max_candidates=MAX_CANDIDATES):
    """Score candidates against the reference vibe with one Sonnet vision call.

    `reference` is the analysis dict (or its `item` block). `candidates` is the
    aggregated product list; only those with an http(s) image are scored. Returns
    (scores, cost) — `scores` is the bare array of per-candidate score objects
    (vibe_match, versatility, glanceable notes), each carrying the candidate's
    original-list `index`.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY must be set for reranking.")

    # Only image-bearing candidates can be visually scored; keep original indices.
    with_images = [
        (i, c) for i, c in enumerate(candidates)
        if str(c.get("image", "")).startswith("http")
    ][:max_candidates]
    if not with_images:
        return [], cost_mod.summarize(model, cost_mod.empty_usage())

    # Fetch each thumbnail ourselves (concurrently) and send the bytes as base64.
    # ThreadPoolExecutor.map preserves input order; drop any we couldn't fetch.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        fetched = list(ex.map(lambda ic: _fetch_image_b64(ic[1]["image"]), with_images))
    usable = [(i, c, img) for (i, c), img in zip(with_images, fetched) if img]
    if not usable:
        return [], cost_mod.summarize(model, cost_mod.empty_usage())

    import anthropic

    client = anthropic.Anthropic()

    content = [{
        "type": "text",
        "text": (
            "REFERENCE (match the vibe of this item):\n"
            + json.dumps(_reference_block(reference), ensure_ascii=False, indent=2)
            + f"\n\nScore each of the {len(usable)} candidate image(s) below; use the "
            "index shown beside each."
        ),
    }]
    for idx, c, (media_type, b64) in usable:
        price = c.get("price") or "unknown"
        content.append({"type": "text", "text": f"[{idx}] price: {price}"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=RERANK_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": _SCORE_SCHEMA}},
    )

    usage = cost_mod.empty_usage()
    cost_mod.add_response_usage(usage, response)
    cost = cost_mod.summarize(model, usage)
    cost["backend"] = "rerank (sonnet)"

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        scores = (json.loads(text) or {}).get("scores") or []
    except (json.JSONDecodeError, AttributeError):
        scores = []

    # Keep only scores for indices we actually sent.
    valid = {idx for idx, _, _ in usable}
    scores = [s for s in scores if isinstance(s.get("index"), int) and s["index"] in valid]
    return scores, cost
