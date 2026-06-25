"""Jacque — upload an image, search Channel3, get product buy links.

Backend: a tiny Flask server that serves the frontend and exposes /api/search.
The browser uploads an image; we base64-encode it and forward it to the
Channel3 search API, then normalize the response into a flat list of
{title, brand, price, image, url} for the frontend to render.
"""

import base64
import json
import os

import requests
from flask import Flask, jsonify, request, send_from_directory

import cost as cost_mod

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

CHANNEL3_SEARCH_URL = "https://api.trychannel3.com/v1/search"
SONNET_MODEL = "claude-sonnet-4-6"

app = Flask(__name__, static_folder="static", static_url_path="")


def _api_key():
    return os.environ.get("CHANNEL3_API_KEY")


# One Sonnet vision call turns an uploaded item photo into: structured
# attributes, a context-free versatility score, and the shopping queries that
# fan out to Google Shopping. "Vibe" matters more than an exact match.
ANALYSIS_PROMPT = (
    "You are a fashion editor reading a garment from an image so a shopper can "
    "find items with the same VIBE — not the exact product. Speak in terse "
    "editorial shorthand: lowercase, fragments, no full sentences, no hedging "
    "words ('likely', 'appears', 'probably'). Think fashion-caption caveman talk.\n\n"
    "Return:\n"
    "- category: the item in 1-3 words, e.g. 'crop top', 'ankle boot'.\n"
    "- color: its dominant color in a word or two, e.g. 'black', 'washed indigo'.\n"
    "- material: best visual read of the fabric, terse, e.g. 'stretch jersey'. "
    "If genuinely unclear, a one-word guess with a '?' is fine.\n"
    "- lines: 2-4 punchy fragments for cut, fit, and aesthetic, e.g. 'asymmetric', "
    "'body-skimming', '90s minimalism'.\n"
    "- occasions: 2-4 places/moments it wears best, e.g. 'evening', 'gallery', 'city'.\n"
    "- queries: 3-5 shopping search strings at VARYING specificity (one tight, one "
    "broad, one aesthetic-led). No price, brand, or size in any query.\n\n"
    "Vibe over exact match."
)

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "item": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "the item in 1-3 words, e.g. 'crop top'"},
                "color": {"type": "string", "description": "dominant color, a word or two, e.g. 'black'"},
                "material": {
                    "type": "string",
                    "description": "terse fabric read, e.g. 'stretch jersey'; one-word guess with '?' if unclear",
                },
                "lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 punchy cut/fit/aesthetic fragments, e.g. 'asymmetric', 'body-skimming', '90s minimalism'",
                },
                "occasions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 places/moments it wears best, e.g. 'evening', 'gallery', 'city'",
                },
            },
            "required": ["category", "color", "material", "lines", "occasions"],
            "additionalProperties": False,
        },
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 shopping search strings at varying specificity "
                           "(one tight, one broad, one aesthetic-led); no price/brand/size",
        },
    },
    "required": ["item", "queries"],
    "additionalProperties": False,
}


def analyze_image(image_bytes, media_type):
    """One Sonnet vision call analyzing the uploaded item.

    Returns (analysis, cost), where `analysis` matches ANALYSIS_SCHEMA: item
    attributes, a versatility_base score, and 3-5 shopping queries.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
    )

    usage = cost_mod.empty_usage()
    cost_mod.add_response_usage(usage, response)

    # output_config.format guarantees the first text block is valid JSON.
    text = next((b.text for b in response.content if b.type == "text"), "")
    return json.loads(text), cost_mod.summarize(SONNET_MODEL, usage)


def _format_price(offer):
    """Channel3 offer price may be a number or an object; render defensively."""
    price = offer.get("price")
    if isinstance(price, dict):
        amount = price.get("price", price.get("amount"))
        currency = price.get("currency", "")
        return f"{currency} {amount}".strip() if amount is not None else ""
    if price is not None:
        return str(price)
    return ""


def _normalize(products):
    items = []
    for p in products:
        brands = p.get("brands") or []
        brand = brands[0].get("name") if brands else ""

        images = p.get("images") or []
        image = ""
        if images:
            main = next((i for i in images if i.get("is_main_image")), images[0])
            image = main.get("url", "")

        # The buy link + price live on offers; take the cheapest/first offer.
        offers = p.get("offers") or []
        url, price, domain = "", "", ""
        if offers:
            offer = offers[0]
            url = offer.get("url", "")
            domain = offer.get("domain", "")
            price = _format_price(offer)

        items.append(
            {
                "title": p.get("title", ""),
                "brand": brand,
                "price": price,
                "domain": domain,
                "image": image,
                "url": url,
            }
        )
    return items


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/assist", methods=["POST"])
def assist():
    """One Sonnet call analyzes the uploaded image: returns item attributes, a
    versatility_base score, and 3-5 shopping queries (no search yet)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY is not set on the server."}), 500

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    media_type = file.mimetype or "image/jpeg"
    try:
        analysis, cost = analyze_image(file.read(), media_type)
    except Exception as e:
        return jsonify({"error": f"Image analysis failed: {e}"}), 502

    cost["backend"] = "analyze (sonnet)"
    return jsonify({"analysis": analysis, "cost": cost})


@app.route("/api/discover", methods=["POST"])
def discover():
    """Image pipeline: one Sonnet call analyzes the uploaded item, then its 3-5
    queries are searched on Google Shopping (SerpApi) and aggregated. The
    aggregated products are the handoff to the reranker (built separately)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY is not set on the server."}), 500
    if not os.environ.get("SERPAPI_API_KEY"):
        return jsonify({"error": "SERPAPI_API_KEY is not set on the server."}), 500

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    media_type = file.mimetype or "image/jpeg"
    try:
        analysis, analyze_cost = analyze_image(file.read(), media_type)
    except Exception as e:
        return jsonify({"error": f"Image analysis failed: {e}"}), 502
    analyze_cost["backend"] = "analyze (sonnet)"

    min_price = _form_price(request.form, "min_price")
    max_price = _form_price(request.form, "max_price")
    try:
        from harness import search_queries

        products, search_cost = search_queries(
            analysis.get("queries", []), min_price=min_price, max_price=max_price
        )
    except Exception as e:
        return jsonify({"error": f"Google Shopping search failed: {e}"}), 502
    search_cost["backend"] = "google shopping (serpapi)"

    # Rerank the aggregated candidates against the reference vibe. Per-dimension
    # scores stay separate so the frontend can re-weight and re-sort them.
    try:
        from rerank import rerank

        scores, rerank_cost = rerank(analysis, products)
    except Exception as e:
        return jsonify({"error": f"Rerank failed: {e}"}), 502
    by_index = {s["index"]: s for s in scores}

    # matched_queries records which analysis queries surfaced each item; `score`
    # is the per-dimension rerank result (None when the item wasn't scored).
    norm = [
        {
            "title": p.get("title", ""),
            "brand": "",
            "price": p.get("price", ""),
            "domain": p.get("site", ""),
            "image": p.get("image", ""),
            "url": p.get("url", ""),
            "matched_queries": p.get("matched_queries", []),
            "score": by_index.get(i),
        }
        for i, p in enumerate(products)
    ]
    return jsonify({
        "analysis": analysis,
        "products": norm,
        "costs": [analyze_cost, search_cost, rerank_cost],
    })


@app.route("/api/search_for", methods=["POST"])
def search_for():
    """Step 2 of the image flow: given an analysis already produced by
    /api/assist, search Google Shopping for its queries and rerank — no
    re-analysis (so the summary the user reviewed matches what's searched)."""
    if not os.environ.get("SERPAPI_API_KEY"):
        return jsonify({"error": "SERPAPI_API_KEY is not set on the server."}), 500
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY is not set on the server."}), 500

    try:
        analysis = json.loads(request.form.get("analysis") or "{}")
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid analysis payload."}), 400
    if not isinstance(analysis, dict) or not analysis.get("queries"):
        return jsonify({"error": "Analysis has no queries to search."}), 400

    min_price = _form_price(request.form, "min_price")
    max_price = _form_price(request.form, "max_price")
    try:
        from harness import search_queries

        products, search_cost = search_queries(
            analysis["queries"], min_price=min_price, max_price=max_price
        )
    except Exception as e:
        return jsonify({"error": f"Google Shopping search failed: {e}"}), 502
    search_cost["backend"] = "google shopping (serpapi)"

    try:
        from rerank import rerank

        scores, rerank_cost = rerank(analysis, products)
    except Exception as e:
        return jsonify({"error": f"Rerank failed: {e}"}), 502
    by_index = {s["index"]: s for s in scores}

    norm = [
        {
            "title": p.get("title", ""),
            "brand": "",
            "price": p.get("price", ""),
            "domain": p.get("site", ""),
            "image": p.get("image", ""),
            "url": p.get("url", ""),
            "matched_queries": p.get("matched_queries", []),
            "score": by_index.get(i),
        }
        for i, p in enumerate(products)
    ]
    return jsonify({"products": norm, "costs": [search_cost, rerank_cost]})


def _form_price(form, key):
    """Parse an optional price bound from the form. Blank/invalid -> None."""
    raw = (form.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@app.route("/api/search_scoped", methods=["POST"])
def search_scoped():
    """Harness backend: search Google Shopping (US) via SerpApi instead of
    Channel3. Consumes the query text plus optional min/max price bounds."""
    if not os.environ.get("SERPAPI_API_KEY"):
        return jsonify(
            {"error": "SERPAPI_API_KEY must be set on the server."}
        ), 500

    query = (request.form.get("query") or "").strip()
    if not query:
        return jsonify(
            {"error": "No query. Use 'Fill description' or type a description first."}
        ), 400

    min_price = _form_price(request.form, "min_price")
    max_price = _form_price(request.form, "max_price")

    try:
        from harness import search_shopping

        products, cost = search_shopping(query, min_price=min_price, max_price=max_price)
    except Exception as e:
        return jsonify({"error": f"Google Shopping search failed: {e}"}), 502

    cost["backend"] = "google shopping (serpapi)"
    # Reshape to the frontend's product shape {title,brand,price,domain,image,url}.
    norm = [
        {
            "title": p.get("title", ""),
            "brand": "",
            "price": p.get("price", ""),
            "domain": p.get("site", ""),  # merchant name (SerpApi `source`)
            "image": p.get("image", ""),
            "url": p.get("url", ""),
        }
        for p in products
    ]
    return jsonify({"products": norm, "cost": cost})


@app.route("/api/search", methods=["POST"])
def search():
    api_key = _api_key()
    if not api_key:
        return jsonify({"error": "CHANNEL3_API_KEY is not set on the server."}), 500

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    b64 = base64.b64encode(file.read()).decode("utf-8")

    payload = {"base64_image": b64, "limit": 20}
    # Optional free-text refinement alongside the image.
    query = (request.form.get("query") or "").strip()
    if query:
        payload["query"] = query

    try:
        resp = requests.post(
            CHANNEL3_SEARCH_URL,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
    except requests.RequestException as e:
        return jsonify({"error": f"Request to Channel3 failed: {e}"}), 502

    if resp.status_code != 200:
        return (
            jsonify(
                {"error": f"Channel3 returned {resp.status_code}: {resp.text[:500]}"}
            ),
            502,
        )

    data = resp.json()
    products = data.get("products", [])
    cost = {
        "backend": "channel3",
        "cost_usd": None,
        "note": "Channel3 request cost not tracked",
    }
    return jsonify({"products": _normalize(products), "cost": cost})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
