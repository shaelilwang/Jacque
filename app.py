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


def _llm_assist(image_bytes, media_type):
    """Ask Anthropic Sonnet to describe the image's items and style separately.

    Returns (items, style). Used to auto-fill the Channel3 search query.
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
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This image is a sample for a product shopping search. "
                            "Describe it for that purpose. Return two separate "
                            "descriptions: (1) the items/products shown in the image, "
                            "and (2) the overall visual style and aesthetic of the "
                            "image (colors, materials, vibe)."
                        ),
                    },
                ],
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "string",
                            "description": "The items or products visible in the image.",
                        },
                        "style": {
                            "type": "string",
                            "description": "The visual style/aesthetic of the sample image.",
                        },
                    },
                    "required": ["items", "style"],
                    "additionalProperties": False,
                },
            }
        },
    )

    usage = cost_mod.empty_usage()
    cost_mod.add_response_usage(usage, response)

    # output_config.format guarantees the first text block is valid JSON.
    text = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(text)
    return data["items"], data["style"], cost_mod.summarize(SONNET_MODEL, usage)


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
    """Text-assist step: Sonnet describes the image's items + style, which the
    frontend drops into the query box before searching Channel3."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY is not set on the server."}), 500

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    media_type = file.mimetype or "image/jpeg"
    try:
        items, style, cost = _llm_assist(file.read(), media_type)
    except Exception as e:
        return jsonify({"error": f"LLM assist failed: {e}"}), 502

    cost["backend"] = "assist (sonnet)"
    query = f"{items} Style: {style}"
    return jsonify({"items": items, "style": style, "query": query, "cost": cost})


@app.route("/api/search_scoped", methods=["POST"])
def search_scoped():
    """Harness backend: search YOUR scoped sites (sites.txt) via the Serper.dev
    search API instead of Channel3. Consumes the query text."""
    if not os.environ.get("SERPER_API_KEY"):
        return jsonify(
            {"error": "SERPER_API_KEY must be set on the server."}
        ), 500

    query = (request.form.get("query") or "").strip()
    if not query:
        return jsonify(
            {"error": "No query. Use 'Fill description' or type a description first."}
        ), 400

    try:
        from harness import load_sites, search_sites

        sites = load_sites()
        products, cost = search_sites(query, sites)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Scoped search failed: {e}"}), 502

    cost["backend"] = "scoped (serper)"

    # Reshape to the frontend's product shape {title,brand,price,domain,image,url}.
    def _shape(p):
        return {
            "title": p.get("title", ""),
            "brand": "",
            "price": p.get("price", ""),
            "domain": p.get("site", ""),
            "image": p.get("image", ""),
            "url": p.get("url", ""),
        }

    # Optional: rank the candidates against the user profile (extra LLM cost).
    if request.form.get("rank") and products:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return jsonify(
                {"error": "ANTHROPIC_API_KEY must be set on the server to rank."}
            ), 500
        try:
            from extract import rank_candidates

            ranked, rank_cost = rank_candidates(products, target=query)
        except Exception as e:
            return jsonify({"error": f"Ranking failed: {e}"}), 502

        by_url = {p.get("url", ""): p for p in products}
        norm = []
        for item in ranked:
            base = _shape(by_url.get(item.garment.url, {"title": item.garment.title,
                                                        "url": item.garment.url}))
            base["ranking"] = _serialize_ranking(item)
            norm.append(base)
        return jsonify({"products": norm, "cost": cost, "rank_cost": rank_cost})

    return jsonify({"products": [_shape(p) for p in products], "cost": cost})


def _serialize_ranking(item):
    """JSON-friendly ranking payload for one RankedItem."""
    return {
        "overall": item.overall,
        "subscores": {
            name: {"value": s.value, "confidence": s.confidence, "reason": s.reason}
            for name, s in item.subscores.items()
        },
    }


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
