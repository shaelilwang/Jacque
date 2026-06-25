"""Product search via SerpApi's Google Shopping API.

Replaces the Serper.dev web-search harness. Google Shopping returns product
metadata directly — title, price, merchant, thumbnail — so there's no URL
filtering, site scoping, or og:image scraping to do anymore: every result is
already a buyable product. Results are localized to the US and can be bounded
by an advanced price filter.

Pipeline elsewhere is unchanged: image -> Sonnet describes items+style -> query.
This module consumes that query text and returns Google Shopping products.

Setup (you supply):
  - Create a SerpApi account, copy your API key -> SERPAPI_API_KEY
Docs: https://serpapi.com/google-shopping-api
"""

import os

import requests

import cost as cost_mod

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# US localization. Google Shopping listings, availability, and prices are
# region-specific, so we pin country / language / domain explicitly.
US_LOCALE = {
    "gl": "us",
    "hl": "en",
    "google_domain": "google.com",
    "location": "United States",
}


def _price_param(value):
    """Normalize a price bound for SerpApi, which requires an integer > 0.

    Coerces floats to a rounded int; returns None for unset, non-numeric, or
    non-positive values so the param is simply omitted.
    """
    if value is None:
        return None
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _to_product(item):
    """Map one SerpApi `shopping_results` entry to our flat product shape.

    Google Shopping links point at the Google product page (`product_link`);
    `source` is the merchant (e.g. "Nordstrom"), `thumbnail` the product image.
    """
    return {
        "title": item.get("title", ""),
        "url": item.get("product_link") or item.get("link") or "",
        "price": item.get("price", ""),
        "extracted_price": item.get("extracted_price"),  # numeric, for sorting/filtering
        "site": item.get("source", ""),                  # merchant name
        "image": item.get("thumbnail", ""),
        "rating": item.get("rating"),
        "reviews": item.get("reviews"),
        "delivery": item.get("delivery", ""),
        "product_id": item.get("product_id", ""),
    }


def search_shopping(query, max_results=10, min_price=None, max_price=None):
    """Search Google Shopping (US) via SerpApi.

    `min_price` / `max_price` (USD) apply Google Shopping's advanced price
    filter; either bound is optional. Returns (products, cost), where each
    product is a flat dict (see `_to_product`).
    """
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError("SERPAPI_API_KEY must be set.")
    if not query or not query.strip():
        raise ValueError("Query is required.")

    params = {
        "engine": "google_shopping",
        "q": query.strip(),
        "api_key": key,
        **US_LOCALE,
    }
    # Advanced price filter — dedicated min_price / max_price params. SerpApi
    # requires each to be an integer > 0, so coerce and drop anything else.
    lo, hi = _price_param(min_price), _price_param(max_price)
    if lo is not None:
        params["min_price"] = lo
    if hi is not None:
        params["max_price"] = hi

    resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"SerpApi returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    results = data.get("shopping_results") or []
    products = [_to_product(it) for it in results if it.get("title")]
    # One API call == one billed SerpApi search.
    return products[:max_results], cost_mod.serpapi_summary(1)


def _agg_key(product):
    """Identity for de-duplicating a product across queries: prefer the Google
    product_id, else a normalized title, else the URL."""
    pid = product.get("product_id")
    if pid:
        return f"id:{pid}"
    title = (product.get("title") or "").lower().strip()
    return f"t:{title}" if title else f"u:{product.get('url', '')}"


def search_queries(queries, max_results_per_query=10, max_total=40,
                   min_price=None, max_price=None):
    """Run several queries through Google Shopping and aggregate the results.

    The analysis step produces 3-5 queries at varying specificity; this fans them
    out, then merges into one de-duplicated list (the same product can surface for
    more than one query). Each product records which queries matched it in
    `matched_queries` — provenance for the downstream reranker. Results keep
    first-seen order, capped at `max_total`.

    Returns (products, cost); cost counts one SerpApi search per non-empty query.
    """
    by_key = {}
    order = []
    searches = 0
    for q in queries or []:
        if not q or not q.strip():
            continue
        items, _ = search_shopping(
            q, max_results=max_results_per_query, min_price=min_price, max_price=max_price
        )
        searches += 1
        for p in items:
            key = _agg_key(p)
            if key in by_key:
                if q not in by_key[key]["matched_queries"]:
                    by_key[key]["matched_queries"].append(q)
            else:
                p = dict(p)
                p["matched_queries"] = [q]
                by_key[key] = p
                order.append(key)

    products = [by_key[k] for k in order][:max_total]
    return products, cost_mod.serpapi_summary(searches)
