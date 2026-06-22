"""Scoped-site search harness — Google Programmable Search edition.

Instead of Channel3 (or Anthropic web_search), this searches ONLY a list of
websites you supply (`sites.txt`) using the Google Custom Search JSON API,
scoped via `site:` operators and paginated via the `start` param. Retrieval is
pure HTTP — no LLM tokens — so it's cheap and supports real pagination.

Pipeline elsewhere is unchanged: image -> Sonnet describes items+style -> query.
This module consumes that query text and returns product links from the sites.

Setup (you supply):
  - Create a Programmable Search Engine set to "Search the entire web", get its
    Search engine ID (cx)  -> GOOGLE_CSE_ID
  - Create an API key for the Custom Search API           -> GOOGLE_API_KEY
"""

import os
import re

import requests

import cost as cost_mod

GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
DEFAULT_SITES_FILE = os.path.join(os.path.dirname(__file__), "sites.txt")
# Google returns up to 10 results per request; paginate with `start`. Each page
# is one billed query, so keep this small. 2 pages = up to 20 results.
MAX_PAGES = 2


def load_sites(path=DEFAULT_SITES_FILE):
    """Read the scoped website list. One domain per line; '#' comments allowed.

    URLs are normalized to bare domains (drop scheme + path).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy sites.example.txt to sites.txt and add your sites."
        )
    sites = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            domain = re.sub(r"^https?://", "", line).strip("/").split("/")[0]
            if domain:
                sites.append(domain)
    return sites


def _extract_price(item):
    """Best-effort price from a Google CSE result's pagemap (often absent)."""
    pm = item.get("pagemap", {}) or {}
    for key in ("offer", "product"):
        entries = pm.get(key) or []
        if entries:
            o = entries[0]
            amt = o.get("price") or o.get("amount")
            cur = o.get("pricecurrency") or o.get("currency") or ""
            if amt:
                return f"{cur} {amt}".strip()
    metas = pm.get("metatags") or []
    if metas:
        m = metas[0]
        amt = m.get("og:price:amount") or m.get("product:price:amount")
        cur = m.get("og:price:currency") or m.get("product:price:currency") or ""
        if amt:
            return f"{cur} {amt}".strip()
    return ""


def google_search(query, sites, max_results=10, max_pages=MAX_PAGES):
    """Query the Google Custom Search JSON API, scoped to `sites`, with paging.

    Returns (products, requests_made). Each element is {title, url, price, site}.
    """
    key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_ID")
    if not key or not cx:
        raise RuntimeError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set.")
    if not sites:
        raise ValueError("No sites configured — add domains to sites.txt.")

    # Scope to the supplied sites with `site:` operators (PSE must be set to
    # "Search the entire web" for these to apply).
    site_clause = " OR ".join(f"site:{d}" for d in sites)
    q = f"{query} ({site_clause})"

    products = []
    requests_made = 0
    start = 1
    while len(products) < max_results and requests_made < max_pages:
        num = min(10, max_results - len(products))
        resp = requests.get(
            GOOGLE_ENDPOINT,
            params={"key": key, "cx": cx, "q": q, "num": num, "start": start},
            timeout=20,
        )
        requests_made += 1
        if resp.status_code != 200:
            raise RuntimeError(
                f"Google CSE returned {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        for it in data.get("items", []):
            products.append(
                {
                    "title": it.get("title", ""),
                    "url": it.get("link", ""),
                    "site": it.get("displayLink", ""),
                    "price": _extract_price(it),
                }
            )
        next_page = (data.get("queries", {}) or {}).get("nextPage")
        if not next_page:
            break
        start = next_page[0].get("startIndex", start + num)

    return products[:max_results], requests_made


def search_sites(query, sites, max_results=10):
    """Search the scoped sites for products matching `query`.

    Returns (products, cost). Raises if the Google keys or sites are missing.
    """
    products, requests_made = google_search(query, sites, max_results)
    return products, cost_mod.google_summary(requests_made)
