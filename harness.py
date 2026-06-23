"""Scoped-site search harness — Serper.dev edition.

Instead of Channel3 (or Anthropic web_search), this searches ONLY a list of
websites you supply (`sites.txt`) using Serper.dev's Google Search API, scoped
via `site:` operators and paginated via the `page` param. Retrieval is pure
HTTP — no LLM tokens — so it's cheap and supports real pagination.

(Replaces the Google Custom Search JSON API, which Google closed to new
customers in Jan 2026 — new projects get a 403 even when the API is "enabled".)

Pipeline elsewhere is unchanged: image -> Sonnet describes items+style -> query.
This module consumes that query text and returns product links from the sites.

Setup (you supply):
  - Create a Serper.dev account, copy your API key -> SERPER_API_KEY
"""

import concurrent.futures
import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests

import cost as cost_mod

SERPER_ENDPOINT = "https://google.serper.dev/search"
DEFAULT_SITES_FILE = os.path.join(os.path.dirname(__file__), "sites.txt")
# Serper returns up to ~10 organic results per page; paginate with `page`. Each
# page is one billed query. The product filter thins each page, so fetch a few
# pages to keep enough buyable results flowing through. 4 pages = up to 40 raw.
# (An `inurl:product` query bias was tried but returns nothing when combined
# with the grouped `site:` OR clause, so we filter post-hoc instead.)
MAX_PAGES = 4
# Serper's free tier rate-limits rapid bursts, so pace pages slightly and retry
# transient 403/429s with backoff before giving up.
SERPER_PAGE_DELAY = 0.4
SERPER_MAX_RETRIES = 2

# Keep only individual *buyable* product pages — drop homepages, category /
# collection listings, and editorial (magazine/lookbook/stylebook) content.
# A URL is treated as a product page if its path matches one of these markers.
# Extend this list if you add a retailer that uses a different URL scheme.
PRODUCT_URL_PATTERNS = [
    r"/products?[/.]",    # /products/, /product/ (Everlane, Uniqlo, COS, SSENSE)
    r"/product-\d",       # /product-12345 numeric ids (not /product-care etc.)
    r"/productpage",      # COS / H&M-style productpage.<id>.html
    r"/p/",               # many retailers
    r"/dp/",              # Amazon-style
    r"/pd/",
    r"/item(s)?[/.]",
    r"/itm/",
    r"/goods/",
    r"-p-?\d",            # slug-p-12345 style product ids
]
# Even with a product marker, never treat these editorial/listing paths as buyable.
NON_PRODUCT_URL_PATTERNS = [
    r"/collections?/",
    r"/contents?/",
    r"/magazine",
    r"/lifewear",
    r"/stylingbook",
    r"/stylehint",
    r"/lookbook",
    r"/feature",
    r"/stor(y|ies)",
    r"/journal",
    r"/blog",
    r"/guide",
    r"/product-care",     # care guides, not products
    r"/reviews?\b",       # product reviews subpage, not the buy page
]
_PRODUCT_RE = re.compile("|".join(PRODUCT_URL_PATTERNS), re.I)
_NON_PRODUCT_RE = re.compile("|".join(NON_PRODUCT_URL_PATTERNS), re.I)


def _is_buyable(url):
    """Heuristic: True only for individual product (buyable) pages.

    Drops homepages and category/editorial pages so results are things you can
    actually add to cart, not catalogs.
    """
    path = urlparse(url).path
    if not path or path == "/":
        return False  # homepage
    if _NON_PRODUCT_RE.search(path):
        return False  # editorial / collection listing
    return bool(_PRODUCT_RE.search(path))


def _dedup_key(title, url):
    """A product identity key that collapses color/locale/brand variants.

    Retailers list the same product once per colorway and per country site, e.g.
    "The Day Glove | Black - Everlane" vs "...| Canvas - Everlane", or the same
    item on /ph/ and /my/. We key on the product name only: the title up to the
    first ` | ` or ` - ` (which separate color/brand/region tails), normalized.
    """
    name = title or url
    name = re.split(r"\s*\|\s*", name, 1)[0]   # drop "| Color - Brand" tail
    name = re.split(r"\s+-\s+", name, 1)[0]    # drop " - Color"/" - Brand" tail
    name = re.sub(r"[®™]", "", name)
    name = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return name or url.lower()


# --- Product images -------------------------------------------------------
# Serper returns no images, so we read each product page's og:image (the
# social-preview image retailers embed in their <head>). Fetches run
# concurrently with a tight timeout, and results are cached across searches.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.I)
_OG_IMAGE_PROP_RE = re.compile(
    r"""(?:property|name)\s*=\s*["'](?:og:image(?::secure_url|:url)?|twitter:image)["']""",
    re.I,
)
_CONTENT_RE = re.compile(r"""content\s*=\s*["']([^"']+)["']""", re.I)
_image_cache = {}  # url -> image url (or "")


def _parse_og_image(html, base_url):
    """Pull the og:image / twitter:image URL from a page's <head> markup."""
    head = html[:200_000]  # meta tags live near the top; cap work
    for tag in _META_TAG_RE.findall(head):
        if _OG_IMAGE_PROP_RE.search(tag):
            m = _CONTENT_RE.search(tag)
            if m and m.group(1).strip():
                img = urljoin(base_url, m.group(1).strip())
                # Upgrade to https so images load on an https-served frontend.
                return re.sub(r"^http://", "https://", img)
    return ""


def _fetch_og_image(url, timeout=5):
    """Best-effort product image for one URL (cached). Returns "" on any failure."""
    if url in _image_cache:
        return _image_cache[url]
    image = ""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _BROWSER_UA, "Accept": "text/html"},
            timeout=timeout,
            stream=True,
        )
        if resp.status_code == 200:
            # Read only the first chunk — enough to reach the <head> meta tags.
            chunk = next(resp.iter_content(200_000, decode_unicode=True), "")
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "ignore")
            image = _parse_og_image(chunk, url)
        resp.close()
    except requests.RequestException:
        image = ""
    _image_cache[url] = image
    return image


def attach_images(products, max_workers=8):
    """Populate each product's `image` by fetching og:image concurrently."""
    todo = [p for p in products if not p.get("image")]
    if not todo:
        return products
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_og_image, p["url"]): p for p in todo}
        for fut in concurrent.futures.as_completed(futures):
            try:
                futures[fut]["image"] = fut.result()
            except Exception:
                futures[fut]["image"] = ""
    return products


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


# A real price has a currency symbol — this avoids grabbing stray numbers like
# delivery thresholds ("Free delivery over $100") or rating counts.
_MONEY_RE = re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d{1,2})?")


def _extract_price(item):
    """Best-effort price from a Serper organic result (often absent for web
    results). Only accepts a currency-prefixed amount to avoid junk."""
    attrs = item.get("attributes") or {}
    for cand in (
        item.get("price"),
        item.get("priceRange"),
        attrs.get("Price"),
        attrs.get("price"),
    ):
        if cand:
            m = _MONEY_RE.search(str(cand))
            if m:
                return m.group(0)
    return ""


def _serper_post(key, q, page, num=10):
    """POST one search page, retrying transient 403/429 rate limits with backoff."""
    resp = None
    for attempt in range(SERPER_MAX_RETRIES + 1):
        resp = requests.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": q, "num": num, "page": page},
            timeout=20,
        )
        if resp.status_code == 200 or resp.status_code not in (403, 429):
            return resp
        if attempt < SERPER_MAX_RETRIES:
            time.sleep(0.8 * (attempt + 1))
    return resp


def serper_search(query, sites, max_results=10, max_pages=MAX_PAGES,
                  products_only=True):
    """Query Serper.dev's Google Search API, scoped to `sites`, with paging.

    With `products_only` (default), only individual buyable product pages are
    kept — homepages, collection listings, and editorial content are dropped.

    Returns (products, requests_made). Each element is {title, url, price, site}.
    """
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY must be set.")
    if not sites:
        raise ValueError("No sites configured — add domains to sites.txt.")

    # Scope to the supplied sites with `site:` operators.
    site_clause = " OR ".join(f"site:{d}" for d in sites)
    q = f"{query} ({site_clause})"

    products = []
    seen_keys = set()  # collapse color/locale/brand variants of the same product
    requests_made = 0
    page = 1
    # Filtering thins each page, so request full pages (10) and trim at the end.
    while len(products) < max_results and requests_made < max_pages:
        if page > 1:
            time.sleep(SERPER_PAGE_DELAY)  # pace pages under the free-tier limit
        resp = _serper_post(key, q, page)
        requests_made += 1
        if resp.status_code != 200:
            # Already have results? Degrade gracefully rather than failing the
            # whole search on a transient later-page rate limit.
            if products:
                break
            raise RuntimeError(
                f"Serper returned {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        organic = data.get("organic", []) or []
        for it in organic:
            link = it.get("link", "")
            # Keep only individual product pages (drop homepages/listings/editorial).
            if products_only and not _is_buyable(link):
                continue
            # Skip color/locale/brand variants of a product we already have.
            dedup_key = _dedup_key(it.get("title", ""), link)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            domain = re.sub(r"^https?://", "", link).split("/")[0]
            products.append(
                {
                    "title": it.get("title", ""),
                    "url": link,
                    "site": domain,
                    "price": _extract_price(it),
                }
            )
        if len(organic) < 10:
            break  # no more results
        page += 1

    return products[:max_results], requests_made


def search_sites(query, sites, max_results=10, with_images=True):
    """Search the scoped sites for products matching `query`.

    Returns (products, cost). Raises if the Serper key or sites are missing.
    When `with_images`, each product is enriched with its og:image URL.
    """
    products, requests_made = serper_search(query, sites, max_results)
    if with_images:
        attach_images(products)
    return products, cost_mod.serper_summary(requests_made)
