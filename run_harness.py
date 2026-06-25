"""CLI to exercise the Google Shopping search harness without the web UI.

Usage:
    .venv/bin/python run_harness.py "red waterproof trail running shoes"
    .venv/bin/python run_harness.py "merino wool sweater" --max-price 150
    .venv/bin/python run_harness.py "leather chelsea boots" --min-price 80 --max-price 250

Requires SERPAPI_API_KEY in the environment (or .env).
"""

import argparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from harness import search_shopping


def main():
    parser = argparse.ArgumentParser(description="Search Google Shopping (US) via SerpApi.")
    parser.add_argument("query", nargs="+", help="What to search for.")
    parser.add_argument("--min-price", type=float, default=None, help="Lowest price (USD).")
    parser.add_argument("--max-price", type=float, default=None, help="Highest price (USD).")
    args = parser.parse_args()

    query = " ".join(args.query)
    bounds = " ".join(
        x for x in (
            f"min ${args.min_price:g}" if args.min_price is not None else "",
            f"max ${args.max_price:g}" if args.max_price is not None else "",
        ) if x
    )
    print(f"Query: {query}" + (f"  (price: {bounds})" if bounds else "") + "\n")

    products, cost = search_shopping(
        query, min_price=args.min_price, max_price=args.max_price
    )

    if not products:
        print("No products found.")
    for i, p in enumerate(products, 1):
        title = p.get("title", "(untitled)")
        meta = " — ".join(x for x in (p.get("site", ""), p.get("price", "")) if x)
        print(f"{i}. {title}" + (f"  [{meta}]" if meta else ""))
        print(f"   {p.get('url', '')}")

    print(
        f"\nCost: ${cost['cost_usd']:.5f}  "
        f"({cost['web_searches']} search request(s), backend {cost['model']})"
    )


if __name__ == "__main__":
    main()
