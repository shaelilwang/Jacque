"""CLI to exercise the scoped-site search harness without the web UI.

Usage:
    .venv/bin/python run_harness.py "red waterproof trail running shoes, earthy palette"

Requires ANTHROPIC_API_KEY in the environment (or .env) and a sites.txt list.
"""

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from harness import load_sites, search_sites


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_harness.py "<query>"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    sites = load_sites()
    print(f"Scoped to {len(sites)} site(s): {', '.join(sites)}")
    print(f"Query: {query}\n")

    products, cost = search_sites(query, sites)

    if not products:
        print("No products found.")
    for i, p in enumerate(products, 1):
        title = p.get("title", "(untitled)")
        price = p.get("price", "")
        site = p.get("site", "")
        url = p.get("url", "")
        meta = " — ".join(x for x in (site, price) if x)
        print(f"{i}. {title}" + (f"  [{meta}]" if meta else ""))
        print(f"   {url}")

    print(
        f"\nCost: ${cost['cost_usd']:.5f}  "
        f"(in {cost['input_tokens']}, out {cost['output_tokens']}, "
        f"{cost['web_searches']} web search(es), model {cost['model']})"
    )


if __name__ == "__main__":
    main()
