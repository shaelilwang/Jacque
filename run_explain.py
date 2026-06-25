"""CLI demo: search, rank, then explain the top picks in a shopper's voice.

Usage:
    .venv/bin/python run_explain.py "minimalist white leather sneakers"

Runs the full pipeline (search -> extract -> rank) and then writes a grounded
"why this is a good pick" for each of the top items, surfacing any explanation
that makes a claim the evidence doesn't support.

Requires SERPER_API_KEY (search) and ANTHROPIC_API_KEY (extract + explain).
"""

from __future__ import annotations

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from explain import explain_items
from extract import rank_candidates
from harness import load_sites, search_sites
from profiles import DEFAULT_PROFILE
from ranker import DEFAULT_WEIGHTS

TOP_N = 10


def main(query: str) -> int:
    sites = load_sites()
    print(f"Searching {len(sites)} site(s) for: {query!r}\n")
    results, _ = search_sites(query, sites)
    print(f"{len(results)} candidate(s) retrieved.")

    ranked, extract_cost = rank_candidates(
        results, target=query, profile=DEFAULT_PROFILE, weights=DEFAULT_WEIGHTS
    )
    top = [it for it in ranked if it.overall is not None][:TOP_N]
    print(f"Extracted + scored (cost ${extract_cost.get('cost_usd')}). "
          f"Explaining top {len(top)}.\n")

    explanations, explain_cost = explain_items(top, profile=DEFAULT_PROFILE, top_n=TOP_N)
    print(f"Explained {explain_cost.get('items')} item(s) "
          f"(cost ${explain_cost.get('cost_usd')}).\n")

    for i, (item, ex) in enumerate(zip(top, explanations), 1):
        score = "  –  " if item.overall is None else f"{item.overall:.3f}"
        print(f"{i:>2}. [{score}] {item.garment.title[:64]}")
        print(f"    lead: {ex.lead_reason}")
        print(f"    {ex.explanation}")
        if ex.confidence_note:
            print(f"    note: {ex.confidence_note}")
        if ex.flags:
            print(f"    ⚠ FLAGGED FOR REVIEW: {'; '.join(ex.flags)}")
        print()

    flagged = sum(1 for ex in explanations if ex.flags)
    print(f"{len(explanations) - flagged}/{len(explanations)} explanations clean; "
          f"{flagged} flagged for review.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(" ".join(sys.argv[1:])))
