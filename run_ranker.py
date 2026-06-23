"""CLI demo: search scoped sites, extract garment attributes, rank for the user.

Usage:
    .venv/bin/python run_ranker.py "minimalist white leather sneakers"

Requires SERPER_API_KEY (search) and ANTHROPIC_API_KEY (extraction) in env/.env.
"""

from __future__ import annotations

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from extract import rank_candidates
from harness import load_sites, search_sites
from profiles import DEFAULT_PROFILE
from ranker import DEFAULT_WEIGHTS


def _fmt(sub):
    if not sub.known:
        return "  –  "
    return f"{sub.value:.2f}@{sub.confidence:.0%}"


def main(query: str) -> int:
    sites = load_sites()
    print(f"Searching {len(sites)} site(s) for: {query!r}\n")
    results, search_cost = search_sites(query, sites)
    print(f"{len(results)} candidate(s) retrieved (cost {search_cost.get('backend')}).")

    ranked, extract_cost = rank_candidates(results, target=query,
                                           profile=DEFAULT_PROFILE, weights=DEFAULT_WEIGHTS)
    print(f"Extracted + scored (cost ${extract_cost.get('cost_usd')}).\n")
    print(f"Profile taste: {', '.join(DEFAULT_PROFILE.taste)} | "
          f"kibbe {DEFAULT_PROFILE.kibbe and DEFAULT_PROFILE.kibbe.value} | "
          f"budget ${DEFAULT_PROFILE.monthly_budget_usd}\n")

    print(f"{'#':>2} {'score':>6}  {'target':>10} {'fit':>10} {'taste':>10} {'budget':>10}  title")
    for i, item in enumerate(ranked, 1):
        s = item.subscores
        overall = "  –  " if item.overall is None else f"{item.overall:.3f}"
        print(f"{i:>2} {overall:>6}  "
              f"{_fmt(s['target_match']):>10} {_fmt(s['fit']):>10} "
              f"{_fmt(s['taste']):>10} {_fmt(s['budget']):>10}  "
              f"{item.garment.title[:48]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(" ".join(sys.argv[1:])))
