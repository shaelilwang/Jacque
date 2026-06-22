"""Scoped-site search harness — an alternative to Channel3.

Instead of querying Channel3, this drives Claude with the server-side web_search
tool, restricted via `allowed_domains` to a list of websites YOU supply
(`sites.txt`). The model searches only those sites and returns real, currently
listed products to buy.

Pipeline unchanged elsewhere: image -> Sonnet describes items+style -> query.
This module consumes that query and returns product links from the scoped sites.
"""

import json
import os
import re

import anthropic

import cost as cost_mod

# Cheapest model that supports the web_search server tool. Haiku does NOT support
# the _20260209 dynamic-filtering variant, so we use the basic web_search_20250305.
SEARCH_MODEL = "claude-haiku-4-5"
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
# Keep this low: each web search adds an internal server round whose accumulated
# context is re-billed as input tokens. Fewer searches = lower cost. Tunable.
MAX_SEARCHES = 3
DEFAULT_SITES_FILE = os.path.join(os.path.dirname(__file__), "sites.txt")


def load_sites(path=DEFAULT_SITES_FILE):
    """Read the scoped website list. One domain per line; '#' comments allowed.

    URLs are normalized to bare domains (drop scheme + path), which is the form
    web_search's allowed_domains expects.
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
            domain = re.sub(r"^https?://", "", line).strip("/")
            domain = domain.split("/")[0]  # drop any path
            if domain:
                sites.append(domain)
    return sites


def _extract_products(text):
    """Pull the JSON array of products out of the model's final message."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def search_sites(query, sites, max_results=10):
    """Search the scoped sites for products matching `query`.

    Returns (products, cost) where products is a list of
    {title, url, price, site} and cost is a cost-summary dict. Raises if
    ANTHROPIC_API_KEY is unset or no sites are configured.
    """
    if not sites:
        raise ValueError("No sites configured — add domains to sites.txt.")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    tools = [
        {
            "type": WEB_SEARCH_TOOL_TYPE,
            "name": "web_search",
            "allowed_domains": sites,
            "max_uses": MAX_SEARCHES,
        }
    ]

    sites_list = ", ".join(sites)
    system = (
        "You are a shopping assistant. Use web_search to find real, currently "
        "listed products the user can buy. Search ONLY the scoped sites listed "
        "in the user's message; never invent products or URLs. Do not ask the "
        "user which sites to use — they are given."
    )
    user = (
        f"Scoped sites (search ONLY these): {sites_list}.\n\n"
        f"Find up to {max_results} products matching this description, only from "
        f"the scoped sites above.\n\nDescription:\n{query}\n\n"
        "When done, output ONLY a JSON array (no prose) where each element is "
        '{"title": str, "url": str, "price": str, "site": str}. '
        "Each url must be a direct product page on one of the scoped sites; "
        "leave price as an empty string if unknown. If nothing matches, output []."
    )

    # Single call — deliberately NO pause_turn resend loop. Re-sending the
    # transcript would re-bill every accumulated web_search result block (that
    # was the dominant cost). With max_uses kept small, the server finishes
    # within its iteration budget in one shot, so pause_turn shouldn't fire; if
    # it does, we take whatever was produced rather than resending.
    usage = cost_mod.empty_usage()
    response = client.messages.create(
        model=SEARCH_MODEL,
        max_tokens=2048,
        system=system,
        tools=tools,
        messages=[{"role": "user", "content": user}],
    )
    cost_mod.add_response_usage(usage, response)

    text = "".join(b.text for b in response.content if b.type == "text")
    return _extract_products(text), cost_mod.summarize(SEARCH_MODEL, usage)
