"""Cost estimation for Jacque, so we can compare spend before/after swapping
the search backend (Anthropic web_search now -> Google PSE later).

All figures are USD estimates. Token prices are list prices per 1M tokens;
cache reads bill ~0.1x input and cache writes ~1.25x input. The web-search
per-call price is an estimate — adjust WEB_SEARCH_USD_PER_CALL to your contract.
"""

# USD per 1M tokens (list prices).
TOKEN_PRICES = {
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}

# Anthropic web search is billed separately from tokens. Estimate ~$10/1000.
WEB_SEARCH_USD_PER_CALL = 0.01

# Google Custom Search JSON API: 100 queries/day free, then ~$5/1000 (cap 10k/day).
GOOGLE_SEARCH_USD_PER_QUERY = 0.005


def _rate(model, kind):
    return TOKEN_PRICES.get(model, {}).get(kind, 0.0)


def empty_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "web_searches": 0,
        "web_search_blocks": 0,
    }


def add_response_usage(acc, response):
    """Accumulate token + web-search usage from one Anthropic response."""
    u = response.usage
    acc["input_tokens"] += getattr(u, "input_tokens", 0) or 0
    acc["output_tokens"] += getattr(u, "output_tokens", 0) or 0
    acc["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
    acc["cache_creation_input_tokens"] += (
        getattr(u, "cache_creation_input_tokens", 0) or 0
    )

    # Web-search count: prefer the usage field, fall back to counting blocks.
    stu = getattr(u, "server_tool_use", None)
    if stu is not None:
        acc["web_searches"] += getattr(stu, "web_search_requests", 0) or 0
    acc["web_search_blocks"] += sum(
        1
        for b in response.content
        if getattr(b, "type", "") == "server_tool_use"
        and getattr(b, "name", "") == "web_search"
    )
    return acc


def google_summary(num_queries):
    """Cost summary for the Google Custom Search backend (no tokens)."""
    total = num_queries * GOOGLE_SEARCH_USD_PER_QUERY
    return {
        "model": "google-cse",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "web_searches": num_queries,  # Google API requests
        "cost_usd": round(total, 6),
        "breakdown_usd": {"google_queries": round(total, 6)},
    }


def summarize(model, usage):
    """Turn accumulated usage into a JSON-serializable cost summary."""
    inp = usage["input_tokens"]
    out = usage["output_tokens"]
    cr = usage["cache_read_input_tokens"]
    cw = usage["cache_creation_input_tokens"]
    searches = usage["web_searches"] or usage["web_search_blocks"]

    input_cost = inp * _rate(model, "input") / 1_000_000
    output_cost = out * _rate(model, "output") / 1_000_000
    cache_read_cost = cr * _rate(model, "input") * 0.1 / 1_000_000
    cache_write_cost = cw * _rate(model, "input") * 1.25 / 1_000_000
    search_cost = searches * WEB_SEARCH_USD_PER_CALL
    total = input_cost + output_cost + cache_read_cost + cache_write_cost + search_cost

    return {
        "model": model,
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cw,
        "web_searches": searches,
        "cost_usd": round(total, 6),
        "breakdown_usd": {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "cache_read": round(cache_read_cost, 6),
            "cache_write": round(cache_write_cost, 6),
            "web_search": round(search_cost, 6),
        },
    }
