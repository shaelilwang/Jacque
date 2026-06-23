# Jacque

Upload an image → search [Channel3](https://trychannel3.com) → get links to products you can buy.

Functionality-first: a plain Flask backend + an unstyled HTML/JS frontend.

## How it works

1. The browser uploads an image to the Flask backend (`/api/search`).
2. The backend base64-encodes it and POSTs to the Channel3 search API
   (`POST https://api.trychannel3.com/v1/search`, auth via `x-api-key`).
3. Channel3 returns visually-similar products; the backend flattens each into
   `{title, brand, price, image, url}` and the frontend renders them as buy links.

### Text assist (Anthropic Sonnet)

Optionally, click **Fill description (Sonnet)** before searching. The backend
(`/api/assist`) sends the image to Anthropic Sonnet (`claude-sonnet-4-6`), which
returns two separate descriptions — the **items** in the image and the sample
image's **style** — via structured output. These are combined into the search
query box (editable), then sent to Channel3 alongside the image. Requires
`ANTHROPIC_API_KEY`.

## Harness: scoped-site search (alternative to Channel3)

Instead of Channel3, the harness searches **only a list of websites you supply**
using the **Serper.dev Google Search API**, scoped via `site:` operators and
paginated via the `page` param. Retrieval is pure HTTP — no LLM tokens — so it's
cheap and supports real pagination.

> Note: this previously used the Google Custom Search JSON API, but Google closed
> that API to new customers in Jan 2026 (new projects get a 403 even when it's
> "enabled"), so the harness now uses Serper.dev.

Pipeline is otherwise unchanged: image → Sonnet describes items+style → query →
**scoped Google search**.

Setup the Serper side once:
1. Sign up at [serper.dev](https://serper.dev) (2,500 free credits) and copy your
   API key → `SERPER_API_KEY`.
2. Put it in `.env` (see `.env.example`).

```bash
# 1. Supply your sites (one per line; gitignored)
cp sites.example.txt sites.txt
# edit sites.txt and add your domains, e.g.
#   www.uniqlo.com
#   www.everlane.com

# 2. Run it from the CLI
.venv/bin/python run_harness.py "minimalist white leather sneakers, clean modern aesthetic"

# ...or in the web UI: the "Search my scoped sites" checkbox (on by default on
# this branch) routes Search to /api/search_scoped instead of Channel3.
```

Requires `SERPER_API_KEY` (the Sonnet assist step still uses
`ANTHROPIC_API_KEY`). Channel3 is not needed for scoped search.

`MAX_PAGES` in `harness.py` controls pagination depth (10 results/page).

- `harness.py` — core: `load_sites()` + `search_sites(query, sites)`.
- `run_harness.py` — CLI runner.
- `/api/search_scoped` — Flask endpoint used by the frontend toggle.

## Ranking: score & rank candidates for the user

Turns retrieved candidates into a ranked list scored against a user profile
(measurements, usual size, fit preference, monthly budget, hardcoded taste spec,
Kibbe body type) and the actual garment.

Two stages, cleanly separated:
1. **Extraction (cheap LLM, Haiku)** — `extract.py` reads each candidate's text
   and pulls garment type, silhouette/aesthetic tags, material/colour, any stated
   dimensions, and whether it *is* the target item — each with a confidence.
   Missing data stays `None` (never guessed).
2. **Scoring (deterministic, pure)** — `ranker.py` computes four sub-scores —
   **target match, fit (Kibbe silhouette + dimensional), taste, budget** — and a
   confidence-weighted overall. Pure functions, no I/O, unit-tested.

```bash
.venv/bin/python run_ranker.py "minimalist white leather sneakers"   # demo
.venv/bin/python test_ranker.py                                       # unit tests
```

- **Tune weights** via `RankingWeights` in `ranker.py` (`DEFAULT_WEIGHTS`):
  `target_match`, `fit`, `taste`, `budget`, plus `confidence_power` (how hard low
  confidence pulls a sub-score's effective weight down).
- **Edit the profile** in `profiles.py` (`DEFAULT_PROFILE`); the taste spec is
  hardcoded, the rest are example placeholders to replace.
- Each sub-score's effective weight is `base * confidence**power`, so uncertain
  signals shrink automatically and `None` drops out — missing data is honest end
  to end, and items with no usable signal sort last (`overall = None`).

Requires `ANTHROPIC_API_KEY` (extraction) on top of `SERPER_API_KEY` (search).

- `profiles.py` — user profile model (`UserProfile`, `Measurements`, `KibbeType`).
- `ranker.py` — `Garment` model + pure scorers + `rank()`.
- `extract.py` — Haiku extraction + `rank_candidates()` orchestrator.
- `test_ranker.py` — unit tests for the deterministic scoring.

## Setup

This project uses a virtual environment (`.venv`).

```bash
# 1. Create the venv and install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Provide your Channel3 API key
cp .env.example .env
# edit .env and set CHANNEL3_API_KEY=...
# (or: export CHANNEL3_API_KEY=...)

# 3. Run
.venv/bin/python app.py
```

Then open http://127.0.0.1:5000 and upload an image.

## Notes

- You supply the Channel3 API key via `CHANNEL3_API_KEY` (env var or `.env`).
- An optional text box lets you refine the image search with keywords.
- No styling yet — functionality first.
