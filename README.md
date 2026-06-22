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

Instead of Channel3, the harness searches **only a list of websites you supply**.
It drives Claude with the server-side `web_search` tool restricted via
`allowed_domains` to your sites, and returns buyable product links from them.

Pipeline is otherwise unchanged: image → Sonnet describes items+style → query →
**scoped search**.

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

Requires `ANTHROPIC_API_KEY`. Channel3 is not needed for scoped search.

- `harness.py` — core: `load_sites()` + `search_sites(query, sites)`.
- `run_harness.py` — CLI runner.
- `/api/search_scoped` — Flask endpoint used by the frontend toggle.

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
