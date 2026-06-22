# Jacque

Upload an image → search [Channel3](https://trychannel3.com) → get links to products you can buy.

Functionality-first: a plain Flask backend + an unstyled HTML/JS frontend.

## How it works

1. The browser uploads an image to the Flask backend (`/api/search`).
2. The backend base64-encodes it and POSTs to the Channel3 search API
   (`POST https://api.trychannel3.com/v1/search`, auth via `x-api-key`).
3. Channel3 returns visually-similar products; the backend flattens each into
   `{title, brand, price, image, url}` and the frontend renders them as buy links.

## Setup

This project uses a virtual environment (`.venv`).

```bash
# 1. Create the venv and install deps (already done if you scaffolded via Claude)
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
