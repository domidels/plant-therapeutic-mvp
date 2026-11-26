# Plant Therapeutic MVP

A web application that helps users discover evidence-based, plant-derived therapeutic options for health conditions. It combines PubMed scientific literature search with LLM-powered summaries to make herbal medicine research accessible to everyone.

## Features

- **Condition search** — Enter any health condition (typos auto-corrected via LLM)
- **Consistent synonyms** — Condition synonyms are picked by the LLM from a closed list of MeSH candidates, filtered by a second LLM pass that drops ambiguous terms, and cached in Turso
- **Plant recommendations** — Finds medicinal plants backed by PubMed studies for your condition, browsable by publication year
- **Evidence ranking** — Prioritizes meta-analyses, systematic reviews, and RCTs
- **Deep dive exploration** — Streams plain-language summaries of individual research papers
- **Plant verdicts** — Each explored article yields a per-plant verdict (positive / negative / no effect / irrelevant); verdicts are aggregated across articles and drive card colors, and cards whose plants are all negative are hidden
- **LLM fallback** — Any OpenAI-compatible provider, with an optional fallback provider on rate-limit or quota errors and a daily email alert
- **Bot protection** — Cloudflare Turnstile CAPTCHA
- **IP-based rate limiting** — no account required; quotas enforced per hashed IP and globally
- **Caching** — Turso edge database caches LLM results to reduce latency and costs

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| Database | Turso (LibSQL) |
| LLM | Any OpenAI-compatible chat-completions API (main + optional fallback; Hugging Face router as legacy) |
| Data sources | NCBI PubMed, MeSH, MedlinePlus |
| Email | Resend API (quota alerts) |
| Bot detection | Cloudflare Turnstile |
| Frontend | Jinja2 + Vanilla JS |

## Getting Started

### Prerequisites

- Python 3.12+
- A virtual environment tool (`venv`)
- All required API keys (see [Environment Variables](#environment-variables))

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd plant-therapeutic-mvp

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements-local.txt` adds optional local-only tooling (`fastembed`) on top of the runtime dependencies.

### Running locally

```bash
uvicorn app.main:app --reload
```

The app will be available at `http://localhost:8000`.

## Environment Variables

Create a `.env` file at the project root with the following variables:

```env
# --- LLM (any OpenAI-compatible chat-completions API) ---
# Main provider (Mistral free plan)
LLM_BASE_URL=https://api.mistral.ai/v1
LLM_API_KEY=
LLM_MODEL=mistral-small-latest
# Optional fallback, used when the main provider is rate-limited, out of quota or down
LLM_FALLBACK_BASE_URL=https://api.groq.com/openai/v1
LLM_FALLBACK_API_KEY=
LLM_FALLBACK_MODEL=openai/gpt-oss-20b
# Legacy: used only when no LLM_* provider is set
HF_TOKEN=                        # Hugging Face API token
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

# --- Quota alert email (sent once per provider per day on HTTP 402/429) ---
RESEND_API_KEY=
EMAIL_FROM=
ALERT_EMAIL=

# --- Database ---
TURSO_DATABASE_URL=              # Turso database URL (libsql://...)
TURSO_AUTH_TOKEN=                # Turso auth token

# --- Security ---
SESSION_SECRET=                  # Random secret key used for HMAC-SHA256 IP hashing
                                 # and Turnstile cookie signing

# --- Bot protection ---
TURNSTILE_SECRET=                # Cloudflare Turnstile secret key

# --- NCBI / PubMed ---
NCBI_TOOL=                       # Tool name identifier for NCBI requests
NCBI_EMAIL=                      # Contact email for NCBI
NCBI_API_KEY=                    # Optional — increases NCBI rate limits

# --- Rate limits: per IP per day ---
DAILY_HF_TOKEN_LIMIT=30000       # LLM tokens (applies to every provider, not only HF)
DAILY_TURSO_OP_LIMIT=5000
DAILY_EXPLORE_LIMIT=25

# --- Rate limits: global ceiling per day (all IPs combined) ---
DAILY_GLOBAL_HF_TOKEN_LIMIT=200000
DAILY_GLOBAL_EXPLORE_LIMIT=150

# --- Debug (optional) ---
PLANTS_DEBUG=                    # Set to 1 to trace plant detection (very verbose)
```

The rate-limit values above are the defaults used when a variable is not set.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/verify_human` | Validate Cloudflare Turnstile token and set the signed cookie |
| `GET` | `/api/auth/me` | Current daily usage and limits for the caller's IP |
| `GET` | `/api/condition_query?condition=` | Correct condition name, resolve synonyms, fetch MedlinePlus summary |
| `GET` | `/api/recommendations?condition=&from_year=&to_year=` | Get plant recommendations from PubMed |
| `GET` | `/api/explore_stream?pmid=&plant=&condition=` | Stream LLM summary and plant verdicts for a specific article |

All `/api/*` endpoints except `/api/verify_human` require a passed Turnstile check.

## Plant Database

The app includes a curated database of **290 medicinal plants** (`app/data/seed_plants.csv`) used for context-aware plant name extraction from PubMed abstracts. Ambiguous terms (e.g. "sage", "tea", "ginseng") are resolved using surrounding context.

## Deployment

The project is configured for deployment on **Vercel** (see `.vercelignore`). Turso is used as the edge database for caching (summaries, condition corrections and synonyms, plant verdicts) and daily usage counters.

## Rate Limiting

No account is required. Requests are rate-limited by hashed client IP using two quota levels:

| Level | LLM tokens/day | Explorations/day |
|-------|---------------|-----------------|
| Per IP | 30,000 | 25 |
| **Global** (all IPs) | **200,000** | **150** |

The global ceiling acts as a hard cost cap regardless of the number of IPs. Client IPs are never stored in plain text — they are hashed with HMAC-SHA256 before being written to the database.

## Project Structure

```
app/
├── api/
│   └── routes.py          # API endpoints, LLM providers, rate limiting
├── core/
│   └── config.py          # Settings and configuration
├── services/
│   ├── plants_v2.py       # Plant name recognition and matching
│   ├── pubmed.py          # PubMed API integration
│   ├── mesh.py            # MeSH synonym candidates for a condition
│   ├── medline.py         # MedlinePlus API integration
│   ├── ranking.py         # Study scoring and ranking
│   └── turso_db.py        # Database client and helpers
├── data/
│   └── seed_plants.csv    # 290 medicinal plants
├── templates/
│   └── index.html         # Single-page frontend
├── static/                # CSS and images
└── main.py                # FastAPI app entry point
```

## License

Copyright (c) 2026 [plant-med.org](https://plant-med.org)

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

You are free to share and adapt this work for non-commercial purposes, provided you give appropriate credit and distribute any derivative works under the same license.

See [LICENSE](LICENSE) for the full license text.
