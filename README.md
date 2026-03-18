# Plant Therapeutic MVP

A web application that helps users discover evidence-based, plant-derived therapeutic options for health conditions. It combines PubMed scientific literature search with LLM-powered summaries to make herbal medicine research accessible to everyone.

## Features

- **Condition search** — Enter any health condition (typos auto-corrected via LLM)
- **Plant recommendations** — Finds medicinal plants backed by PubMed studies for your condition
- **Evidence ranking** — Prioritizes meta-analyses, systematic reviews, and RCTs
- **Deep dive exploration** — Streams plain-language summaries of individual research papers
- **Bot protection** — Cloudflare Turnstile CAPTCHA
- **IP-based rate limiting** — no account required; quotas enforced per hashed IP and globally
- **Caching** — Turso edge database caches LLM results to reduce latency and costs

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| Database | Turso (LibSQL) |
| LLM | Hugging Face Inference API (Llama 3.1 8B Instruct) |
| Data sources | NCBI PubMed, MedlinePlus |
| Email | Resend API |
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

### Running locally

```bash
uvicorn app.main:app --reload
```

The app will be available at `http://localhost:8000`.

## Environment Variables

Create a `.env` file at the project root with the following variables:

```env
# --- LLM ---
HF_TOKEN=                        # Hugging Face API token
HF_BASE_URL=https://router.huggingface.co/v1
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

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
DAILY_HF_TOKEN_LIMIT=4000
DAILY_TURSO_OP_LIMIT=5000
DAILY_EXPLORE_LIMIT=25

# --- Rate limits: global ceiling per day (all IPs combined) ---
DAILY_GLOBAL_HF_TOKEN_LIMIT=50000
DAILY_GLOBAL_EXPLORE_LIMIT=150
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/verify_human` | Validate Cloudflare Turnstile token |
| `POST` | `/api/condition_query` | Correct condition name + fetch MedlinePlus summary |
| `POST` | `/api/recommendations` | Get plant recommendations from PubMed |
| `GET` | `/api/explore_stream` | Stream LLM summary for a specific article |
| `POST` | `/api/auth/request_code` | Send OTP to email |
| `POST` | `/api/auth/verify_code` | Verify OTP and create session |
| `POST` | `/api/auth/logout` | Invalidate session |

## Plant Database

The app includes a curated database of **280 medicinal plants** (`app/data/seed_plants.csv`) used for context-aware plant name extraction from PubMed abstracts. Ambiguous terms (e.g. "sage", "tea", "ginseng") are resolved using surrounding context.

## Deployment

The project is configured for deployment on **Vercel** (see `.vercelignore`). Turso is used as the edge database for caching and session storage.

## Rate Limiting

No account is required. Requests are rate-limited by hashed client IP using two quota levels:

| Level | LLM tokens/day | Explorations/day |
|-------|---------------|-----------------|
| Per IP | 4,000 | 25 |
| **Global** (all IPs) | **50,000** | **150** |

The global ceiling acts as a hard cost cap regardless of the number of IPs. Client IPs are never stored in plain text — they are hashed with HMAC-SHA256 before being written to the database.

## Project Structure

```
app/
├── api/
│   └── routes.py          # All API endpoints
├── core/
│   └── config.py          # Settings and configuration
├── services/
│   ├── plants_v2.py       # Plant name recognition and matching
│   ├── pubmed.py          # PubMed API integration
│   ├── medline.py         # MedlinePlus API integration
│   ├── ranking.py         # Study scoring and ranking
│   └── turso_db.py        # Database client and helpers
├── data/
│   └── seed_plants.csv    # 280 medicinal plants
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
