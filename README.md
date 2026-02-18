# SEC RAG System

A retrieval-augmented generation (RAG) system for querying SEC filings (10-K, 10-Q, 8-K) using natural language. Covers the top 10 S&P 500 companies from 2010 to present.

**Live Demo:** [sec-rag-system.vercel.app](https://sec-rag-system.vercel.app)

## Architecture

```
User Query
    |
    v
React Frontend (Vercel)
    |
    v
FastAPI Backend (Railway)
    |
    +---> Query Classifier (GPT-4o-mini)
    |         |
    |         v
    +---> Retrieval Router
    |       |
    |       +---> metric_lookup   (XBRL facts from PostgreSQL)
    |       +---> timeseries      (multi-period XBRL data)
    |       +---> full_statement  (complete financial statements)
    |       +---> narrative       (semantic search via pgvector)
    |       +---> hybrid          (vector + relational combined)
    |       |
    |       v
    +---> Answer Generation (GPT-4o-mini)
    |       |
    |       v
    +---> Guardrails + Confidence Scoring
    |
    v
Streamed Response (SSE)
```

For detailed documentation, see:
- [System Architecture](docs/architecture.md) - Component design, data flow, caching, guardrails
- [Database Design](docs/database.md) - Table schemas, indexes, vector search, data volumes
- [Retrieval Routes](docs/retrieval-routes.md) - How each query route works with examples

## Features

- **Intelligent Query Routing** - Classifies queries and routes to the optimal retrieval strategy
- **5 Retrieval Pipelines** - Metric lookup, timeseries, full statements, narrative search, and hybrid
- **Semantic Search** - Vector similarity search over 10-K/10-Q sections using pgvector embeddings
- **XBRL Data Extraction** - Structured financial data from SEC EDGAR XBRL filings
- **Confidence Scoring** - Investor-grade confidence tiers with signal breakdown
- **Contradiction Detection** - Identifies conflicting data across sources
- **Source Attribution** - Every answer links back to SEC EDGAR filings
- **Streaming UI** - Real-time classification, retrieval plan, and answer streaming via SSE
- **Redis Caching** - Three-layer cache (query results, classifications, retrievals)
- **Cost Tracking** - Per-query OpenAI token usage and cost estimates

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Tailwind CSS |
| Backend | FastAPI, Python |
| Database | PostgreSQL + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | GPT-4o-mini |
| Data Source | SEC EDGAR (XBRL + full filings) |

## Coverage

**Tickers:** AAPL, MSFT, NVDA, AMZN, GOOGL, META, BRK-B, LLY, AVGO, JPM

**Filings:** 10-K (annual), 10-Q (quarterly), 8-K (earnings) from 2010 to present

## Project Structure

```
sec_rag_system/
├── api_server.py              # FastAPI server (SSE streaming + REST)
├── rag_query.py               # Query engine: classifier, router, retrieval, generation
├── config.py                  # Tickers, years, fiscal year mappings
├── guardrails.py              # Retrieval filtering + confidence scoring
├── guardrails.yaml            # Guardrail thresholds and config
├── cache.py                   # Redis caching layer
├── chunk_and_embed.py         # Section chunking + OpenAI embeddings
├── xbrl_to_postgres.py        # XBRL parsing + PostgreSQL storage
├── fetch_financials_to_postgres.py  # Financial statement fetching
├── filing_sections.py         # 10-K/10-Q section extraction
├── section_vector_tables.py   # pgvector table setup
├── backfill_pipeline.py       # Unified data ingestion pipeline
├── requirements.txt           # Python dependencies
├── railway.toml               # Railway deployment config
├── Procfile                   # Process start command
└── frontend/                  # React frontend
    └── src/
        └── App.js             # Main UI with streaming, charts, confidence display
```

## Local Development

### Prerequisites

- Python 3.11+
- PostgreSQL 17 with pgvector extension
- Node.js 18+
- Redis (optional, for caching)

### Setup

```bash
# Clone
git clone https://github.com/bhattaraisubal-eng/sec-rag-system.git
cd sec-rag-system

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
PG_HOST=localhost
PG_PORT=5432
PG_USER=your_user
PG_PASSWORD=your_password
PG_DATABASE=sec_filings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EOF

# Start backend
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm start
```

### Data Ingestion

```bash
# Run the backfill pipeline to populate the database
python backfill_pipeline.py
```

## Deployment

Deployed on **Railway** (backend + PostgreSQL) and **Vercel** (frontend).

| Component | Service |
|-----------|---------|
| Frontend | [Vercel](https://vercel.com) (free) |
| Backend | [Railway](https://railway.app) ($5/mo Hobby) |
| Database | Railway PostgreSQL (pgvector Docker image) |

### Environment Variables (Railway)

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (auto-parsed into PG_* vars) |
| `OPENAI_API_KEY` | OpenAI API key |
| `FRONTEND_URL` | Vercel frontend URL (for CORS) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | `1536` |

### Environment Variables (Vercel)

| Variable | Description |
|----------|-------------|
| `REACT_APP_BACKEND_URL` | Railway backend URL |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query/stream` | SSE streaming query (classification + plan + result) |
| POST | `/query` | Non-streaming query |
| GET | `/health` | Health check |
| GET | `/cache/stats` | Redis cache statistics |
| POST | `/cache/clear` | Clear cache (optional `layer` param) |

## Example Queries

- "What was Apple's revenue in 2023?"
- "Compare NVIDIA and AMD gross margins from 2020 to 2024"
- "What are the key risk factors in Meta's latest 10-K?"
- "Show me JPMorgan's balance sheet for Q2 2024"
- "How has Microsoft's R&D spending changed over time?"
