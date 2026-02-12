# LogicMaster AI

> AI-Native Adaptive Learning Platform for GMAT Critical Reasoning

## Features

### Adaptive Learning Engine
- **IRT-based Ability Estimation** — 3-Parameter Logistic model maps latent ability (theta) to GMAT Verbal scale (V20-V51)
- **BKT Skill Tracking** — Bayesian Knowledge Tracing across 10+ cognitive skills (Causal Reasoning, Assumption Identification, Alternative Explanations, etc.)
- **Hybrid Recommendation** — IRT ability matching + BKT weakness-first strategy to target specific logical gaps

### AI-Powered Tutoring
- **RAG System** — Qdrant vector database + OpenAI `text-embedding-3-small` (1536 dims) for retrieval-augmented explanation generation
- **LangChain Socratic Agent** — Multi-turn dialogue with progressive hint strength (gentle → moderate → direct) and student understanding evaluation
- **3-Tier Explanation Fallback** — Cached → RAG-enhanced → plain LLM, ensuring reliable output

### Data Science Infrastructure
- **A/B Testing Framework** — Consistent hashing for deterministic variant assignment; supports `tutor_strategy` (socratic vs direct) and `explanation_source` (RAG vs baseline) experiments
- **Statistical Analysis** — Two-sample t-tests, Cohen's d effect size, per-variant metrics
- **LLM-as-Judge Evaluation** — Automated explanation quality scoring (correctness, clarity, completeness, pedagogical value)

### Frontend
- **Sidebar Navigation** — Practice, Analytics, Settings pages
- **Styled Question Cards** — Color-coded badges by question type and difficulty, custom CSS
- **Real-time Health Indicators** — API, database, and Qdrant status in sidebar
- **Loading Spinners** — Feedback during RAG generation, question loading, and tutor responses

## Architecture

```
┌─────────────────────┐     HTTP      ┌────────────────────────────┐
│   Streamlit (8501)  │ ────────────→ │   FastAPI Backend (8000)   │
│   app.py            │               │   backend/main.py          │
│                     │               │                            │
│ - Practice page     │               │ 5 Routers:                 │
│ - Analytics page    │               │  - /api/theta/update       │
│ - Settings page     │               │  - /api/questions/next     │
└─────────────────────┘               │  - /api/tutor/*            │
                                      │  - /api/explanations/*     │
                                      │  - /api/analytics/*        │
                                      │                            │
                                      └──┬──────────┬──────────┬───┘
                                         │          │          │
                                    ┌────┘    ┌─────┘    ┌─────┘
                                    ▼         ▼          ▼
                              ┌──────────┐ ┌──────┐ ┌──────────────┐
                              │  SQLite  │ │Qdrant│ │ DeepSeek LLM │
                              │  (~50 Q) │ │(6333)│ │ via OpenAI   │
                              └──────────┘ └──────┘ │ SDK          │
                                                    └──────────────┘
```

- **Frontend**: Streamlit — calls FastAPI via HTTP
- **Backend**: FastAPI — wraps existing `engine/` modules as REST endpoints
- **Database**: SQLite (`logicmaster.db`) — ~50 verified GMAT CR questions
- **Vector DB**: Qdrant (Docker, port 6333) — collection `gmat_explanations`
- **LLM**: DeepSeek API via OpenAI SDK (custom `base_url`)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Qdrant)
- API keys: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`

### Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd logicmaster
pip install -r requirements.txt
pip install -r requirements-backend.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start Qdrant
docker-compose up -d

# 4. Index questions into vector DB
python scripts/index_to_rag.py

# 5. Start FastAPI backend
cd backend && uvicorn main:app --reload --port 8000

# 6. Start Streamlit frontend (new terminal)
streamlit run app.py
```

### Docker (Full Stack)

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

## Performance

| Metric | Value |
|---|---|
| Test cases | 82 passing (4 skipped — empty test DB) |
| API endpoints | 10 across 5 routers |
| RAG embedding model | `text-embedding-3-small` (1536 dims) |
| Indexed questions | ~50 |
| Tutor hint levels | 3 (gentle → moderate → direct) |
| A/B experiments | 2 (tutor_strategy, explanation_source) |
| Explanation fallback tiers | 3 (cached → RAG → plain LLM) |

## Tech Stack

**Backend**: FastAPI, Pydantic v2, LangChain, LangChain-OpenAI
**AI/ML**: DeepSeek LLM, OpenAI Embeddings, Qdrant vector DB
**Data Science**: SciPy (t-tests, Cohen's d), IRT 3PL, Bayesian Knowledge Tracing
**Frontend**: Streamlit (custom CSS, Plotly charts)
**Database**: SQLite, Qdrant
**Testing**: pytest (86 test cases)
**Infrastructure**: Docker, Docker Compose

## Documentation

- [API Reference](docs/api.md) — All endpoints with request/response examples
- [Resume Bullet Points](docs/resume_bullet_points.md) — Project highlights for job applications
- [Demo Script](docs/demo_script.md) — 5-minute walkthrough

## Evaluation Scripts

```bash
# Analyze A/B test results
python scripts/analyze_ab_tests.py

# Evaluate LLM explanation quality
python scripts/evaluate_llm_quality.py

# Re-index questions into Qdrant
python scripts/index_to_rag.py --force
```

## Project Structure

```
├── app.py                          # Streamlit frontend
├── backend/
│   ├── main.py                     # FastAPI entry point
│   ├── config.py                   # pydantic-settings configuration
│   ├── routers/
│   │   ├── theta.py                # IRT theta update
│   │   ├── questions.py            # Adaptive question recommendation
│   │   ├── tutor.py                # Socratic tutor (LangChain Agent)
│   │   ├── explanations.py         # RAG-enhanced explanations
│   │   └── analytics.py            # A/B test results, RAG performance
│   ├── services/
│   │   ├── rag_service.py          # Qdrant + OpenAI embedding
│   │   ├── explanation_service.py  # 3-tier fallback logic
│   │   ├── tutor_agent.py          # SocraticTutorAgent (LangChain)
│   │   ├── conversation_manager.py # Multi-turn state management
│   │   └── ab_testing.py           # A/B experiment framework
│   ├── ml/
│   │   ├── rag_evaluator.py        # Precision@K, Recall@K, MRR, F1@K
│   │   └── llm_evaluator.py        # LLM-as-Judge quality scoring
│   └── tests/                      # 86 pytest test cases
├── engine/
│   ├── scoring.py                  # IRT 3PL scoring
│   └── recommender.py              # BKT hybrid recommendation
├── scripts/
│   ├── index_to_rag.py             # Batch indexer → Qdrant
│   ├── analyze_ab_tests.py         # A/B statistical analysis
│   └── evaluate_llm_quality.py     # LLM quality evaluation
├── docker-compose.yml              # Qdrant + Backend + Frontend
└── docs/                           # Documentation
```

## License

MIT
