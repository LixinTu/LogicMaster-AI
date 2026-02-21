# LogicMaster AI

> AI-Native GMAT Critical Reasoning Tutor — A GlitchMind Product

**Live:** [gmat.glitchmind.io](https://gmat.glitchmind.io) &nbsp;|&nbsp; **API:** [api.glitchmind.io](https://api.glitchmind.io)

---

## Overview

LogicMaster AI is a full-stack adaptive learning platform for GMAT Critical Reasoning. It combines real-time ability estimation, neural knowledge tracing, and an AI Socratic tutor to deliver a personalized study experience — built with React + FastAPI.

---

## Key Features

### Adaptive Learning Engine
- **3PL IRT Adaptive Scoring** — Three-Parameter Logistic model (discrimination *a*, difficulty *b*, guessing *c*) with item information function I(θ) and MLE calibration via L-BFGS-B; maps latent ability to GMAT Verbal scale (V20–V51)
- **Thompson Sampling Question Selection** — Multi-armed bandit balancing exploration (Beta-distributed uncertainty) and exploitation (3PL item information); per-question Beta(α, β) priors updated after each response
- **Deep Knowledge Tracing (LSTM)** — PyTorch LSTM (Piech et al. 2015) predicts per-skill mastery from temporal learning sequences; auto-selects numpy logistic regression fallback on cold-start (< 50 interactions)
- **Spaced Repetition** — Half-Life Regression (Settles & Meeder 2016) models per-item forgetting curves with Ebbinghaus decay; review candidates injected at 40% probability when recall probability < 0.5
- **BKT Skill Tracking** — Bayesian Knowledge Tracing across 10+ cognitive skills (Causal Reasoning, Assumption Identification, Alternative Explanations, etc.)
- **Hybrid Recommendation Pipeline** — BKT/DKT weakness scoring → spaced repetition injection → Thompson Sampling final selection

### AI-Powered Tutoring
- **Socratic AI Tutor** — When a student answers incorrectly, a LangChain agent guides them through the logic using progressive hints (gentle → moderate → direct), never revealing the answer outright
- **Bloom's Taxonomy Evaluation** — LLM-based 6-level cognitive classification (Remember → Create) scaffolds hint strategy and tracks progression across turns
- **RAG Explanations** — Qdrant vector DB + OpenAI `text-embedding-3-small` (1536 dims) for retrieval-augmented explanation generation with 3-tier fallback (cached → RAG → plain LLM)

### User Features
- **Authentication** — JWT-based registration, login, and profile management (bcrypt password hashing, 7-day tokens)
- **Dashboard** — Real-time stats: GMAT score estimate, streak, theta, weak skills, reviews due
- **Wrong Book** — Auto-bookmarked incorrect answers with redo functionality and wrong-answer stats
- **Learning Goals** — Set a target GMAT score and daily question goal; track estimated questions remaining and on-track status

### Data Science Infrastructure
- **A/B Testing Framework** — Deterministic variant assignment (consistent hashing); supports `tutor_strategy` (socratic vs direct) and `explanation_source` (RAG vs baseline) experiments with t-test significance and Cohen's d analysis
- **LLM-as-Judge Evaluation** — Automated explanation quality scoring (correctness, clarity, completeness, pedagogical value)

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Zustand, Framer Motion, Recharts |
| **Backend** | FastAPI, Python 3.12, Pydantic v2, LangChain |
| **AI / ML** | DeepSeek LLM, OpenAI Embeddings, PyTorch (DKT LSTM), Qdrant (vector DB) |
| **Algorithms** | 3PL IRT, Thompson Sampling, Deep Knowledge Tracing, Half-Life Regression, Bloom's Taxonomy, BKT |
| **Database** | SQLite, Qdrant |
| **Auth** | JWT (PyJWT), bcrypt |
| **Testing** | pytest — 297 tests, 100% pass rate |
| **Infrastructure** | Docker, Cloudflare Tunnel (dev), AWS EC2 + S3 + CloudFront (production) |

---

## Architecture

```
React Frontend (S3 + CloudFront)
        ↓ HTTPS
FastAPI Backend (EC2)
    ├── Auth Service (JWT + bcrypt)
    ├── IRT Scoring Engine (3PL)
    ├── Thompson Sampling Bandit
    ├── Socratic Tutor Agent (DeepSeek LLM + LangChain)
    ├── RAG Pipeline (Qdrant + OpenAI Embeddings)
    ├── Deep Knowledge Tracing (LSTM)
    ├── Spaced Repetition Scheduler (Half-Life Regression)
    └── A/B Testing Framework
        ↓
    SQLite (questions, users, bookmarks, goals, stats)
    Qdrant (gmat_explanations, 1536-dim embeddings)
    DeepSeek API (LLM inference)
```

---

## Quick Start

### Prerequisites
- Python 3.11+, Node.js 18+
- Docker (for Qdrant)
- API keys: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`

### Backend

```bash
git clone <repo-url>
cd logicmaster

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Install backend dependencies
pip install -r requirements-backend.txt

# Start Qdrant
docker-compose up -d qdrant

# Index questions into vector DB
python scripts/index_to_rag.py

# Start FastAPI backend
cd backend && uvicorn main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

### Full Stack (Docker)

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

---

## Performance

| Metric | Value |
|---|---|
| Test cases | 297 passing |
| API endpoints | 20+ across 8 routers |
| IRT model | 3PL (discrimination, difficulty, guessing) |
| Knowledge tracing | DKT LSTM (numpy fallback on cold start) |
| Question selection | Thompson Sampling (Beta priors, explore/exploit) |
| Spaced repetition | Half-Life Regression (Ebbinghaus forgetting curve) |
| MLE calibration | L-BFGS-B, bounds a∈[0.5,2.5] b∈[-3,3] c∈[0,0.35] |
| Cognitive evaluation | Bloom's Taxonomy 6-level (Remember → Create) |
| RAG embedding model | `text-embedding-3-small` (1536 dims) |
| Explanation fallback tiers | 3 (cached → RAG → plain LLM) |
| Tutor hint levels | 3 (gentle → moderate → direct), Bloom's-scaffolded |
| A/B experiments | 2 (tutor_strategy, explanation_source) |

---

## Project Structure

```
├── frontend/                       # React + TypeScript frontend
│   └── src/
│       ├── pages/                  # Dashboard, Practice, WrongBook, Analytics, Profile, Settings
│       ├── components/             # Reusable UI components
│       ├── lib/                    # API client, config, utilities
│       └── store/                  # Zustand state management
├── backend/
│   ├── main.py                     # FastAPI entry point
│   ├── config.py                   # pydantic-settings configuration
│   ├── routers/
│   │   ├── auth.py                 # Register, login, profile, stats
│   │   ├── questions.py            # Adaptive question recommendation + bandit update
│   │   ├── tutor.py                # Socratic tutor (LangChain Agent)
│   │   ├── explanations.py         # RAG-enhanced explanations
│   │   ├── analytics.py            # A/B test results, RAG performance
│   │   ├── dashboard.py            # Summary stats endpoint
│   │   ├── bookmarks.py            # Wrong book (add, remove, list)
│   │   ├── goals.py                # Learning goal setting + progress
│   │   └── theta.py                # IRT theta update
│   ├── services/
│   │   ├── auth_service.py         # JWT + bcrypt authentication
│   │   ├── rag_service.py          # Qdrant + OpenAI embedding
│   │   ├── explanation_service.py  # 3-tier fallback logic
│   │   ├── tutor_agent.py          # SocraticTutorAgent (LangChain)
│   │   ├── conversation_manager.py # Multi-turn state + Bloom's tracking
│   │   ├── ab_testing.py           # A/B experiment framework
│   │   └── email_service.py        # Daily reminder emails (SMTP)
│   ├── ml/
│   │   ├── rag_evaluator.py        # Precision@K, Recall@K, MRR, F1@K
│   │   └── llm_evaluator.py        # LLM-as-Judge quality scoring
│   └── tests/                      # 297 pytest test cases
├── engine/
│   ├── scoring.py                  # IRT 3PL (probability, information, MLE calibration)
│   ├── bandit_selector.py          # Thompson Sampling question selector
│   ├── spaced_repetition.py        # Half-Life Regression forgetting curve
│   ├── skill_encoder.py            # Skill ↔ vector encoding for DKT
│   ├── dkt_model.py                # Deep Knowledge Tracing (LSTM + numpy)
│   └── recommender.py              # BKT/DKT scoring + SR injection + bandit selection
├── scripts/
│   ├── train_dkt.py                # DKT model training + evaluation
│   ├── index_to_rag.py             # Batch indexer → Qdrant
│   ├── analyze_ab_tests.py         # A/B statistical analysis (t-test, Cohen's d)
│   ├── evaluate_llm_quality.py     # LLM explanation quality evaluation
│   └── send_reminders.py           # Daily email reminder cron job
├── utils/
│   └── db_handler.py               # SQLite schema, queries, migrations
├── docker-compose.yml              # Qdrant + Backend + Frontend
├── backend/Dockerfile              # FastAPI container image
└── docs/
    ├── api.md                      # Full API endpoint reference
    ├── resume_bullet_points.md     # Project highlights
    └── demo_script.md              # 5-minute demo walkthrough
```

---

## Evaluation Scripts

```bash
# Train DKT model (auto-selects LSTM or numpy)
python scripts/train_dkt.py --epochs 10 --compare

# Re-index questions into Qdrant
python scripts/index_to_rag.py --force

# Analyze A/B test results (significance + effect size)
python scripts/analyze_ab_tests.py

# Evaluate LLM explanation quality
python scripts/evaluate_llm_quality.py
```

---

## Documentation

- [API Reference](docs/api.md) — All endpoints with request/response examples
- [Demo Script](docs/demo_script.md) — 5-minute walkthrough
- [Resume Bullet Points](docs/resume_bullet_points.md) — Project highlights for job applications

---

## License

MIT — © 2025 GlitchMind
