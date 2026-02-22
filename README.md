# LogicMaster AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-gmat.glitchmind.io-brightgreen?style=for-the-badge)](https://gmat.glitchmind.io)

> AI-Native GMAT Critical Reasoning Tutor — A GlitchMind Product

**Live:** [gmat.glitchmind.io](https://gmat.glitchmind.io) &nbsp;|&nbsp; **API:** [api.glitchmind.io](https://api.glitchmind.io) &nbsp;|&nbsp; **API Docs:** [api.glitchmind.io/docs](https://api.glitchmind.io/docs)

---

## 🌐 Live Demo

**👉 [https://gmat.glitchmind.io](https://gmat.glitchmind.io)**

---

## 1. Project Overview

LogicMaster AI is a **full-stack adaptive learning platform** for GMAT Critical Reasoning preparation.

**Core problems solved:**
- Traditional practice platforms use fixed questions with no adaptive difficulty, resulting in poor efficiency
- After answering incorrectly, students receive no personalized guidance — just answers without logical explanation
- No way to track students' true cognitive weak points and forgetting patterns

**Solution:**
- IRT 3PL model estimates student ability (θ) in real time and dynamically adjusts question difficulty
- When a student answers incorrectly, the Socratic AI tutor guides them to discover their logical errors through dialogue — never revealing the answer directly
- Deep Knowledge Tracing (DKT LSTM) tracks mastery across each skill dimension
- Half-Life Regression forgetting curve calculates optimal review timing

---

## 2. Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Zustand, Framer Motion, Recharts, Lucide |
| **Backend** | FastAPI, Python 3.12, Pydantic v2, LangChain, Uvicorn |
| **AI / LLM** | DeepSeek API (LLM inference), LangChain-OpenAI wrapper |
| **Vector Search** | OpenAI `text-embedding-3-small` (1536 dims), Qdrant vector database |
| **ML Algorithms** | IRT 3PL, Thompson Sampling, DKT LSTM (PyTorch), Half-Life Regression, BKT, Bloom's Taxonomy |
| **Database** | SQLite (application data), Qdrant (vector index) |
| **Authentication** | JWT (PyJWT, 7-day expiry), bcrypt password hashing |
| **Testing** | pytest — 297 test cases, 100% passing |
| **Deployment** | AWS EC2 (backend), AWS S3 + CloudFront (frontend static), Nginx reverse proxy, systemd |
| **Containers** | Docker Compose (local dev: Qdrant + Backend + Frontend) |

---

## 3. Project Directory Structure

```
mathquest_logicmaster/
│
├── frontend/                        # React + TypeScript frontend
│   ├── src/
│   │   ├── pages/                   # Page components
│   │   │   ├── Login.tsx            # Login / registration page
│   │   │   ├── Dashboard.tsx        # Learning dashboard
│   │   │   ├── Practice.tsx         # Practice page (core interaction)
│   │   │   ├── WrongBook.tsx        # Wrong answers notebook
│   │   │   ├── Review.tsx           # Spaced repetition review page
│   │   │   ├── Analytics.tsx        # Learning analytics charts
│   │   │   ├── Profile.tsx          # User profile
│   │   │   └── Settings.tsx         # Settings page
│   │   ├── components/              # Reusable UI components
│   │   ├── lib/
│   │   │   ├── api.ts               # API client (wraps all HTTP requests)
│   │   │   └── config.ts            # API_BASE_URL configuration
│   │   ├── store/
│   │   │   ├── useAppStore.ts       # Zustand global state (questions / progress)
│   │   │   └── useAuthStore.ts      # Zustand auth state (JWT token)
│   │   └── hooks/                   # Custom React hooks
│   ├── .env.production              # Production env vars (VITE_API_BASE_URL)
│   ├── vite.config.ts               # Vite configuration
│   └── package.json
│
├── backend/                         # FastAPI backend
│   ├── main.py                      # App entry: CORS, router registration, /health
│   ├── config.py                    # pydantic-settings config (reads .env)
│   ├── Dockerfile                   # Backend container image
│   ├── pytest.ini                   # pytest configuration
│   ├── routers/
│   │   ├── auth.py                  # /api/auth — register / login / profile / stats
│   │   ├── questions.py             # /api/questions — adaptive selection / bandit / review
│   │   ├── tutor.py                 # /api/tutor — Socratic tutor conversation
│   │   ├── explanations.py          # /api/explanations — RAG explanation generation
│   │   ├── analytics.py             # /api/analytics — A/B testing / learning analytics
│   │   ├── dashboard.py             # /api/dashboard — dashboard statistics
│   │   ├── bookmarks.py             # /api/bookmarks — bookmarks / wrong answers
│   │   ├── goals.py                 # /api/goals — learning goals
│   │   └── theta.py                 # /api/theta — IRT ability update
│   ├── services/
│   │   ├── auth_service.py          # JWT issuance / verification, bcrypt hashing
│   │   ├── rag_service.py           # RAGService: Qdrant embedding retrieval
│   │   ├── explanation_service.py   # 3-tier fallback: cache → RAG → plain LLM
│   │   ├── tutor_agent.py           # SocraticTutorAgent (LangChain Agent)
│   │   ├── conversation_manager.py  # Multi-turn conversation state + Bloom's tracking
│   │   ├── ab_testing.py            # A/B testing framework (deterministic assignment + logging)
│   │   └── email_service.py         # SMTP daily reminder email service
│   ├── ml/
│   │   ├── rag_evaluator.py         # RAG evaluation: Precision@K, Recall@K, MRR, F1@K
│   │   └── llm_evaluator.py         # LLM-as-Judge explanation quality scoring
│   └── tests/                       # 297 pytest tests
│       ├── test_api.py              # Basic API tests (16)
│       ├── test_rag.py              # RAG system tests (21)
│       ├── test_tutor_agent.py      # Tutor agent tests (26)
│       ├── test_blooms_taxonomy.py  # Bloom's taxonomy tests (15)
│       ├── test_ab_testing.py       # A/B testing framework tests (20)
│       ├── test_bandit.py           # Thompson Sampling tests (24)
│       ├── test_spaced_repetition.py # Spaced repetition tests (20)
│       ├── test_dkt.py              # DKT model tests (39)
│       ├── test_irt_3pl.py          # IRT 3PL tests (18)
│       ├── test_features.py         # Feature tests (51): dashboard / bookmarks / goals / email
│       └── test_auth.py             # Authentication tests (44)
│
├── engine/                          # Core ML engine (no FastAPI dependency)
│   ├── scoring.py                   # IRT 3PL: probability_3pl, calculate_new_theta, estimate_gmat_score
│   ├── bandit_selector.py           # Thompson Sampling question selector (Beta dist + SQLite)
│   ├── spaced_repetition.py         # Half-Life Regression forgetting curve
│   ├── skill_encoder.py             # SkillEncoder: skill ↔ vector encoding (DKT input)
│   ├── dkt_model.py                 # DKTModelNumpy (cold start) + DKTModelLSTM (PyTorch)
│   ├── recommender.py               # Recommendation pipeline: BKT/DKT → SR injection → Bandit selection
│   └── __init__.py                  # SkillEncoder / get_dkt_model exports
│
├── utils/
│   └── db_handler.py                # DatabaseManager: SQLite schema + all DB operations
│
├── scripts/
│   ├── train_dkt.py                 # DKT model training (argparse, user-level split, early stopping)
│   ├── index_to_rag.py              # Batch index SQLite → Qdrant
│   ├── analyze_ab_tests.py          # A/B statistical analysis (t-test, Cohen's d)
│   ├── evaluate_llm_quality.py      # Batch LLM explanation quality evaluation
│   └── send_reminders.py            # Daily reminder email cron script
│
├── docs/
│   ├── api.md                       # Complete API endpoint reference
│   ├── demo_script.md               # 5-minute demo walkthrough script
│   └── resume_bullet_points.md      # Resume bullet points
│
├── app.py                           # Legacy Streamlit frontend (retained, superseded by React)
├── llm_service.py                   # DeepSeek LLM wrapper (tutor_reply backward compat)
├── logicmaster.db                   # SQLite database (~50 questions, ~421KB in production)
├── docker-compose.yml               # Local dev: Qdrant + Backend + Frontend
├── Dockerfile.streamlit             # Streamlit frontend container (legacy)
├── .env                             # Local environment variables (not committed to git)
├── .env.example                     # Environment variable template
├── requirements-backend.txt         # Python backend dependencies
└── requirements.txt                 # Legacy Streamlit dependencies
```

---

## 4. Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```dotenv
# ===== Required: LLM API Keys =====
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
OPENAI_API_KEY=sk-your-openai-key-here

# ===== Qdrant Vector Database (Docker defaults, no change needed) =====
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=gmat_explanations

# ===== OpenAI Embedding Config =====
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMS=1536

# ===== JWT Authentication (MUST change SECRET in production!) =====
JWT_SECRET_KEY=glitchmind-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

# ===== SMTP Email Reminders (optional, leave blank to disable) =====
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# ===== Application Config =====
APP_ENV=development        # development | production
DEBUG=True
DAILY_QUESTION_GOAL=5      # Default daily question target
```

Frontend production environment variable (`frontend/.env.production`):

```dotenv
VITE_API_BASE_URL=https://api.glitchmind.io
```

---

## 5. Local Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)
- `DEEPSEEK_API_KEY` + `OPENAI_API_KEY`

### Backend

```bash
# 1. Clone the repository
git clone <repo-url>
cd mathquest_logicmaster

# 2. Configure environment variables
cp .env.example .env
# Edit .env and fill in API keys

# 3. Install Python dependencies (virtual environment recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-backend.txt

# 4. Start Qdrant (Docker)
docker-compose up -d qdrant

# 5. Index questions into the vector database (first run only)
python scripts/index_to_rag.py

# 6. Start FastAPI backend
cd backend
uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Visit: http://localhost:5173
```

### Full Docker Setup (local one-command)

```bash
cp .env.example .env
# Edit .env
docker-compose up -d
# Backend: http://localhost:8000
# Frontend (legacy Streamlit): http://localhost:8501
```

### Running Tests

```bash
cd backend
pytest tests/ -v
# Expected: 297 tests, all passing
```

---

## 6. Production Deployment Architecture

```
User Browser
    │ HTTPS
    ▼
AWS CloudFront + S3
  (gmat.glitchmind.io)
  Static files: React build dist/
    │
    │ HTTPS API requests
    ▼
Nginx (EC2, api.glitchmind.io)
  ┌─────────────────────────────────┐
  │  server { listen 443 ssl; }     │
  │    proxy_pass → 127.0.0.1:8000 │
  │  server { listen 80; }          │
  │    OPTIONS → direct 204         │
  │    other   → 301 → HTTPS        │
  └─────────────────────────────────┘
    │
    ▼
FastAPI (uvicorn, systemd service: logicmaster)
  Port 8000, process-managed
    │
    ├── SQLite (logicmaster.db)
    └── Qdrant (Docker, ports 6333/6334)
```

### systemd Service Management

```bash
# Service file location: /etc/systemd/system/logicmaster.service
sudo systemctl start logicmaster
sudo systemctl stop logicmaster
sudo systemctl restart logicmaster
sudo systemctl status logicmaster

# View live logs
journalctl -u logicmaster -f
```

### Nginx Configuration (key: CORS + OPTIONS handling)

```nginx
server {
    listen 80;
    server_name api.glitchmind.io;

    # OPTIONS preflight bypasses redirect and responds directly
    if ($request_method = OPTIONS) {
        return 204;
    }
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name api.glitchmind.io;

    ssl_certificate     /etc/letsencrypt/live/api.glitchmind.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.glitchmind.io/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Note: CORS headers are NOT added at the Nginx layer; handled by FastAPI CORSMiddleware
    }
}
```

### Deployment Update Flow

```bash
# Run on the server
cd /path/to/mathquest_logicmaster
git pull
sudo systemctl restart logicmaster

# Frontend update (build locally, then upload to S3)
cd frontend
npm run build
aws s3 sync dist/ s3://your-bucket-name/ --delete
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

---

## 7. API Endpoints

**Base URL (production):** `https://api.glitchmind.io`
**Swagger UI:** `https://api.glitchmind.io/docs`

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check (DB + Qdrant status) |

### Authentication `/api/auth`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/register` | None | Register (email + password), returns JWT |
| POST | `/api/auth/login` | None | Login, returns JWT |
| GET | `/api/auth/me` | Bearer | Get current user profile |
| PUT | `/api/auth/profile` | Bearer | Update display name |
| PUT | `/api/auth/change-password` | Bearer | Change password |
| DELETE | `/api/auth/account` | Bearer | Delete account and all data |
| GET | `/api/auth/stats` | Bearer | Get learning stats (total questions / accuracy / best streak / estimated GMAT score) |

### Questions `/api/questions`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/questions/next` | Get next adaptive question (`strategy: bandit\|legacy`) |
| POST | `/api/questions/bandit-update` | Update Bandit / SR / DKT / wrong book after answering |
| GET | `/api/questions/review-schedule` | Get spaced repetition review queue |
| GET | `/api/questions/{question_id}` | Get a specific question by ID |

### Socratic Tutor `/api/tutor`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/tutor/start-remediation` | Start tutor session (diagnose error + first hint + A/B assignment) |
| POST | `/api/tutor/continue` | Continue session (Bloom's evaluation → next hint or conclusion) |
| POST | `/api/tutor/conclude` | End session, return summary |
| POST | `/api/tutor/chat` | Legacy stateless chat (backward compatible) |

### Explanations `/api/explanations`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/explanations/generate-with-rag` | Generate RAG-enhanced explanation (3-tier fallback) |
| POST | `/api/explanations/search-similar` | Vector search for similar questions |

### Dashboard `/api/dashboard`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dashboard/summary` | Dashboard summary (today's progress / streak / theta / estimated GMAT score / weak skills / reviews due) |

### Bookmarks / Wrong Book `/api/bookmarks`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/bookmarks/add` | Add bookmark / wrong answer (idempotent) |
| DELETE | `/api/bookmarks/remove` | Remove bookmark |
| GET | `/api/bookmarks/list` | Query list (supports type / skill filtering) |
| GET | `/api/bookmarks/wrong-stats` | Wrong answer statistics (by skill / question type distribution) |

### Learning Goals `/api/goals`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/goals/set` | Set target GMAT score + daily question count |
| GET | `/api/goals/progress` | Get goal progress (score gap / estimated remaining questions / on-track status) |

### Analytics `/api/analytics`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analytics/log-outcome` | Record A/B experiment outcome |
| GET | `/api/analytics/ab-test-results` | Get A/B test aggregated statistics |
| GET | `/api/analytics/summary` | Learning analytics summary (answer history / wrong analysis / skill mastery) |
| GET | `/api/analytics/rag-performance` | RAG system performance metrics |

### IRT Ability `/api/theta`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/theta/update` | Update theta (3PL IRT), returns new theta + estimated GMAT score |

---

## 8. Implemented Features

### Adaptive Learning Engine

- **IRT 3PL Scoring** — Three-parameter logistic model (discrimination *a*, difficulty *b*, guessing *c*), MLE calibration (L-BFGS-B), mapped to GMAT Verbal scale (V20–V51)
- **Thompson Sampling Question Selection** — Multi-armed bandit, Beta(α, β) prior, updated after each answer; automatic explore/exploit balance
- **Deep Knowledge Tracing** — PyTorch LSTM (Piech et al. 2015) predicts skill mastery per dimension; cold start (<50 interactions) automatically falls back to numpy logistic regression
- **Spaced Repetition** — Half-Life Regression (Settles & Meeder 2016) models the forgetting curve; injects review questions when recall probability < 0.5 (40% injection rate)
- **BKT Skill Tracking** — Bayesian Knowledge Tracing covering 10+ cognitive skills (Causal Reasoning, Assumption Identification, etc.)
- **Hybrid Recommendation Pipeline** — BKT/DKT weakness scoring → SR injection → Thompson Sampling final selection

### AI Tutor System

- **Socratic Tutor** — LangChain Agent (DeepSeek LLM), progressive hints: gentle (hint 1) → moderate (hint 2) → direct (hint 3), never reveals the answer directly
- **Bloom's Taxonomy Cognitive Evaluation** — LLM evaluates students' cognitive level in real time (levels 1–6: Remember → Create) and adjusts hint strategy accordingly
- **RAG Explanations** — Qdrant + OpenAI embeddings, 3-tier fallback: cache hit → RAG-enhanced → plain LLM

### User Features

- **JWT Authentication** — Register / login / profile management / password change / account deletion (7-day token, bcrypt hashing)
- **Dashboard** — Real-time stats: estimated GMAT score, streak days, theta, weak skills, reviews due, 7-day calendar
- **Wrong Book** — Wrong answers added automatically; supports filtering by skill / question type with statistical analysis
- **Learning Goals** — Set target GMAT score and daily question count; track progress and achievement
- **Learning Analytics** — Answer history curve, wrong answer distribution pie chart, skill mastery radar

### Data Science Infrastructure

- **A/B Testing Framework** — Deterministic variant assignment (hash-based); two experiments: `tutor_strategy` (socratic_standard / socratic_aggressive / direct_explanation) and `explanation_source` (rag_enhanced / baseline); t-test significance + Cohen's d effect size analysis
- **LLM-as-Judge** — DeepSeek automatically evaluates explanation quality (correctness / clarity / completeness / pedagogical value, 0–10)

---

## 9. Known Issues / TODO

### Known Issues

| Issue | Status | Description |
|-------|--------|-------------|
| CORS | In progress | Nginx HTTP→HTTPS redirect intercepts OPTIONS preflight; `backend/main.py` CORS config corrected (explicit domain allowlist), pending `git pull + restart` on server |
| `config.ts` fallback URL | Fixed | Legacy fallback used `http://` (non-HTTPS); changed to `https://api.glitchmind.io` |
| SQLite concurrency | Known limitation | SQLite is not suitable for high-concurrency writes; production scaling requires migration to PostgreSQL |
| DKT cold start | Handled | Automatically falls back to numpy model when <50 interactions, but initial recommendation quality is limited |
| Bandit stats no user_id | Known | `bandit_stats` table has no `user_id` column; Bandit statistics are globally shared (not per-user) |

### TODO

- [ ] PostgreSQL migration (replace SQLite)
- [ ] Expand question bank (currently ~50 questions)
- [ ] Per-user Bandit statistics (currently globally shared)
- [ ] Question generation pipeline (LLM auto-generation + human review)
- [ ] Mobile layout optimization
- [ ] DKT model periodic online learning (currently offline training)
- [ ] Qdrant persistent backup solution
- [ ] Rate limiting (no request limits currently)

---

## 10. Database Schema Overview

Database file: `logicmaster.db` (SQLite)

### `questions` — Question Bank

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Unique question ID |
| `question_type` | TEXT | Type (Weaken / Strengthen / Assumption / Inference / Flaw / Evaluate / Boldface) |
| `difficulty` | TEXT | Difficulty (easy / medium / hard) |
| `content` | TEXT | Question JSON (stimulus, question, choices, correct, skills, explanation) |
| `elo_difficulty` | REAL | IRT difficulty parameter b (default 1500.0) |
| `is_verified` | INTEGER | Review status (1 = approved; only verified questions are recommended) |
| `discrimination` | REAL | 3PL discrimination parameter a (default 1.0) |
| `guessing` | REAL | 3PL guessing parameter c (default 0.2) |

### `users` — User Accounts

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `email` | TEXT UNIQUE | Email (unique index) |
| `password_hash` | TEXT | bcrypt hash |
| `display_name` | TEXT | Display name (optional) |
| `created_at` | TIMESTAMP | Registration time |

### `answer_history` — Answer History (DKT training data)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `user_id` | TEXT | User ID (default "default") |
| `question_id` | TEXT | Question ID |
| `skill_ids` | TEXT | Skill list JSON |
| `is_correct` | INTEGER | 0 / 1 |
| `theta_at_time` | REAL | Theta value at time of answer |
| `created_at` | TIMESTAMP | Answer timestamp (indexed: user_id + created_at) |

### `spaced_repetition_stats` — Spaced Repetition Statistics

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | TEXT | User ID (composite PK) |
| `question_id` | TEXT | Question ID (composite PK) |
| `half_life` | REAL | Memory half-life (days) |
| `last_practiced` | TIMESTAMP | Last practice time |
| `n_correct` | INTEGER | Number of correct answers |
| `n_attempts` | INTEGER | Total attempts |

### `bookmarks` — Bookmarks / Wrong Book

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | TEXT | User ID (composite PK) |
| `question_id` | TEXT | Question ID (composite PK) |
| `bookmark_type` | TEXT | "favorite" or "wrong" (composite PK) |
| `created_at` | TIMESTAMP | Time added |

### `learning_goals` — Learning Goals

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | TEXT PK | User ID |
| `target_gmat_score` | INTEGER | Target GMAT score (20–51) |
| `daily_question_goal` | INTEGER | Daily question target |
| `created_at` / `updated_at` | TIMESTAMP | Created / updated time |

### `experiment_logs` — A/B Test Logs

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `user_id` | TEXT | User ID |
| `experiment_name` | TEXT | Experiment name (tutor_strategy / explanation_source) |
| `variant` | TEXT | Variant name |
| `event_type` | TEXT | "exposure" or "outcome" |
| `outcome_metric` | TEXT | Metric name (is_correct, theta_gain, etc.) |
| `outcome_value` | REAL | Metric value |
| `created_at` | TIMESTAMP | Record time |

### `email_logs` — Email Send Records

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `user_id` | TEXT | User ID |
| `email_type` | TEXT | Email type |
| `sent_at` | TIMESTAMP | Send time (24h dedup basis) |

### `bandit_stats` (engine SQLite, independent of users) — Thompson Sampling Statistics

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | TEXT PK | Question ID |
| `alpha` | REAL | Beta distribution α parameter (successes + 1) |
| `beta` | REAL | Beta distribution β parameter (failures + 1) |
| `n_trials` | INTEGER | Total selection count |
| `n_successes` | INTEGER | Correct answer count |

---

## Evaluation Scripts

```bash
# Train DKT model (automatically selects LSTM or numpy)
python scripts/train_dkt.py --epochs 10 --compare

# Re-index questions into Qdrant
python scripts/index_to_rag.py --force

# A/B test statistical analysis (significance + effect size)
python scripts/analyze_ab_tests.py

# LLM explanation quality evaluation
python scripts/evaluate_llm_quality.py

# Manually send daily reminder emails
python scripts/send_reminders.py
```

---

## License

MIT — © 2025 GlitchMind
