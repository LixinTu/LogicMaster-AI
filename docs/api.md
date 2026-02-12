# API Reference

**Base URL**: `http://localhost:8000`

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Health Check

### GET /health

Returns service status for the API, SQLite database, and Qdrant vector DB.

**Response:**
```json
{
  "status": "ok",
  "env": "development",
  "db_status": "connected",
  "qdrant_status": "connected"
}
```

---

## Theta Router (`/api/theta`)

### POST /api/theta/update

Update user ability estimate (theta) based on an answer attempt. Uses the IRT 3PL model from `engine/scoring.py`.

**Request:**
```json
{
  "current_theta": 0.5,
  "question_difficulty": 1500.0,
  "is_correct": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `current_theta` | float | Yes | Current ability value, range [-3.0, 3.0] |
| `question_difficulty` | float | Yes | Item difficulty parameter |
| `is_correct` | bool | Yes | Whether the answer was correct |

**Response:**
```json
{
  "new_theta": 0.62,
  "gmat_score": 38
}
```

---

## Questions Router (`/api/questions`)

### POST /api/questions/next

Get the next adaptively recommended question. Uses IRT + BKT hybrid recommendation from `engine/recommender.py`. The `correct` answer is intentionally excluded from the response to prevent cheating.

**Request:**
```json
{
  "user_theta": 0.5,
  "current_q_id": "q001",
  "questions_log": [
    {
      "question_id": "q001",
      "skills": ["Causal Reasoning"],
      "is_correct": false
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `user_theta` | float | Yes | Current ability value, range [-3.0, 3.0] |
| `current_q_id` | string | No | Current question ID to exclude from recommendation |
| `questions_log` | array | No | History of past attempts |

**Response:**
```json
{
  "question_id": "q015",
  "question_type": "Weaken",
  "difficulty": "medium",
  "elo_difficulty": 1520.0,
  "stimulus": "A company recently introduced a new product...",
  "question": "Which of the following, if true, most seriously weakens the argument?",
  "choices": [
    "A. The company has a strong brand reputation...",
    "B. Competitors have launched similar products...",
    "C. The market research was conducted...",
    "D. Consumer preferences have shifted...",
    "E. The product was tested extensively..."
  ],
  "skills": ["Causal Reasoning", "Alternative Explanations"]
}
```

**Error (404):** No suitable question found.

---

## Tutor Router (`/api/tutor`)

### POST /api/tutor/chat

Backward-compatible stateless Socratic dialogue endpoint (Week 1). Uses `llm_service.tutor_reply` directly.

**Request:**
```json
{
  "message": "Why is A wrong?",
  "chat_history": [
    {"role": "user", "content": "I chose A"},
    {"role": "assistant", "content": "Let's think about the assumption..."}
  ],
  "question_id": "q001",
  "current_q": { "stimulus": "...", "question": "...", "choices": ["..."] },
  "socratic_context": { "diagnosis": "...", "logic_gap": "..." }
}
```

**Response:**
```json
{
  "reply": "Consider what the argument assumes about the causal link...",
  "is_error": false
}
```

### POST /api/tutor/start-remediation

Start a new multi-turn remediation conversation. Performs A/B variant assignment, error diagnosis via LangChain Agent, and generates the first Socratic hint.

**Request:**
```json
{
  "question_id": "q001",
  "question": {
    "stimulus": "A company recently introduced...",
    "question": "Which of the following weakens...",
    "choices": ["A. ...", "B. ...", "C. ...", "D. ...", "E. ..."]
  },
  "user_choice": "A",
  "correct_choice": "C",
  "user_id": "session_abc123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question_id` | string | Yes | Question identifier |
| `question` | object | Yes | Full question dictionary |
| `user_choice` | string | Yes | Student's answer (A-E) |
| `correct_choice` | string | Yes | Correct answer (A-E) |
| `user_id` | string | No | User ID for A/B test assignment |

**Response:**
```json
{
  "conversation_id": "a1b2c3d4-...",
  "first_hint": "Let's examine what the argument assumes. What connects the premises to the conclusion?",
  "logic_gap": "Student confused correlation with causation",
  "error_type": "causal_confusion",
  "hint_count": 1,
  "student_understanding": "confused",
  "current_state": "hinting",
  "variant": "socratic_standard"
}
```

### POST /api/tutor/continue

Continue an existing remediation conversation. Evaluates student understanding and generates the next hint or concludes.

**Request:**
```json
{
  "conversation_id": "a1b2c3d4-...",
  "student_message": "I think the argument assumes the new product will succeed because of brand reputation",
  "question": { "...": "..." },
  "correct_choice": "C"
}
```

**Response:**
```json
{
  "reply": "Good observation! Now consider whether brand reputation alone guarantees...",
  "hint_count": 2,
  "student_understanding": "partial",
  "should_continue": true,
  "current_state": "hinting"
}
```

When `should_continue` is `false`, the tutor has concluded the dialogue.

### POST /api/tutor/conclude

Explicitly end a remediation conversation and get a summary.

**Request:**
```json
{
  "conversation_id": "a1b2c3d4-...",
  "question": { "...": "..." },
  "correct_choice": "C"
}
```

**Response:**
```json
{
  "conclusion": "The correct answer is C. The argument's key assumption was...",
  "summary": {
    "total_turns": 4,
    "hint_count": 2,
    "final_understanding": "clear",
    "time_spent_seconds": 95.3
  }
}
```

---

## Explanations Router (`/api/explanations`)

### POST /api/explanations/generate-with-rag

Generate an explanation using 3-tier fallback: cached → RAG-enhanced → plain LLM.

**Request:**
```json
{
  "question_id": "q001",
  "question": {
    "stimulus": "...",
    "question": "...",
    "choices": ["..."],
    "detailed_explanation": "..."
  },
  "user_choice": "A",
  "is_correct": false
}
```

**Response:**
```json
{
  "explanation": "The correct answer is C because...",
  "similar_references": [
    {"question_id": "q015", "similarity": 0.92},
    {"question_id": "q032", "similarity": 0.87}
  ],
  "source": "rag_enhanced"
}
```

| `source` value | Meaning |
|---|---|
| `cached` | Used pre-existing explanation from the database |
| `rag_enhanced` | Generated with similar questions as few-shot context |
| `llm_only` | RAG unavailable; generated with LLM alone |

### POST /api/explanations/search-similar

Search for similar questions in the Qdrant vector database.

**Request:**
```json
{
  "query": "causal reasoning argument about company market share",
  "top_k": 5,
  "skills": ["Causal Reasoning"]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query text |
| `top_k` | int | No | Number of results (default: 5) |
| `skills` | array | No | Filter by skill tags |

**Response:**
```json
{
  "results": [
    {
      "question_id": "q015",
      "explanation": "This question tests causal reasoning...",
      "similarity_score": 0.9234,
      "question_type": "Weaken",
      "skills": ["Causal Reasoning"]
    }
  ]
}
```

---

## Analytics Router (`/api/analytics`)

### POST /api/analytics/log-outcome

Log an experimental outcome metric from the frontend.

**Request:**
```json
{
  "user_id": "session_abc123",
  "experiment_name": "tutor_strategy",
  "variant": "socratic_standard",
  "metric": "is_correct",
  "value": 1.0,
  "metadata": {"question_id": "q001", "attempt": 2}
}
```

**Response:**
```json
{
  "ok": true,
  "message": "logged"
}
```

### GET /api/analytics/ab-test-results

Get aggregated A/B test statistics for an experiment.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment` | string | `tutor_strategy` | Experiment name |

**Response:**
```json
{
  "experiment": "tutor_strategy",
  "active": true,
  "description": "Socratic vs Direct explanation tutoring strategies",
  "total_exposures": 150,
  "total_outcomes": 120,
  "variants": {
    "socratic_standard": {
      "exposures": 80,
      "outcomes": {
        "is_correct": {"count": 65, "mean": 0.78, "sum": 51.0},
        "theta_gain": {"count": 65, "mean": 0.15, "sum": 9.75}
      }
    },
    "direct_explanation": {
      "exposures": 70,
      "outcomes": {
        "is_correct": {"count": 55, "mean": 0.65, "sum": 35.75}
      }
    }
  }
}
```

### GET /api/analytics/rag-performance

Get RAG system performance metrics.

**Response:**
```json
{
  "retrieval_metrics": {
    "embedding_model": "text-embedding-3-small",
    "embedding_dims": 1536,
    "collection": "gmat_explanations"
  },
  "quality_metrics": {
    "rag_avg_score": 4.28,
    "baseline_avg_score": 3.80,
    "improvement_pct": 12.7
  },
  "system_metrics": {
    "indexed_questions": 50,
    "qdrant_host": "localhost",
    "qdrant_port": 6333
  }
}
```
