# LogicMaster AI Demo Script (5 Minutes)

## Slide 1: Problem (30 seconds)

"GMAT test-takers need adaptive practice systems, but existing platforms have key limitations:
- Static question banks with no personalization
- Low-quality feedback — answers without guided reasoning
- No data-driven optimization of teaching strategies"

## Slide 2: Solution Overview (30 seconds)

"LogicMaster AI is an AI-native adaptive learning platform for GMAT Critical Reasoning:
- **IRT engine** dynamically adjusts question difficulty to student ability
- **RAG-enhanced LLM** generates high-quality explanations using similar question context
- **LangChain Socratic Agent** guides students through multi-turn reasoning dialogue
- **A/B testing framework** continuously evaluates and optimizes tutoring strategies"

## Slide 3: Live Demo — Core Practice Flow (2 minutes)

*[Open Streamlit at http://localhost:8501]*

1. **Show the Practice page** — point out the sidebar navigation (Practice / Analytics / Settings) and health indicators (API, DB, Qdrant status)

2. **Display a question** — highlight the styled question card with color-coded badges (question type, difficulty, skills) and the full-text radio choices

3. **Answer incorrectly on purpose** — the Socratic Agent activates:
   - Show the first hint (gentle guidance)
   - Type a partial response → Agent evaluates understanding as "partial"
   - Show the second hint (moderate, more specific)
   - Type a correct insight → Agent evaluates as "clear" and concludes

4. **Show the RAG-enhanced explanation** — point out:
   - The explanation source label (rag_enhanced / cached / llm_only)
   - Similar question references with similarity scores
   - The loading spinner during generation

5. **Move to next question** — show the adaptive recommendation adjusting difficulty based on the updated theta

## Slide 4: Technical Architecture (1 minute)

*[Show architecture diagram from README or a slide]*

"The system follows a clean API-driven architecture:

- **Streamlit frontend** communicates with the **FastAPI backend** over HTTP
- Backend wraps existing `engine/` modules (IRT scoring, BKT recommendation) as REST endpoints
- **Qdrant** stores vector embeddings of all 50 questions for retrieval
- **DeepSeek LLM** powers both the Socratic agent and explanation generation via OpenAI SDK
- **SQLite** stores the question bank and user logs

Key design decisions:
- 3-tier explanation fallback ensures reliability (cached → RAG → plain LLM)
- Conversation state is managed in-memory with TTL cleanup
- A/B variant assignment uses consistent hashing for reproducibility"

## Slide 5: Data Science Capabilities (1 minute)

*[Open the Analytics page in Streamlit]*

"The platform includes a full data science evaluation stack:

1. **A/B Testing** — comparing Socratic vs. direct explanation strategies
   - Show per-variant metrics (accuracy, theta gain, sample sizes)
   - Highlight statistical significance results (p-value, Cohen's d)

2. **RAG Evaluation** — Precision@K, Recall@K, MRR metrics
   - Show the embedding model config and indexed question count

3. **LLM Quality Assessment** — GPT-4-as-judge scoring
   - 4 criteria: correctness, clarity, completeness, pedagogical value

All evaluation scripts are reproducible:
```bash
python scripts/analyze_ab_tests.py
python scripts/evaluate_llm_quality.py
```"

## Slide 6: Impact and Next Steps (30 seconds)

"Results from the 6-week upgrade:
- **86 automated tests** covering all major components
- **10 API endpoints** across 5 routers with full Swagger documentation
- **3-tier explanation fallback** for reliable output
- **Multi-turn Socratic dialogue** with progressive hint strength
- **A/B testing framework** for continuous optimization

Next steps:
- Multi-modal support (diagram-based CR questions)
- Reinforcement learning for agent strategy optimization
- Cloud deployment (AWS/GCP) with CI/CD pipeline
- Mobile-responsive frontend"
