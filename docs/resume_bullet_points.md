# Resume Bullet Points for LogicMaster AI

## For AI Education Companies

- Architected AI-native GMAT training platform with FastAPI backend serving 10 REST endpoints across 5 routers, with Streamlit frontend consuming the API over HTTP

- Integrated RAG (Retrieval-Augmented Generation) system using Qdrant vector database and OpenAI `text-embedding-3-small` embeddings (1536 dims), implementing 3-tier explanation fallback (cached → RAG-enhanced → plain LLM)

- Developed LangChain-based Socratic tutoring agent with multi-turn dialogue management, progressive hint strength (gentle → moderate → direct), and automated student understanding evaluation (confused → partial → clear)

- Built conversation state manager supporting concurrent sessions with UUID-based tracking, TTL-based cleanup, and graceful degradation to stateless fallback on agent failure

- Evaluated LLM-generated explanations using GPT-4-as-judge methodology with 4-criteria rubric (correctness, clarity, completeness, pedagogical value), establishing automated quality baseline

## For DA/DS Positions

- Implemented A/B testing framework using consistent hashing (MD5-based) for deterministic variant assignment across `tutor_strategy` and `explanation_source` experiments, with exposure/outcome logging and per-variant metric aggregation

- Performed statistical analysis of experiment results using two-sample t-tests (scipy), computing p-values, Cohen's d effect size, and per-variant accuracy/theta-gain metrics for data-driven strategy selection

- Calibrated 3-Parameter Logistic IRT model for adaptive difficulty estimation, mapping latent ability (theta) to GMAT Verbal scale (V20-V51) with Bayesian Knowledge Tracing across 10+ cognitive skills

- Designed retrieval evaluation framework computing Precision@K, Recall@K, MRR, and F1@K metrics for the RAG system; automated quality assessment using LLM-as-judge with structured scoring rubrics

- Built real-time analytics API endpoints aggregating A/B test results and RAG performance metrics, with Streamlit dashboard visualizing experiment outcomes and system health indicators

## Technical Skills Section

**Languages**: Python (FastAPI, LangChain, SQLAlchemy, Pandas, NumPy, SciPy)
**AI/ML**: OpenAI API, LangChain, RAG Systems (Qdrant + Embeddings), Prompt Engineering
**Databases**: SQLite, Qdrant (Vector DB)
**Data Science**: Hypothesis Testing (t-tests, Cohen's d), A/B Testing, IRT (3PL), Bayesian Knowledge Tracing
**Tools**: Docker, Docker Compose, Git, pytest (86 test cases)
**Visualization**: Plotly, Matplotlib, Streamlit

## Project Link

GitHub: github.com/yourusername/logicmaster
