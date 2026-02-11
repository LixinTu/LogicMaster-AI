# MathQuest Labs — LogicMaster
LogicMaster AI: Adaptive GMAT Training System

Overview
LogicMaster AI is an advanced adaptive learning platform designed for GMAT Critical Reasoning (CR). The system utilizes educational psychometrics and machine learning to provide a personalized high-stakes testing experience.

Technical Core
1. Dynamic Ability Estimation (IRT/Elo-based)
The system uses Item Response Theory (IRT) principles to estimate user ability:

Latent Ability (Theta): The engine calculates a latent ability value (theta) typically ranging from -3.0 to +3.0.

Probability Modeling: A Logistic Sigmoid function is used to calculate the expected probability of a correct answer based on current theta and item difficulty.

GMAT Score Mapping: Theta values are mapped to the GMAT Verbal scale (V20 - V51) for statistically grounded score estimation.

2. Knowledge Tracing and Recommendation (BKT)
The platform models mastery of specific logical skills using Bayesian Knowledge Tracing (BKT) concepts:

Skill Profiling: Real-time tracking of performance across sub-skills such as Causal Reasoning, Assumption Identification, and Alternative Explanations.

Hybrid Recommendation: The engine combines Ability Matching (Elo) with Weakness-First (BKT) strategies to prioritize questions targeting specific logical gaps.

3. Socratic AI Tutor (Multi-Agent System)
Powered by DeepSeek LLM, the system features a specialized teaching agent:

Error Diagnosis: Upon an incorrect response, the system performs a diagnostic analysis of the student's logical misconception.

Socratic Dialogue: The AI Tutor uses the Socratic Method to guide students through their own reasoning errors instead of revealing answers.

Architecture
Frontend: Streamlit-based responsive dashboard.

Engine: IRT scoring engine and BKT-driven recommendation logic (located in engine/).

Intelligence: DeepSeek API for Socratic tutoring and item generation.

Database: SQLite with management logic located in utils/db_handler.py.

Quick Start
Environment Setup: Run: pip install -r requirements.txt

Initialize Database: Run: python database.py (Note: Ensure this script references the correct relative paths for initialization)

Populate Adaptive Item Pool: Run: python generate_pool.py This script generates items with latent difficulty tags and diagnostic metadata.

Launch Application: Run: streamlit run app.py

Engineering Notes
Environment Security: API Keys are accessed via environment variables from a .env file.

Defensive Programming: Includes error handling for database access and API rate limits.

Modular Design: Separates mathematical scoring functions (engine/) from UI (app.py) and database utilities (utils/).