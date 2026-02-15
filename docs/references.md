# Academic References

Papers and foundational work behind each component of LogicMaster AI.

## IRT 3PL (Item Response Theory — Three-Parameter Logistic Model)

**Lord, F. M. (1980).** *Applications of Item Response Theory to Practical Testing Problems.* Erlbaum.

Lord's monograph formalized the 3PL model: P(θ) = c + (1-c) / (1 + exp(-a(θ-b))). Our `engine/scoring.py` implements this directly — `probability_3pl()` computes response probability given ability θ, difficulty b, discrimination a, and guessing c. The `item_information()` function implements Lord's information formula I(θ) = a²(P-c)²(1-P)/((1-c)²P), which we use in the bandit's exploit score to select maximally informative questions for each student.

**Baker, F. B. & Kim, S.-H. (2004).** *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker.

Baker & Kim detail Maximum Likelihood Estimation for IRT parameters. Our `calibrate_item_parameters()` minimizes the negative log-likelihood using scipy's L-BFGS-B optimizer with bounds a∈[0.5, 2.5], b∈[-3, 3], c∈[0, 0.35], recovering discrimination, difficulty, and guessing from student response histories.

## BKT (Bayesian Knowledge Tracing)

**Corbett, A. T. & Anderson, J. R. (1995).** Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253-278.

Corbett & Anderson's BKT tracks per-skill mastery as a hidden variable updated after each observation. Our `engine/recommender.py:analyze_weak_skills()` adapts this idea — it maintains per-skill correct/total counts across the question log, computes error rates, and identifies the 3 weakest skills. These weak skills drive the candidate filtering stage: questions targeting weak skills receive a +0.5 scoring bonus, implementing a weakness-first recommendation strategy before the bandit makes the final selection.

## Thompson Sampling (Multi-Armed Bandit)

**Clement, B., Roy, D., Oudeyer, P.-Y., & Lopes, M. (2015).** Multi-armed bandits for intelligent tutoring systems. *Journal of Educational Data Mining*, 7(2), 20-48.

Clement et al. applied Thompson Sampling to the problem of selecting learning activities for students, showing it outperforms random and expert-designed curricula. Our `engine/bandit_selector.py` implements their approach: each question maintains a Beta(α, β) distribution tracking correct/incorrect responses. At selection time, we sample from each question's Beta distribution (explore) and compute 3PL item information (exploit), combining them as `(1-w)*exploit + w*explore` with configurable weight w=0.3. After each student response, α (correct) or β (incorrect) is incremented, sharpening the posterior for future selections.

## DKT (Deep Knowledge Tracing)

**Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L. J. & Sohl-Dickstein, J. (2015).** Deep Knowledge Tracing. *Advances in Neural Information Processing Systems (NeurIPS)*, 28.

Piech et al. replaced BKT's discrete Hidden Markov Model with an LSTM recurrent neural network that learns to predict student performance from raw interaction sequences, capturing complex temporal dependencies. Our `engine/dkt_model.py` implements `DKTModelLSTM` with architecture LSTM(input_size=2K, hidden_size=64, num_layers=1) → Linear(K) → Sigmoid, where K is the number of skills and the 2K input encodes correct/incorrect channels per skill. The model is trained with Adam optimizer, binary cross-entropy loss, and early stopping (patience=5 on validation loss). The system auto-selects via `get_dkt_model()`: LSTM when PyTorch is available AND >= 50 interactions exist, otherwise falls back to `DKTModelNumpy` — a windowed logistic regression that provides reasonable predictions during cold start. Both models share the `predict_mastery(history) → Dict[str, float]` interface, enabling transparent upgrades as data accumulates.

## Spaced Repetition (Half-Life Regression)

**Settles, B. & Meeder, B. (2016).** A Trainable Spaced Repetition Model for Language Learning. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, 1848-1858.

Developed at Duolingo, Settles & Meeder model forgetting as an exponential decay parameterized by a per-learner-item half-life. Our `engine/spaced_repetition.py` implements the Ebbinghaus forgetting curve P(recall) = 2^(-Δt/h), where the half-life h doubles on correct responses and halves on incorrect ones (clamped to [0.5, 365] days). The recommender (`engine/recommender.py`) queries review candidates with P(recall) < 0.5 and injects them into the recommendation pipeline with 40% probability, balancing new learning with retention maintenance.

## Bloom's Taxonomy

**Bloom, B. S. (ed.) (1956).** *Taxonomy of Educational Objectives: The Classification of Educational Goals. Handbook I: Cognitive Domain.* David McKay Company.

Bloom's original taxonomy defined 6 levels of cognitive complexity: Knowledge, Comprehension, Application, Analysis, Synthesis, Evaluation. This hierarchical framework provides the theoretical basis for our cognitive level assessment.

**Anderson, L. W. & Krathwohl, D. R. (eds.) (2001).** *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives.* Longman.

Anderson & Krathwohl revised the taxonomy to: Remember, Understand, Apply, Analyze, Evaluate, Create. Our `backend/services/tutor_agent.py` uses this 6-level framework in `evaluate_blooms_level()`, prompting the LLM to classify student responses into cognitive levels with structured reasoning. The classification maps to hint strategies: levels 1-2 (Remember/Understand) trigger additional scaffolding, levels 3-4 (Apply/Analyze) push analytical thinking, and levels 5-6 (Evaluate/Create) lead toward conclusion. Bloom's progression is tracked per conversation via `blooms_history` in the conversation manager.

## Scaffolding

**Wood, D., Bruner, J. S. & Ross, G. (1976).** The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry*, 17(2), 89-100.

Wood, Bruner & Ross introduced scaffolding: structured support that an expert provides and gradually withdraws as the learner gains competence. Our progressive hint system directly implements this principle — hint #1 is gentle (minimal guidance), hint #2 is moderate (more explicit), hint #3 is direct (near-reveal). Combined with Bloom's Taxonomy evaluation, the scaffold intensity is dynamically adjusted: students at low cognitive levels receive stronger structural support, while those demonstrating higher-order thinking receive prompts that push further analysis.

## Socratic Tutoring

**Bloom, B. S. (1984).** The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring. *Educational Researcher*, 13(6), 4-16.

Bloom demonstrated that one-on-one tutoring produces a 2 standard deviation improvement over conventional instruction. Our Socratic tutor agent (`backend/services/tutor_agent.py`) implements an automated approximation: LangChain chains generate diagnostic assessments, progressive hints (gentle → moderate → direct), and understanding evaluations, creating a structured one-on-one dialogue flow.

**VanLehn, K. (2011).** The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist*, 46(4), 197-221.

VanLehn's meta-analysis showed that ITS systems are most effective when they provide step-level feedback and adapt to student understanding. Our conversation manager (`backend/services/conversation_manager.py`) tracks hint count and student understanding level (confused → partial → clear) across turns, adjusting hint strength progressively and concluding remediation when the student demonstrates understanding — matching VanLehn's step-level adaptation pattern.

## RAG (Retrieval-Augmented Generation)

**Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020).** Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Lewis et al. introduced RAG as a way to ground LLM generation in retrieved documents. Our `backend/services/rag_service.py` implements this: when generating explanations, we embed the question with OpenAI `text-embedding-3-small`, retrieve similar questions from the Qdrant vector database, and inject the retrieved explanations as context for the LLM. The 3-tier fallback (cached → RAG-enhanced → plain LLM) ensures degradation when retrieval produces low-relevance results. We evaluate retrieval quality using Precision@K, Recall@K, MRR, and F1@K (`backend/ml/rag_evaluator.py`).

## LLM-as-Judge

**Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023).** Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.

Zheng et al. validated that strong LLMs can serve as reliable evaluators of text quality, correlating well with human judgments. Our `backend/ml/llm_evaluator.py` applies this to educational content: the LLMEvaluator prompts DeepSeek to score generated explanations on a 4-criteria rubric (correctness, clarity, completeness, pedagogical value), producing a structured quality score. This enables automated evaluation of explanation quality across RAG vs baseline variants in our A/B testing framework.
