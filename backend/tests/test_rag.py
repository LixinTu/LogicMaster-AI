"""
RAG 系统测试
- RAGEvaluator 纯逻辑测试（不需要外部服务）
- RAG API 端点测试（mock Qdrant + OpenAI）
- ExplanationService 降级测试
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.ml.rag_evaluator import RAGEvaluator

client = TestClient(app)


# ========== RAGEvaluator 纯逻辑测试 ==========

class TestRAGEvaluator:
    def test_precision_at_k_perfect(self):
        relevant = ["q1", "q2", "q3"]
        retrieved = ["q1", "q2", "q3", "q4", "q5"]
        assert RAGEvaluator.precision_at_k(relevant, retrieved, 3) == 1.0

    def test_precision_at_k_partial(self):
        relevant = ["q1", "q3"]
        retrieved = ["q1", "q2", "q3", "q4", "q5"]
        # 2 hits in top 5
        assert RAGEvaluator.precision_at_k(relevant, retrieved, 5) == 0.4

    def test_precision_at_k_zero(self):
        relevant = ["q1"]
        retrieved = ["q2", "q3"]
        assert RAGEvaluator.precision_at_k(relevant, retrieved, 2) == 0.0

    def test_precision_at_k_zero_k(self):
        assert RAGEvaluator.precision_at_k(["q1"], ["q1"], 0) == 0.0

    def test_recall_at_k_perfect(self):
        relevant = ["q1", "q2"]
        retrieved = ["q1", "q2", "q3"]
        assert RAGEvaluator.recall_at_k(relevant, retrieved, 3) == 1.0

    def test_recall_at_k_partial(self):
        relevant = ["q1", "q2", "q3"]
        retrieved = ["q1", "q4", "q5"]
        # 1 out of 3 relevant
        assert RAGEvaluator.recall_at_k(relevant, retrieved, 3) == pytest.approx(1 / 3)

    def test_recall_at_k_empty_relevant(self):
        assert RAGEvaluator.recall_at_k([], ["q1"], 1) == 0.0

    def test_mrr_first_hit(self):
        relevant = ["q1"]
        retrieved = ["q1", "q2", "q3"]
        assert RAGEvaluator.mrr(relevant, retrieved) == 1.0

    def test_mrr_second_hit(self):
        relevant = ["q2"]
        retrieved = ["q1", "q2", "q3"]
        assert RAGEvaluator.mrr(relevant, retrieved) == 0.5

    def test_mrr_no_hit(self):
        relevant = ["q9"]
        retrieved = ["q1", "q2", "q3"]
        assert RAGEvaluator.mrr(relevant, retrieved) == 0.0

    def test_f1_at_k(self):
        relevant = ["q1", "q2"]
        retrieved = ["q1", "q3", "q4"]
        # P@3 = 1/3, R@3 = 1/2, F1 = 2*(1/3)*(1/2)/((1/3)+(1/2)) = 0.4
        assert RAGEvaluator.f1_at_k(relevant, retrieved, 3) == pytest.approx(0.4)

    def test_f1_at_k_zero(self):
        assert RAGEvaluator.f1_at_k(["q1"], ["q2"], 1) == 0.0

    def test_evaluate_retrieval(self):
        result = RAGEvaluator.evaluate_retrieval(
            relevant_ids=["q1", "q2"],
            retrieved_ids=["q1", "q2", "q3"],
            k=3,
        )
        assert f"precision@3" in result
        assert f"recall@3" in result
        assert "mrr" in result
        assert f"f1@3" in result
        assert result["precision@3"] == pytest.approx(2 / 3)
        assert result["recall@3"] == 1.0
        assert result["mrr"] == 1.0

    def test_create_evaluation_report(self):
        cases = [
            {"relevant_ids": ["q1", "q2"], "retrieved_ids": ["q1", "q2", "q3"]},
            {"relevant_ids": ["q3"], "retrieved_ids": ["q1", "q3"]},
        ]
        report = RAGEvaluator.create_evaluation_report(cases, k=3)
        assert report["num_cases"] == 2
        assert f"avg_precision@3" in report
        assert f"avg_recall@3" in report
        assert "avg_mrr" in report
        assert len(report["per_case"]) == 2

    def test_create_evaluation_report_empty(self):
        report = RAGEvaluator.create_evaluation_report([], k=5)
        assert report["num_cases"] == 0
        assert report["avg_mrr"] == 0.0


# ========== API 端点测试（mock 外部服务）==========

class TestExplanationEndpoints:
    """测试解析 API 端点（mock RAG 和 LLM 调用）"""

    @patch("backend.services.explanation_service.get_rag_service")
    def test_generate_with_rag_cached(self, mock_get_rag):
        """Tier 1: 如果题目已有 detailed_explanation 则直接返回 cached"""
        resp = client.post("/api/explanations/generate-with-rag", json={
            "question_id": "q001",
            "question": {
                "detailed_explanation": "A" * 200,  # 超过 100 字符
                "stimulus": "Test",
                "question": "Test?",
                "choices": ["A. x", "B. y", "C. z", "D. w", "E. v"],
                "correct": "A",
                "question_type": "Weaken",
            },
            "user_choice": "B",
            "is_correct": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "cached"
        assert len(data["explanation"]) > 100
        # RAG 不应被调用
        mock_get_rag.assert_not_called()

    @patch("backend.services.explanation_service._call_llm_plain")
    @patch("backend.services.explanation_service.get_rag_service")
    def test_generate_with_rag_fallback_to_llm(self, mock_get_rag, mock_llm):
        """Tier 3: 当 RAG 检索为空时，降级到 plain LLM"""
        # RAG 返回空
        mock_rag_instance = MagicMock()
        mock_rag_instance.retrieve_similar.return_value = []
        mock_get_rag.return_value = mock_rag_instance

        # LLM 返回解析
        mock_llm.return_value = "This is a plain LLM explanation " * 10

        resp = client.post("/api/explanations/generate-with-rag", json={
            "question_id": "q002",
            "question": {
                "stimulus": "Test",
                "question": "Test?",
                "choices": ["A. x", "B. y", "C. z", "D. w", "E. v"],
                "correct": "B",
                "question_type": "Strengthen",
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "llm_only"
        assert data["similar_references"] == []

    @patch("backend.services.rag_service.get_rag_service")
    def test_search_similar_endpoint(self, mock_get_rag):
        """测试相似搜索端点"""
        mock_rag_instance = MagicMock()
        mock_rag_instance.retrieve_similar.return_value = [
            {
                "question_id": "q001",
                "explanation": "test explanation",
                "question_type": "Weaken",
                "skills": ["Causal Reasoning"],
                "score": 0.95,
            }
        ]
        mock_get_rag.return_value = mock_rag_instance

        resp = client.post("/api/explanations/search-similar", json={
            "query": "company market share",
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)


# ========== ExplanationService 降级测试 ==========

class TestExplanationServiceFallback:
    """测试 3-tier fallback 逻辑"""

    @patch("backend.services.explanation_service.get_rag_service")
    def test_tier1_cached_wins(self, mock_get_rag):
        """有缓存时直接返回，不调 RAG"""
        from backend.services.explanation_service import generate_rag_enhanced_explanation

        question = {
            "detailed_explanation": "Cached explanation text. " * 20,
            "stimulus": "Test",
            "question": "Test?",
        }
        result = generate_rag_enhanced_explanation(question)
        assert result["source"] == "cached"
        mock_get_rag.assert_not_called()

    @patch("backend.services.explanation_service._call_llm_with_rag")
    @patch("backend.services.explanation_service.get_rag_service")
    def test_tier2_rag_enhanced(self, mock_get_rag, mock_llm_rag):
        """无缓存 + RAG 有结果 → rag_enhanced"""
        from backend.services.explanation_service import generate_rag_enhanced_explanation

        mock_rag_instance = MagicMock()
        mock_rag_instance.retrieve_similar.return_value = [
            {"question_id": "q1", "explanation": "similar expl", "score": 0.9}
        ]
        mock_get_rag.return_value = mock_rag_instance
        mock_llm_rag.return_value = "RAG-enhanced explanation text. " * 10

        question = {
            "stimulus": "Test stimulus",
            "question": "Test question?",
            "choices": ["A. x"],
            "correct": "A",
            "question_type": "Weaken",
        }
        result = generate_rag_enhanced_explanation(question)
        assert result["source"] == "rag_enhanced"
        assert len(result["similar_references"]) > 0

    @patch("backend.services.explanation_service._call_llm_plain")
    @patch("backend.services.explanation_service.get_rag_service")
    def test_tier3_plain_llm(self, mock_get_rag, mock_llm_plain):
        """无缓存 + RAG 为空 → plain LLM"""
        from backend.services.explanation_service import generate_rag_enhanced_explanation

        mock_rag_instance = MagicMock()
        mock_rag_instance.retrieve_similar.return_value = []
        mock_get_rag.return_value = mock_rag_instance
        mock_llm_plain.return_value = "Plain LLM explanation text. " * 10

        question = {
            "stimulus": "Test",
            "question": "Test?",
            "choices": ["A. x"],
            "correct": "A",
            "question_type": "Weaken",
        }
        result = generate_rag_enhanced_explanation(question)
        assert result["source"] == "llm_only"
        assert result["similar_references"] == []
