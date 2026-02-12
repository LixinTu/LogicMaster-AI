"""
Week 4 测试：A/B Testing + Analytics + LLM Evaluator
- ABTestService: 一致性哈希、日志记录、聚合统计
- Analytics 端点：log-outcome, ab-test-results, rag-performance
- LLM Evaluator: 评分解析
- 分析脚本：分组统计、t 检验、Cohen's d
"""

import sys
import os
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ========== ABTestService 单元测试 ==========

class TestABTestService:
    def setup_method(self):
        """每个测试用新实例，使用临时数据库"""
        from backend.services.ab_testing import ABTestService
        self.ab = ABTestService()

    def test_assign_variant_deterministic(self):
        """同 user+experiment 总是返回同 variant"""
        v1 = self.ab.assign_variant("user_A", "tutor_strategy")
        v2 = self.ab.assign_variant("user_A", "tutor_strategy")
        assert v1 == v2
        assert v1 is not None

    def test_assign_variant_covers_all_variants(self):
        """不同 user_id 应覆盖多个 variant"""
        variants_seen = set()
        for i in range(100):
            v = self.ab.assign_variant(f"user_{i}", "tutor_strategy")
            if v:
                variants_seen.add(v)
        # 100 个用户应至少覆盖 2 个 variant（概率上 3 个都有）
        assert len(variants_seen) >= 2

    def test_assign_variant_nonexistent_experiment(self):
        """不存在的实验返回 None"""
        v = self.ab.assign_variant("user_X", "nonexistent_experiment")
        assert v is None

    def test_is_experiment_active(self):
        assert self.ab.is_experiment_active("tutor_strategy") is True
        assert self.ab.is_experiment_active("explanation_source") is True
        assert self.ab.is_experiment_active("nonexistent") is False

    def test_log_exposure(self):
        ok = self.ab.log_exposure("u1", "tutor_strategy", "socratic_standard")
        assert ok is True

    def test_log_outcome(self):
        ok = self.ab.log_outcome("u1", "tutor_strategy", "socratic_standard", "is_correct", 1.0)
        assert ok is True

    def test_get_experiment_results_structure(self):
        """聚合结果结构正确"""
        # 先写入一些数据
        self.ab.log_exposure("u_r1", "tutor_strategy", "socratic_standard")
        self.ab.log_outcome("u_r1", "tutor_strategy", "socratic_standard", "is_correct", 1.0)
        self.ab.log_outcome("u_r1", "tutor_strategy", "socratic_standard", "is_correct", 0.0)

        results = self.ab.get_experiment_results("tutor_strategy")
        assert results["experiment"] == "tutor_strategy"
        assert results["active"] is True
        assert "variants" in results
        assert "socratic_standard" in results["variants"]


# ========== Analytics 端点测试 ==========

class TestAnalyticsEndpoints:
    def test_log_outcome_endpoint(self):
        resp = client.post("/api/analytics/log-outcome", json={
            "user_id": "test_user_1",
            "experiment_name": "tutor_strategy",
            "variant": "socratic_standard",
            "metric": "is_correct",
            "value": 1.0,
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_ab_test_results_endpoint(self):
        resp = client.get("/api/analytics/ab-test-results?experiment=tutor_strategy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["experiment"] == "tutor_strategy"
        assert "variants" in data

    def test_rag_performance_endpoint(self):
        resp = client.get("/api/analytics/rag-performance")
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieval_metrics" in data
        assert "system_metrics" in data

    def test_log_outcome_missing_field(self):
        resp = client.post("/api/analytics/log-outcome", json={
            "user_id": "u",
            # missing experiment_name
        })
        assert resp.status_code == 422  # validation error


# ========== Tutor /start-remediation with A/B ==========

SAMPLE_QUESTION = {
    "question_id": "ab_test_q",
    "question_type": "Weaken",
    "stimulus": "Coffee drinkers live longer.",
    "question": "Which weakens the argument?",
    "choices": ["A. Wealth confound", "B. Tea also", "C. Antioxidants", "D. One city", "E. Decaf same"],
    "correct": "A",
    "explanation": "Wealth is a confound.",
}


class TestStartRemediationWithAB:
    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    @patch("backend.routers.tutor.get_ab_test_service")
    def test_socratic_variant(self, mock_ab, mock_cm, mock_agent):
        from backend.services.conversation_manager import ConversationManager
        cm = ConversationManager()
        mock_cm.return_value = cm

        ab = MagicMock()
        ab.assign_variant.return_value = "socratic_standard"
        ab.log_exposure.return_value = True
        mock_ab.return_value = ab

        agent = MagicMock()
        agent.diagnose_error.return_value = {
            "logic_gap": "Missed confound",
            "error_type": "correlation_causation",
            "key_assumption": "No confound",
            "why_wrong": "B doesn't address causation",
        }
        agent.generate_socratic_hint.return_value = "What is the main conclusion?"
        mock_agent.return_value = agent

        resp = client.post("/api/tutor/start-remediation", json={
            "question_id": "ab_q1",
            "question": SAMPLE_QUESTION,
            "user_choice": "B",
            "correct_choice": "A",
            "user_id": "test_ab_user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["variant"] == "socratic_standard"
        assert data["current_state"] == "hinting"
        assert data["first_hint"] == "What is the main conclusion?"
        ab.assign_variant.assert_called_once_with("test_ab_user", "tutor_strategy")
        ab.log_exposure.assert_called_once()

    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    @patch("backend.routers.tutor.get_ab_test_service")
    def test_direct_explanation_variant(self, mock_ab, mock_cm, mock_agent):
        from backend.services.conversation_manager import ConversationManager
        cm = ConversationManager()
        mock_cm.return_value = cm

        ab = MagicMock()
        ab.assign_variant.return_value = "direct_explanation"
        ab.log_exposure.return_value = True
        mock_ab.return_value = ab

        agent = MagicMock()
        agent.diagnose_error.return_value = {
            "logic_gap": "Missed confound",
            "error_type": "correlation_causation",
            "key_assumption": "No confound",
            "why_wrong": "B doesn't weaken the causal claim.",
        }
        mock_agent.return_value = agent

        resp = client.post("/api/tutor/start-remediation", json={
            "question_id": "ab_q2",
            "question": SAMPLE_QUESTION,
            "user_choice": "B",
            "correct_choice": "A",
            "user_id": "test_ab_user_direct",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["variant"] == "direct_explanation"
        assert data["current_state"] == "concluded"
        assert "correct answer is A" in data["first_hint"]


# ========== LLM Evaluator 单元测试 ==========

class TestLLMEvaluator:
    def test_evaluate_single_empty_explanation(self):
        from backend.ml.llm_evaluator import LLMQualityEvaluator
        evaluator = LLMQualityEvaluator()
        result = evaluator.evaluate_single({"question_type": "Weaken"}, "")
        assert result["error"] == "Empty explanation"
        assert result["overall"] == 0.0

    @patch("backend.ml.llm_evaluator.LLMQualityEvaluator._get_client")
    def test_evaluate_single_mocked(self, mock_client_fn):
        from backend.ml.llm_evaluator import LLMQualityEvaluator

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"correctness": 4, "clarity": 5, "completeness": 3, "pedagogical_value": 4, "justification": "Good"}'

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        evaluator = LLMQualityEvaluator()
        result = evaluator.evaluate_single(
            {"question_type": "Weaken", "stimulus": "test", "question": "test?", "correct": "A"},
            "The answer is A because..."
        )
        assert result["error"] is None
        assert result["correctness"] == 4
        assert result["clarity"] == 5
        assert result["overall"] == 4.0  # (4+5+3+4)/4

    @patch("backend.ml.llm_evaluator.LLMQualityEvaluator._get_client")
    def test_evaluate_batch_mocked(self, mock_client_fn):
        from backend.ml.llm_evaluator import LLMQualityEvaluator

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"correctness": 4, "clarity": 4, "completeness": 4, "pedagogical_value": 4, "justification": "OK"}'

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        evaluator = LLMQualityEvaluator()
        items = [
            {"question": {"question_type": "Weaken"}, "explanation": "Explanation 1"},
            {"question": {"question_type": "Weaken"}, "explanation": "Explanation 2"},
        ]
        result = evaluator.evaluate_batch(items)
        assert result["count"] == 2
        assert result["avg_overall"] == 4.0


# ========== 分析脚本函数测试 ==========

class TestAnalysisScriptFunctions:
    def test_calculate_metrics_by_variant(self):
        from scripts.analyze_ab_tests import calculate_metrics_by_variant

        rows = [
            {"variant": "A", "outcome_metric": "is_correct", "outcome_value": 1.0},
            {"variant": "A", "outcome_metric": "is_correct", "outcome_value": 0.0},
            {"variant": "B", "outcome_metric": "is_correct", "outcome_value": 1.0},
            {"variant": "B", "outcome_metric": "is_correct", "outcome_value": 1.0},
        ]
        metrics = calculate_metrics_by_variant(rows)
        assert metrics["A"]["is_correct"]["count"] == 2
        assert metrics["A"]["is_correct"]["mean"] == 0.5
        assert metrics["B"]["is_correct"]["mean"] == 1.0

    def test_statistical_significance_test_sufficient_samples(self):
        from scripts.analyze_ab_tests import statistical_significance_test

        # 明确差异的样本
        a = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]  # mean=0.8
        b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # mean=0.2
        result = statistical_significance_test(a, b)
        assert result["n_a"] == 10
        assert result["n_b"] == 10
        assert result["mean_a"] > result["mean_b"]
        assert result["cohens_d"] > 0
        # p_value 应该很小（差异很大）
        assert result["p_value"] < 0.05

    def test_statistical_significance_test_insufficient_samples(self):
        from scripts.analyze_ab_tests import statistical_significance_test

        result = statistical_significance_test([1.0], [0.0])
        assert "error" in result

    def test_cohens_d_calculation(self):
        from scripts.analyze_ab_tests import statistical_significance_test

        # 相同分布 → d 接近 0
        same = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = statistical_significance_test(same, same.copy())
        assert abs(result["cohens_d"]) < 0.01
