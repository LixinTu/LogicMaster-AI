"""
间隔重复 (Half-Life Regression) 测试

测试 SpacedRepetitionModel 的核心功能：
- 回忆概率随时间衰减（Ebbinghaus 遗忘曲线）
- 答对 → half_life 翻倍
- 答错 → half_life 减半
- 复习候选筛选
- half_life 钳制
- 冷启动默认值
- 与 recommender 的集成
- API 端点
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import patch, MagicMock

from engine.spaced_repetition import SpacedRepetitionModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """创建临时 SQLite 数据库用于测试"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def sr(tmp_db):
    """创建使用临时数据库的 SpacedRepetitionModel 实例"""
    return SpacedRepetitionModel(db_path=tmp_db, user_id="test_user")


@pytest.fixture
def now():
    """固定的当前时间"""
    return datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# TestRecallDecay - 回忆概率衰减
# ---------------------------------------------------------------------------

class TestRecallDecay:
    """回忆概率应随时间递减（Ebbinghaus 遗忘曲线）"""

    def test_recall_decays_over_time(self, sr, now):
        """P = 2^(-elapsed / half_life)，时间越长概率越低"""
        # 在 now 时刻练习，half_life=1天
        sr.update_half_life("q1", is_correct=False, current_time=now)
        # 答错后 half_life = 1.0 * 0.5 = 0.5

        # 刚练完：概率接近 1.0
        p_0h = sr.recall_probability("q1", current_time=now)
        assert p_0h > 0.99

        # 6小时后
        p_6h = sr.recall_probability("q1", current_time=now + timedelta(hours=6))
        # 12小时后
        p_12h = sr.recall_probability("q1", current_time=now + timedelta(hours=12))
        # 24小时后
        p_24h = sr.recall_probability("q1", current_time=now + timedelta(hours=24))

        assert p_6h < p_0h
        assert p_12h < p_6h
        assert p_24h < p_12h

    def test_recall_at_half_life(self, sr, now):
        """经过恰好 1 个 half_life 后，概率应为 0.5"""
        sr.update_half_life("q1", is_correct=True, current_time=now)
        # 答对后 half_life = 1.0 * 2.0 = 2.0 天
        hl = 2.0

        p = sr.recall_probability("q1", current_time=now + timedelta(days=hl))
        assert abs(p - 0.5) < 1e-9

    def test_recall_never_practiced(self, sr, now):
        """从未练过的题目，回忆概率应为 0.0"""
        p = sr.recall_probability("never_seen", current_time=now)
        assert p == 0.0


# ---------------------------------------------------------------------------
# TestHalfLifeUpdate - half_life 更新
# ---------------------------------------------------------------------------

class TestHalfLifeUpdate:
    """答对/答错对 half_life 的影响"""

    def test_correct_doubles_half_life(self, sr, now):
        """答对一次后 half_life 从默认 1.0 → 2.0"""
        hl = sr.update_half_life("q1", is_correct=True, current_time=now)
        assert hl == 2.0

    def test_incorrect_halves_half_life(self, sr, now):
        """答对两次 (hl=4.0) 再答错一次 → hl=2.0"""
        sr.update_half_life("q1", is_correct=True, current_time=now)  # 1→2
        sr.update_half_life("q1", is_correct=True, current_time=now)  # 2→4
        hl = sr.update_half_life("q1", is_correct=False, current_time=now)  # 4→2
        assert hl == 2.0

    def test_half_life_clamping_upper(self, sr, now):
        """half_life 不应超过 90.0 天"""
        # 连续答对足够多次，使 half_life 超过上限
        for _ in range(10):
            sr.update_half_life("q1", is_correct=True, current_time=now)
        stats = sr.get_all_stats(current_time=now)
        assert stats["q1"]["half_life"] <= 90.0

    def test_half_life_clamping_lower(self, sr, now):
        """half_life 不应低于 0.25 天"""
        # 连续答错足够多次
        for _ in range(10):
            sr.update_half_life("q1", is_correct=False, current_time=now)
        stats = sr.get_all_stats(current_time=now)
        assert stats["q1"]["half_life"] >= 0.25


# ---------------------------------------------------------------------------
# TestReviewCandidates - 复习候选
# ---------------------------------------------------------------------------

class TestReviewCandidates:
    """get_review_candidates 应只返回回忆概率低于阈值的题目"""

    def test_review_candidates_below_threshold(self, sr, now):
        """只有 recall_prob < threshold 的题目应被返回"""
        # q1: 练过，等很久 → 概率低
        sr.update_half_life("q1", is_correct=True, current_time=now)  # hl=2
        # q2: 练过，刚做完 → 概率高
        later = now + timedelta(days=10)
        sr.update_half_life("q2", is_correct=True, current_time=later)  # hl=2

        # 在 later 时刻查询：q1 已过 10 天 (hl=2)，q2 刚做
        candidates = sr.get_review_candidates(current_time=later, threshold=0.5)
        q_ids = [c["question_id"] for c in candidates]
        assert "q1" in q_ids
        assert "q2" not in q_ids

    def test_review_candidates_empty_when_all_fresh(self, sr, now):
        """所有题目刚练完时，不应有复习候选"""
        sr.update_half_life("q1", is_correct=True, current_time=now)
        sr.update_half_life("q2", is_correct=True, current_time=now)
        candidates = sr.get_review_candidates(current_time=now, threshold=0.5)
        assert len(candidates) == 0

    def test_review_candidates_sorted_by_recall(self, sr, now):
        """候选应按回忆概率升序排序（最易忘的在前）"""
        sr.update_half_life("q1", is_correct=True, current_time=now)  # hl=2
        sr.update_half_life("q2", is_correct=False, current_time=now)  # hl=0.5

        later = now + timedelta(days=3)
        candidates = sr.get_review_candidates(current_time=later, threshold=0.9)
        assert len(candidates) >= 2
        # q2 (hl=0.5, 3天后) 概率远低于 q1 (hl=2, 3天后)
        assert candidates[0]["question_id"] == "q2"


# ---------------------------------------------------------------------------
# TestColdStart - 冷启动
# ---------------------------------------------------------------------------

class TestColdStart:
    """新题目应获得默认 half_life"""

    def test_cold_start_default(self, sr, now):
        """首次答对后 half_life 应为 2.0（默认 1.0 * 2.0）"""
        hl = sr.update_half_life("new_q", is_correct=True, current_time=now)
        assert hl == 2.0

    def test_cold_start_wrong(self, sr, now):
        """首次答错后 half_life 应为 0.5（默认 1.0 * 0.5）"""
        hl = sr.update_half_life("new_q", is_correct=False, current_time=now)
        assert hl == 0.5

    def test_cold_start_stats(self, sr, now):
        """冷启动后 n_attempts=1"""
        sr.update_half_life("new_q", is_correct=True, current_time=now)
        stats = sr.get_all_stats(current_time=now)
        assert stats["new_q"]["n_attempts"] == 1
        assert stats["new_q"]["n_correct"] == 1


# ---------------------------------------------------------------------------
# TestGetAllStats - 统计查询
# ---------------------------------------------------------------------------

class TestGetAllStats:
    """get_all_stats 返回值测试"""

    def test_empty(self, sr, now):
        """无数据时应返回空字典"""
        stats = sr.get_all_stats(current_time=now)
        assert stats == {}

    def test_structure(self, sr, now):
        """返回值应包含所有字段"""
        sr.update_half_life("q1", is_correct=True, current_time=now)
        stats = sr.get_all_stats(current_time=now)
        assert "half_life" in stats["q1"]
        assert "last_practiced" in stats["q1"]
        assert "recall_prob" in stats["q1"]
        assert "n_correct" in stats["q1"]
        assert "n_attempts" in stats["q1"]

    def test_multiple_users(self, tmp_db, now):
        """不同 user_id 的统计应独立"""
        sr_a = SpacedRepetitionModel(db_path=tmp_db, user_id="user_a")
        sr_b = SpacedRepetitionModel(db_path=tmp_db, user_id="user_b")
        sr_a.update_half_life("q1", is_correct=True, current_time=now)
        sr_b.update_half_life("q1", is_correct=False, current_time=now)
        stats_a = sr_a.get_all_stats(current_time=now)
        stats_b = sr_b.get_all_stats(current_time=now)
        assert stats_a["q1"]["half_life"] == 2.0   # 答对
        assert stats_b["q1"]["half_life"] == 0.5   # 答错


# ---------------------------------------------------------------------------
# TestRecommenderIntegration - 与 recommender 集成
# ---------------------------------------------------------------------------

class TestRecommenderIntegration:
    """间隔重复与 generate_next_question 的集成"""

    def test_spaced_repetition_can_inject_review(self):
        """use_spaced_repetition=True 时，如果有复习题且随机命中，应返回复习题"""
        from engine.recommender import generate_next_question

        mock_db = MagicMock()
        mock_db.get_adaptive_candidates.return_value = [
            {"id": "q1", "elo_difficulty": 1500.0, "question_type": "Weaken",
             "difficulty": "medium", "stimulus": "s", "question": "q",
             "choices": ["A", "B"], "correct": "A", "skills": []},
            {"id": "q2", "elo_difficulty": 1600.0, "question_type": "Weaken",
             "difficulty": "medium", "stimulus": "s", "question": "q",
             "choices": ["A", "B"], "correct": "A", "skills": []},
        ]
        mock_state = MagicMock()
        mock_state.radio_key = 0

        mock_sr = MagicMock()
        mock_sr.get_review_candidates.return_value = [
            {"question_id": "q1", "recall_probability": 0.2, "half_life": 1.0, "elapsed_days": 3.0}
        ]

        import random
        random.seed(0)  # random.random() < 0.4 可能命中也可能不命中

        with patch("engine.recommender.get_spaced_repetition_model", return_value=mock_sr), \
             patch("engine.recommender.get_bandit_selector") as mock_bandit_get:
            mock_bandit = MagicMock()
            mock_bandit.select_question.return_value = mock_db.get_adaptive_candidates.return_value[1]
            mock_bandit_get.return_value = mock_bandit

            result = generate_next_question(
                user_theta=0.0, current_q_id="", questions_log=[],
                session_state=mock_state, db_manager=mock_db,
                use_bandit=True, use_spaced_repetition=True,
            )
            assert result is not None
            # 不管是否命中复习，都应返回一个有效题目
            assert result["question_id"] in ("q1", "q2")

    def test_spaced_repetition_disabled(self):
        """use_spaced_repetition=False 时不应调用间隔重复模型"""
        from engine.recommender import generate_next_question

        mock_db = MagicMock()
        mock_db.get_adaptive_candidates.return_value = [
            {"id": "q1", "elo_difficulty": 1500.0, "question_type": "Weaken",
             "difficulty": "medium", "stimulus": "s", "question": "q",
             "choices": ["A", "B"], "correct": "A", "skills": []},
        ]
        mock_state = MagicMock()
        mock_state.radio_key = 0

        with patch("engine.recommender.get_spaced_repetition_model") as mock_get:
            result = generate_next_question(
                user_theta=0.0, current_q_id="", questions_log=[],
                session_state=mock_state, db_manager=mock_db,
                use_bandit=False, use_spaced_repetition=False,
            )
            assert result is not None
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# TestReviewScheduleAPI - API 端点
# ---------------------------------------------------------------------------

class TestReviewScheduleAPI:
    """review-schedule API 端点测试"""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_review_schedule_empty(self, client):
        """无复习数据时应返回空列表"""
        with patch("backend.routers.questions.get_spaced_repetition_model") as mock_get:
            mock_sr = MagicMock()
            mock_sr.get_review_candidates.return_value = []
            mock_get.return_value = mock_sr

            resp = client.get("/api/questions/review-schedule?user_id=test")
            assert resp.status_code == 200
            data = resp.json()
            assert data["due_count"] == 0
            assert data["reviews"] == []

    def test_review_schedule_with_data(self, client):
        """有复习数据时应返回题目列表"""
        with patch("backend.routers.questions.get_spaced_repetition_model") as mock_get:
            mock_sr = MagicMock()
            mock_sr.get_review_candidates.return_value = [
                {"question_id": "q1", "recall_probability": 0.3, "half_life": 1.0, "elapsed_days": 2.0},
                {"question_id": "q2", "recall_probability": 0.1, "half_life": 0.5, "elapsed_days": 3.0},
            ]
            mock_get.return_value = mock_sr

            resp = client.get("/api/questions/review-schedule?user_id=test&threshold=0.5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["due_count"] == 2
            assert data["threshold"] == 0.5
            assert len(data["reviews"]) == 2
            assert data["reviews"][0]["question_id"] == "q1"
