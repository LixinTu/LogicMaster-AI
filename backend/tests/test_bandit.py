"""
Thompson Sampling Bandit 选题器测试

测试 BanditQuestionSelector 的核心功能：
- Thompson Sampling 选题
- Explore/Exploit 平衡
- 统计更新
- 回退机制
- 与 recommender 的集成
- API 端点
"""

import os
import sys
import sqlite3
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import patch, MagicMock

from engine.bandit_selector import BanditQuestionSelector, get_bandit_selector


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
def bandit(tmp_db):
    """创建使用临时数据库的 BanditQuestionSelector 实例"""
    return BanditQuestionSelector(db_path=tmp_db)


@pytest.fixture
def sample_candidates():
    """5 道不同难度的候选题目"""
    return [
        {"id": "q1", "elo_difficulty": 1300.0, "discrimination": 1.0, "guessing": 0.2},
        {"id": "q2", "elo_difficulty": 1400.0, "discrimination": 1.2, "guessing": 0.2},
        {"id": "q3", "elo_difficulty": 1500.0, "discrimination": 1.5, "guessing": 0.2},
        {"id": "q4", "elo_difficulty": 1600.0, "discrimination": 1.0, "guessing": 0.2},
        {"id": "q5", "elo_difficulty": 1700.0, "discrimination": 0.8, "guessing": 0.2},
    ]


# ---------------------------------------------------------------------------
# TestBanditTable - 数据库表创建
# ---------------------------------------------------------------------------

class TestBanditTable:
    """bandit_stats 表的创建与结构"""

    def test_table_created(self, tmp_db):
        """BanditQuestionSelector 初始化时应创建 bandit_stats 表"""
        BanditQuestionSelector(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bandit_stats'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_table_schema(self, tmp_db):
        """bandit_stats 表应包含 question_id, alpha, beta 列"""
        BanditQuestionSelector(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(bandit_stats)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "question_id" in columns
        assert "alpha" in columns
        assert "beta" in columns


# ---------------------------------------------------------------------------
# TestThompsonSampling - 核心选题逻辑
# ---------------------------------------------------------------------------

class TestThompsonSampling:
    """Thompson Sampling 选题测试"""

    def test_returns_candidate(self, bandit, sample_candidates):
        """select_question 应返回候选列表中的一个题目"""
        result = bandit.select_question(theta=0.0, candidates=sample_candidates)
        assert result is not None
        assert result["id"] in [c["id"] for c in sample_candidates]

    def test_empty_candidates_returns_none(self, bandit):
        """空候选列表应返回 None"""
        result = bandit.select_question(theta=0.0, candidates=[])
        assert result is None

    def test_single_candidate(self, bandit):
        """只有一个候选时应返回该题目"""
        candidates = [{"id": "q1", "elo_difficulty": 1500.0}]
        result = bandit.select_question(theta=0.0, candidates=candidates)
        assert result["id"] == "q1"

    def test_deterministic_with_seed(self, bandit, sample_candidates):
        """固定随机种子时结果应可复现"""
        import random
        random.seed(42)
        r1 = bandit.select_question(theta=0.0, candidates=sample_candidates)
        random.seed(42)
        r2 = bandit.select_question(theta=0.0, candidates=sample_candidates)
        assert r1["id"] == r2["id"]


# ---------------------------------------------------------------------------
# TestExploreExploit - 探索/利用平衡
# ---------------------------------------------------------------------------

class TestExploreExploit:
    """探索/利用权重测试"""

    def test_pure_exploit(self, bandit, sample_candidates):
        """explore_weight=0 时应选择信息量最大的题目（纯 exploit）"""
        import random
        random.seed(0)
        # 纯 exploit：最终选择取决于 item_information，不受 betavariate 影响
        result = bandit.select_question(
            theta=0.0, candidates=sample_candidates, explore_weight=0.0
        )
        assert result is not None

    def test_pure_explore(self, bandit, sample_candidates):
        """explore_weight=1 时应完全依赖 Thompson 采样（纯 explore）"""
        import random
        random.seed(0)
        result = bandit.select_question(
            theta=0.0, candidates=sample_candidates, explore_weight=1.0
        )
        assert result is not None

    def test_exploit_prefers_matched_difficulty(self, bandit):
        """纯 exploit 时应倾向于选择难度匹配的题目（信息量最大）"""
        # theta=0 对应 elo=1500，q_match 难度最接近
        candidates = [
            {"id": "q_easy", "elo_difficulty": 1000.0, "discrimination": 1.0, "guessing": 0.2},
            {"id": "q_match", "elo_difficulty": 1500.0, "discrimination": 1.0, "guessing": 0.2},
            {"id": "q_hard", "elo_difficulty": 2000.0, "discrimination": 1.0, "guessing": 0.2},
        ]
        # 多次运行，纯 exploit 应总是选 q_match
        for _ in range(5):
            result = bandit.select_question(theta=0.0, candidates=candidates, explore_weight=0.0)
            assert result["id"] == "q_match"


# ---------------------------------------------------------------------------
# TestBanditUpdate - 统计更新
# ---------------------------------------------------------------------------

class TestBanditUpdate:
    """bandit 统计更新测试"""

    def test_correct_increments_alpha(self, bandit):
        """答对应增加 alpha"""
        bandit.update("q1", is_correct=True)
        stats = bandit.get_stats()
        assert stats["q1"]["alpha"] == 2.0  # 初始 1.0 + 1
        assert stats["q1"]["beta"] == 1.0   # 未变

    def test_wrong_increments_beta(self, bandit):
        """答错应增加 beta"""
        bandit.update("q1", is_correct=False)
        stats = bandit.get_stats()
        assert stats["q1"]["alpha"] == 1.0  # 未变
        assert stats["q1"]["beta"] == 2.0   # 初始 1.0 + 1

    def test_multiple_updates(self, bandit):
        """多次更新应累积"""
        bandit.update("q1", is_correct=True)
        bandit.update("q1", is_correct=True)
        bandit.update("q1", is_correct=False)
        stats = bandit.get_stats()
        assert stats["q1"]["alpha"] == 3.0  # 1 + 2
        assert stats["q1"]["beta"] == 2.0   # 1 + 1

    def test_upsert_creates_row(self, bandit):
        """首次更新应自动创建行（UPSERT）"""
        bandit.update("new_q", is_correct=True)
        stats = bandit.get_stats()
        assert "new_q" in stats


# ---------------------------------------------------------------------------
# TestGetStats - 统计查询
# ---------------------------------------------------------------------------

class TestGetStats:
    """get_stats 返回值测试"""

    def test_empty_stats(self, bandit):
        """无数据时应返回空字典"""
        stats = bandit.get_stats()
        assert stats == {}

    def test_expected_value(self, bandit):
        """expected_value 应为 alpha / (alpha + beta)"""
        bandit.update("q1", is_correct=True)   # alpha=2, beta=1
        stats = bandit.get_stats()
        expected = 2.0 / 3.0
        assert abs(stats["q1"]["expected_value"] - expected) < 1e-9

    def test_uncertainty(self, bandit):
        """uncertainty 应为 alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))"""
        bandit.update("q1", is_correct=True)   # alpha=2, beta=1
        stats = bandit.get_stats()
        a, b = 2.0, 1.0
        expected = (a * b) / ((a + b) ** 2 * (a + b + 1))
        assert abs(stats["q1"]["uncertainty"] - expected) < 1e-9

    def test_multiple_questions(self, bandit):
        """多道题目的统计应独立"""
        bandit.update("q1", is_correct=True)
        bandit.update("q2", is_correct=False)
        stats = bandit.get_stats()
        assert len(stats) == 2
        assert stats["q1"]["alpha"] == 2.0
        assert stats["q2"]["beta"] == 2.0


# ---------------------------------------------------------------------------
# TestFallback - 回退机制
# ---------------------------------------------------------------------------

class TestFallback:
    """缺失统计时的回退行为"""

    def test_missing_stats_uses_uniform_prior(self, bandit, sample_candidates):
        """无统计数据的题目应使用 Beta(1,1) 均匀先验"""
        # 不调用 update，直接 select — 所有题目应使用默认 (1.0, 1.0)
        result = bandit.select_question(theta=0.0, candidates=sample_candidates)
        assert result is not None

    def test_partial_stats(self, bandit, sample_candidates):
        """部分题目有统计、部分没有时应正常工作"""
        bandit.update("q1", is_correct=True)
        bandit.update("q1", is_correct=True)
        # q2-q5 无统计
        result = bandit.select_question(theta=0.0, candidates=sample_candidates)
        assert result is not None


# ---------------------------------------------------------------------------
# TestRecommenderIntegration - 与 recommender.py 的集成
# ---------------------------------------------------------------------------

class TestRecommenderIntegration:
    """bandit 与 generate_next_question 的集成测试"""

    def test_use_bandit_true(self):
        """use_bandit=True 时应调用 BanditQuestionSelector"""
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

        with patch("engine.recommender.get_bandit_selector") as mock_get:
            mock_bandit = MagicMock()
            mock_bandit.select_question.return_value = mock_db.get_adaptive_candidates.return_value[1]
            mock_get.return_value = mock_bandit

            result = generate_next_question(
                user_theta=0.0, current_q_id="", questions_log=[],
                session_state=mock_state, db_manager=mock_db, use_bandit=True,
            )
            assert result is not None
            mock_bandit.select_question.assert_called_once()

    def test_use_bandit_false(self):
        """use_bandit=False 时不应调用 BanditQuestionSelector"""
        from engine.recommender import generate_next_question

        mock_db = MagicMock()
        mock_db.get_adaptive_candidates.return_value = [
            {"id": "q1", "elo_difficulty": 1500.0, "question_type": "Weaken",
             "difficulty": "medium", "stimulus": "s", "question": "q",
             "choices": ["A", "B"], "correct": "A", "skills": []},
        ]

        mock_state = MagicMock()
        mock_state.radio_key = 0

        with patch("engine.recommender.get_bandit_selector") as mock_get:
            result = generate_next_question(
                user_theta=0.0, current_q_id="", questions_log=[],
                session_state=mock_state, db_manager=mock_db, use_bandit=False,
            )
            assert result is not None
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# TestBanditAPI - API 端点测试
# ---------------------------------------------------------------------------

class TestBanditAPI:
    """bandit-update API 端点测试"""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_bandit_update_correct(self, client, tmp_db):
        """POST /api/questions/bandit-update 答对应返回 200"""
        with patch("backend.routers.questions.get_bandit_selector") as mock_get:
            mock_bandit = MagicMock()
            mock_get.return_value = mock_bandit
            resp = client.post("/api/questions/bandit-update", json={
                "question_id": "q1", "is_correct": True,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["question_id"] == "q1"
            mock_bandit.update.assert_called_once_with(question_id="q1", is_correct=True)

    def test_bandit_update_wrong(self, client, tmp_db):
        """POST /api/questions/bandit-update 答错应返回 200"""
        with patch("backend.routers.questions.get_bandit_selector") as mock_get:
            mock_bandit = MagicMock()
            mock_get.return_value = mock_bandit
            resp = client.post("/api/questions/bandit-update", json={
                "question_id": "q2", "is_correct": False,
            })
            assert resp.status_code == 200
            mock_bandit.update.assert_called_once_with(question_id="q2", is_correct=False)

    def test_strategy_field_accepted(self, client):
        """NextQuestionRequest 应接受 strategy 字段"""
        # 验证 strategy 字段不会导致 422 验证错误
        # 实际调用会因为 DB 问题而失败，但不应是 422
        resp = client.post("/api/questions/next", json={
            "user_theta": 0.0,
            "current_q_id": "",
            "strategy": "legacy",
        })
        # 404（无题目）或 200 均可接受，关键是不能是 422（字段不合法）
        assert resp.status_code != 422
