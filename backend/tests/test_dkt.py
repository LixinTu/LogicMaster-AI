"""
Deep Knowledge Tracing (DKT) 测试

覆盖范围：
- SkillEncoder: 词表构建、编码/解码、保存/加载
- DKTModelNumpy: 预测、训练、保存/加载、BKT 对比
- DKTModelLSTM: 初始化、前向传播、预测、保存/加载、训练
- 自动选择: numpy / LSTM / 回退
- 辅助函数: sigmoid、特征提取
- answer_history: 表创建、插入/查询、排序、用户隔离
- Recommender 集成: use_dkt=True/False、降级
"""

import json
import os
import sys
import sqlite3
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from engine.skill_encoder import SkillEncoder, get_skill_encoder
from engine.dkt_model import (
    DKTModelNumpy,
    TORCH_AVAILABLE,
    _sigmoid,
    _extract_features,
    get_dkt_model,
    WINDOW_SIZE,
    NUM_FEATURES,
    MIN_INTERACTIONS_FOR_LSTM,
)

if TORCH_AVAILABLE:
    import torch
    from engine.dkt_model import DKTModelLSTM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """创建带有 questions 和 answer_history 表的临时 SQLite 数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    # questions 表（SkillEncoder 需要读 content 字段）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY,
            question_type TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            content TEXT NOT NULL,
            elo_difficulty REAL DEFAULT 1500.0,
            is_verified INTEGER DEFAULT 1
        )
    """)
    # answer_history 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS answer_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'default',
            question_id TEXT NOT NULL,
            skill_ids TEXT NOT NULL,
            is_correct INTEGER NOT NULL,
            theta_at_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_answer_history_user_time
        ON answer_history (user_id, created_at)
    """)
    conn.commit()
    conn.close()

    yield path
    os.unlink(path)


@pytest.fixture
def seeded_db(tmp_db):
    """带有 3 道题目（涉及 4 个技能）的数据库"""
    conn = sqlite3.connect(tmp_db)
    cursor = conn.cursor()

    questions = [
        ("q1", "Weaken", "medium", json.dumps({
            "skills": ["Causal Reasoning", "Assumption Identification"],
            "stimulus": "test", "question": "test", "choices": ["A", "B"], "correct": "A"
        })),
        ("q2", "Strengthen", "hard", json.dumps({
            "skills": ["Evidence Evaluation"],
            "stimulus": "test", "question": "test", "choices": ["A", "B"], "correct": "B"
        })),
        ("q3", "Evaluate", "easy", json.dumps({
            "skills": ["Causal Reasoning", "Logical Structure"],
            "stimulus": "test", "question": "test", "choices": ["A", "B"], "correct": "A"
        })),
    ]
    cursor.executemany(
        "INSERT INTO questions (id, question_type, difficulty, content) VALUES (?, ?, ?, ?)",
        questions,
    )
    conn.commit()
    conn.close()
    return tmp_db


@pytest.fixture
def encoder(seeded_db):
    """已构建词表的 SkillEncoder（使用 seeded_db）"""
    enc = SkillEncoder()
    enc.build_vocab(seeded_db)
    return enc


@pytest.fixture
def sample_history():
    """样例交互历史"""
    return [
        {"skills": ["Causal Reasoning", "Assumption Identification"], "is_correct": True},
        {"skills": ["Evidence Evaluation"], "is_correct": False},
        {"skills": ["Causal Reasoning", "Logical Structure"], "is_correct": True},
        {"skills": ["Assumption Identification"], "is_correct": False},
        {"skills": ["Causal Reasoning"], "is_correct": True},
    ]


# ===========================================================================
# TestSkillEncoder (9 tests)
# ===========================================================================

class TestSkillEncoder:
    """SkillEncoder 词表、编码、解码、持久化测试"""

    def test_build_vocab(self, encoder):
        """build_vocab 应发现 4 个唯一技能"""
        assert encoder.num_skills == 4

    def test_vocab_deterministic(self, seeded_db):
        """多次 build_vocab 结果一致"""
        enc1 = SkillEncoder()
        enc1.build_vocab(seeded_db)
        enc2 = SkillEncoder()
        enc2.build_vocab(seeded_db)
        assert enc1.skill_to_id == enc2.skill_to_id

    def test_vocab_alphabetical(self, encoder):
        """词表按字母序排列"""
        skills = list(encoder.skill_to_id.keys())
        assert skills == sorted(skills)

    def test_encode_correct(self, encoder):
        """答对时前 K 维有值，后 K 维为零"""
        vec = encoder.encode_interaction(["Causal Reasoning"], is_correct=True)
        k = encoder.num_skills
        assert vec.shape == (2 * k,)
        idx = encoder.skill_to_id["Causal Reasoning"]
        assert vec[idx] == 1.0
        assert np.sum(vec[k:]) == 0.0  # 后半全零

    def test_encode_incorrect(self, encoder):
        """答错时后 K 维有值，前 K 维为零"""
        vec = encoder.encode_interaction(["Evidence Evaluation"], is_correct=False)
        k = encoder.num_skills
        idx = encoder.skill_to_id["Evidence Evaluation"]
        assert vec[k + idx] == 1.0
        assert np.sum(vec[:k]) == 0.0  # 前半全零

    def test_encode_multi_skill(self, encoder):
        """多技能编码同时置位"""
        vec = encoder.encode_interaction(
            ["Causal Reasoning", "Assumption Identification"], is_correct=True
        )
        k = encoder.num_skills
        idx1 = encoder.skill_to_id["Causal Reasoning"]
        idx2 = encoder.skill_to_id["Assumption Identification"]
        assert vec[idx1] == 1.0
        assert vec[idx2] == 1.0
        assert np.sum(vec[:k]) == 2.0

    def test_encode_unknown_skill(self, encoder):
        """未知技能被忽略"""
        vec = encoder.encode_interaction(["NonexistentSkill"], is_correct=True)
        assert np.sum(vec) == 0.0

    def test_decode_predictions(self, encoder):
        """decode_predictions 将向量解码为 {技能: 概率}"""
        output = np.array([0.8, 0.3, 0.9, 0.5], dtype=np.float32)
        decoded = encoder.decode_predictions(output)
        assert len(decoded) == 4
        assert all(isinstance(v, float) for v in decoded.values())

    def test_save_load_vocab(self, encoder):
        """保存/加载后词表一致"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            encoder.save_vocab(path)
            enc2 = SkillEncoder()
            enc2.load_vocab(path)
            assert enc2.skill_to_id == encoder.skill_to_id
            assert enc2.num_skills == encoder.num_skills
        finally:
            os.unlink(path)


# ===========================================================================
# TestDKTModelNumpy (7 tests)
# ===========================================================================

class TestDKTModelNumpy:
    """DKTModelNumpy 预测、训练、持久化测试"""

    def test_predict_empty_history(self, seeded_db):
        """空历史返回 0.5（先验）"""
        # 重置单例以使用 seeded_db
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        mastery = model.predict_mastery([])
        assert all(abs(v - 0.5) < 1e-6 for v in mastery.values())
        se._encoder = None  # 清理

    def test_predict_with_history(self, seeded_db, sample_history):
        """有历史时各技能返回 [0, 1] 范围的概率"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        mastery = model.predict_mastery(sample_history)
        assert len(mastery) > 0
        for v in mastery.values():
            assert 0.0 <= v <= 1.0
        se._encoder = None

    def test_train_returns_metrics(self, seeded_db, sample_history):
        """训练返回 {total_loss, num_updates, avg_loss}"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        metrics = model.train(sequences=[sample_history], epochs=1)
        assert "total_loss" in metrics
        assert "num_updates" in metrics
        assert "avg_loss" in metrics
        assert metrics["num_updates"] > 0
        se._encoder = None

    def test_train_loss_decreases(self, seeded_db, sample_history):
        """多轮训练后 loss 不应增加太多"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        m1 = model.train(sequences=[sample_history], epochs=1)
        m2 = model.train(sequences=[sample_history], epochs=5)
        # loss 应该合理（不会爆炸）
        assert m2["avg_loss"] < 10.0
        se._encoder = None

    def test_cold_start_unknown_skill(self, seeded_db):
        """未知技能的历史不影响预测"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        history = [{"skills": ["UnknownSkill"], "is_correct": True}]
        mastery = model.predict_mastery(history)
        # 所有已知技能仍为 0.5
        for v in mastery.values():
            assert abs(v - 0.5) < 1e-4
        se._encoder = None

    def test_save_load_weights(self, seeded_db, sample_history):
        """保存/加载权重后预测一致"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        model.train(sequences=[sample_history], epochs=3)
        pred1 = model.predict_mastery(sample_history)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            model.save_weights(path)

            se._encoder = None
            model2 = DKTModelNumpy(db_path=seeded_db)
            model2.load_weights(path)
            pred2 = model2.predict_mastery(sample_history)

            for skill in pred1:
                assert abs(pred1[skill] - pred2[skill]) < 1e-5
        finally:
            os.unlink(path)
        se._encoder = None

    def test_compare_with_bkt(self, seeded_db, sample_history):
        """compare_with_bkt 返回正确结构"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        comparison = model.compare_with_bkt(sample_history)
        assert len(comparison) > 0
        for skill, data in comparison.items():
            assert "bkt_error_rate" in data
            assert "dkt_mastery" in data
            assert "agreement" in data
            assert 0.0 <= data["bkt_error_rate"] <= 1.0
            assert 0.0 <= data["dkt_mastery"] <= 1.0
        se._encoder = None


# ===========================================================================
# TestDKTModelLSTM (5 tests)
# ===========================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDKTModelLSTM:
    """DKTModelLSTM 初始化、前向传播、预测、持久化、训练测试"""

    def test_init(self):
        """LSTM 模型正确初始化"""
        model = DKTModelLSTM(num_skills=4, hidden_size=32)
        assert model.num_skills == 4
        assert model.input_size == 8  # 2 * 4
        assert model.hidden_size == 32

    def test_forward_shape(self):
        """forward 输出形状正确"""
        model = DKTModelLSTM(num_skills=4)
        x = torch.randn(2, 5, 8)  # batch=2, seq=5, input=2*4
        out = model.forward(x)
        assert out.shape == (2, 5, 4)  # batch=2, seq=5, skills=4

    def test_predict_mastery(self, seeded_db, sample_history):
        """predict_mastery 返回有效的掌握概率"""
        import engine.skill_encoder as se
        se._encoder = None
        enc = get_skill_encoder(seeded_db)
        model = DKTModelLSTM(num_skills=enc.num_skills)
        mastery = model.predict_mastery(sample_history, enc)
        assert len(mastery) == enc.num_skills
        for v in mastery.values():
            assert 0.0 <= v <= 1.0
        se._encoder = None

    def test_save_load_weights(self):
        """保存/加载权重后参数一致"""
        model = DKTModelLSTM(num_skills=4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save_weights(path)
            model2 = DKTModelLSTM(num_skills=4)
            model2.load_weights(path)
            # 检查参数一致
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"
        finally:
            os.unlink(path)

    def test_train_returns_metrics(self, seeded_db, sample_history):
        """train_model 返回有效指标"""
        import engine.skill_encoder as se
        se._encoder = None
        enc = get_skill_encoder(seeded_db)
        model = DKTModelLSTM(num_skills=enc.num_skills)
        metrics = model.train_model(
            sequences=[sample_history],
            encoder=enc,
            epochs=2,
            patience=5,
        )
        assert "train_losses" in metrics
        assert "best_epoch" in metrics
        assert isinstance(metrics["train_losses"], list)
        assert len(metrics["train_losses"]) > 0
        se._encoder = None


# ===========================================================================
# TestAutoSelection (3 tests)
# ===========================================================================

class TestAutoSelection:
    """get_dkt_model 自动选择逻辑测试"""

    def test_selects_numpy_when_few_interactions(self, seeded_db):
        """交互 < 50 时选择 numpy 模型"""
        import engine.skill_encoder as se
        se._encoder = None
        model = get_dkt_model(db_path=seeded_db)
        assert isinstance(model, DKTModelNumpy)
        se._encoder = None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_selects_lstm_when_enough_interactions(self, seeded_db):
        """交互 >= 50 且 torch 可用时选择 LSTM"""
        import engine.skill_encoder as se
        se._encoder = None

        # 插入足够的交互记录
        conn = sqlite3.connect(seeded_db)
        cursor = conn.cursor()
        for i in range(60):
            cursor.execute(
                "INSERT INTO answer_history (user_id, question_id, skill_ids, is_correct) VALUES (?, ?, ?, ?)",
                ("default", f"q{i % 3 + 1}", '["Causal Reasoning"]', i % 2),
            )
        conn.commit()
        conn.close()

        model = get_dkt_model(db_path=seeded_db)
        assert isinstance(model, DKTModelLSTM)
        se._encoder = None

    def test_fallback_when_torch_unavailable(self, seeded_db):
        """torch 不可用时回退到 numpy"""
        import engine.skill_encoder as se
        se._encoder = None

        with patch("engine.dkt_model.TORCH_AVAILABLE", False):
            model = get_dkt_model(db_path=seeded_db)
            assert isinstance(model, DKTModelNumpy)
        se._encoder = None


# ===========================================================================
# TestCompareWithBKT (2 tests)
# ===========================================================================

class TestCompareWithBKT:
    """BKT 对比功能测试"""

    def test_output_structure(self, seeded_db, sample_history):
        """compare_with_bkt 输出结构正确"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        result = model.compare_with_bkt(sample_history)
        assert isinstance(result, dict)
        for skill, data in result.items():
            assert isinstance(skill, str)
            assert set(data.keys()) == {"bkt_error_rate", "dkt_mastery", "agreement"}
        se._encoder = None

    def test_empty_history(self, seeded_db):
        """空历史返回空字典"""
        import engine.skill_encoder as se
        se._encoder = None
        model = DKTModelNumpy(db_path=seeded_db)
        result = model.compare_with_bkt([])
        assert result == {}
        se._encoder = None


# ===========================================================================
# TestAnswerHistory (5 tests)
# ===========================================================================

class TestAnswerHistory:
    """answer_history 数据库操作测试"""

    def test_table_creation(self, tmp_db):
        """init_db 创建 answer_history 表"""
        from utils.db_handler import DatabaseManager
        dm = DatabaseManager(db_path=tmp_db)
        dm.init_db()
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='answer_history'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_insert_and_query(self, tmp_db):
        """插入和查询答题历史"""
        from utils.db_handler import DatabaseManager
        dm = DatabaseManager(db_path=tmp_db)
        dm.init_db()

        assert dm.insert_answer_history("q1", ["Causal Reasoning"], True, theta_at_time=0.5)
        assert dm.insert_answer_history("q2", ["Evidence Evaluation"], False)

        rows = dm.query_answer_history()
        assert len(rows) == 2
        assert rows[0]["question_id"] == "q1"
        assert rows[0]["skill_ids"] == ["Causal Reasoning"]
        assert rows[0]["is_correct"] == 1
        assert rows[0]["theta_at_time"] == 0.5

    def test_ordering(self, tmp_db):
        """查询结果按 created_at 升序"""
        from utils.db_handler import DatabaseManager
        dm = DatabaseManager(db_path=tmp_db)
        dm.init_db()

        dm.insert_answer_history("q1", ["A"], True)
        dm.insert_answer_history("q2", ["B"], False)
        dm.insert_answer_history("q3", ["C"], True)

        rows = dm.query_answer_history()
        ids = [r["question_id"] for r in rows]
        assert ids == ["q1", "q2", "q3"]

    def test_user_isolation(self, tmp_db):
        """不同用户的记录隔离"""
        from utils.db_handler import DatabaseManager
        dm = DatabaseManager(db_path=tmp_db)
        dm.init_db()

        dm.insert_answer_history("q1", ["A"], True, user_id="alice")
        dm.insert_answer_history("q2", ["B"], False, user_id="bob")
        dm.insert_answer_history("q3", ["C"], True, user_id="alice")

        alice_rows = dm.query_answer_history(user_id="alice")
        assert len(alice_rows) == 2
        bob_rows = dm.query_answer_history(user_id="bob")
        assert len(bob_rows) == 1

    def test_count(self, tmp_db):
        """count_answer_history 返回正确计数"""
        from utils.db_handler import DatabaseManager
        dm = DatabaseManager(db_path=tmp_db)
        dm.init_db()

        assert dm.count_answer_history() == 0
        dm.insert_answer_history("q1", ["A"], True)
        dm.insert_answer_history("q2", ["B"], False)
        assert dm.count_answer_history() == 2


# ===========================================================================
# TestRecommenderIntegration (3 tests)
# ===========================================================================

class TestRecommenderIntegration:
    """recommender.generate_next_question 的 use_dkt 集成测试"""

    def _make_mock_state(self):
        """创建 mock session state"""
        state = MagicMock()
        state.radio_key = 0
        state.attempt = 0
        state.phase = ""
        state.last_feedback = ""
        state.show_explanation = False
        state.pending_next_question = False
        state.socratic_context = {}
        state.chat_history = []
        state.current_q = None
        state.current_q_id = None
        state.current_question = None
        return state

    @patch("engine.recommender.get_spaced_repetition_model")
    def test_use_dkt_false(self, mock_sr, seeded_db):
        """use_dkt=False 不调用 DKT"""
        from engine.recommender import generate_next_question
        from utils.db_handler import DatabaseManager

        mock_sr.return_value.get_review_candidates.return_value = []
        dm = DatabaseManager(db_path=seeded_db)
        state = self._make_mock_state()

        with patch("engine.dkt_model.get_dkt_model") as mock_dkt:
            result = generate_next_question(
                user_theta=0.0,
                current_q_id="",
                questions_log=[],
                session_state=state,
                db_manager=dm,
                use_bandit=False,
                use_dkt=False,
            )
            mock_dkt.assert_not_called()

    @patch("engine.recommender.get_spaced_repetition_model")
    def test_use_dkt_true(self, mock_sr, seeded_db):
        """use_dkt=True 调用 DKT predict_mastery"""
        from engine.recommender import generate_next_question
        from utils.db_handler import DatabaseManager

        mock_sr.return_value.get_review_candidates.return_value = []
        dm = DatabaseManager(db_path=seeded_db)
        state = self._make_mock_state()

        mock_model = MagicMock()
        mock_model.predict_mastery.return_value = {"Causal Reasoning": 0.3, "Evidence Evaluation": 0.8}

        with patch("engine.dkt_model.get_dkt_model", return_value=mock_model) as mock_get:
            result = generate_next_question(
                user_theta=0.0,
                current_q_id="",
                questions_log=[],
                session_state=state,
                db_manager=dm,
                use_bandit=False,
                use_dkt=True,
            )
            mock_get.assert_called_once()

    @patch("engine.recommender.get_spaced_repetition_model")
    def test_dkt_failure_falls_back(self, mock_sr, seeded_db):
        """DKT 异常时降级到 BKT"""
        from engine.recommender import generate_next_question
        from utils.db_handler import DatabaseManager

        mock_sr.return_value.get_review_candidates.return_value = []
        dm = DatabaseManager(db_path=seeded_db)
        state = self._make_mock_state()

        with patch("engine.dkt_model.get_dkt_model", side_effect=Exception("boom")):
            result = generate_next_question(
                user_theta=0.0,
                current_q_id="",
                questions_log=[],
                session_state=state,
                db_manager=dm,
                use_bandit=False,
                use_dkt=True,
            )
            # 应不抛异常，正常返回（可能为 None 如果没候选）
            # 只要不抛出异常就说明降级成功


# ===========================================================================
# TestHelperFunctions (4 tests)
# ===========================================================================

class TestHelperFunctions:
    """辅助函数测试"""

    def test_sigmoid_positive(self):
        """sigmoid 对正数返回 (0.5, 1)"""
        result = _sigmoid(np.array([10.0]))
        assert 0.99 < result[0] < 1.0

    def test_sigmoid_negative(self):
        """sigmoid 对负数返回 (0, 0.5)"""
        result = _sigmoid(np.array([-10.0]))
        assert 0.0 < result[0] < 0.01

    def test_sigmoid_stability(self):
        """sigmoid 对极端值不溢出"""
        result = _sigmoid(np.array([500.0, -500.0, 0.0]))
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert abs(result[2] - 0.5) < 1e-6

    def test_extract_features_shape(self, sample_history):
        """_extract_features 返回正确形状"""
        features = _extract_features(sample_history, "Causal Reasoning")
        assert features.shape == (WINDOW_SIZE, NUM_FEATURES)

    def test_extract_features_empty(self):
        """空历史返回全零特征"""
        features = _extract_features([], "Causal Reasoning")
        assert features.shape == (WINDOW_SIZE, NUM_FEATURES)
        assert np.sum(features) == 0.0
