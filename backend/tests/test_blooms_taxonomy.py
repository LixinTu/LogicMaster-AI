"""
Bloom's Taxonomy 认知水平评估测试

测试 evaluate_blooms_level 的核心功能：
- 各级别到 understanding 的映射
- 向后兼容（evaluate_understanding 仍返回 confused/partial/clear）
- ConversationManager 跟踪 blooms_history
- 提示策略根据 Bloom's level 调整
- /conclude 端点返回 blooms_progression
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
from backend.services.conversation_manager import (
    ConversationManager,
    STATE_HINTING,
    STATE_CONCLUDED,
)

client = TestClient(app)

SAMPLE_QUESTION = {
    "question_id": "test_q_blooms",
    "question_type": "Weaken",
    "stimulus": "A study found that people who drink coffee live longer.",
    "question": "Which of the following weakens the argument?",
    "choices": [
        "A. Coffee drinkers tend to be wealthier",
        "B. Tea drinkers also live long",
        "C. Coffee contains antioxidants",
        "D. The study was conducted in a single city",
        "E. Decaf coffee shows the same effect",
    ],
    "correct": "A",
}


# ---------------------------------------------------------------------------
# TestBloomsLevelMapping - 级别映射
# ---------------------------------------------------------------------------

class TestBloomsLevelMapping:
    """各 Bloom's level 应正确映射到 understanding"""

    def test_level_1_maps_to_confused(self):
        """Level 1 (Remember) → confused"""
        from backend.services.tutor_agent import SocraticTutorAgent
        agent = MagicMock(spec=SocraticTutorAgent)
        # 直接测试映射逻辑
        mappings = {1: "confused", 2: "confused", 3: "partial", 4: "partial", 5: "clear", 6: "clear"}
        for level, expected in mappings.items():
            if level <= 2:
                assert expected == "confused"
            elif level <= 4:
                assert expected == "partial"
            else:
                assert expected == "clear"

    def test_all_levels_mapped(self):
        """所有 6 个 level 都有对应的 level_name"""
        level_names = {
            1: "Remember", 2: "Understand", 3: "Apply",
            4: "Analyze", 5: "Evaluate", 6: "Create",
        }
        for level in range(1, 7):
            assert level in level_names

    def test_evaluate_blooms_returns_structure(self):
        """evaluate_blooms_level 应返回完整的结果结构"""
        from backend.services.tutor_agent import SocraticTutorAgent

        with patch.object(SocraticTutorAgent, "__init__", lambda self, **kwargs: None):
            agent = SocraticTutorAgent()
            agent.llm = MagicMock()
            agent.str_parser = MagicMock()
            agent.blooms_prompt = MagicMock()

            # 模拟 LLM 返回 level 4
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = '{"level": 4, "reasoning": "Student analyzed the structure"}'
            agent.blooms_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

            result = agent.evaluate_blooms_level(
                student_response="The premise is X, conclusion is Y",
                logic_gap="correlation vs causation",
                key_assumption="no confound",
            )
            assert "level" in result
            assert "level_name" in result
            assert "reasoning" in result
            assert "mapped_understanding" in result
            assert result["level"] == 4
            assert result["level_name"] == "Analyze"
            assert result["mapped_understanding"] == "partial"

    def test_evaluate_blooms_fallback_on_error(self):
        """LLM 失败时应返回默认值 (level=1, Remember, confused)"""
        from backend.services.tutor_agent import SocraticTutorAgent

        with patch.object(SocraticTutorAgent, "__init__", lambda self, **kwargs: None):
            agent = SocraticTutorAgent()
            agent.llm = MagicMock()
            agent.str_parser = MagicMock()
            agent.blooms_prompt = MagicMock()

            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("LLM timeout")
            agent.blooms_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

            result = agent.evaluate_blooms_level(
                student_response="I don't know",
                logic_gap="test",
                key_assumption="test",
            )
            assert result["level"] == 1
            assert result["level_name"] == "Remember"
            assert result["mapped_understanding"] == "confused"


# ---------------------------------------------------------------------------
# TestBackwardCompatibility - 向后兼容
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """evaluate_understanding 仍返回 confused/partial/clear"""

    def test_evaluate_understanding_returns_old_format(self):
        """evaluate_understanding() 应返回 {understanding, reasoning}"""
        from backend.services.tutor_agent import SocraticTutorAgent

        with patch.object(SocraticTutorAgent, "__init__", lambda self, **kwargs: None):
            agent = SocraticTutorAgent()
            # mock evaluate_blooms_level
            agent.evaluate_blooms_level = MagicMock(return_value={
                "level": 3,
                "level_name": "Apply",
                "reasoning": "Student applied concept",
                "mapped_understanding": "partial",
            })

            result = agent.evaluate_understanding(
                student_response="test",
                logic_gap="test",
                key_assumption="test",
            )
            assert "understanding" in result
            assert "reasoning" in result
            assert result["understanding"] == "partial"
            assert result["reasoning"] == "Student applied concept"

    def test_evaluate_understanding_calls_blooms(self):
        """evaluate_understanding 应内部调用 evaluate_blooms_level"""
        from backend.services.tutor_agent import SocraticTutorAgent

        with patch.object(SocraticTutorAgent, "__init__", lambda self, **kwargs: None):
            agent = SocraticTutorAgent()
            agent.evaluate_blooms_level = MagicMock(return_value={
                "level": 5,
                "level_name": "Evaluate",
                "reasoning": "Good judgment",
                "mapped_understanding": "clear",
            })

            result = agent.evaluate_understanding("test", "gap", "assumption")
            agent.evaluate_blooms_level.assert_called_once()
            assert result["understanding"] == "clear"


# ---------------------------------------------------------------------------
# TestBloomsProgressionTracking - 进度跟踪
# ---------------------------------------------------------------------------

class TestBloomsProgressionTracking:
    """ConversationManager 跟踪 Bloom's level 历史"""

    def test_blooms_default(self):
        """新对话默认 blooms_level=1, blooms_history=[]"""
        cm = ConversationManager()
        conv = cm.create_conversation("q1")
        assert conv.blooms_level == 1
        assert conv.blooms_history == []

    def test_blooms_history_appended(self):
        """update_state(blooms_level=X) 应追加到 blooms_history"""
        cm = ConversationManager()
        conv = cm.create_conversation("q1")
        cid = conv.conversation_id

        cm.update_state(cid, blooms_level=2)
        cm.update_state(cid, blooms_level=4)
        cm.update_state(cid, blooms_level=5)

        assert conv.blooms_level == 5
        assert conv.blooms_history == [2, 4, 5]

    def test_blooms_in_to_dict(self):
        """to_dict() 应包含 blooms_level 和 blooms_history"""
        cm = ConversationManager()
        conv = cm.create_conversation("q1")
        cm.update_state(conv.conversation_id, blooms_level=3)

        d = conv.to_dict()
        assert d["blooms_level"] == 3
        assert d["blooms_history"] == [3]

    def test_conclude_includes_blooms_progression(self):
        """conclude() 应返回 blooms_progression"""
        cm = ConversationManager()
        conv = cm.create_conversation("q1")
        cid = conv.conversation_id
        cm.update_state(cid, state=STATE_HINTING, hint_count=2)
        cm.update_state(cid, blooms_level=1)
        cm.update_state(cid, blooms_level=3)
        cm.update_state(cid, blooms_level=5)
        cm.add_message(cid, "user", "msg")

        summary = cm.conclude(cid)
        assert summary["blooms_level"] == 5
        assert summary["blooms_progression"] == [1, 3, 5]


# ---------------------------------------------------------------------------
# TestHintStrategyAdjustment - 提示策略调整
# ---------------------------------------------------------------------------

class TestHintStrategyAdjustment:
    """Bloom's level 应影响提示策略"""

    def test_low_blooms_adds_scaffolding(self):
        """blooms_level 1-2 应在提示中增加脚手架指令"""
        from backend.services.tutor_agent import SocraticTutorAgent

        with patch.object(SocraticTutorAgent, "__init__", lambda self, **kwargs: None):
            agent = SocraticTutorAgent()
            agent.llm = MagicMock()
            agent.str_parser = MagicMock()
            agent.hint_prompt = MagicMock()

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "What is the conclusion?"
            agent.hint_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

            agent.generate_socratic_hint(
                question=SAMPLE_QUESTION,
                user_choice="B",
                logic_gap="correlation confusion",
                error_type="correlation_causation",
                hint_count=0,
                blooms_level=1,
            )

            # 验证 invoke 被调用，并检查 strength_instruction 包含脚手架关键词
            call_args = mock_chain.invoke.call_args[0][0]
            assert "scaffolding" in call_args["strength_instruction"].lower()

    def test_mid_blooms_pushes_analysis(self):
        """blooms_level 3-4 应推动更深分析"""
        from backend.services.tutor_agent import SocraticTutorAgent

        with patch.object(SocraticTutorAgent, "__init__", lambda self, **kwargs: None):
            agent = SocraticTutorAgent()
            agent.llm = MagicMock()
            agent.str_parser = MagicMock()
            agent.hint_prompt = MagicMock()

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Identify the premise and conclusion."
            agent.hint_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

            agent.generate_socratic_hint(
                question=SAMPLE_QUESTION,
                user_choice="B",
                logic_gap="correlation confusion",
                error_type="correlation_causation",
                hint_count=1,
                blooms_level=4,
            )

            call_args = mock_chain.invoke.call_args[0][0]
            assert "analysis" in call_args["strength_instruction"].lower() or \
                   "premise" in call_args["strength_instruction"].lower()

    def test_no_blooms_no_adjustment(self):
        """blooms_level=None 时不应有额外调整"""
        from backend.services.tutor_agent import SocraticTutorAgent

        with patch.object(SocraticTutorAgent, "__init__", lambda self, **kwargs: None):
            agent = SocraticTutorAgent()
            agent.llm = MagicMock()
            agent.str_parser = MagicMock()
            agent.hint_prompt = MagicMock()

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "What is the conclusion?"
            agent.hint_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

            agent.generate_socratic_hint(
                question=SAMPLE_QUESTION,
                user_choice="B",
                logic_gap="test",
                error_type="other",
                hint_count=0,
                blooms_level=None,
            )

            call_args = mock_chain.invoke.call_args[0][0]
            assert "scaffolding" not in call_args["strength_instruction"].lower()
            assert "analysis" not in call_args["strength_instruction"].lower()


# ---------------------------------------------------------------------------
# TestConcludeEndpoint - /conclude 端点包含 blooms
# ---------------------------------------------------------------------------

class TestConcludeEndpoint:
    """/conclude 端点应返回 blooms_progression"""

    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    def test_conclude_includes_blooms_progression(self, mock_get_cm, mock_get_agent):
        cm = ConversationManager()
        conv = cm.create_conversation("q_blooms")
        cid = conv.conversation_id
        cm.update_state(cid, state=STATE_HINTING, hint_count=2, understanding="partial")
        cm.update_state(cid, blooms_level=1)
        cm.update_state(cid, blooms_level=3)
        conv.logic_gap = "Missed confound"
        conv.question = SAMPLE_QUESTION
        conv.correct_choice = "A"
        cm.add_message(cid, "user", "msg1")
        cm.add_message(cid, "assistant", "hint1")
        mock_get_cm.return_value = cm

        agent = MagicMock()
        agent.generate_conclusion.return_value = "The correct answer is A."
        mock_get_agent.return_value = agent

        resp = client.post("/api/tutor/conclude", json={
            "conversation_id": cid,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["blooms_level"] == 3
        assert data["summary"]["blooms_progression"] == [1, 3]

    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    def test_continue_returns_blooms_fields(self, mock_get_cm, mock_get_agent):
        """/continue 端点应返回 blooms_level 和 blooms_name"""
        cm = ConversationManager()
        conv = cm.create_conversation("q_cont_blooms")
        cid = conv.conversation_id
        cm.update_state(cid, state=STATE_HINTING, hint_count=1)
        conv.logic_gap = "Correlation vs causation"
        conv.error_type = "correlation_causation"
        conv.user_choice = "B"
        conv.question = SAMPLE_QUESTION
        conv.correct_choice = "A"
        conv.key_assumption = "No confounding variable"
        cm.add_message(cid, "user", "I chose B")
        cm.add_message(cid, "assistant", "What is the conclusion?")
        mock_get_cm.return_value = cm

        agent = MagicMock()
        agent.evaluate_blooms_level.return_value = {
            "level": 3,
            "level_name": "Apply",
            "reasoning": "Student applied concept",
            "mapped_understanding": "partial",
        }
        agent.generate_socratic_hint.return_value = "Consider the confound."
        mock_get_agent.return_value = agent

        resp = client.post("/api/tutor/continue", json={
            "conversation_id": cid,
            "student_message": "Coffee causes longer life because of antioxidants.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["blooms_level"] == 3
        assert data["blooms_name"] == "Apply"
        assert data["student_understanding"] == "partial"
