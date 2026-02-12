"""
Week 3 测试：ConversationManager + Tutor Agent 端点
- ConversationManager 生命周期
- /api/tutor/start-remediation, /continue, /conclude 端点（Mock LLM）
- 降级回退逻辑
"""

import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.conversation_manager import (
    ConversationManager,
    Conversation,
    STATE_DIAGNOSING,
    STATE_HINTING,
    STATE_CONCLUDED,
    UNDERSTANDING_CONFUSED,
    UNDERSTANDING_PARTIAL,
    UNDERSTANDING_CLEAR,
    MAX_HINTS,
    CONVERSATION_TTL_SECONDS,
)

client = TestClient(app)


# ========== ConversationManager 单元测试 ==========

class TestConversationManager:
    def setup_method(self):
        self.cm = ConversationManager()

    def test_create_conversation(self):
        conv = self.cm.create_conversation("q1")
        assert conv.question_id == "q1"
        assert conv.current_state == STATE_DIAGNOSING
        assert conv.hint_count == 0
        assert conv.student_understanding == UNDERSTANDING_CONFUSED
        assert len(conv.chat_history) == 0
        assert self.cm.active_count == 1

    def test_get_conversation_exists(self):
        conv = self.cm.create_conversation("q2")
        fetched = self.cm.get_conversation(conv.conversation_id)
        assert fetched is not None
        assert fetched.question_id == "q2"

    def test_get_conversation_not_found(self):
        assert self.cm.get_conversation("nonexistent") is None

    def test_add_message(self):
        conv = self.cm.create_conversation("q3")
        cid = conv.conversation_id
        assert self.cm.add_message(cid, "user", "hello")
        assert self.cm.add_message(cid, "assistant", "hi there")
        assert len(conv.chat_history) == 2
        assert conv.chat_history[0]["role"] == "user"
        assert conv.chat_history[1]["content"] == "hi there"

    def test_add_message_nonexistent(self):
        assert self.cm.add_message("fake_id", "user", "msg") is False

    def test_get_context_for_llm(self):
        conv = self.cm.create_conversation("q4")
        cid = conv.conversation_id
        for i in range(10):
            self.cm.add_message(cid, "user", f"msg_{i}")
        # 默认 max_messages=8
        ctx = self.cm.get_context_for_llm(cid)
        assert len(ctx) == 8
        assert ctx[0]["content"] == "msg_2"  # 截取最后 8 条

    def test_get_context_for_llm_nonexistent(self):
        assert self.cm.get_context_for_llm("fake") == []

    def test_update_state(self):
        conv = self.cm.create_conversation("q5")
        cid = conv.conversation_id
        self.cm.update_state(cid, state=STATE_HINTING, hint_count=2, understanding="partial")
        assert conv.current_state == STATE_HINTING
        assert conv.hint_count == 2
        assert conv.student_understanding == "partial"

    def test_update_state_nonexistent(self):
        assert self.cm.update_state("fake", state=STATE_HINTING) is False

    def test_should_continue_under_max_hints(self):
        conv = self.cm.create_conversation("q6")
        cid = conv.conversation_id
        self.cm.update_state(cid, state=STATE_HINTING, hint_count=1)
        assert self.cm.should_continue_remediation(cid) is True

    def test_should_not_continue_at_max_hints(self):
        conv = self.cm.create_conversation("q7")
        cid = conv.conversation_id
        self.cm.update_state(cid, state=STATE_HINTING, hint_count=MAX_HINTS)
        assert self.cm.should_continue_remediation(cid) is False

    def test_should_not_continue_when_clear(self):
        conv = self.cm.create_conversation("q8")
        cid = conv.conversation_id
        self.cm.update_state(cid, state=STATE_HINTING, hint_count=1, understanding=UNDERSTANDING_CLEAR)
        assert self.cm.should_continue_remediation(cid) is False

    def test_should_not_continue_when_concluded(self):
        conv = self.cm.create_conversation("q9")
        cid = conv.conversation_id
        self.cm.update_state(cid, state=STATE_CONCLUDED)
        assert self.cm.should_continue_remediation(cid) is False

    def test_conclude(self):
        conv = self.cm.create_conversation("q10")
        cid = conv.conversation_id
        self.cm.update_state(cid, state=STATE_HINTING, hint_count=2, understanding="partial")
        self.cm.add_message(cid, "user", "msg1")
        self.cm.add_message(cid, "assistant", "hint1")
        self.cm.add_message(cid, "user", "msg2")

        summary = self.cm.conclude(cid)
        assert summary is not None
        assert summary["total_turns"] == 2  # 2 user messages
        assert summary["hint_count"] == 2
        assert summary["final_understanding"] == "partial"
        assert summary["time_spent_seconds"] >= 0
        assert conv.current_state == STATE_CONCLUDED

    def test_conclude_nonexistent(self):
        assert self.cm.conclude("fake") is None

    def test_expired_conversation_is_removed(self):
        conv = self.cm.create_conversation("q_old")
        cid = conv.conversation_id
        # 模拟过期
        conv.created_at = time.time() - CONVERSATION_TTL_SECONDS - 1
        assert self.cm.get_conversation(cid) is None

    def test_conversation_to_dict(self):
        conv = self.cm.create_conversation("q11")
        d = conv.to_dict()
        assert d["question_id"] == "q11"
        assert d["current_state"] == STATE_DIAGNOSING
        assert d["chat_history_length"] == 0


# ========== Tutor 端点测试（Mock LLM） ==========

SAMPLE_QUESTION = {
    "question_id": "test_q_001",
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
    "explanation": "Coffee drinkers tend to be wealthier, which is a confound.",
}

MOCK_DIAGNOSIS = {
    "logic_gap": "Student confused correlation with causation.",
    "error_type": "correlation_causation",
    "core_conclusion": "Coffee causes longer life.",
    "key_assumption": "There is no confounding variable.",
    "why_wrong": "Option B does not address the causal claim.",
}


class TestStartRemediation:
    """测试 /api/tutor/start-remediation 端点"""

    @patch("backend.routers.tutor.get_ab_test_service")
    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    def test_start_remediation_success(self, mock_get_cm, mock_get_agent, mock_get_ab):
        # 设置 mock
        cm = ConversationManager()
        mock_get_cm.return_value = cm

        agent = MagicMock()
        agent.diagnose_error.return_value = MOCK_DIAGNOSIS
        agent.generate_socratic_hint.return_value = "What is the main conclusion?"
        mock_get_agent.return_value = agent

        ab = MagicMock()
        ab.assign_variant.return_value = "socratic_standard"
        ab.log_exposure.return_value = True
        mock_get_ab.return_value = ab

        resp = client.post("/api/tutor/start-remediation", json={
            "question_id": "test_q_001",
            "question": SAMPLE_QUESTION,
            "user_choice": "B",
            "correct_choice": "A",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert data["first_hint"] == "What is the main conclusion?"
        assert data["logic_gap"] == "Student confused correlation with causation."
        assert data["error_type"] == "correlation_causation"
        assert data["hint_count"] == 1
        assert data["current_state"] == STATE_HINTING
        assert data["variant"] == "socratic_standard"

    @patch("backend.routers.tutor.get_ab_test_service")
    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    def test_start_remediation_fallback_on_error(self, mock_get_cm, mock_get_agent, mock_get_ab):
        """LLM 失败时的降级回退"""
        cm = ConversationManager()
        mock_get_cm.return_value = cm

        agent = MagicMock()
        agent.diagnose_error.side_effect = Exception("LLM timeout")
        mock_get_agent.return_value = agent

        resp = client.post("/api/tutor/start-remediation", json={
            "question_id": "test_q_002",
            "question": SAMPLE_QUESTION,
            "user_choice": "C",
            "correct_choice": "A",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "step back" in data["first_hint"].lower() or len(data["first_hint"]) > 0
        assert data["hint_count"] == 1
        assert data["current_state"] == STATE_HINTING


class TestContinueRemediation:
    """测试 /api/tutor/continue 端点"""

    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    def test_continue_generates_next_hint(self, mock_get_cm, mock_get_agent):
        # 先创建一个对话
        cm = ConversationManager()
        conv = cm.create_conversation("q_cont")
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
        agent.evaluate_understanding.return_value = {"understanding": "partial", "reasoning": "ok"}
        agent.generate_socratic_hint.return_value = "Consider whether wealth is a factor."
        mock_get_agent.return_value = agent

        resp = client.post("/api/tutor/continue", json={
            "conversation_id": cid,
            "student_message": "The conclusion is that coffee helps.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Consider whether wealth is a factor."
        assert data["hint_count"] == 2
        assert data["student_understanding"] == "partial"
        assert data["should_continue"] is True
        assert data["current_state"] == STATE_HINTING

    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    def test_continue_concludes_when_clear(self, mock_get_cm, mock_get_agent):
        cm = ConversationManager()
        conv = cm.create_conversation("q_clear")
        cid = conv.conversation_id
        cm.update_state(cid, state=STATE_HINTING, hint_count=1)
        conv.logic_gap = "Missed confound"
        conv.error_type = "correlation_causation"
        conv.user_choice = "B"
        conv.question = SAMPLE_QUESTION
        conv.correct_choice = "A"
        conv.key_assumption = "No confounding variable"
        mock_get_cm.return_value = cm

        agent = MagicMock()
        agent.evaluate_understanding.return_value = {"understanding": "clear", "reasoning": "Student got it"}
        agent.generate_conclusion.return_value = "Great! The answer is A."
        mock_get_agent.return_value = agent

        resp = client.post("/api/tutor/continue", json={
            "conversation_id": cid,
            "student_message": "I see, wealth is a confounding variable!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["should_continue"] is False
        assert data["student_understanding"] == "clear"
        assert data["current_state"] == STATE_CONCLUDED

    def test_continue_404_on_missing_conversation(self):
        resp = client.post("/api/tutor/continue", json={
            "conversation_id": "nonexistent_id",
            "student_message": "hello",
        })
        assert resp.status_code == 404


class TestConcludeRemediation:
    """测试 /api/tutor/conclude 端点"""

    @patch("backend.routers.tutor.get_tutor_agent")
    @patch("backend.routers.tutor.get_conversation_manager")
    def test_conclude_success(self, mock_get_cm, mock_get_agent):
        cm = ConversationManager()
        conv = cm.create_conversation("q_conc")
        cid = conv.conversation_id
        cm.update_state(cid, state=STATE_HINTING, hint_count=2, understanding="partial")
        conv.logic_gap = "Missed confound"
        conv.question = SAMPLE_QUESTION
        conv.correct_choice = "A"
        cm.add_message(cid, "user", "msg1")
        cm.add_message(cid, "assistant", "hint1")
        mock_get_cm.return_value = cm

        agent = MagicMock()
        agent.generate_conclusion.return_value = "The correct answer is A. Well done!"
        mock_get_agent.return_value = agent

        resp = client.post("/api/tutor/conclude", json={
            "conversation_id": cid,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "correct answer is A" in data["conclusion"]
        assert data["summary"]["hint_count"] == 2
        assert data["summary"]["total_turns"] == 1  # 1 user message
        assert data["summary"]["final_understanding"] == "partial"

    def test_conclude_404_on_missing_conversation(self):
        resp = client.post("/api/tutor/conclude", json={
            "conversation_id": "nonexistent_id",
        })
        assert resp.status_code == 404


# ========== 向后兼容 /api/tutor/chat ==========

class TestTutorChatBackwardCompat:
    """确保旧 /api/tutor/chat 端点仍然可用"""

    @patch("backend.routers.tutor.tutor_reply")
    def test_chat_returns_reply(self, mock_reply):
        mock_reply.return_value = "Think about the conclusion."

        resp = client.post("/api/tutor/chat", json={
            "message": "Why is B wrong?",
            "chat_history": [],
            "question_id": "q_compat",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Think about the conclusion."
        assert data["is_error"] is False

    @patch("backend.routers.tutor.tutor_reply")
    def test_chat_error_flag(self, mock_reply):
        mock_reply.return_value = "[LLM ERROR] timeout"

        resp = client.post("/api/tutor/chat", json={
            "message": "hello",
            "chat_history": [],
        })
        data = resp.json()
        assert data["is_error"] is True
