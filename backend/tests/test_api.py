"""
API 端点测试
使用 FastAPI TestClient 测试所有端点
"""

import sys
import os

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ========== /health ==========

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "env" in data


# ========== /api/theta/update ==========

class TestThetaUpdate:
    def test_correct_answer_increases_theta(self):
        resp = client.post("/api/theta/update", json={
            "current_theta": 0.0,
            "question_difficulty": 0.0,
            "is_correct": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["new_theta"] > 0.0
        assert 20 <= data["gmat_score"] <= 51

    def test_wrong_answer_decreases_theta(self):
        resp = client.post("/api/theta/update", json={
            "current_theta": 0.0,
            "question_difficulty": 0.0,
            "is_correct": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["new_theta"] < 0.0

    def test_theta_clamped_to_range(self):
        # 极高 theta + 答对，不应超过 3.0
        resp = client.post("/api/theta/update", json={
            "current_theta": 3.0,
            "question_difficulty": -3.0,
            "is_correct": True,
        })
        data = resp.json()
        assert data["new_theta"] <= 3.0

    def test_gmat_score_range(self):
        for theta in [-3.0, 0.0, 3.0]:
            resp = client.post("/api/theta/update", json={
                "current_theta": theta,
                "question_difficulty": 0.0,
                "is_correct": True,
            })
            score = resp.json()["gmat_score"]
            assert 20 <= score <= 51

    def test_invalid_theta_rejected(self):
        resp = client.post("/api/theta/update", json={
            "current_theta": 5.0,  # 超出范围
            "question_difficulty": 0.0,
            "is_correct": True,
        })
        assert resp.status_code == 422  # Pydantic 验证失败


# ========== /api/questions/next ==========

class TestQuestionsNext:
    def test_returns_question(self):
        resp = client.post("/api/questions/next", json={
            "user_theta": 0.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["question_id"]
        assert data["question_type"]
        assert data["stimulus"]
        assert data["question"]
        assert len(data["choices"]) == 5

    def test_no_correct_field_exposed(self):
        resp = client.post("/api/questions/next", json={
            "user_theta": 0.0,
        })
        data = resp.json()
        assert "correct" not in data
        assert "correct_choice" not in data

    def test_skills_returned(self):
        resp = client.post("/api/questions/next", json={
            "user_theta": 0.0,
        })
        data = resp.json()
        assert isinstance(data["skills"], list)

    def test_with_history_log(self):
        resp = client.post("/api/questions/next", json={
            "user_theta": 0.5,
            "current_q_id": "nonexistent",
            "questions_log": [
                {"question_id": "q1", "skills": ["因果推理"], "is_correct": False},
                {"question_id": "q2", "skills": ["假设识别"], "is_correct": True},
            ],
        })
        assert resp.status_code == 200

    def test_different_theta_returns_question(self):
        # 不同 theta 值都应能返回题目
        for theta in [-2.0, 0.0, 2.0]:
            resp = client.post("/api/questions/next", json={
                "user_theta": theta,
            })
            assert resp.status_code == 200, f"Failed for theta={theta}"


# ========== /api/tutor/chat ==========

class TestTutorChat:
    def test_simple_chat(self):
        resp = client.post("/api/tutor/chat", json={
            "message": "Hello, can you help me?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0
        assert isinstance(data["is_error"], bool)

    def test_chat_with_question_context(self):
        resp = client.post("/api/tutor/chat", json={
            "message": "我选了A",
            "question_id": "test_q1",
            "current_q": {
                "stimulus": "某公司销售额增长。",
                "question": "以下哪项削弱结论？",
                "choices": ["A. 原因1", "B. 原因2", "C. 原因3", "D. 原因4", "E. 原因5"],
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert not data["is_error"]
        assert len(data["reply"]) > 0

    def test_chat_with_history(self):
        resp = client.post("/api/tutor/chat", json={
            "message": "那B呢？",
            "chat_history": [
                {"role": "user", "content": "我选了A"},
                {"role": "assistant", "content": "为什么选A？"},
            ],
            "question_id": "test_q1",
            "current_q": {
                "stimulus": "某公司销售额增长。",
                "question": "以下哪项削弱结论？",
                "choices": ["A. 原因1", "B. 原因2", "C. 原因3", "D. 原因4", "E. 原因5"],
            },
        })
        assert resp.status_code == 200

    def test_missing_message_rejected(self):
        resp = client.post("/api/tutor/chat", json={})
        assert resp.status_code == 422  # message 是必填字段
