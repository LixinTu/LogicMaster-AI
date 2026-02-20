"""
Feature tests: Dashboard, Bookmarks, Goals, Email Reminder
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

# ---------- Test DB fixture helpers ----------

def _make_test_db() -> str:
    """Create a temp SQLite DB with all tables, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    from utils.db_handler import DatabaseManager
    db = DatabaseManager(db_path=tmp.name)
    db.init_db()
    return tmp.name


def _seed_question(db_path: str, q_id: str = "q001", skills=None) -> None:
    """Insert a dummy verified question into the test DB."""
    skills = skills or ["Causal Reasoning", "Weaken"]
    content = json.dumps({
        "stimulus": "A company's revenue increased after a marketing campaign.",
        "question": "Which weakens the causal claim?",
        "choices": ["A", "B", "C", "D", "E"],
        "correct": "C",
        "skills": skills,
        "explanation": "Test explanation.",
        "detailed_explanation": "",
        "diagnoses": {},
        "label_source": "test",
        "skills_rationale": "",
    })
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute(
        """INSERT OR IGNORE INTO questions
           (id, question_type, difficulty, content, elo_difficulty, is_verified)
           VALUES (?, 'Weaken', 'medium', ?, 1500.0, 1)""",
        (q_id, content),
    )
    conn.commit()
    conn.close()


def _seed_answer_history(db_path: str, user_id: str, days_ago: int, is_correct: bool, skills=None, theta=0.5) -> None:
    """Insert answer_history record n days ago."""
    skills = skills or ["Causal Reasoning"]
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute(
        """INSERT INTO answer_history (user_id, question_id, skill_ids, is_correct, theta_at_time, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, f"q_{days_ago}", json.dumps(skills), 1 if is_correct else 0, theta, ts),
    )
    conn.commit()
    conn.close()


# ===========================================================================
# TestDashboardDB — unit tests for db_handler methods used by dashboard
# ===========================================================================

class TestDashboardDB(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        from utils.db_handler import DatabaseManager
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def test_today_count_empty(self):
        count = self.db.count_today_answers("user_x")
        self.assertEqual(count, 0)

    def test_today_count_today_records(self):
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=True)
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=False)
        count = self.db.count_today_answers("u1")
        self.assertEqual(count, 2)

    def test_today_count_excludes_old(self):
        _seed_answer_history(self.db_path, "u1", days_ago=1, is_correct=True)
        count = self.db.count_today_answers("u1")
        self.assertEqual(count, 0)

    def test_today_count_user_isolation(self):
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=True)
        self.assertEqual(self.db.count_today_answers("u1"), 1)
        self.assertEqual(self.db.count_today_answers("u2"), 0)

    def test_streak_empty(self):
        self.assertEqual(self.db.calculate_streak("noone"), 0)

    def test_streak_today_only(self):
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=True)
        streak = self.db.calculate_streak("u1")
        self.assertEqual(streak, 1)

    def test_streak_consecutive_days(self):
        for d in range(3):  # today, yesterday, 2 days ago
            _seed_answer_history(self.db_path, "u1", days_ago=d, is_correct=True)
        self.assertEqual(self.db.calculate_streak("u1"), 3)

    def test_streak_broken(self):
        # today + 2 days ago (gap on day 1)
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=True)
        _seed_answer_history(self.db_path, "u1", days_ago=2, is_correct=True)
        self.assertEqual(self.db.calculate_streak("u1"), 1)

    def test_streak_yesterday_no_today(self):
        # Only yesterday — should count as 1 (active streak)
        _seed_answer_history(self.db_path, "u1", days_ago=1, is_correct=True)
        self.assertGreaterEqual(self.db.calculate_streak("u1"), 1)

    def test_weak_skills_empty(self):
        result = self.db.get_skill_error_rates("u1")
        self.assertEqual(result, [])

    def test_weak_skills_ordered_by_error_rate(self):
        # 2 wrong for "Weaken", 0 wrong for "Strengthen"
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute(
            "INSERT INTO answer_history (user_id, question_id, skill_ids, is_correct, theta_at_time) VALUES (?, ?, ?, 0, 0.0)",
            ("u1", "qa", json.dumps(["Weaken"]))
        )
        conn.execute(
            "INSERT INTO answer_history (user_id, question_id, skill_ids, is_correct, theta_at_time) VALUES (?, ?, ?, 0, 0.0)",
            ("u1", "qb", json.dumps(["Weaken"]))
        )
        conn.execute(
            "INSERT INTO answer_history (user_id, question_id, skill_ids, is_correct, theta_at_time) VALUES (?, ?, ?, 1, 0.0)",
            ("u1", "qc", json.dumps(["Strengthen"]))
        )
        conn.commit()
        conn.close()
        result = self.db.get_skill_error_rates("u1")
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["skill_name"], "Weaken")
        self.assertAlmostEqual(result[0]["error_rate"], 1.0)
        self.assertAlmostEqual(result[0]["mastery"], 0.0)

    def test_get_latest_theta_none(self):
        self.assertIsNone(self.db.get_latest_theta("nobody"))

    def test_get_latest_theta(self):
        _seed_answer_history(self.db_path, "u1", days_ago=1, is_correct=True, theta=0.8)
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=True, theta=1.2)
        theta = self.db.get_latest_theta("u1")
        self.assertAlmostEqual(theta, 1.2, places=1)

    def test_get_last_practiced_none(self):
        self.assertIsNone(self.db.get_last_practiced_time("nobody"))

    def test_get_last_practiced(self):
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=True)
        ts = self.db.get_last_practiced_time("u1")
        self.assertIsNotNone(ts)


# ===========================================================================
# TestBookmarks — DB methods + API endpoint
# ===========================================================================

class TestBookmarkDB(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        from utils.db_handler import DatabaseManager
        self.db = DatabaseManager(db_path=self.db_path)
        _seed_question(self.db_path, "q001")

    def tearDown(self):
        os.unlink(self.db_path)

    def test_insert_bookmark(self):
        ok = self.db.insert_bookmark("u1", "q001", "favorite")
        self.assertTrue(ok)

    def test_insert_bookmark_duplicate_idempotent(self):
        self.db.insert_bookmark("u1", "q001", "wrong")
        ok = self.db.insert_bookmark("u1", "q001", "wrong")  # INSERT OR IGNORE
        self.assertTrue(ok)

    def test_remove_bookmark(self):
        self.db.insert_bookmark("u1", "q001", "favorite")
        ok = self.db.remove_bookmark("u1", "q001", "favorite")
        self.assertTrue(ok)
        items = self.db.query_bookmarks("u1", bookmark_type="favorite")
        self.assertEqual(len(items), 0)

    def test_query_bookmarks_by_type(self):
        self.db.insert_bookmark("u1", "q001", "wrong")
        items = self.db.query_bookmarks("u1", bookmark_type="wrong")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question_id"], "q001")

    def test_query_bookmarks_skill_filter(self):
        self.db.insert_bookmark("u1", "q001", "favorite")
        items_match = self.db.query_bookmarks("u1", skill_filter="Causal Reasoning")
        items_no_match = self.db.query_bookmarks("u1", skill_filter="NoSuchSkill")
        self.assertEqual(len(items_match), 1)
        self.assertEqual(len(items_no_match), 0)

    def test_query_bookmarks_stimulus_preview(self):
        self.db.insert_bookmark("u1", "q001", "favorite")
        items = self.db.query_bookmarks("u1")
        self.assertIn("stimulus_preview", items[0])
        self.assertIsInstance(items[0]["stimulus_preview"], str)

    def test_wrong_stats_empty(self):
        stats = self.db.get_wrong_stats("u1")
        self.assertEqual(stats["total_wrong"], 0)
        self.assertEqual(stats["by_skill"], [])
        self.assertEqual(stats["by_type"], [])

    def test_wrong_stats_with_data(self):
        self.db.insert_bookmark("u1", "q001", "wrong")
        stats = self.db.get_wrong_stats("u1")
        self.assertEqual(stats["total_wrong"], 1)
        self.assertIsInstance(stats["by_type"], list)
        self.assertIsInstance(stats["by_skill"], list)


class TestBookmarkAPI(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        _seed_question(self.db_path)
        # Patch DB path in router
        import backend.routers.bookmarks as bm_router
        self._orig_get_db = bm_router._get_db
        from utils.db_handler import DatabaseManager
        bm_router._get_db = lambda: DatabaseManager(db_path=self.db_path)
        from backend.main import app
        self.client = TestClient(app)

    def tearDown(self):
        import backend.routers.bookmarks as bm_router
        bm_router._get_db = self._orig_get_db
        os.unlink(self.db_path)

    def test_add_bookmark(self):
        resp = self.client.post("/api/bookmarks/add", json={
            "user_id": "u1", "question_id": "q001", "bookmark_type": "favorite"
        })
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["status"], "ok")

    def test_add_bookmark_invalid_type(self):
        resp = self.client.post("/api/bookmarks/add", json={
            "user_id": "u1", "question_id": "q001", "bookmark_type": "invalid"
        })
        self.assertEqual(resp.status_code, 422)

    def test_list_bookmarks(self):
        self.client.post("/api/bookmarks/add", json={
            "user_id": "u1", "question_id": "q001", "bookmark_type": "wrong"
        })
        resp = self.client.get("/api/bookmarks/list?user_id=u1&type=wrong")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["question_id"], "q001")

    def test_remove_bookmark(self):
        self.client.post("/api/bookmarks/add", json={
            "user_id": "u1", "question_id": "q001", "bookmark_type": "favorite"
        })
        resp = self.client.request("DELETE", "/api/bookmarks/remove", json={
            "user_id": "u1", "question_id": "q001", "bookmark_type": "favorite"
        })
        self.assertEqual(resp.status_code, 200)
        # verify gone
        items = self.client.get("/api/bookmarks/list?user_id=u1&type=favorite").json()
        self.assertEqual(len(items), 0)

    def test_wrong_stats(self):
        self.client.post("/api/bookmarks/add", json={
            "user_id": "u1", "question_id": "q001", "bookmark_type": "wrong"
        })
        resp = self.client.get("/api/bookmarks/wrong-stats?user_id=u1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_wrong"], 1)
        self.assertIn("by_skill", data)
        self.assertIn("by_type", data)


# ===========================================================================
# TestAutoWrongBookmark — bandit-update auto-bookmark
# ===========================================================================

class TestAutoWrongBookmark(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        _seed_question(self.db_path)
        import backend.routers.questions as q_router
        self._orig_db = q_router._db_manager
        from utils.db_handler import DatabaseManager
        q_router._db_manager = DatabaseManager(db_path=self.db_path)
        from backend.main import app
        self.client = TestClient(app)

    def tearDown(self):
        import backend.routers.questions as q_router
        q_router._db_manager = self._orig_db
        os.unlink(self.db_path)

    def test_wrong_answer_creates_bookmark(self):
        with patch("backend.routers.questions.get_bandit_selector") as mock_bandit, \
             patch("backend.routers.questions.get_spaced_repetition_model") as mock_sr:
            mock_bandit.return_value.update = MagicMock()
            mock_sr.return_value.update_half_life = MagicMock()
            resp = self.client.post("/api/questions/bandit-update", json={
                "question_id": "q001",
                "is_correct": False,
                "skills": ["Causal Reasoning"],
                "user_id": "u1",
            })
        self.assertEqual(resp.status_code, 200)
        from utils.db_handler import DatabaseManager
        db = DatabaseManager(db_path=self.db_path)
        items = db.query_bookmarks("u1", bookmark_type="wrong")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question_id"], "q001")

    def test_correct_answer_no_bookmark(self):
        with patch("backend.routers.questions.get_bandit_selector") as mock_bandit, \
             patch("backend.routers.questions.get_spaced_repetition_model") as mock_sr:
            mock_bandit.return_value.update = MagicMock()
            mock_sr.return_value.update_half_life = MagicMock()
            self.client.post("/api/questions/bandit-update", json={
                "question_id": "q001",
                "is_correct": True,
                "skills": ["Causal Reasoning"],
                "user_id": "u1",
            })
        from utils.db_handler import DatabaseManager
        db = DatabaseManager(db_path=self.db_path)
        items = db.query_bookmarks("u1", bookmark_type="wrong")
        self.assertEqual(len(items), 0)


# ===========================================================================
# TestGoalsDB — learning goals db methods
# ===========================================================================

class TestGoalsDB(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        from utils.db_handler import DatabaseManager
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def test_get_goal_defaults(self):
        goal = self.db.get_learning_goal("new_user")
        self.assertEqual(goal["target_gmat_score"], 40)
        self.assertEqual(goal["daily_question_goal"], 5)

    def test_upsert_and_get_goal(self):
        self.db.upsert_learning_goal("u1", target_gmat_score=48, daily_question_goal=10)
        goal = self.db.get_learning_goal("u1")
        self.assertEqual(goal["target_gmat_score"], 48)
        self.assertEqual(goal["daily_question_goal"], 10)

    def test_upsert_updates_existing(self):
        self.db.upsert_learning_goal("u1", 45, 5)
        self.db.upsert_learning_goal("u1", 50, 8)
        goal = self.db.get_learning_goal("u1")
        self.assertEqual(goal["target_gmat_score"], 50)
        self.assertEqual(goal["daily_question_goal"], 8)


class TestGoalsAPI(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        import backend.routers.goals as g_router
        import backend.routers.dashboard as d_router
        self._orig_goals_db = g_router._get_db
        self._orig_dash_db = d_router._get_db
        from utils.db_handler import DatabaseManager
        g_router._get_db = lambda: DatabaseManager(db_path=self.db_path)
        d_router._get_db = lambda: DatabaseManager(db_path=self.db_path)
        from backend.main import app
        self.client = TestClient(app)

    def tearDown(self):
        import backend.routers.goals as g_router
        import backend.routers.dashboard as d_router
        g_router._get_db = self._orig_goals_db
        d_router._get_db = self._orig_dash_db
        os.unlink(self.db_path)

    def test_set_goal(self):
        resp = self.client.post("/api/goals/set", json={
            "user_id": "u1",
            "target_gmat_score": 48,
            "daily_question_goal": 10,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_set_goal_invalid_score(self):
        resp = self.client.post("/api/goals/set", json={
            "user_id": "u1",
            "target_gmat_score": 99,   # exceeds max 51
            "daily_question_goal": 5,
        })
        self.assertEqual(resp.status_code, 422)

    def test_get_progress_defaults(self):
        resp = self.client.get("/api/goals/progress?user_id=new_user")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("target_gmat_score", data)
        self.assertIn("current_gmat_score", data)
        self.assertIn("score_gap", data)
        self.assertIn("estimated_questions_remaining", data)
        self.assertIn("daily_question_goal", data)
        self.assertIn("today_completed", data)
        self.assertIn("today_progress_pct", data)
        self.assertIn("on_track", data)

    def test_get_progress_on_track_false_by_default(self):
        resp = self.client.get("/api/goals/progress?user_id=new_user")
        data = resp.json()
        self.assertFalse(data["on_track"])
        self.assertEqual(data["today_completed"], 0)

    def test_score_gap_non_negative(self):
        resp = self.client.get("/api/goals/progress?user_id=new_user")
        data = resp.json()
        self.assertGreaterEqual(data["score_gap"], 0)


# ===========================================================================
# TestDashboardAPI
# ===========================================================================

class TestDashboardAPI(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        import backend.routers.dashboard as d_router
        self._orig_get_db = d_router._get_db
        from utils.db_handler import DatabaseManager
        d_router._get_db = lambda: DatabaseManager(db_path=self.db_path)
        from backend.main import app
        self.client = TestClient(app)

    def tearDown(self):
        import backend.routers.dashboard as d_router
        d_router._get_db = self._orig_get_db
        os.unlink(self.db_path)

    def test_summary_returns_all_fields(self):
        with patch("backend.routers.dashboard.get_spaced_repetition_model") as mock_sr:
            mock_sr.return_value.get_review_candidates.return_value = []
            resp = self.client.get("/api/dashboard/summary?user_id=new_user")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for field in [
            "today_goal", "today_completed", "streak_days",
            "current_theta", "gmat_score", "weak_skills",
            "reviews_due", "last_practiced",
        ]:
            self.assertIn(field, data, f"Missing field: {field}")

    def test_summary_today_completed_counts(self):
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=True)
        _seed_answer_history(self.db_path, "u1", days_ago=0, is_correct=False)
        with patch("backend.routers.dashboard.get_spaced_repetition_model") as mock_sr:
            mock_sr.return_value.get_review_candidates.return_value = []
            resp = self.client.get("/api/dashboard/summary?user_id=u1")
        self.assertEqual(resp.json()["today_completed"], 2)

    def test_summary_streak_days(self):
        for d in range(3):
            _seed_answer_history(self.db_path, "u1", days_ago=d, is_correct=True)
        with patch("backend.routers.dashboard.get_spaced_repetition_model") as mock_sr:
            mock_sr.return_value.get_review_candidates.return_value = []
            resp = self.client.get("/api/dashboard/summary?user_id=u1")
        self.assertGreaterEqual(resp.json()["streak_days"], 3)

    def test_summary_reviews_due(self):
        fake_reviews = [
            {"question_id": "q1", "recall_probability": 0.1, "half_life": 1.0, "elapsed_days": 5.0},
            {"question_id": "q2", "recall_probability": 0.3, "half_life": 2.0, "elapsed_days": 3.0},
        ]
        with patch("backend.routers.dashboard.get_spaced_repetition_model") as mock_sr:
            mock_sr.return_value.get_review_candidates.return_value = fake_reviews
            resp = self.client.get("/api/dashboard/summary?user_id=u1")
        self.assertEqual(resp.json()["reviews_due"], 2)

    def test_summary_weak_skills_structure(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute(
            "INSERT INTO answer_history (user_id, question_id, skill_ids, is_correct, theta_at_time) VALUES (?,?,?,0,0.0)",
            ("u1", "qa", json.dumps(["Weaken"]))
        )
        conn.commit()
        conn.close()
        with patch("backend.routers.dashboard.get_spaced_repetition_model") as mock_sr:
            mock_sr.return_value.get_review_candidates.return_value = []
            resp = self.client.get("/api/dashboard/summary?user_id=u1")
        data = resp.json()
        self.assertIsInstance(data["weak_skills"], list)
        if data["weak_skills"]:
            skill = data["weak_skills"][0]
            self.assertIn("skill_name", skill)
            self.assertIn("error_rate", skill)
            self.assertIn("mastery", skill)


# ===========================================================================
# TestEmailService
# ===========================================================================

class TestEmailService(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        from backend.services.email_service import EmailReminderService
        self.service = EmailReminderService(
            smtp_host="smtp.example.com",
            smtp_port=587,
            sender_email="noreply@example.com",
            sender_password="secret",
            db_path=self.db_path,
        )

    def tearDown(self):
        os.unlink(self.db_path)

    def test_service_configured(self):
        self.assertTrue(self.service._configured)

    def test_service_unconfigured(self):
        from backend.services.email_service import EmailReminderService
        svc = EmailReminderService("", 587, "", "", db_path=self.db_path)
        self.assertFalse(svc._configured)

    def test_html_template_contains_branding(self):
        from backend.services.email_service import _HTML_TEMPLATE
        self.assertIn("GlitchMind", _HTML_TEMPLATE)
        self.assertIn("losing signal", _HTML_TEMPLATE)
        self.assertIn("{due_count}", _HTML_TEMPLATE)
        self.assertIn("{user_name}", _HTML_TEMPLATE)

    def test_send_review_reminder_unconfigured_returns_false(self):
        from backend.services.email_service import EmailReminderService
        svc = EmailReminderService("", 587, "", "", db_path=self.db_path)
        result = svc.send_review_reminder("a@b.com", "Alice", 3, [])
        self.assertFalse(result)

    def test_send_review_reminder_smtp_called(self):
        questions = [
            {"question_id": "q1", "recall_probability": 0.2, "half_life": 1.0, "elapsed_days": 4.0},
        ]
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_server = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = self.service.send_review_reminder(
                "student@example.com", "Bob", 5, questions
            )
        # SMTP constructor was called (connection attempted)
        mock_smtp_cls.assert_called_once()

    def test_email_log_insert_and_retrieve(self):
        from utils.db_handler import DatabaseManager
        db = DatabaseManager(db_path=self.db_path)
        db.insert_email_log("u1", "review_reminder")
        last = db.get_last_reminder_time("u1")
        self.assertIsNotNone(last)

    def test_email_log_get_last_none(self):
        from utils.db_handler import DatabaseManager
        db = DatabaseManager(db_path=self.db_path)
        last = db.get_last_reminder_time("nobody")
        self.assertIsNone(last)

    def test_check_and_send_unconfigured_returns_zero(self):
        from backend.services.email_service import EmailReminderService
        svc = EmailReminderService("", 587, "", "", db_path=self.db_path)
        count = svc.check_and_send_reminders()
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
