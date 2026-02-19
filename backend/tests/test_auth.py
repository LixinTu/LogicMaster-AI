"""
认证 + 资料管理端点测试
Covers: register, login, /me, update_profile, change_password, delete_account, stats
+ DB method unit tests: insert/get user, update, delete cascade, get_user_stats
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from utils.db_handler import DatabaseManager
from backend.services.auth_service import hash_password


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_db() -> str:
    """Create a temp SQLite DB with all tables, return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    DatabaseManager(db_path=tmp.name).init_db()
    return tmp.name


def _seed_user(
    db_path: str,
    user_id: str = "uid-test-1",
    email: str = "test@example.com",
    password: str = "secret123",
    display_name: str = "Tester",
) -> dict:
    """Insert a user, return their info dict."""
    DatabaseManager(db_path=db_path).insert_user(
        user_id=user_id,
        email=email,
        password_hash=hash_password(password),
        display_name=display_name,
    )
    return {"user_id": user_id, "email": email, "password": password, "display_name": display_name}


def _seed_answers(db_path: str, user_id: str, records: list) -> None:
    """
    Bulk-insert answer_history rows.
    records: list of (days_ago: int, is_correct: bool, theta: float)
    """
    conn = sqlite3.connect(db_path, timeout=10)
    for days_ago, is_correct, theta in records:
        ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        conn.execute(
            """INSERT INTO answer_history
               (user_id, question_id, skill_ids, is_correct, theta_at_time, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, f"q_{days_ago}", json.dumps(["Weaken"]), 1 if is_correct else 0, theta, ts),
        )
    conn.commit()
    conn.close()


# ===========================================================================
# TestAuthDB — unit tests for DatabaseManager user methods
# ===========================================================================

class TestAuthDB(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    # ---- insert / get ----

    def test_insert_and_get_by_email_returns_hash(self):
        self.db.insert_user("u1", "alice@b.com", hash_password("pass123"), "Alice")
        user = self.db.get_user_by_email("alice@b.com")
        self.assertIsNotNone(user)
        self.assertEqual(user["email"], "alice@b.com")
        self.assertIn("password_hash", user)   # email query must include hash

    def test_get_user_by_id_excludes_password_hash(self):
        self.db.insert_user("u1", "alice@b.com", hash_password("pass123"), "Alice")
        user = self.db.get_user_by_id("u1")
        self.assertIsNotNone(user)
        self.assertEqual(user["id"], "u1")
        self.assertNotIn("password_hash", user)

    def test_insert_duplicate_email_returns_false(self):
        self.db.insert_user("u1", "dup@b.com", hash_password("pass123"))
        result = self.db.insert_user("u2", "dup@b.com", hash_password("other"))
        self.assertFalse(result)

    def test_display_name_optional_defaults_to_none(self):
        self.db.insert_user("u1", "anon@b.com", hash_password("pass123"))
        user = self.db.get_user_by_id("u1")
        self.assertIsNone(user.get("display_name"))

    def test_email_stored_lowercase(self):
        self.db.insert_user("u1", "Upper@B.com", hash_password("pass123"))
        user = self.db.get_user_by_email("upper@b.com")
        self.assertIsNotNone(user)

    # ---- update ----

    def test_update_display_name_succeeds(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"), "Old")
        ok = self.db.update_user_display_name("u1", "New Name")
        self.assertTrue(ok)
        self.assertEqual(self.db.get_user_by_id("u1")["display_name"], "New Name")

    def test_update_password_persists_new_hash(self):
        old_hash = hash_password("oldpass")
        self.db.insert_user("u1", "a@b.com", old_hash)
        new_hash = hash_password("newpass456")
        ok = self.db.update_user_password("u1", new_hash)
        self.assertTrue(ok)
        stored = self.db.get_user_by_email("a@b.com")["password_hash"]
        self.assertEqual(stored, new_hash)

    # ---- delete cascade ----

    def test_delete_removes_user_row(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        self.db.delete_user_and_data("u1")
        self.assertIsNone(self.db.get_user_by_id("u1"))

    def test_delete_cascades_answer_history(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        _seed_answers(self.db_path, "u1", [(0, True, 0.5), (1, False, 0.3)])
        self.db.delete_user_and_data("u1")
        self.assertEqual(len(self.db.query_answer_history(user_id="u1")), 0)

    def test_delete_cascades_bookmarks(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        self.db.insert_bookmark("u1", "q001", "wrong")
        self.db.delete_user_and_data("u1")
        self.assertEqual(len(self.db.query_bookmarks("u1")), 0)

    def test_delete_cascades_learning_goals(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        self.db.upsert_learning_goal("u1", 45, 10)
        self.db.delete_user_and_data("u1")
        # After delete the goal row should be gone; default values returned
        goal = self.db.get_learning_goal("u1")
        self.assertEqual(goal["daily_question_goal"], 5)   # back to defaults

    def test_delete_nonexistent_user_returns_true(self):
        # Deleting a non-existent user should not raise; just 0 rows affected
        ok = self.db.delete_user_and_data("ghost_user")
        self.assertTrue(ok)

    # ---- get_user_stats ----

    def test_get_stats_empty_user_all_zeros(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        stats = self.db.get_user_stats("u1")
        self.assertEqual(stats["total_questions"], 0)
        self.assertEqual(stats["total_correct"], 0)
        self.assertEqual(stats["accuracy_pct"], 0.0)
        self.assertEqual(stats["best_streak"], 0)
        self.assertIsNotNone(stats["member_since"])
        self.assertIsNone(stats["current_theta"])
        self.assertIsNone(stats["favorite_question_type"])

    def test_get_stats_accuracy_calculation(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        _seed_answers(self.db_path, "u1", [
            (0, True, 0.5), (1, True, 0.6),
            (2, False, 0.4), (3, False, 0.3),
        ])
        stats = self.db.get_user_stats("u1")
        self.assertEqual(stats["total_questions"], 4)
        self.assertEqual(stats["total_correct"], 2)
        self.assertAlmostEqual(stats["accuracy_pct"], 50.0, places=0)

    def test_get_stats_best_streak_consecutive_days(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        # 4 consecutive days
        _seed_answers(self.db_path, "u1", [
            (0, True, 0.5), (1, True, 0.5), (2, True, 0.5), (3, True, 0.5),
        ])
        stats = self.db.get_user_stats("u1")
        self.assertGreaterEqual(stats["best_streak"], 4)

    def test_get_stats_best_streak_broken_chain(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        # days 0, 1 → streak 2; then gap; day 5 alone
        _seed_answers(self.db_path, "u1", [
            (0, True, 0.5), (1, True, 0.5), (5, True, 0.5),
        ])
        stats = self.db.get_user_stats("u1")
        self.assertEqual(stats["best_streak"], 2)

    def test_get_stats_current_theta_is_latest(self):
        self.db.insert_user("u1", "a@b.com", hash_password("pass123"))
        # Older record has higher theta — but newest (days_ago=0) wins
        _seed_answers(self.db_path, "u1", [(2, True, 2.5), (0, True, 1.0)])
        stats = self.db.get_user_stats("u1")
        self.assertAlmostEqual(stats["current_theta"], 1.0, places=1)


# ===========================================================================
# TestAuthAPI — HTTP endpoint integration tests
# ===========================================================================

class TestAuthAPI(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()
        import backend.routers.auth as _auth_mod
        self._orig_get_db = _auth_mod._get_db
        _auth_mod._get_db = lambda: DatabaseManager(db_path=self.db_path)
        from backend.main import app
        self.client = TestClient(app)

    def tearDown(self):
        import backend.routers.auth as _auth_mod
        _auth_mod._get_db = self._orig_get_db
        os.unlink(self.db_path)

    # ---- tiny helpers ----

    def _register(self, email="user@test.com", password="secret123", display_name="Tester"):
        return self.client.post("/api/auth/register", json={
            "email": email, "password": password, "display_name": display_name,
        })

    def _login(self, email="user@test.com", password="secret123"):
        return self.client.post("/api/auth/login", json={"email": email, "password": password})

    def _hdrs(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    # ==== POST /api/auth/register ====

    def test_register_returns_201_with_token(self):
        resp = self._register()
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("token", data)
        self.assertIn("user_id", data)
        self.assertEqual(data["email"], "user@test.com")
        self.assertEqual(data["display_name"], "Tester")

    def test_register_token_is_non_empty_string(self):
        token = self._register().json()["token"]
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 20)

    def test_register_duplicate_email_409(self):
        self._register()
        resp = self._register()
        self.assertEqual(resp.status_code, 409)

    def test_register_short_password_422(self):
        resp = self.client.post("/api/auth/register", json={
            "email": "x@y.com", "password": "abc",   # < 6 chars
        })
        self.assertEqual(resp.status_code, 422)

    def test_register_without_display_name_returns_none(self):
        resp = self.client.post("/api/auth/register", json={
            "email": "anon@test.com", "password": "pass123",
        })
        self.assertEqual(resp.status_code, 201)
        self.assertIsNone(resp.json()["display_name"])

    # ==== POST /api/auth/login ====

    def test_login_success_returns_token(self):
        self._register()
        resp = self._login()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("token", data)
        self.assertIn("user_id", data)

    def test_login_wrong_password_401(self):
        self._register()
        resp = self._login(password="wrongpass")
        self.assertEqual(resp.status_code, 401)

    def test_login_unknown_email_401(self):
        resp = self._login(email="nobody@test.com")
        self.assertEqual(resp.status_code, 401)

    def test_login_case_insensitive_email(self):
        self._register(email="lower@test.com")
        resp = self._login(email="Lower@test.com")
        self.assertEqual(resp.status_code, 200)

    # ==== GET /api/auth/me ====

    def test_get_me_with_valid_token(self):
        token = self._register().json()["token"]
        resp = self.client.get("/api/auth/me", headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["email"], "user@test.com")
        self.assertEqual(data["display_name"], "Tester")
        self.assertIn("user_id", data)
        self.assertIn("created_at", data)

    def test_get_me_no_token_401(self):
        resp = self.client.get("/api/auth/me")
        self.assertEqual(resp.status_code, 401)

    def test_get_me_bad_token_401(self):
        resp = self.client.get("/api/auth/me",
                               headers={"Authorization": "Bearer bad.token.here"})
        self.assertEqual(resp.status_code, 401)

    # ==== PUT /api/auth/profile ====

    def test_update_profile_success(self):
        token = self._register().json()["token"]
        resp = self.client.put("/api/auth/profile",
                               json={"display_name": "Updated Name"},
                               headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["display_name"], "Updated Name")
        self.assertEqual(data["message"], "Profile updated")
        self.assertIn("user_id", data)
        self.assertIn("email", data)

    def test_update_profile_reflected_in_get_me(self):
        token = self._register().json()["token"]
        self.client.put("/api/auth/profile",
                        json={"display_name": "Reflected Name"},
                        headers=self._hdrs(token))
        me = self.client.get("/api/auth/me", headers=self._hdrs(token)).json()
        self.assertEqual(me["display_name"], "Reflected Name")

    def test_update_profile_no_token_401(self):
        resp = self.client.put("/api/auth/profile", json={"display_name": "X"})
        self.assertEqual(resp.status_code, 401)

    # ==== PUT /api/auth/change-password ====

    def test_change_password_returns_success_message(self):
        token = self._register().json()["token"]
        resp = self.client.put("/api/auth/change-password",
                               json={"current_password": "secret123", "new_password": "newpass456"},
                               headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["message"], "Password updated successfully")

    def test_change_password_new_password_works_on_login(self):
        self._register()
        token = self._login().json()["token"]
        self.client.put("/api/auth/change-password",
                        json={"current_password": "secret123", "new_password": "newpass456"},
                        headers=self._hdrs(token))
        # old password rejected
        self.assertEqual(self._login(password="secret123").status_code, 401)
        # new password accepted
        self.assertEqual(self._login(password="newpass456").status_code, 200)

    def test_change_password_wrong_current_401(self):
        token = self._register().json()["token"]
        resp = self.client.put("/api/auth/change-password",
                               json={"current_password": "wrongpass", "new_password": "newpass456"},
                               headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 401)
        self.assertIn("incorrect", resp.json()["detail"])

    def test_change_password_short_new_password_422(self):
        token = self._register().json()["token"]
        resp = self.client.put("/api/auth/change-password",
                               json={"current_password": "secret123", "new_password": "abc"},
                               headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 422)

    def test_change_password_no_token_401(self):
        resp = self.client.put("/api/auth/change-password",
                               json={"current_password": "secret123", "new_password": "newpass456"})
        self.assertEqual(resp.status_code, 401)

    # ==== DELETE /api/auth/account ====

    def test_delete_account_returns_200(self):
        token = self._register().json()["token"]
        resp = self.client.delete("/api/auth/account", headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["message"], "Account deleted")

    def test_delete_account_user_absent_from_db(self):
        reg = self._register().json()
        token, user_id = reg["token"], reg["user_id"]
        self.client.delete("/api/auth/account", headers=self._hdrs(token))
        db = DatabaseManager(db_path=self.db_path)
        self.assertIsNone(db.get_user_by_id(user_id))

    def test_delete_account_no_token_401(self):
        resp = self.client.delete("/api/auth/account")
        self.assertEqual(resp.status_code, 401)

    # ==== GET /api/auth/stats ====

    def test_get_stats_fresh_user_all_zeros(self):
        token = self._register().json()["token"]
        resp = self.client.get("/api/auth/stats", headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_questions"], 0)
        self.assertEqual(data["total_correct"], 0)
        self.assertEqual(data["accuracy_pct"], 0.0)
        self.assertEqual(data["best_streak"], 0)
        self.assertEqual(data["current_theta"], 0.0)
        self.assertIn("current_gmat_score", data)
        self.assertIn("member_since", data)
        self.assertIsNone(data["favorite_question_type"])

    def test_get_stats_with_answer_history(self):
        reg = self._register().json()
        token, user_id = reg["token"], reg["user_id"]
        # 3 correct + 1 wrong across 4 consecutive days
        _seed_answers(self.db_path, user_id, [
            (0, True, 1.0), (1, True, 0.8),
            (2, True, 0.6), (3, False, 0.4),
        ])
        resp = self.client.get("/api/auth/stats", headers=self._hdrs(token))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_questions"], 4)
        self.assertEqual(data["total_correct"], 3)
        self.assertAlmostEqual(data["accuracy_pct"], 75.0, places=0)
        self.assertGreaterEqual(data["best_streak"], 3)
        self.assertAlmostEqual(data["current_theta"], 1.0, places=1)
        self.assertIsNotNone(data["member_since"])
        self.assertGreaterEqual(data["current_gmat_score"], 20)
        self.assertLessEqual(data["current_gmat_score"], 51)

    def test_get_stats_gmat_score_in_valid_range(self):
        token = self._register().json()["token"]
        data = self.client.get("/api/auth/stats", headers=self._hdrs(token)).json()
        self.assertGreaterEqual(data["current_gmat_score"], 20)
        self.assertLessEqual(data["current_gmat_score"], 51)

    def test_get_stats_no_token_401(self):
        resp = self.client.get("/api/auth/stats")
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
