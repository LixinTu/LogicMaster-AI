"""
间隔重复模型：Half-Life Regression (Settles & Meeder 2016, Duolingo)

基于 Ebbinghaus 遗忘曲线的间隔重复系统：
- 每个 (user, question) 对维护 half_life、last_practiced、n_correct、n_attempts
- recall_probability = 2^(-elapsed_days / half_life)
- 答对 → half_life *= 2.0（记住了 → 间隔加长）
- 答错 → half_life *= 0.5（忘了 → 间隔缩短）
- half_life 钳制在 [0.25, 90.0] 天
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# SQLite 表创建
# ---------------------------------------------------------------------------

def _get_default_db_path() -> str:
    """项目根目录下的 logicmaster.db"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "logicmaster.db")


def _ensure_sr_table(db_path: str) -> None:
    """创建 spaced_repetition_stats 表（如果不存在）"""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spaced_repetition_stats (
            user_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            half_life REAL NOT NULL DEFAULT 1.0,
            last_practiced TIMESTAMP NOT NULL,
            n_correct INTEGER NOT NULL DEFAULT 0,
            n_attempts INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, question_id)
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SpacedRepetitionModel
# ---------------------------------------------------------------------------

class SpacedRepetitionModel:
    """
    Half-Life Regression 间隔重复模型。

    跟踪每个 (user, question) 对的遗忘曲线参数，
    基于 recall_probability = 2^(-elapsed / half_life) 判断是否需要复习。
    """

    # half_life 的上下界（天）
    MIN_HALF_LIFE = 0.25
    MAX_HALF_LIFE = 90.0

    def __init__(self, db_path: Optional[str] = None, user_id: str = "default"):
        self.db_path = db_path or _get_default_db_path()
        self.user_id = user_id
        _ensure_sr_table(self.db_path)

    # ------ 核心方法 ------

    def recall_probability(
        self,
        question_id: str,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        计算某道题的当前回忆概率（Ebbinghaus 遗忘曲线）。

        P = 2^(-elapsed_days / half_life)

        Args:
            question_id: 题目 ID
            current_time: 当前时间（默认 UTC now）

        Returns:
            回忆概率 [0, 1]。若无记录返回 0.0（从未练过 → 需要学习）。
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        row = self._get_row(question_id)
        if row is None:
            return 0.0

        half_life, last_practiced_str, _, _ = row
        last_practiced = self._parse_timestamp(last_practiced_str)
        elapsed_days = (current_time - last_practiced).total_seconds() / 86400.0

        if elapsed_days <= 0:
            return 1.0

        return 2.0 ** (-elapsed_days / half_life)

    def update_half_life(
        self,
        question_id: str,
        is_correct: bool,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        根据答题结果更新 half_life。

        答对 → half_life *= 2.0（间隔加长）
        答错 → half_life *= 0.5（间隔缩短）
        钳制在 [MIN_HALF_LIFE, MAX_HALF_LIFE]

        Args:
            question_id: 题目 ID
            is_correct: 是否答对
            current_time: 当前时间（默认 UTC now）

        Returns:
            更新后的 half_life
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        ts_str = current_time.isoformat()
        row = self._get_row(question_id)

        if row is None:
            # 冷启动：新题目，默认 half_life=1.0
            half_life = 1.0
            n_correct = 1 if is_correct else 0
            n_attempts = 1
            if is_correct:
                half_life *= 2.0
            else:
                half_life *= 0.5
            half_life = max(self.MIN_HALF_LIFE, min(self.MAX_HALF_LIFE, half_life))

            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute(
                """INSERT INTO spaced_repetition_stats
                   (user_id, question_id, half_life, last_practiced, n_correct, n_attempts)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.user_id, question_id, half_life, ts_str, n_correct, n_attempts),
            )
            conn.commit()
            conn.close()
            return half_life

        old_half_life, _, n_correct, n_attempts = row
        if is_correct:
            new_half_life = old_half_life * 2.0
            n_correct += 1
        else:
            new_half_life = old_half_life * 0.5

        new_half_life = max(self.MIN_HALF_LIFE, min(self.MAX_HALF_LIFE, new_half_life))
        n_attempts += 1

        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute(
            """UPDATE spaced_repetition_stats
               SET half_life = ?, last_practiced = ?, n_correct = ?, n_attempts = ?
               WHERE user_id = ? AND question_id = ?""",
            (new_half_life, ts_str, n_correct, n_attempts, self.user_id, question_id),
        )
        conn.commit()
        conn.close()
        return new_half_life

    def get_review_candidates(
        self,
        current_time: Optional[datetime] = None,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        返回回忆概率低于阈值的题目（学生可能已忘记，需要复习）。

        Args:
            current_time: 当前时间（默认 UTC now）
            threshold: 回忆概率阈值（默认 0.5）

        Returns:
            需要复习的题目列表，每项包含 question_id 和 recall_probability
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question_id, half_life, last_practiced FROM spaced_repetition_stats WHERE user_id = ?",
            (self.user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        candidates = []
        for q_id, half_life, last_practiced_str in rows:
            last_practiced = self._parse_timestamp(last_practiced_str)
            elapsed_days = (current_time - last_practiced).total_seconds() / 86400.0
            if elapsed_days <= 0:
                continue
            recall_prob = 2.0 ** (-elapsed_days / half_life)
            if recall_prob < threshold:
                candidates.append({
                    "question_id": q_id,
                    "recall_probability": round(recall_prob, 4),
                    "half_life": half_life,
                    "elapsed_days": round(elapsed_days, 2),
                })

        # 按回忆概率升序排序（最容易忘记的排前面）
        candidates.sort(key=lambda x: x["recall_probability"])
        return candidates

    def get_all_stats(
        self,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        返回所有题目的间隔重复统计。

        Returns:
            {question_id: {half_life, last_practiced, recall_prob, n_correct, n_attempts}}
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question_id, half_life, last_practiced, n_correct, n_attempts "
            "FROM spaced_repetition_stats WHERE user_id = ?",
            (self.user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        result: Dict[str, Dict[str, Any]] = {}
        for q_id, half_life, last_practiced_str, n_correct, n_attempts in rows:
            last_practiced = self._parse_timestamp(last_practiced_str)
            elapsed_days = (current_time - last_practiced).total_seconds() / 86400.0
            recall_prob = 2.0 ** (-elapsed_days / half_life) if elapsed_days > 0 else 1.0
            result[q_id] = {
                "half_life": half_life,
                "last_practiced": last_practiced_str,
                "recall_prob": round(recall_prob, 4),
                "n_correct": n_correct,
                "n_attempts": n_attempts,
            }
        return result

    # ------ 内部方法 ------

    def _get_row(self, question_id: str) -> Optional[tuple]:
        """读取单条记录，返回 (half_life, last_practiced, n_correct, n_attempts) 或 None"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT half_life, last_practiced, n_correct, n_attempts "
            "FROM spaced_repetition_stats WHERE user_id = ? AND question_id = ?",
            (self.user_id, question_id),
        )
        row = cursor.fetchone()
        conn.close()
        return row

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        """解析 ISO 格式时间戳，兼容有无时区信息"""
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_model: Optional[SpacedRepetitionModel] = None


def get_spaced_repetition_model(
    db_path: Optional[str] = None,
    user_id: str = "default",
) -> SpacedRepetitionModel:
    """获取全局 SpacedRepetitionModel 实例"""
    global _model
    if _model is None or _model.user_id != user_id:
        _model = SpacedRepetitionModel(db_path=db_path, user_id=user_id)
    return _model
