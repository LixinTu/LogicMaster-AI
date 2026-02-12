"""
Multi-Armed Bandit 题目选择器（Thompson Sampling）

在 explore（不确定性高的题目）和 exploit（信息量大的题目）之间平衡，
为每个学生选择最优下一题。

每道题维护 Beta 分布参数 (alpha, beta)：
- alpha: 学生答对该题的累计次数 + 1
- beta:  学生答错该题的累计次数 + 1
- 初始先验 Beta(1, 1) = 均匀分布
"""

import os
import random
import sqlite3
from typing import Any, Dict, List, Optional

from engine.scoring import item_information


# ---------------------------------------------------------------------------
# Bandit 统计存储（SQLite bandit_stats 表）
# ---------------------------------------------------------------------------

def _get_default_db_path() -> str:
    """项目根目录下的 logicmaster.db"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "logicmaster.db")


def _ensure_bandit_table(db_path: str) -> None:
    """创建 bandit_stats 表（如果不存在）"""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bandit_stats (
            question_id TEXT PRIMARY KEY,
            alpha REAL NOT NULL DEFAULT 1.0,
            beta  REAL NOT NULL DEFAULT 1.0
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# BanditQuestionSelector
# ---------------------------------------------------------------------------

class BanditQuestionSelector:
    """
    Thompson Sampling 题目选择器。

    对每个候选题目计算两个分数并加权合并：
    - exploit_score: item_information(θ, b, a, c) — 该题对当前学生的信息量
    - explore_score: Beta(α, β) 随机采样 — 不确定性奖励
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _get_default_db_path()
        _ensure_bandit_table(self.db_path)

    # ------ 核心方法 ------

    def select_question(
        self,
        theta: float,
        candidates: List[Dict[str, Any]],
        explore_weight: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """
        从候选题目中选择最优题目。

        combined_score = (1 - explore_weight) * exploit_score + explore_weight * explore_score

        Args:
            theta: 学生当前能力值
            candidates: 候选题目字典列表（需包含 id/elo_difficulty 等字段）
            explore_weight: 探索权重（0.0 = 纯 exploit，1.0 = 纯 explore）

        Returns:
            选中的题目字典，或 None（无候选时）
        """
        if not candidates:
            return None

        stats = self._load_stats_batch([c.get("id", "") for c in candidates])

        best_candidate = None
        best_score = -float("inf")

        for candidate in candidates:
            q_id = candidate.get("id", "")

            # --- exploit: 3PL 信息函数 ---
            elo = candidate.get("elo_difficulty", 1500.0)
            b = (elo - 1500.0) / 100.0
            a = candidate.get("discrimination", 1.0)
            c = candidate.get("guessing", 0.2)
            exploit_score = item_information(theta, b, a, c)

            # --- explore: Thompson 采样 ---
            alpha, beta_val = stats.get(q_id, (1.0, 1.0))
            explore_score = random.betavariate(max(alpha, 0.01), max(beta_val, 0.01))

            combined = (1.0 - explore_weight) * exploit_score + explore_weight * explore_score

            if combined > best_score:
                best_score = combined
                best_candidate = candidate

        return best_candidate

    def update(self, question_id: str, is_correct: bool) -> None:
        """
        更新题目的 bandit 统计。

        Args:
            question_id: 题目 ID
            is_correct: 学生是否答对
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()

        # UPSERT: 如果行不存在则插入默认值
        cursor.execute(
            "INSERT OR IGNORE INTO bandit_stats (question_id, alpha, beta) VALUES (?, 1.0, 1.0)",
            (question_id,),
        )
        if is_correct:
            cursor.execute(
                "UPDATE bandit_stats SET alpha = alpha + 1 WHERE question_id = ?",
                (question_id,),
            )
        else:
            cursor.execute(
                "UPDATE bandit_stats SET beta = beta + 1 WHERE question_id = ?",
                (question_id,),
            )

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        返回所有题目的 bandit 统计。

        Returns:
            {question_id: {"alpha": ..., "beta": ..., "expected_value": ..., "uncertainty": ...}}
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT question_id, alpha, beta FROM bandit_stats")
        rows = cursor.fetchall()
        conn.close()

        result: Dict[str, Dict[str, float]] = {}
        for q_id, alpha, beta_val in rows:
            total = alpha + beta_val
            result[q_id] = {
                "alpha": alpha,
                "beta": beta_val,
                "expected_value": alpha / total if total > 0 else 0.5,
                "uncertainty": (alpha * beta_val) / (total ** 2 * (total + 1)) if total > 0 else 0.25,
            }
        return result

    # ------ 内部方法 ------

    def _load_stats_batch(self, question_ids: List[str]) -> Dict[str, tuple]:
        """批量读取 bandit 统计，返回 {question_id: (alpha, beta)}"""
        if not question_ids:
            return {}
        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in question_ids)
        cursor.execute(
            f"SELECT question_id, alpha, beta FROM bandit_stats WHERE question_id IN ({placeholders})",
            question_ids,
        )
        stats = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        conn.close()
        return stats


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_selector: Optional[BanditQuestionSelector] = None


def get_bandit_selector(db_path: Optional[str] = None) -> BanditQuestionSelector:
    """获取全局 BanditQuestionSelector 实例"""
    global _selector
    if _selector is None:
        _selector = BanditQuestionSelector(db_path=db_path)
    return _selector
