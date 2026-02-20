"""
用户学习仪表盘 API
汇总今日进度、连续学习天数、能力值、薄弱技能和复习待办
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from engine.scoring import estimate_gmat_score
from engine.spaced_repetition import get_spaced_repetition_model
from utils.db_handler import DatabaseManager

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "logicmaster.db")


def _get_db() -> DatabaseManager:
    return DatabaseManager(db_path=_DB_PATH)


# ---------- 响应模型 ----------

class WeakSkill(BaseModel):
    skill_name: str
    error_rate: float
    mastery: float


class DashboardSummary(BaseModel):
    today_goal: int
    today_completed: int
    streak_days: int
    current_theta: float
    gmat_score: int
    accuracy_pct: float
    total_questions: int
    weak_skills: List[WeakSkill]
    reviews_due: int
    last_practiced: Optional[str]
    last_7_days: List[bool]


# ---------- 端点 ----------

@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(user_id: str = "default"):
    """
    获取用户学习仪表盘汇总数据。

    - today_completed: 今日答题数（answer_history）
    - streak_days: 连续答题天数
    - current_theta: 最近一次答题的能力值
    - gmat_score: 对应 GMAT 估分
    - weak_skills: 错误率最高的 3 个技能
    - reviews_due: 回忆概率 < 0.5 的复习题数
    - last_practiced: 最后一次答题时间
    """
    db = _get_db()

    # 今日目标（从 learning_goals 取，若无则用默认值 5）
    goal = db.get_learning_goal(user_id)
    today_goal: int = goal.get("daily_question_goal", 5)

    # 今日完成数
    today_completed: int = db.count_today_answers(user_id)

    # 连续学习天数
    streak_days: int = db.calculate_streak(user_id)

    # 当前 theta / GMAT 估分
    theta: float = db.get_latest_theta(user_id) or 0.0
    gmat_score: int = estimate_gmat_score(theta)

    # 总体统计（accuracy_pct / total_questions）
    user_stats = db.get_user_stats(user_id)
    accuracy_pct: float = user_stats.get("accuracy_pct", 0.0)
    total_questions: int = user_stats.get("total_questions", 0)

    # 薄弱技能（DKT 错误率）
    raw_skills = db.get_skill_error_rates(user_id, limit=3)
    weak_skills = [WeakSkill(**s) for s in raw_skills]

    # 复习待办数（Half-Life Regression，recall < 0.5）
    reviews_due: int = 0
    try:
        sr_model = get_spaced_repetition_model(user_id=user_id)
        candidates = sr_model.get_review_candidates(threshold=0.5)
        reviews_due = len(candidates)
    except Exception:
        pass

    # 最后答题时间
    last_practiced: Optional[str] = db.get_last_practiced_time(user_id)

    # 最近 7 天练习情况
    last_7_days: List[bool] = db.get_last_7_days(user_id)

    return DashboardSummary(
        today_goal=today_goal,
        today_completed=today_completed,
        streak_days=streak_days,
        current_theta=round(theta, 4),
        gmat_score=gmat_score,
        accuracy_pct=accuracy_pct,
        total_questions=total_questions,
        weak_skills=weak_skills,
        reviews_due=reviews_due,
        last_practiced=last_practiced,
        last_7_days=last_7_days,
    )
