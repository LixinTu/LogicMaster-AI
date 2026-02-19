"""
学习目标 API
设置目标 GMAT 分数和每日刷题量，跟踪进度
"""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from engine.scoring import estimate_gmat_score
from utils.db_handler import DatabaseManager

router = APIRouter(prefix="/api/goals", tags=["goals"])

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "logicmaster.db")


def _get_db() -> DatabaseManager:
    return DatabaseManager(db_path=_DB_PATH)


# ---------- 请求/响应模型 ----------

class SetGoalRequest(BaseModel):
    user_id: str = Field("default", description="用户标识")
    target_gmat_score: int = Field(..., ge=20, le=51, description="目标 GMAT 分数（20-51）")
    daily_question_goal: int = Field(..., ge=1, le=100, description="每日目标题数")


class GoalProgressResponse(BaseModel):
    target_gmat_score: int
    current_gmat_score: int
    score_gap: int
    estimated_questions_remaining: int
    daily_goal: int
    today_completed: int
    today_progress_pct: float
    on_track: bool


# ---------- 端点 ----------

@router.post("/set", status_code=status.HTTP_200_OK)
def set_goal(req: SetGoalRequest):
    """
    设置或更新用户学习目标（UPSERT）。
    """
    db = _get_db()
    success = db.upsert_learning_goal(
        user_id=req.user_id,
        target_gmat_score=req.target_gmat_score,
        daily_question_goal=req.daily_question_goal,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存学习目标失败",
        )
    return {
        "status": "ok",
        "user_id": req.user_id,
        "target_gmat_score": req.target_gmat_score,
        "daily_question_goal": req.daily_question_goal,
    }


@router.get("/progress", response_model=GoalProgressResponse)
def get_goal_progress(user_id: str = "default"):
    """
    获取用户学习目标进度。

    - score_gap: 目标分数与当前分数的差距
    - estimated_questions_remaining: 粗估剩余所需题数（score_gap × 10）
    - today_progress_pct: 今日进度百分比
    - on_track: 是否达成今日目标
    """
    db = _get_db()

    # 读取目标设置
    goal = db.get_learning_goal(user_id)
    target_gmat: int = goal.get("target_gmat_score", 40)
    daily_goal: int = goal.get("daily_question_goal", 5)

    # 当前 theta / GMAT 估分
    theta: float = db.get_latest_theta(user_id) or 0.0
    current_gmat: int = estimate_gmat_score(theta)

    # 分数差距（低于目标时为正值）
    score_gap: int = max(0, target_gmat - current_gmat)

    # 粗估剩余题数：每道题平均提升约 1 分（IRT theta 增益约 0.1，对应约 0.7 GMAT 分）
    # 保守估计：score_gap × 10 题
    estimated_remaining: int = score_gap * 10

    # 今日完成数
    today_completed: int = db.count_today_answers(user_id)

    # 今日进度百分比
    today_pct: float = min(100.0, (today_completed / daily_goal * 100)) if daily_goal > 0 else 0.0

    # 是否达成今日目标
    on_track: bool = today_completed >= daily_goal

    return GoalProgressResponse(
        target_gmat_score=target_gmat,
        current_gmat_score=current_gmat,
        score_gap=score_gap,
        estimated_questions_remaining=estimated_remaining,
        daily_goal=daily_goal,
        today_completed=today_completed,
        today_progress_pct=round(today_pct, 1),
        on_track=on_track,
    )
