"""
题目推荐 API
复用 engine/recommender.py 的 generate_next_question（IRT + BKT 混合推荐）
"""

import io
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.recommender import generate_next_question
from engine.bandit_selector import get_bandit_selector
from engine.spaced_repetition import get_spaced_repetition_model
from utils.db_handler import DatabaseManager

router = APIRouter(prefix="/api/questions", tags=["questions"])

# 数据库路径：项目根目录下的 logicmaster.db
# __file__ 在 backend/routers/ 下，需要上溯两级到项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "logicmaster.db")
_db_manager = DatabaseManager(db_path=_DB_PATH)


# ---------- Mock session_state ----------

class _MockSessionState:
    """
    轻量级 mock 对象，替代 Streamlit session_state。
    generate_next_question 内部会写入这些属性，API 层不需要它们。
    """
    def __init__(self):
        self.radio_key = 0
        self.attempt = 0
        self.phase = ""
        self.last_feedback = ""
        self.show_explanation = False
        self.pending_next_question = False
        self.socratic_context = {}
        self.chat_history = []
        self.current_q = None
        self.current_q_id = None
        self.current_question = None


# ---------- 请求/响应模型 ----------

class QuestionLogItem(BaseModel):
    question_id: str = ""
    skills: List[str] = []
    is_correct: bool = False


class NextQuestionRequest(BaseModel):
    user_theta: float = Field(..., description="用户当前能力值", ge=-3.0, le=3.0)
    current_q_id: str = Field("", description="当前题目 ID（排除推荐）")
    questions_log: List[QuestionLogItem] = Field(default_factory=list, description="历史作答记录")
    strategy: str = Field("bandit", description="选题策略：bandit（Thompson Sampling）或 legacy（加权排序）")


class NextQuestionResponse(BaseModel):
    question_id: str
    question_type: str
    difficulty: str
    elo_difficulty: float
    stimulus: str
    question: str
    choices: List[str]
    skills: List[str] = []
    # 注意：不返回 correct 字段，防止前端作弊


# ---------- 端点 ----------

@router.post("/next", response_model=NextQuestionResponse)
def get_next_question(req: NextQuestionRequest):
    """
    获取下一道自适应推荐题目。
    直接调用 engine.recommender.generate_next_question，用 mock session_state 解耦 Streamlit 依赖。
    """
    mock_state = _MockSessionState()

    # 将 Pydantic 模型转为 dict 列表（generate_next_question 期望的格式）
    log_dicts = [item.model_dump() for item in req.questions_log]

    # engine/recommender.py 中的 print 含 emoji，在 Windows GBK 环境下会抛 UnicodeEncodeError
    # 临时将 stdout/stderr 重定向到 UTF-8 buffer 来绕过
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
        use_bandit = req.strategy != "legacy"
        result = generate_next_question(
            user_theta=req.user_theta,
            current_q_id=req.current_q_id,
            questions_log=log_dicts,
            session_state=mock_state,
            history_limit=10,
            db_manager=_db_manager,
            use_bandit=use_bandit,
        )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    if result is None:
        raise HTTPException(status_code=404, detail="没有找到合适的题目")

    return NextQuestionResponse(
        question_id=result.get("question_id", ""),
        question_type=result.get("question_type", ""),
        difficulty=result.get("difficulty", ""),
        elo_difficulty=result.get("elo_difficulty", 1500.0),
        stimulus=result.get("stimulus", ""),
        question=result.get("question", ""),
        choices=result.get("choices", []),
        skills=result.get("skills", []),
    )


# ---------- Bandit 更新端点 ----------

class BanditUpdateRequest(BaseModel):
    question_id: str = Field(..., description="题目 ID")
    is_correct: bool = Field(..., description="是否答对")
    skills: List[str] = Field(default_factory=list, description="涉及的技能列表（DKT 用）")
    theta_at_time: Optional[float] = Field(None, description="答题时的能力值")
    user_id: str = Field("default", description="用户标识")


class BanditUpdateResponse(BaseModel):
    status: str
    question_id: str


@router.post("/bandit-update", response_model=BanditUpdateResponse)
def update_bandit_stats(req: BanditUpdateRequest):
    """
    更新 bandit 统计（答题后调用）。
    将答题结果反馈给 Thompson Sampling，更新 Beta 分布参数。
    同时更新间隔重复的 half-life。
    """
    bandit = get_bandit_selector()
    bandit.update(question_id=req.question_id, is_correct=req.is_correct)
    # 同步更新间隔重复统计
    try:
        sr_model = get_spaced_repetition_model()
        sr_model.update_half_life(question_id=req.question_id, is_correct=req.is_correct)
    except Exception:
        pass  # 间隔重复更新失败时静默降级
    # 记录答题历史（DKT 训练数据）
    try:
        if req.skills:
            _db_manager.insert_answer_history(
                question_id=req.question_id,
                skill_ids=req.skills,
                is_correct=req.is_correct,
                theta_at_time=req.theta_at_time,
                user_id=req.user_id,
            )
    except Exception:
        pass  # 答题历史记录失败时静默降级
    # 自动添加错题书签（答错时）
    if not req.is_correct:
        try:
            _db_manager.insert_bookmark(
                user_id=req.user_id,
                question_id=req.question_id,
                bookmark_type="wrong",
            )
        except Exception:
            pass  # 书签写入失败时静默降级
    return BanditUpdateResponse(status="ok", question_id=req.question_id)


# ---------- 间隔重复端点 ----------

class ReviewItem(BaseModel):
    question_id: str
    recall_probability: float
    half_life: float
    elapsed_days: float


class ReviewScheduleResponse(BaseModel):
    user_id: str
    threshold: float
    due_count: int
    reviews: List[ReviewItem]


@router.get("/review-schedule", response_model=ReviewScheduleResponse)
def get_review_schedule(user_id: str = "default", threshold: float = 0.5):
    """
    返回需要复习的题目列表（回忆概率低于阈值）。
    基于 Half-Life Regression 遗忘曲线计算。
    """
    sr_model = get_spaced_repetition_model(user_id=user_id)
    candidates = sr_model.get_review_candidates(threshold=threshold)
    reviews = [ReviewItem(**c) for c in candidates]
    return ReviewScheduleResponse(
        user_id=user_id,
        threshold=threshold,
        due_count=len(reviews),
        reviews=reviews,
    )
