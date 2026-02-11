"""
IRT theta 更新 API
直接复用 engine/scoring.py 中的 calculate_new_theta 和 estimate_gmat_score
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from engine.scoring import calculate_new_theta, estimate_gmat_score

router = APIRouter(prefix="/api/theta", tags=["theta"])


# ---------- 请求/响应模型 ----------

class ThetaUpdateRequest(BaseModel):
    current_theta: float = Field(..., description="当前能力值 theta", ge=-3.0, le=3.0)
    question_difficulty: float = Field(..., description="题目难度参数")
    is_correct: bool = Field(..., description="是否答对")


class ThetaUpdateResponse(BaseModel):
    new_theta: float
    gmat_score: int


# ---------- 端点 ----------

@router.post("/update", response_model=ThetaUpdateResponse)
def update_theta(req: ThetaUpdateRequest):
    """
    根据作答结果更新用户能力值 theta，并返回 GMAT 估分。
    直接调用 engine.scoring 中的现有函数。
    """
    new_theta = calculate_new_theta(
        current_theta=req.current_theta,
        question_difficulty=req.question_difficulty,
        is_correct=req.is_correct,
    )
    gmat_score = estimate_gmat_score(new_theta)

    return ThetaUpdateResponse(new_theta=new_theta, gmat_score=gmat_score)
