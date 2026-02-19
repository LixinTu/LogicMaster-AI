"""
收藏 & 错题本 API
支持收藏题目、标记错题、查询错题本、统计分析
"""

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from utils.db_handler import DatabaseManager

router = APIRouter(prefix="/api/bookmarks", tags=["bookmarks"])

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "logicmaster.db")


def _get_db() -> DatabaseManager:
    return DatabaseManager(db_path=_DB_PATH)


# ---------- 请求/响应模型 ----------

class BookmarkRequest(BaseModel):
    user_id: str = Field("default", description="用户标识")
    question_id: str = Field(..., description="题目 ID")
    bookmark_type: str = Field(..., description='"favorite" 或 "wrong"')


class BookmarkItem(BaseModel):
    question_id: str
    question_type: str
    difficulty: str
    stimulus_preview: str
    skills: List[str]
    bookmark_type: str
    created_at: Optional[str]


class SkillStat(BaseModel):
    skill_name: str
    count: int


class TypeStat(BaseModel):
    question_type: str
    count: int


class WrongStatsResponse(BaseModel):
    total_wrong: int
    by_skill: List[SkillStat]
    by_type: List[TypeStat]


# ---------- 端点 ----------

_VALID_TYPES = {"favorite", "wrong"}


@router.post("/add", status_code=status.HTTP_201_CREATED)
def add_bookmark(req: BookmarkRequest):
    """
    添加书签（收藏或错题）。重复添加幂等，不报错。
    """
    if req.bookmark_type not in _VALID_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"bookmark_type 必须是 'favorite' 或 'wrong'，收到：{req.bookmark_type}",
        )
    db = _get_db()
    db.insert_bookmark(
        user_id=req.user_id,
        question_id=req.question_id,
        bookmark_type=req.bookmark_type,
    )
    return {"status": "ok", "question_id": req.question_id, "bookmark_type": req.bookmark_type}


@router.delete("/remove")
def remove_bookmark(req: BookmarkRequest):
    """
    删除书签。不存在时静默成功。
    """
    if req.bookmark_type not in _VALID_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"bookmark_type 必须是 'favorite' 或 'wrong'",
        )
    db = _get_db()
    db.remove_bookmark(
        user_id=req.user_id,
        question_id=req.question_id,
        bookmark_type=req.bookmark_type,
    )
    return {"status": "ok", "question_id": req.question_id}


@router.get("/list", response_model=List[BookmarkItem])
def list_bookmarks(
    user_id: str = "default",
    type: Optional[str] = None,
    skill: Optional[str] = None,
):
    """
    查询书签列表。

    - type: 过滤 "favorite" 或 "wrong"（不传则全部）
    - skill: 按技能名过滤（如 "Causal Reasoning"）
    """
    if type and type not in _VALID_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"type 必须是 'favorite' 或 'wrong'",
        )
    db = _get_db()
    items = db.query_bookmarks(user_id=user_id, bookmark_type=type, skill_filter=skill)
    return [BookmarkItem(**item) for item in items]


@router.get("/wrong-stats", response_model=WrongStatsResponse)
def get_wrong_stats(user_id: str = "default"):
    """
    获取错题本统计：总数、按技能分布、按题型分布。
    """
    db = _get_db()
    stats = db.get_wrong_stats(user_id=user_id)
    return WrongStatsResponse(
        total_wrong=stats["total_wrong"],
        by_skill=[SkillStat(**s) for s in stats["by_skill"]],
        by_type=[TypeStat(**t) for t in stats["by_type"]],
    )
