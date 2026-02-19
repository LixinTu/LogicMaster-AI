"""
用户认证 API：注册、登录、获取当前用户信息
"""

import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from backend.services.auth_service import (
    hash_password,
    verify_password,
    create_jwt_token,
    get_current_user,
)
from utils.db_handler import DatabaseManager

router = APIRouter(prefix="/api/auth", tags=["auth"])

# 使用与其他路由相同的数据库路径解析
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "logicmaster.db")


def _get_db() -> DatabaseManager:
    return DatabaseManager(db_path=_DB_PATH)


# ---------- 请求/响应模型 ----------

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="用户邮箱（唯一）")
    password: str = Field(..., min_length=6, description="密码（最少6位）")
    display_name: Optional[str] = Field(None, max_length=64, description="显示名称")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="用户邮箱")
    password: str = Field(..., description="密码")


class AuthResponse(BaseModel):
    user_id: str
    email: str
    display_name: Optional[str]
    token: str


class ProfileResponse(BaseModel):
    user_id: str
    email: str
    display_name: Optional[str]
    created_at: str


class UpdateProfileRequest(BaseModel):
    display_name: str = Field(..., max_length=64, description="新显示名称")


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., description="当前密码")
    new_password: str = Field(..., min_length=6, description="新密码（最少6位）")


class UserStatsResponse(BaseModel):
    total_questions: int
    total_correct: int
    accuracy_pct: float
    best_streak: int
    member_since: Optional[str]
    current_theta: float
    current_gmat_score: int
    favorite_question_type: Optional[str]


# ---------- 端点 ----------

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """
    注册新用户。
    - 邮箱唯一；重复注册返回 409
    - 密码使用 bcrypt 哈希存储
    - 注册成功后直接返回 JWT token（免除二次登录）
    """
    db = _get_db()

    # 检查邮箱是否已注册
    existing = db.get_user_by_email(req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"邮箱 {req.email} 已被注册",
        )

    user_id = str(uuid.uuid4())
    pw_hash = hash_password(req.password)

    success = db.insert_user(
        user_id=user_id,
        email=req.email,
        password_hash=pw_hash,
        display_name=req.display_name,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="注册失败，邮箱可能已存在",
        )

    token = create_jwt_token(user_id=user_id, email=req.email)
    return AuthResponse(
        user_id=user_id,
        email=req.email,
        display_name=req.display_name,
        token=token,
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """
    用户登录。
    - 验证邮箱 + 密码
    - 返回 JWT token（7 天有效）
    """
    db = _get_db()
    user = db.get_user_by_email(req.email)

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
        )

    token = create_jwt_token(user_id=user["id"], email=user["email"])
    return AuthResponse(
        user_id=user["id"],
        email=user["email"],
        display_name=user.get("display_name"),
        token=token,
    )


@router.get("/me", response_model=ProfileResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """
    获取当前登录用户的资料。
    - 需要 Authorization: Bearer <token> 请求头
    """
    db = _get_db()
    user = db.get_user_by_id(current_user["user_id"])

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )

    return ProfileResponse(
        user_id=user["id"],
        email=user["email"],
        display_name=user.get("display_name"),
        created_at=str(user.get("created_at", "")),
    )


@router.put("/profile")
def update_profile(
    req: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    更新当前用户的显示名称。
    - 需要 Authorization: Bearer <token> 请求头
    """
    db = _get_db()
    user_id: str = current_user["user_id"]
    success = db.update_user_display_name(user_id, req.display_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新失败，请稍后重试",
        )
    return {
        "user_id": user_id,
        "email": current_user["email"],
        "display_name": req.display_name,
        "message": "Profile updated",
    }


@router.put("/change-password")
def change_password(
    req: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    修改当前用户密码。
    - 需要先验证当前密码；不匹配则返回 401
    - 需要 Authorization: Bearer <token> 请求头
    """
    db = _get_db()
    # get_user_by_email 返回含 password_hash 的完整记录
    user = db.get_user_by_email(current_user["email"])
    if not user or not verify_password(req.current_password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )
    new_hash = hash_password(req.new_password)
    db.update_user_password(current_user["user_id"], new_hash)
    return {"message": "Password updated successfully"}


@router.delete("/account")
def delete_account(current_user: dict = Depends(get_current_user)):
    """
    永久删除当前用户账户及其所有数据。
    - 删除 users + answer_history + spaced_repetition_stats +
      bookmarks + learning_goals + email_logs + experiment_logs
    - 需要 Authorization: Bearer <token> 请求头
    """
    db = _get_db()
    db.delete_user_and_data(current_user["user_id"])
    return {"message": "Account deleted"}


@router.get("/stats", response_model=UserStatsResponse)
def get_stats(current_user: dict = Depends(get_current_user)):
    """
    获取当前用户的学习统计数据。
    - 需要 Authorization: Bearer <token> 请求头
    """
    from engine.scoring import estimate_gmat_score

    db = _get_db()
    raw = db.get_user_stats(current_user["user_id"])
    theta: float = raw.get("current_theta") or 0.0
    return UserStatsResponse(
        total_questions=raw["total_questions"],
        total_correct=raw["total_correct"],
        accuracy_pct=raw["accuracy_pct"],
        best_streak=raw["best_streak"],
        member_since=raw["member_since"],
        current_theta=round(theta, 4),
        current_gmat_score=estimate_gmat_score(theta),
        favorite_question_type=raw["favorite_question_type"],
    )
