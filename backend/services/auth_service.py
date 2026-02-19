"""
认证服务：密码哈希、JWT 生成/验证、FastAPI 依赖注入
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import bcrypt
import jwt
from fastapi import HTTPException, Request, status

from backend.config import settings


# ---------- 密码处理 ----------

def hash_password(password: str) -> str:
    """
    使用 bcrypt 对密码进行哈希处理

    Args:
        password: 明文密码

    Returns:
        bcrypt 哈希字符串
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    验证明文密码是否与 bcrypt 哈希匹配

    Args:
        password: 用户输入的明文密码
        password_hash: 存储的 bcrypt 哈希

    Returns:
        匹配返回 True，否则 False
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ---------- JWT 处理 ----------

def create_jwt_token(user_id: str, email: str) -> str:
    """
    生成 JWT Token（7 天有效期）

    Args:
        user_id: 用户 ID
        email: 用户邮箱

    Returns:
        JWT token 字符串
    """
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    解码并验证 JWT Token

    Args:
        token: JWT token 字符串

    Returns:
        包含 user_id 和 email 的字典

    Raises:
        HTTPException 401: token 无效或已过期
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id: str = payload.get("sub", "")
        email: str = payload.get("email", "")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload 缺少 user_id",
            )
        return {"user_id": user_id, "email": email}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期，请重新登录",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 无效：{e}",
        )


# ---------- FastAPI 依赖 ----------

def get_current_user(request: Request) -> Dict[str, Any]:
    """
    FastAPI 依赖函数：从 Authorization: Bearer <token> 中提取并验证用户信息

    Args:
        request: FastAPI Request 对象

    Returns:
        包含 user_id 和 email 的字典

    Raises:
        HTTPException 401: 未提供 token 或 token 无效
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Authorization 头，格式：Bearer <token>",
        )
    token = auth_header[len("Bearer "):]
    return decode_jwt_token(token)
