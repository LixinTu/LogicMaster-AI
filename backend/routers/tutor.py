"""
Tutor 对话 API
复用 llm_service.py 的 tutor_reply（苏格拉底式追问）
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_service import tutor_reply
from backend.config import settings

router = APIRouter(prefix="/api/tutor", tags=["tutor"])


# ---------- 请求/响应模型 ----------

class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: user 或 assistant")
    content: str = Field(..., description="消息内容")


class TutorChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的文本")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="历史对话")
    question_id: str = Field("", description="当前题目 ID")
    current_q: Optional[Dict[str, Any]] = Field(None, description="当前题目快照")
    socratic_context: Optional[Dict[str, Any]] = Field(None, description="苏格拉底上下文")


class TutorChatResponse(BaseModel):
    reply: str = Field(..., description="AI 回复")
    is_error: bool = Field(False, description="是否为错误响应")


# ---------- 端点 ----------

@router.post("/chat", response_model=TutorChatResponse)
def tutor_chat(req: TutorChatRequest):
    """
    Tutor 对话端点。
    直接调用 llm_service.tutor_reply，传入 DeepSeek API Key。
    """
    api_key = settings.DEEPSEEK_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY 未配置")

    # 将 Pydantic 模型转为 dict 列表（tutor_reply 期望的格式）
    history_dicts = [msg.model_dump() for msg in req.chat_history]

    reply = tutor_reply(
        user_text=req.message,
        api_key=api_key,
        chat_history=history_dicts,
        current_q=req.current_q,
        current_q_id=req.question_id,
        socratic_context=req.socratic_context,
    )

    is_error = reply.startswith("[LLM ERROR]")
    if is_error:
        return TutorChatResponse(reply=reply, is_error=True)

    return TutorChatResponse(reply=reply, is_error=False)
