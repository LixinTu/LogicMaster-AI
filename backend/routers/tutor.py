"""
Tutor 对话 API
- POST /api/tutor/chat               — (Week 1) 向后兼容：无状态 Socratic 对话
- POST /api/tutor/start-remediation   — (Week 3) 新建对话，诊断 + 首条提示
- POST /api/tutor/continue            — (Week 3) 继续对话：评估理解 → 下一提示/结论
- POST /api/tutor/conclude            — (Week 3) 结束对话，返回总结
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_service import tutor_reply
from backend.config import settings
from backend.services.tutor_agent import get_tutor_agent
from backend.services.conversation_manager import (
    get_conversation_manager,
    STATE_HINTING,
    STATE_CONCLUDED,
)
from backend.services.ab_testing import get_ab_test_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tutor", tags=["tutor"])


# ====================================================================
# Week 1 向后兼容模型 + 端点（保持不变）
# ====================================================================

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


@router.post("/chat", response_model=TutorChatResponse)
def tutor_chat(req: TutorChatRequest):
    """Week 1 向后兼容端点：无状态 Socratic 对话"""
    api_key = settings.DEEPSEEK_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")

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
    return TutorChatResponse(reply=reply, is_error=is_error)


# ====================================================================
# Week 3 新端点：LangChain Agent + ConversationManager
# ====================================================================

# ---------- 请求/响应模型 ----------

class StartRemediationRequest(BaseModel):
    question_id: str = Field(..., description="题目 ID")
    question: Dict[str, Any] = Field(..., description="完整题目字典")
    user_choice: str = Field(..., description="学生选择 (A-E)")
    correct_choice: str = Field(..., description="正确答案 (A-E)")
    user_id: Optional[str] = Field(None, description="用户 ID（用于 A/B 分组）")


class StartRemediationResponse(BaseModel):
    conversation_id: str
    first_hint: str
    logic_gap: str
    error_type: str
    hint_count: int
    student_understanding: str
    current_state: str
    variant: str = "socratic_standard"


class ContinueRequest(BaseModel):
    conversation_id: str = Field(..., description="对话 ID")
    student_message: str = Field(..., description="学生回复")
    question: Optional[Dict[str, Any]] = Field(None, description="题目字典（用于生成提示）")
    correct_choice: Optional[str] = Field(None, description="正确答案")


class ContinueResponse(BaseModel):
    reply: str
    hint_count: int
    student_understanding: str
    should_continue: bool
    current_state: str
    blooms_level: int = 1
    blooms_name: str = "Remember"


class ConcludeRequest(BaseModel):
    conversation_id: str = Field(..., description="对话 ID")
    question: Optional[Dict[str, Any]] = Field(None, description="题目字典（用于生成结论）")
    correct_choice: Optional[str] = Field(None, description="正确答案")


class ConversationSummary(BaseModel):
    total_turns: int
    hint_count: int
    final_understanding: str
    time_spent_seconds: float
    blooms_level: int = 1
    blooms_progression: List[int] = []


class ConcludeResponse(BaseModel):
    conclusion: str
    summary: ConversationSummary


# ---------- 端点实现 ----------

@router.post("/start-remediation", response_model=StartRemediationResponse)
def start_remediation(req: StartRemediationRequest):
    """
    新建 remediation 对话：A/B 分组 → 诊断错误 → 生成第一条提示（或直接解析）
    """
    try:
        cm = get_conversation_manager()
        agent = get_tutor_agent()
        ab = get_ab_test_service()

        # 0. A/B 分组
        user_id = req.user_id or "anonymous"
        variant = ab.assign_variant(user_id, "tutor_strategy") or "socratic_standard"
        ab.log_exposure(user_id, "tutor_strategy", variant, metadata={"question_id": req.question_id})

        # 1. 创建对话
        conv = cm.create_conversation(question_id=req.question_id)
        cid = conv.conversation_id

        # 保存 A/B 信息到对话
        conv.variant = variant
        conv.user_id = user_id

        # 2. 诊断错误（所有变体都需要诊断）
        diagnosis = agent.diagnose_error(
            question=req.question,
            user_choice=req.user_choice,
            correct_choice=req.correct_choice,
        )
        logic_gap = diagnosis.get("logic_gap", "")
        error_type = diagnosis.get("error_type", "other")
        key_assumption = diagnosis.get("key_assumption", "")

        # 保存诊断结果到对话状态
        cm.update_state(cid, state=STATE_HINTING, logic_gap=logic_gap, error_type=error_type)
        conv.key_assumption = key_assumption
        conv.question = req.question
        conv.correct_choice = req.correct_choice
        conv.user_choice = req.user_choice

        # 3. 记录学生选择
        cm.add_message(cid, "user", f"I chose answer: {req.user_choice}")

        # 4. 根据变体生成不同的首条回复
        if variant == "direct_explanation":
            # 直接给出解析（跳过苏格拉底对话）
            explanation = (
                f"The correct answer is {req.correct_choice}. "
                f"{diagnosis.get('why_wrong', '')} "
                f"The key assumption here: {key_assumption}"
            )
            cm.add_message(cid, "assistant", explanation)
            cm.update_state(cid, state=STATE_CONCLUDED, hint_count=0)

            return StartRemediationResponse(
                conversation_id=cid,
                first_hint=explanation,
                logic_gap=logic_gap,
                error_type=error_type,
                hint_count=0,
                student_understanding="confused",
                current_state=STATE_CONCLUDED,
                variant=variant,
            )
        else:
            # socratic_standard 或 socratic_aggressive: 生成苏格拉底提示
            hint_count_start = 0
            if variant == "socratic_aggressive":
                hint_count_start = 1  # 从 moderate 强度开始

            first_hint = agent.generate_socratic_hint(
                question=req.question,
                user_choice=req.user_choice,
                logic_gap=logic_gap,
                error_type=error_type,
                hint_count=hint_count_start,
                chat_history=cm.get_context_for_llm(cid),
            )
            cm.add_message(cid, "assistant", first_hint)
            cm.update_state(cid, hint_count=0)  # hint_count only increments via /continue

            return StartRemediationResponse(
                conversation_id=cid,
                first_hint=first_hint,
                logic_gap=logic_gap,
                error_type=error_type,
                hint_count=0,
                student_understanding="confused",
                current_state=STATE_HINTING,
                variant=variant,
            )

    except Exception as e:
        logger.error("start_remediation failed: %s", e)
        # 优雅降级：仍然创建对话，返回默认提示
        cm = get_conversation_manager()
        conv = cm.create_conversation(question_id=req.question_id)
        cid = conv.conversation_id
        cm.update_state(cid, state=STATE_HINTING, hint_count=0)  # hint_count only increments via /continue
        cm.add_message(cid, "user", f"I chose answer: {req.user_choice}")

        fallback_hint = "Let's take a step back. What is the main conclusion of the argument?"
        cm.add_message(cid, "assistant", fallback_hint)

        conv.question = req.question
        conv.correct_choice = req.correct_choice
        conv.user_choice = req.user_choice
        conv.variant = "socratic_standard"
        conv.user_id = req.user_id or "anonymous"

        return StartRemediationResponse(
            conversation_id=cid,
            first_hint=fallback_hint,
            logic_gap="Unable to diagnose — using default guidance.",
            error_type="other",
            hint_count=0,
            student_understanding="confused",
            current_state=STATE_HINTING,
            variant="socratic_standard",
        )


@router.post("/continue", response_model=ContinueResponse)
def continue_remediation(req: ContinueRequest):
    """
    继续对话：评估学生理解 → 决定下一步（继续提示 or 结论）
    """
    cm = get_conversation_manager()
    conv = cm.get_conversation(req.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found or expired")

    agent = get_tutor_agent()
    cid = req.conversation_id

    # 获取题目上下文（优先用请求中传入的，否则用对话中保存的）
    question = req.question or getattr(conv, "question", {}) or {}
    correct_choice = req.correct_choice or getattr(conv, "correct_choice", "")

    try:
        # 1. 记录学生消息
        cm.add_message(cid, "user", req.student_message)

        # 2. 用 Bloom's Taxonomy 评估认知水平
        blooms_result = agent.evaluate_blooms_level(
            student_response=req.student_message,
            logic_gap=conv.logic_gap,
            key_assumption=getattr(conv, "key_assumption", ""),
            chat_history=cm.get_context_for_llm(cid),
        )
        blooms_level = blooms_result["level"]
        blooms_name = blooms_result["level_name"]
        understanding = blooms_result["mapped_understanding"]
        cm.update_state(cid, understanding=understanding, blooms_level=blooms_level)

        # 3. 判断是否继续（Bloom's 5-6 也触发结束）
        should_continue = cm.should_continue_remediation(cid)

        if should_continue:
            # 生成下一条提示（强度递增 + Bloom's 策略调整）
            hint = agent.generate_socratic_hint(
                question=question,
                user_choice=getattr(conv, "user_choice", ""),
                logic_gap=conv.logic_gap,
                error_type=conv.error_type,
                hint_count=conv.hint_count,
                chat_history=cm.get_context_for_llm(cid),
                blooms_level=blooms_level,
            )
            cm.add_message(cid, "assistant", hint)
            new_hint_count = conv.hint_count + 1
            cm.update_state(cid, hint_count=new_hint_count)

            return ContinueResponse(
                reply=hint,
                hint_count=new_hint_count,
                student_understanding=understanding,
                should_continue=True,
                current_state=STATE_HINTING,
                blooms_level=blooms_level,
                blooms_name=blooms_name,
            )
        else:
            # 生成结论（揭示正确答案）
            conclusion = agent.generate_conclusion(
                question=question,
                correct_choice=correct_choice,
                logic_gap=conv.logic_gap,
                student_understanding=understanding,
            )
            cm.add_message(cid, "assistant", conclusion)
            cm.update_state(cid, state=STATE_CONCLUDED)

            return ContinueResponse(
                reply=conclusion,
                hint_count=conv.hint_count,
                student_understanding=understanding,
                should_continue=False,
                current_state=STATE_CONCLUDED,
                blooms_level=blooms_level,
                blooms_name=blooms_name,
            )

    except Exception as e:
        logger.error("continue_remediation failed: %s", e)
        # 降级：返回一条默认提示
        fallback = "Think about the assumption connecting the premises to the conclusion. Is it valid?"
        cm.add_message(cid, "assistant", fallback)
        return ContinueResponse(
            reply=fallback,
            hint_count=conv.hint_count,
            student_understanding=conv.student_understanding,
            should_continue=cm.should_continue_remediation(cid),
            current_state=conv.current_state,
        )


@router.post("/conclude", response_model=ConcludeResponse)
def conclude_remediation(req: ConcludeRequest):
    """
    结束对话，生成最终总结
    """
    cm = get_conversation_manager()
    conv = cm.get_conversation(req.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found or expired")

    agent = get_tutor_agent()

    question = req.question or getattr(conv, "question", {}) or {}
    correct_choice = req.correct_choice or getattr(conv, "correct_choice", "")

    try:
        # 生成结论
        conclusion = agent.generate_conclusion(
            question=question,
            correct_choice=correct_choice,
            logic_gap=conv.logic_gap,
            student_understanding=conv.student_understanding,
        )
        cm.add_message(req.conversation_id, "assistant", conclusion)

        # 结束对话并获取摘要
        summary_dict = cm.conclude(req.conversation_id)

        return ConcludeResponse(
            conclusion=conclusion,
            summary=ConversationSummary(
                total_turns=summary_dict.get("total_turns", 0),
                hint_count=summary_dict.get("hint_count", 0),
                final_understanding=summary_dict.get("final_understanding", "confused"),
                time_spent_seconds=summary_dict.get("time_spent_seconds", 0.0),
                blooms_level=summary_dict.get("blooms_level", 1),
                blooms_progression=summary_dict.get("blooms_progression", []),
            ),
        )

    except Exception as e:
        logger.error("conclude_remediation failed: %s", e)
        # 降级
        summary_dict = cm.conclude(req.conversation_id) or {
            "total_turns": 0, "hint_count": 0,
            "final_understanding": "confused", "time_spent_seconds": 0.0,
            "blooms_level": 1, "blooms_progression": [],
        }
        fallback_conclusion = (
            f"The correct answer is {correct_choice}. "
            "Review the argument's hidden assumption for similar questions."
        )
        return ConcludeResponse(
            conclusion=fallback_conclusion,
            summary=ConversationSummary(
                total_turns=summary_dict.get("total_turns", 0),
                hint_count=summary_dict.get("hint_count", 0),
                final_understanding=summary_dict.get("final_understanding", "confused"),
                time_spent_seconds=summary_dict.get("time_spent_seconds", 0.0),
                blooms_level=summary_dict.get("blooms_level", 1),
                blooms_progression=summary_dict.get("blooms_progression", []),
            ),
        )
