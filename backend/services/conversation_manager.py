"""
对话状态管理器：服务端管理多轮 Socratic Tutor 对话的状态
使用内存字典存储，按 conversation_id (uuid4) 索引
"""

import time
import uuid
from typing import Dict, List, Optional, Any


# 对话状态常量
STATE_DIAGNOSING = "diagnosing"
STATE_HINTING = "hinting"
STATE_CONCLUDED = "concluded"

# 理解程度常量
UNDERSTANDING_CONFUSED = "confused"
UNDERSTANDING_PARTIAL = "partial"
UNDERSTANDING_CLEAR = "clear"

# 配置
MAX_HINTS = 3
CONVERSATION_TTL_SECONDS = 3600  # 1 小时过期


class Conversation:
    """单个对话的完整状态"""

    def __init__(self, conversation_id: str, question_id: str):
        self.conversation_id: str = conversation_id
        self.question_id: str = question_id
        self.chat_history: List[Dict[str, str]] = []
        self.current_state: str = STATE_DIAGNOSING
        self.hint_count: int = 0
        self.student_understanding: str = UNDERSTANDING_CONFUSED
        self.logic_gap: str = ""
        self.error_type: str = ""
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于 API 响应）"""
        return {
            "conversation_id": self.conversation_id,
            "question_id": self.question_id,
            "current_state": self.current_state,
            "hint_count": self.hint_count,
            "student_understanding": self.student_understanding,
            "logic_gap": self.logic_gap,
            "error_type": self.error_type,
            "chat_history_length": len(self.chat_history),
        }


class ConversationManager:
    """
    对话管理器，维护所有活跃对话的状态。
    使用内存字典存储（单进程足够，后续可替换为 Redis）。
    """

    def __init__(self):
        self._store: Dict[str, Conversation] = {}

    def create_conversation(self, question_id: str) -> Conversation:
        """
        创建新对话，返回 Conversation 对象

        Args:
            question_id: 关联的题目 ID

        Returns:
            新创建的 Conversation 实例
        """
        # 先清理过期对话
        self._cleanup_expired()

        conv_id = str(uuid.uuid4())
        conv = Conversation(conversation_id=conv_id, question_id=question_id)
        self._store[conv_id] = conv
        return conv

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        获取对话，不存在则返回 None

        Args:
            conversation_id: 对话 ID

        Returns:
            Conversation 实例或 None
        """
        conv = self._store.get(conversation_id)
        if conv is None:
            return None
        # 检查是否过期
        if time.time() - conv.created_at > CONVERSATION_TTL_SECONDS:
            del self._store[conversation_id]
            return None
        return conv

    def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> bool:
        """
        向对话添加一条消息

        Args:
            conversation_id: 对话 ID
            role: "user" 或 "assistant"
            content: 消息内容

        Returns:
            成功返回 True，对话不存在返回 False
        """
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return False
        conv.chat_history.append({"role": role, "content": content})
        conv.updated_at = time.time()
        return True

    def get_context_for_llm(
        self, conversation_id: str, max_messages: int = 8
    ) -> List[Dict[str, str]]:
        """
        获取最近 N 条消息用于 LLM 调用（截断以控制 token）

        Args:
            conversation_id: 对话 ID
            max_messages: 最大消息条数

        Returns:
            消息列表 [{"role": "user/assistant", "content": "..."}]
        """
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return []
        return conv.chat_history[-max_messages:]

    def should_continue_remediation(self, conversation_id: str) -> bool:
        """
        判断是否应继续 remediation 对话

        Returns:
            False 当 hint_count >= MAX_HINTS 或 student_understanding == "clear"
        """
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return False
        if conv.current_state == STATE_CONCLUDED:
            return False
        if conv.hint_count >= MAX_HINTS:
            return False
        if conv.student_understanding == UNDERSTANDING_CLEAR:
            return False
        return True

    def update_state(
        self,
        conversation_id: str,
        state: Optional[str] = None,
        hint_count: Optional[int] = None,
        understanding: Optional[str] = None,
        logic_gap: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> bool:
        """
        更新对话状态字段

        Returns:
            成功返回 True
        """
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return False
        if state is not None:
            conv.current_state = state
        if hint_count is not None:
            conv.hint_count = hint_count
        if understanding is not None:
            conv.student_understanding = understanding
        if logic_gap is not None:
            conv.logic_gap = logic_gap
        if error_type is not None:
            conv.error_type = error_type
        conv.updated_at = time.time()
        return True

    def conclude(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        结束对话，返回摘要

        Returns:
            对话摘要字典，或 None（对话不存在）
        """
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return None
        conv.current_state = STATE_CONCLUDED
        conv.updated_at = time.time()
        return {
            "conversation_id": conv.conversation_id,
            "total_turns": len([m for m in conv.chat_history if m["role"] == "user"]),
            "hint_count": conv.hint_count,
            "final_understanding": conv.student_understanding,
            "time_spent_seconds": round(conv.updated_at - conv.created_at, 1),
        }

    def _cleanup_expired(self) -> int:
        """
        清理超过 TTL 的过期对话

        Returns:
            清理的对话数量
        """
        now = time.time()
        expired_ids = [
            cid
            for cid, conv in self._store.items()
            if now - conv.created_at > CONVERSATION_TTL_SECONDS
        ]
        for cid in expired_ids:
            del self._store[cid]
        return len(expired_ids)

    @property
    def active_count(self) -> int:
        """当前活跃对话数"""
        return len(self._store)


# 模块级单例
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """获取 ConversationManager 单例"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
