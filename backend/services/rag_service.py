"""
RAG 服务：基于 Qdrant 向量数据库的检索增强生成
提供题目索引、相似题目检索、技能过滤检索等功能
"""

import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 检索服务，封装 Qdrant 向量数据库操作和 OpenAI Embedding 调用。
    采用懒初始化，首次调用时才连接 Qdrant 和创建 collection。
    所有方法均有 graceful degradation：出错时返回空结果而非抛异常。
    """

    def __init__(self):
        self._qdrant: Optional[QdrantClient] = None
        self._openai: Optional[OpenAI] = None
        self._collection_ready: bool = False

    # ---------- 懒初始化 ----------

    def _get_qdrant(self) -> QdrantClient:
        """懒初始化 Qdrant 客户端"""
        if self._qdrant is None:
            self._qdrant = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
        return self._qdrant

    def _get_openai(self) -> OpenAI:
        """懒初始化 OpenAI 客户端（用于 embedding）"""
        if self._openai is None:
            self._openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai

    def _ensure_collection(self) -> bool:
        """确保 Qdrant collection 存在，不存在则创建"""
        if self._collection_ready:
            return True
        try:
            client = self._get_qdrant()
            collections = client.get_collections().collections
            exists = any(c.name == settings.QDRANT_COLLECTION for c in collections)
            if not exists:
                client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=settings.OPENAI_EMBEDDING_DIMS,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", settings.QDRANT_COLLECTION)
            self._collection_ready = True
            return True
        except Exception as e:
            logger.error("Failed to ensure Qdrant collection: %s", e)
            return False

    # ---------- Embedding ----------

    def embed(self, text: str) -> Optional[List[float]]:
        """
        调用 OpenAI Embedding API 生成向量

        Args:
            text: 待嵌入的文本

        Returns:
            浮点数列表（向量），失败时返回 None
        """
        try:
            client = self._get_openai()
            resp = client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error("OpenAI embedding failed: %s", e)
            return None

    # ---------- 索引 ----------

    def index_question(
        self,
        question_id: str,
        question_text: str,
        explanation: str,
        question_type: str = "",
        skills: Optional[List[str]] = None,
        difficulty: str = "",
    ) -> bool:
        """
        将一道题目索引到 Qdrant（upsert，幂等）

        Args:
            question_id: 题目 ID（用作 point ID 的一部分）
            question_text: 题干 + 问题文本
            explanation: 题目解析
            question_type: 题型（Weaken/Strengthen/...）
            skills: 技能标签列表
            difficulty: 难度等级

        Returns:
            成功返回 True，失败返回 False
        """
        if not self._ensure_collection():
            return False

        # 构建文档文本用于 embedding
        document = f"Question: {question_text}\n\nExplanation: {explanation}"
        vector = self.embed(document)
        if vector is None:
            return False

        try:
            # 使用 question_id 的 hash 作为 Qdrant point id（整数）
            point_id = abs(hash(question_id)) % (2**63)

            self._get_qdrant().upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "question_id": question_id,
                            "question_text": question_text,
                            "explanation": explanation,
                            "question_type": question_type,
                            "skills": skills or [],
                            "difficulty": difficulty,
                        },
                    )
                ],
            )
            return True
        except Exception as e:
            logger.error("Failed to index question %s: %s", question_id, e)
            return False

    # ---------- 检索 ----------

    def retrieve_similar(
        self, query_text: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        根据文本检索最相似的题目

        Args:
            query_text: 查询文本
            top_k: 返回数量

        Returns:
            相似题目列表，每项包含 question_id, explanation, score 等
        """
        if not self._ensure_collection():
            return []

        vector = self.embed(query_text)
        if vector is None:
            return []

        try:
            results = self._get_qdrant().search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=vector,
                limit=top_k,
            )
            return [
                {
                    "question_id": hit.payload.get("question_id", ""),
                    "explanation": hit.payload.get("explanation", ""),
                    "question_type": hit.payload.get("question_type", ""),
                    "skills": hit.payload.get("skills", []),
                    "score": hit.score,
                }
                for hit in results
            ]
        except Exception as e:
            logger.error("Qdrant search failed: %s", e)
            return []

    def retrieve_by_skills(
        self,
        query_text: str,
        required_skills: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        带技能过滤的相似检索

        Args:
            query_text: 查询文本
            required_skills: 要求至少包含其中一项技能
            top_k: 返回数量

        Returns:
            匹配技能的相似题目列表
        """
        if not self._ensure_collection():
            return []

        vector = self.embed(query_text)
        if vector is None:
            return []

        try:
            # 使用 Qdrant 的 payload filter：skills 数组中至少包含一个 required_skill
            skill_filter = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="skills",
                        match=qmodels.MatchValue(value=skill),
                    )
                    for skill in required_skills
                ]
            )

            results = self._get_qdrant().search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=vector,
                query_filter=skill_filter,
                limit=top_k,
            )
            return [
                {
                    "question_id": hit.payload.get("question_id", ""),
                    "explanation": hit.payload.get("explanation", ""),
                    "question_type": hit.payload.get("question_type", ""),
                    "skills": hit.payload.get("skills", []),
                    "score": hit.score,
                }
                for hit in results
            ]
        except Exception as e:
            logger.error("Qdrant skill-filtered search failed: %s", e)
            return []


# 模块级单例
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """获取 RAGService 单例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
