"""
解析相关 API 端点
- POST /api/explanations/generate-with-rag  — 生成 RAG 增强解析
- POST /api/explanations/search-similar     — 搜索相似题目
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.explanation_service import generate_rag_enhanced_explanation
from backend.services.rag_service import get_rag_service

router = APIRouter(prefix="/api/explanations", tags=["explanations"])


# ---------- Pydantic Models ----------

class GenerateRequest(BaseModel):
    question_id: str
    question: Dict[str, Any]
    user_choice: Optional[str] = None
    is_correct: bool = False


class SimilarRef(BaseModel):
    question_id: str
    similarity: float


class GenerateResponse(BaseModel):
    explanation: str
    similar_references: List[SimilarRef]
    source: str  # "cached" | "rag_enhanced" | "llm_only"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    skills: Optional[List[str]] = None


class SearchResult(BaseModel):
    question_id: str
    explanation: str
    similarity_score: float
    question_type: str = ""
    skills: List[str] = []


class SearchResponse(BaseModel):
    results: List[SearchResult]


# ---------- Endpoints ----------

@router.post("/generate-with-rag", response_model=GenerateResponse)
def generate_with_rag(req: GenerateRequest):
    """
    生成 RAG 增强的题目解析（3-tier fallback: cached → RAG → plain LLM）
    """
    result = generate_rag_enhanced_explanation(
        question=req.question,
        user_choice=req.user_choice,
        is_correct=req.is_correct,
    )
    return GenerateResponse(
        explanation=result["explanation"],
        similar_references=[
            SimilarRef(question_id=r["question_id"], similarity=r["similarity"])
            for r in result.get("similar_references", [])
        ],
        source=result["source"],
    )


@router.post("/search-similar", response_model=SearchResponse)
def search_similar(req: SearchRequest):
    """
    搜索相似题目（可选技能过滤）
    """
    rag = get_rag_service()

    if req.skills:
        hits = rag.retrieve_by_skills(req.query, req.skills, top_k=req.top_k)
    else:
        hits = rag.retrieve_similar(req.query, top_k=req.top_k)

    return SearchResponse(
        results=[
            SearchResult(
                question_id=h["question_id"],
                explanation=h["explanation"],
                similarity_score=round(h["score"], 4),
                question_type=h.get("question_type", ""),
                skills=h.get("skills", []),
            )
            for h in hits
        ]
    )
