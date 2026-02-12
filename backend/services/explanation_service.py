"""
解析生成服务：3-tier fallback 策略
  Tier 1: 缓存（数据库中的 detailed_explanation）
  Tier 2: RAG-enhanced LLM（检索相似题目作为 few-shot 示例）
  Tier 3: Plain LLM（直接调用 DeepSeek 生成）
"""

import logging
from typing import Dict, Any, List, Optional

from openai import OpenAI

from backend.config import settings
from backend.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)


def generate_rag_enhanced_explanation(
    question: Dict[str, Any],
    user_choice: Optional[str] = None,
    is_correct: bool = False,
) -> Dict[str, Any]:
    """
    生成 RAG 增强的题目解析（3-tier fallback）

    Args:
        question: 完整题目字典
        user_choice: 学生选择的选项
        is_correct: 学生是否答对

    Returns:
        {
            "explanation": str,           # 解析文本
            "similar_references": list,   # 相似题目参考
            "source": str,               # "cached" | "rag_enhanced" | "llm_only"
        }
    """
    # ---------- Tier 1: 缓存 ----------
    cached = question.get("detailed_explanation", "")
    if cached and len(cached) > 100:
        return {
            "explanation": cached,
            "similar_references": [],
            "source": "cached",
        }

    # ---------- Tier 2: RAG-enhanced LLM ----------
    query = f"{question.get('stimulus', '')} {question.get('question', '')}"
    rag = get_rag_service()
    similar = rag.retrieve_similar(query, top_k=2)

    if similar:
        try:
            explanation = _call_llm_with_rag(question, similar, user_choice, is_correct)
            if explanation and len(explanation) > 100:
                refs = [
                    {"question_id": s["question_id"], "similarity": round(s["score"], 2)}
                    for s in similar
                ]
                return {
                    "explanation": explanation,
                    "similar_references": refs,
                    "source": "rag_enhanced",
                }
        except Exception as e:
            logger.warning("RAG-enhanced generation failed, falling back to plain LLM: %s", e)

    # ---------- Tier 3: Plain LLM ----------
    try:
        explanation = _call_llm_plain(question, user_choice, is_correct)
        return {
            "explanation": explanation,
            "similar_references": [],
            "source": "llm_only",
        }
    except Exception as e:
        logger.error("Plain LLM generation also failed: %s", e)
        # 最终 fallback：返回基础 explanation
        return {
            "explanation": question.get("explanation", "No explanation available."),
            "similar_references": [],
            "source": "cached",
        }


def _call_llm_with_rag(
    question: Dict[str, Any],
    similar: List[Dict[str, Any]],
    user_choice: Optional[str],
    is_correct: bool,
) -> str:
    """使用 RAG 检索到的相似题目作为 few-shot 示例，调用 LLM 生成解析"""
    client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    # 构建 few-shot 示例
    examples = ""
    for i, s in enumerate(similar, 1):
        expl = s.get("explanation", "")
        if expl:
            examples += f"\nExample {i} (similar question explanation):\n{expl}\n"

    prompt = f"""You are a GMAT Critical Reasoning expert. Generate a detailed explanation (150-250 words in English) for the following question.

Here are high-quality explanation examples from similar questions for reference:
{examples}

Current question:
- Type: {question.get('question_type', 'Weaken')}
- Stimulus: {question.get('stimulus', '')}
- Question: {question.get('question', '')}
- Choices:
{chr(10).join(['  ' + c for c in question.get('choices', [])])}
- Correct answer: {question.get('correct', '')}
"""
    if user_choice:
        prompt += f"- Student's choice: {user_choice} ({'correct' if is_correct else 'incorrect'})\n"

    prompt += """
Generate a structured explanation covering:
1) Correct answer and question type
2) Argument structure (conclusion, premises, hidden assumption)
3) Why the correct option works
4) Why key wrong options fail
5) One-sentence takeaway

Output explanation text only."""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a GMAT Critical Reasoning explanation expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def _call_llm_plain(
    question: Dict[str, Any],
    user_choice: Optional[str],
    is_correct: bool,
) -> str:
    """不使用 RAG，直接调用 LLM 生成解析（Tier 3）"""
    client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    prompt = f"""Generate a detailed explanation (150-250 words in English) for the following GMAT Critical Reasoning question.

- Type: {question.get('question_type', 'Weaken')}
- Stimulus: {question.get('stimulus', '')}
- Question: {question.get('question', '')}
- Choices:
{chr(10).join(['  ' + c for c in question.get('choices', [])])}
- Correct answer: {question.get('correct', '')}
"""
    if user_choice:
        prompt += f"- Student's choice: {user_choice} ({'correct' if is_correct else 'incorrect'})\n"

    prompt += """
Generate a structured explanation covering:
1) Correct answer and question type
2) Argument structure (conclusion, premises, hidden assumption)
3) Why the correct option works
4) Why key wrong options fail
5) One-sentence takeaway

Output explanation text only."""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a GMAT Critical Reasoning explanation expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()
