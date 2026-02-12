"""
LLM 质量评估器：使用 GPT-4o-mini 作为 Judge 评估解析质量
评分维度: correctness, clarity, completeness, pedagogical_value（各 1-5 分）
"""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

# Judge prompt 模板
JUDGE_SYSTEM_PROMPT = """\
You are a GMAT Critical Reasoning expert evaluator.
Score the given explanation on 4 criteria, each from 1 (worst) to 5 (best).
Output strict JSON only, no extra text."""

JUDGE_USER_TEMPLATE = """\
Question type: {question_type}
Stimulus: {stimulus}
Question: {question_stem}
Correct answer: {correct_choice}

Explanation to evaluate:
\"\"\"{explanation}\"\"\"

Score the explanation on these criteria:
1. Correctness (1-5): Is the reasoning factually and logically correct?
2. Clarity (1-5): Is it easy to understand?
3. Completeness (1-5): Does it fully explain why the answer is correct and others are wrong?
4. Pedagogical Value (1-5): Does it teach a transferable reasoning pattern?

Output strict JSON:
{{
  "correctness": <int 1-5>,
  "clarity": <int 1-5>,
  "completeness": <int 1-5>,
  "pedagogical_value": <int 1-5>,
  "justification": "<one sentence reasoning>"
}}"""


def _extract_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON（兼容 markdown 代码块）"""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


class LLMQualityEvaluator:
    """
    使用 GPT-4o-mini 作为 Judge 评估 GMAT 解析质量

    Methods:
        evaluate_single: 评估单个解析 → {correctness, clarity, completeness, pedagogical_value, overall, justification}
        evaluate_batch: 批量评估 → 各维度平均分
    """

    CRITERIA = ["correctness", "clarity", "completeness", "pedagogical_value"]

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._model = model
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def evaluate_single(
        self,
        question: Dict[str, Any],
        explanation: str,
    ) -> Dict[str, Any]:
        """
        评估单个解析的质量

        Args:
            question: 题目字典（含 stimulus, question, correct 等）
            explanation: 待评估的解析文本

        Returns:
            {
                "correctness": 4,
                "clarity": 5,
                "completeness": 3,
                "pedagogical_value": 4,
                "overall": 4.0,
                "justification": "...",
                "error": None
            }
        """
        default = {c: 0 for c in self.CRITERIA}
        default.update({"overall": 0.0, "justification": "", "error": None})

        if not explanation or not explanation.strip():
            default["error"] = "Empty explanation"
            return default

        try:
            client = self._get_client()
            user_msg = JUDGE_USER_TEMPLATE.format(
                question_type=question.get("question_type", "Weaken"),
                stimulus=question.get("stimulus", "")[:500],
                question_stem=question.get("question", ""),
                correct_choice=question.get("correct", question.get("correct_choice", "")),
                explanation=explanation[:1000],
            )

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            raw = response.choices[0].message.content or ""
            scores = _extract_json(raw)

            # 校验并裁剪分数到 1-5
            result: Dict[str, Any] = {}
            for c in self.CRITERIA:
                val = scores.get(c, 3)
                result[c] = max(1, min(5, int(val)))

            result["overall"] = round(sum(result[c] for c in self.CRITERIA) / len(self.CRITERIA), 2)
            result["justification"] = scores.get("justification", "")
            result["error"] = None
            return result

        except Exception as e:
            logger.warning("evaluate_single failed: %s", e)
            default["error"] = str(e)
            return default

    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量评估多个解析

        Args:
            items: [{"question": {...}, "explanation": "..."}, ...]

        Returns:
            {
                "count": 10,
                "avg_correctness": 4.2,
                "avg_clarity": 4.5,
                "avg_completeness": 3.8,
                "avg_pedagogical_value": 4.0,
                "avg_overall": 4.13,
                "results": [...]   # 每个 item 的完整评分
            }
        """
        results = []
        for item in items:
            score = self.evaluate_single(
                question=item.get("question", {}),
                explanation=item.get("explanation", ""),
            )
            results.append(score)

        # 只统计成功的评估（error is None）
        valid = [r for r in results if r.get("error") is None]
        count = len(valid)

        summary: Dict[str, Any] = {
            "count": count,
            "total_evaluated": len(items),
            "results": results,
        }

        if count > 0:
            for c in self.CRITERIA:
                summary[f"avg_{c}"] = round(sum(r[c] for r in valid) / count, 2)
            summary["avg_overall"] = round(sum(r["overall"] for r in valid) / count, 2)
        else:
            for c in self.CRITERIA:
                summary[f"avg_{c}"] = 0.0
            summary["avg_overall"] = 0.0

        return summary


# 模块级单例
_evaluator: Optional[LLMQualityEvaluator] = None


def get_llm_evaluator() -> LLMQualityEvaluator:
    """获取 LLMQualityEvaluator 单例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = LLMQualityEvaluator()
    return _evaluator
