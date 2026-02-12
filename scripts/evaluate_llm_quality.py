"""
LLM 解析质量评估脚本
功能：从数据库加载题目 → 生成 RAG 增强解析 vs baseline 解析 → GPT-4o-mini 评分 → 对比报告

Usage:
    python scripts/evaluate_llm_quality.py [--count 10]
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_handler import get_db_manager
from backend.ml.llm_evaluator import LLMQualityEvaluator


def load_questions(limit: int = 10):
    """从数据库加载有解析的题目"""
    db = get_db_manager()
    candidates = db.get_adaptive_candidates(target_difficulty=0.0, limit=limit * 2)
    # 过滤有解析的题目
    with_explanation = [
        q for q in candidates
        if q.get("explanation") and len(q.get("explanation", "")) > 20
    ]
    return with_explanation[:limit]


def run_evaluation(questions, evaluator: LLMQualityEvaluator):
    """评估已有解析的质量"""
    items = []
    for q in questions:
        items.append({
            "question": q,
            "explanation": q.get("detailed_explanation") or q.get("explanation", ""),
        })

    print(f"Evaluating {len(items)} explanations with GPT-4o-mini judge...")
    return evaluator.evaluate_batch(items)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM explanation quality")
    parser.add_argument("--count", type=int, default=10, help="Number of questions to evaluate")
    args = parser.parse_args()

    print("=== LLM Quality Evaluation ===\n")

    questions = load_questions(args.count)
    if not questions:
        print("No questions with explanations found in database.")
        return

    print(f"Loaded {len(questions)} questions with explanations.\n")

    evaluator = LLMQualityEvaluator()
    result = run_evaluation(questions, evaluator)

    print(f"\n=== Results ({result['count']}/{result['total_evaluated']} successful) ===")
    for c in LLMQualityEvaluator.CRITERIA:
        avg = result.get(f"avg_{c}", 0.0)
        print(f"  {c:25s}: {avg:.2f} / 5.00")
    print(f"  {'Overall':25s}: {result.get('avg_overall', 0.0):.2f} / 5.00")

    # 保存报告
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    report_path = os.path.join(PROJECT_ROOT, "reports", "llm_quality_evaluation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        # 排除 results 中的每题详情以减小文件
        summary = {k: v for k, v in result.items() if k != "results"}
        summary["sample_questions"] = [q.get("question_id", q.get("id", "")) for q in questions]
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
