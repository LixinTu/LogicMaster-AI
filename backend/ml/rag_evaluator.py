"""
RAG 检索质量评估器
计算 Precision@K, Recall@K, MRR, F1@K 等检索指标
"""

from typing import Dict, List, Any


class RAGEvaluator:
    """RAG 检索质量评估器"""

    @staticmethod
    def precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        Precision@K = |relevant ∩ retrieved[:k]| / k

        Args:
            relevant_ids: 标注的相关题目 ID 列表
            retrieved_ids: 检索返回的题目 ID 列表（按相似度降序）
            k: 截取前 k 个

        Returns:
            Precision@K 值 (0.0 ~ 1.0)
        """
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for rid in top_k if rid in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

        Returns:
            Recall@K 值 (0.0 ~ 1.0)
        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for rid in top_k if rid in relevant_set)
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR (Mean Reciprocal Rank) = 1 / 第一个相关结果的位置

        Returns:
            MRR 值 (0.0 ~ 1.0)
        """
        relevant_set = set(relevant_ids)
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def f1_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        F1@K = 2 * P@K * R@K / (P@K + R@K)

        Returns:
            F1@K 值 (0.0 ~ 1.0)
        """
        p = RAGEvaluator.precision_at_k(relevant_ids, retrieved_ids, k)
        r = RAGEvaluator.recall_at_k(relevant_ids, retrieved_ids, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def evaluate_retrieval(
        relevant_ids: List[str], retrieved_ids: List[str], k: int = 5
    ) -> Dict[str, float]:
        """
        一次性计算所有检索指标

        Returns:
            {
                "precision@k": float,
                "recall@k": float,
                "mrr": float,
                "f1@k": float,
            }
        """
        return {
            f"precision@{k}": RAGEvaluator.precision_at_k(relevant_ids, retrieved_ids, k),
            f"recall@{k}": RAGEvaluator.recall_at_k(relevant_ids, retrieved_ids, k),
            "mrr": RAGEvaluator.mrr(relevant_ids, retrieved_ids),
            f"f1@{k}": RAGEvaluator.f1_at_k(relevant_ids, retrieved_ids, k),
        }

    @staticmethod
    def create_evaluation_report(
        test_cases: List[Dict[str, Any]], k: int = 5
    ) -> Dict[str, Any]:
        """
        对多个 test case 批量评估，返回平均指标

        Args:
            test_cases: [{"relevant_ids": [...], "retrieved_ids": [...]}, ...]
            k: K 值

        Returns:
            {
                "num_cases": int,
                "avg_precision@k": float,
                "avg_recall@k": float,
                "avg_mrr": float,
                "avg_f1@k": float,
                "per_case": [...]
            }
        """
        if not test_cases:
            return {
                "num_cases": 0,
                f"avg_precision@{k}": 0.0,
                f"avg_recall@{k}": 0.0,
                "avg_mrr": 0.0,
                f"avg_f1@{k}": 0.0,
                "per_case": [],
            }

        per_case = []
        sum_p = sum_r = sum_mrr = sum_f1 = 0.0

        for tc in test_cases:
            relevant = tc.get("relevant_ids", [])
            retrieved = tc.get("retrieved_ids", [])
            metrics = RAGEvaluator.evaluate_retrieval(relevant, retrieved, k)
            per_case.append(metrics)

            sum_p += metrics[f"precision@{k}"]
            sum_r += metrics[f"recall@{k}"]
            sum_mrr += metrics["mrr"]
            sum_f1 += metrics[f"f1@{k}"]

        n = len(test_cases)
        return {
            "num_cases": n,
            f"avg_precision@{k}": round(sum_p / n, 4),
            f"avg_recall@{k}": round(sum_r / n, 4),
            "avg_mrr": round(sum_mrr / n, 4),
            f"avg_f1@{k}": round(sum_f1 / n, 4),
            "per_case": per_case,
        }
