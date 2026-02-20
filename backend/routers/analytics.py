"""
Analytics API（Week 4）
- POST /api/analytics/log-outcome    — 前端提交实验结果
- GET  /api/analytics/ab-test-results — 获取 A/B 测试聚合统计
- GET  /api/analytics/rag-performance  — 获取 RAG 系统性能指标
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.ab_testing import get_ab_test_service
from backend.config import settings
from engine.scoring import estimate_gmat_score
from utils.db_handler import DatabaseManager

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "logicmaster.db")


def _get_db() -> DatabaseManager:
    return DatabaseManager(db_path=_DB_PATH)


_TYPE_COLORS: Dict[str, str] = {
    "Weaken": "hsl(345, 100%, 60%)",
    "Strengthen": "hsl(152, 100%, 50%)",
    "Assumption": "hsl(56, 100%, 50%)",
    "Inference": "hsl(180, 100%, 50%)",
    "Flaw": "hsl(20, 100%, 60%)",
    "Evaluate": "hsl(270, 80%, 60%)",
    "Boldface": "hsl(210, 80%, 55%)",
}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ========== 请求/响应模型 ==========

class LogOutcomeRequest(BaseModel):
    user_id: str = Field(..., description="用户标识（session UUID）")
    experiment_name: str = Field(..., description="实验名称")
    variant: str = Field(..., description="变体名称")
    metric: str = Field(..., description="指标名称 (is_correct, theta_gain, hint_count, ...)")
    value: float = Field(..., description="指标数值")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class LogOutcomeResponse(BaseModel):
    ok: bool
    message: str = ""


class VariantOutcomeStats(BaseModel):
    count: int = 0
    mean: float = 0.0
    sum: float = 0.0


class VariantStats(BaseModel):
    exposures: int = 0
    outcomes: Dict[str, VariantOutcomeStats] = {}


class ABTestResultsResponse(BaseModel):
    experiment: str
    active: bool
    description: str = ""
    total_exposures: int = 0
    total_outcomes: int = 0
    variants: Dict[str, VariantStats] = {}


class RAGPerformanceResponse(BaseModel):
    retrieval_metrics: Dict[str, Any] = {}
    quality_metrics: Dict[str, Any] = {}
    system_metrics: Dict[str, Any] = {}


# ========== 端点 ==========

@router.post("/log-outcome", response_model=LogOutcomeResponse)
def log_outcome(req: LogOutcomeRequest):
    """前端提交实验结果（答题后调用）"""
    ab = get_ab_test_service()
    ok = ab.log_outcome(
        user_id=req.user_id,
        experiment_name=req.experiment_name,
        variant=req.variant,
        metric=req.metric,
        value=req.value,
        metadata=req.metadata,
    )
    return LogOutcomeResponse(ok=ok, message="logged" if ok else "failed")


@router.get("/ab-test-results", response_model=ABTestResultsResponse)
def get_ab_test_results(experiment: str = Query("tutor_strategy", description="实验名称")):
    """获取 A/B 测试聚合统计"""
    ab = get_ab_test_service()
    raw = ab.get_experiment_results(experiment)

    # 将 raw dict 转换为响应模型
    variants_out: Dict[str, VariantStats] = {}
    for vname, vdata in raw.get("variants", {}).items():
        outcome_stats = {}
        for mname, mdata in vdata.get("outcomes", {}).items():
            outcome_stats[mname] = VariantOutcomeStats(
                count=mdata.get("count", 0),
                mean=mdata.get("mean", 0.0),
                sum=mdata.get("sum", 0.0),
            )
        variants_out[vname] = VariantStats(
            exposures=vdata.get("exposures", 0),
            outcomes=outcome_stats,
        )

    return ABTestResultsResponse(
        experiment=raw.get("experiment", experiment),
        active=raw.get("active", False),
        description=raw.get("description", ""),
        total_exposures=raw.get("total_exposures", 0),
        total_outcomes=raw.get("total_outcomes", 0),
        variants=variants_out,
    )


@router.get("/summary")
def get_analytics_summary(user_id: str = "default"):
    """
    获取用户学习统计汇总（供 Analytics 页面图表使用）。
    返回 answer_history, wrong_by_type, wrong_by_skill, skill_mastery 等。
    """
    db = _get_db()

    # 基础统计
    stats = db.get_user_stats(user_id)
    theta: float = stats.get("current_theta") or 0.0

    # 答题历史（学习曲线）
    history = db.query_answer_history(user_id=user_id)
    answer_history = [
        {
            "question_id": h["question_id"],
            "is_correct": bool(h["is_correct"]),
            "theta_at_time": h.get("theta_at_time") or 0.0,
            "timestamp": h.get("created_at"),
        }
        for h in history
    ]

    # 错题分析
    wrong_stats = db.get_wrong_stats(user_id)
    wrong_by_type = [
        {
            "name": t["question_type"],
            "value": t["count"],
            "color": _TYPE_COLORS.get(t["question_type"], "hsl(260, 60%, 50%)"),
        }
        for t in wrong_stats.get("by_type", [])
    ]
    wrong_by_skill = [
        {"skill": s["skill_name"], "count": s["count"]}
        for s in wrong_stats.get("by_skill", [])
    ]

    # 技能掌握度（错误率取反，转换为 0-100）
    skill_rates = db.get_skill_error_rates(user_id, limit=10)
    skill_mastery = [
        {"skill": s["skill_name"], "value": round(s["mastery"] * 100)}
        for s in skill_rates
    ]

    return {
        "total_questions": stats["total_questions"],
        "total_correct": stats["total_correct"],
        "accuracy_pct": stats["accuracy_pct"],
        "current_theta": round(theta, 4),
        "current_gmat_score": estimate_gmat_score(theta),
        "best_streak": stats["best_streak"],
        "answer_history": answer_history,
        "wrong_by_type": wrong_by_type,
        "wrong_by_skill": wrong_by_skill,
        "skill_mastery": skill_mastery,
    }


@router.get("/rag-performance", response_model=RAGPerformanceResponse)
def get_rag_performance():
    """
    获取 RAG 系统性能指标（静态 + 动态混合）
    静态指标来自配置/已知值，动态指标后续可从评估记录中读取
    """
    try:
        # 尝试获取 Qdrant 中已索引的文档数
        indexed_count = 0
        try:
            from backend.services.rag_service import get_rag_service
            rag = get_rag_service()
            if rag._client is not None:
                info = rag._client.get_collection(settings.QDRANT_COLLECTION)
                indexed_count = info.points_count
        except Exception:
            pass  # Qdrant 未运行时降级

        # 尝试获取 explanation_source 实验的 RAG vs baseline 数据
        quality_metrics: Dict[str, Any] = {}
        try:
            ab = get_ab_test_service()
            exp_data = ab.get_experiment_results("explanation_source")
            rag_variant = exp_data.get("variants", {}).get("rag_enhanced", {})
            baseline_variant = exp_data.get("variants", {}).get("baseline", {})
            rag_score = rag_variant.get("outcomes", {}).get("quality_score", {})
            baseline_score = baseline_variant.get("outcomes", {}).get("quality_score", {})
            if rag_score.get("count", 0) > 0 and baseline_score.get("count", 0) > 0:
                improvement = (rag_score["mean"] - baseline_score["mean"]) / max(baseline_score["mean"], 0.01) * 100
                quality_metrics = {
                    "rag_avg_score": rag_score["mean"],
                    "baseline_avg_score": baseline_score["mean"],
                    "improvement_pct": round(improvement, 1),
                }
        except Exception:
            pass

        return RAGPerformanceResponse(
            retrieval_metrics={
                "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                "embedding_dims": settings.OPENAI_EMBEDDING_DIMS,
                "collection": settings.QDRANT_COLLECTION,
            },
            quality_metrics=quality_metrics,
            system_metrics={
                "indexed_questions": indexed_count,
                "qdrant_host": settings.QDRANT_HOST,
                "qdrant_port": settings.QDRANT_PORT,
            },
        )
    except Exception as e:
        logger.warning("get_rag_performance failed: %s", e)
        return RAGPerformanceResponse()
