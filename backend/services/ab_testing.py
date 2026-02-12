"""
A/B Testing 服务：一致性分组、曝光/结果记录、聚合统计
使用 MD5 哈希确保同一 user_id 在同一实验中总是分配到相同变体
"""

import hashlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import settings
from utils.db_handler import get_db_manager

logger = logging.getLogger(__name__)


class ABTestService:
    """
    A/B Testing 服务

    - assign_variant: 一致性哈希分组（同 user 同 experiment → 同 variant）
    - log_exposure / log_outcome: 记录到 SQLite experiment_logs 表
    - get_experiment_results: 聚合统计每个 variant 的指标
    - is_experiment_active: 检查实验是否启用
    """

    def __init__(self):
        self.db = get_db_manager()
        # 确保 experiment_logs 表存在
        self.db.init_db()

    # ---------- 分组 ----------

    def assign_variant(self, user_id: str, experiment_name: str) -> Optional[str]:
        """
        为 user_id 分配实验变体（确定性：同用户同实验 → 同变体）

        使用 MD5(user_id + experiment_name) 的整数值，按权重累加区间分配。

        Args:
            user_id: 用户标识（session UUID）
            experiment_name: 实验名称（如 "tutor_strategy"）

        Returns:
            变体名称（如 "socratic_standard"），实验不存在或未启用返回 None
        """
        if not self.is_experiment_active(experiment_name):
            return None

        experiment = settings.AB_EXPERIMENTS.get(experiment_name, {})
        variants = experiment.get("variants", {})
        if not variants:
            return None

        # MD5 哈希 → 0~1 之间的浮点数
        hash_input = f"{user_id}:{experiment_name}"
        hash_hex = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        hash_val = int(hash_hex, 16) / (16 ** 32)  # 归一化到 [0, 1)

        # 按权重累加区间，分配变体
        cumulative = 0.0
        for variant_name, weight in variants.items():
            cumulative += weight
            if hash_val < cumulative:
                return variant_name

        # 兜底：返回最后一个变体（浮点精度边界情况）
        return list(variants.keys())[-1]

    # ---------- 记录 ----------

    def log_exposure(
        self,
        user_id: str,
        experiment_name: str,
        variant: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        记录实验曝光（用户被分配到某变体）

        Args:
            user_id: 用户标识
            experiment_name: 实验名称
            variant: 分配的变体
            metadata: 额外元数据（如 question_id）

        Returns:
            成功返回 True
        """
        try:
            return self.db.insert_experiment_log(
                user_id=user_id,
                experiment_name=experiment_name,
                variant=variant,
                event_type="exposure",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("log_exposure failed: %s", e)
            return False

    def log_outcome(
        self,
        user_id: str,
        experiment_name: str,
        variant: str,
        metric: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        记录实验结果（某指标的观测值）

        Args:
            user_id: 用户标识
            experiment_name: 实验名称
            variant: 变体名称
            metric: 指标名称（如 "is_correct", "theta_gain", "hint_count"）
            value: 指标数值
            metadata: 额外元数据

        Returns:
            成功返回 True
        """
        try:
            return self.db.insert_experiment_log(
                user_id=user_id,
                experiment_name=experiment_name,
                variant=variant,
                event_type="outcome",
                outcome_metric=metric,
                outcome_value=value,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("log_outcome failed: %s", e)
            return False

    # ---------- 统计 ----------

    def get_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """
        聚合某个实验的结果统计

        Returns:
            {
                "experiment": "tutor_strategy",
                "active": True,
                "total_exposures": 120,
                "total_outcomes": 80,
                "variants": {
                    "socratic_standard": {
                        "exposures": 40,
                        "outcomes": {
                            "is_correct": {"count": 25, "mean": 0.72, "sum": 18.0},
                            "theta_gain": {"count": 25, "mean": 0.12, "sum": 3.0},
                        },
                    },
                    ...
                },
            }
        """
        experiment = settings.AB_EXPERIMENTS.get(experiment_name, {})
        result: Dict[str, Any] = {
            "experiment": experiment_name,
            "active": self.is_experiment_active(experiment_name),
            "description": experiment.get("description", ""),
            "total_exposures": 0,
            "total_outcomes": 0,
            "variants": {},
        }

        try:
            # 查询曝光
            exposures = self.db.query_logs_by_experiment(experiment_name, event_type="exposure")
            # 查询结果
            outcomes = self.db.query_logs_by_experiment(experiment_name, event_type="outcome")

            result["total_exposures"] = len(exposures)
            result["total_outcomes"] = len(outcomes)

            # 初始化所有配置中的变体
            variant_names = list(experiment.get("variants", {}).keys())
            for vn in variant_names:
                result["variants"][vn] = {"exposures": 0, "outcomes": {}}

            # 统计曝光数
            for row in exposures:
                v = row.get("variant", "")
                if v in result["variants"]:
                    result["variants"][v]["exposures"] += 1
                else:
                    result["variants"][v] = {"exposures": 1, "outcomes": {}}

            # 统计结果指标
            for row in outcomes:
                v = row.get("variant", "")
                metric = row.get("outcome_metric", "")
                value = row.get("outcome_value")
                if value is None:
                    continue

                if v not in result["variants"]:
                    result["variants"][v] = {"exposures": 0, "outcomes": {}}

                outcomes_dict = result["variants"][v]["outcomes"]
                if metric not in outcomes_dict:
                    outcomes_dict[metric] = {"count": 0, "sum": 0.0, "mean": 0.0}

                outcomes_dict[metric]["count"] += 1
                outcomes_dict[metric]["sum"] += value

            # 计算均值
            for vn, vdata in result["variants"].items():
                for metric, mdata in vdata["outcomes"].items():
                    if mdata["count"] > 0:
                        mdata["mean"] = round(mdata["sum"] / mdata["count"], 4)

        except Exception as e:
            logger.warning("get_experiment_results failed: %s", e)

        return result

    # ---------- 状态 ----------

    def is_experiment_active(self, experiment_name: str) -> bool:
        """
        检查实验是否处于活跃状态

        Returns:
            True 当全局 AB_TEST_ENABLED 且实验 active=True
        """
        if not settings.AB_TEST_ENABLED:
            return False
        experiment = settings.AB_EXPERIMENTS.get(experiment_name)
        if experiment is None:
            return False
        return experiment.get("active", False)


# 模块级单例
_ab_test_service: Optional[ABTestService] = None


def get_ab_test_service() -> ABTestService:
    """获取 ABTestService 单例"""
    global _ab_test_service
    if _ab_test_service is None:
        _ab_test_service = ABTestService()
    return _ab_test_service
