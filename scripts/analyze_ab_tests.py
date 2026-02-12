"""
A/B 测试统计分析脚本
功能：加载实验数据 → 分组统计 → t 检验 + Cohen's d → JSON 报告

Usage:
    python scripts/analyze_ab_tests.py [experiment_name]
    python scripts/analyze_ab_tests.py tutor_strategy
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_handler import get_db_manager


# ========== 数据加载 ==========

def load_experiment_data(experiment_name: str) -> List[Dict[str, Any]]:
    """从 SQLite 加载实验 outcome 日志"""
    db = get_db_manager()
    return db.query_logs_by_experiment(experiment_name, event_type="outcome")


# ========== 分组统计 ==========

def calculate_metrics_by_variant(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    按 variant 分组，计算每个指标的 count, mean, std

    Returns:
        {
            "socratic_standard": {
                "is_correct": {"count": 20, "mean": 0.75, "std": 0.43, "values": [...]},
                "theta_change": {...},
            },
            ...
        }
    """
    grouped: Dict[str, Dict[str, List[float]]] = {}

    for row in rows:
        variant = row.get("variant", "unknown")
        metric = row.get("outcome_metric", "")
        value = row.get("outcome_value")
        if value is None or not metric:
            continue

        if variant not in grouped:
            grouped[variant] = {}
        if metric not in grouped[variant]:
            grouped[variant][metric] = []
        grouped[variant][metric].append(float(value))

    result: Dict[str, Dict[str, Any]] = {}
    for variant, metrics in grouped.items():
        result[variant] = {}
        for metric, values in metrics.items():
            n = len(values)
            mean = sum(values) / n if n > 0 else 0.0
            variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
            std = math.sqrt(variance)
            result[variant][metric] = {
                "count": n,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "values": values,
            }

    return result


# ========== 统计检验 ==========

def statistical_significance_test(
    values_a: List[float],
    values_b: List[float],
) -> Dict[str, Any]:
    """
    独立样本 t 检验 + Cohen's d

    Returns:
        {
            "n_a": 20, "n_b": 18,
            "mean_a": 0.75, "mean_b": 0.61,
            "t_statistic": 2.45,
            "p_value": 0.012,
            "cohens_d": 0.28,
            "is_significant": True
        }
    """
    n_a, n_b = len(values_a), len(values_b)
    result: Dict[str, Any] = {
        "n_a": n_a, "n_b": n_b,
        "mean_a": 0.0, "mean_b": 0.0,
        "t_statistic": 0.0, "p_value": 1.0,
        "cohens_d": 0.0, "is_significant": False,
    }

    if n_a < 2 or n_b < 2:
        result["error"] = "Insufficient samples (need >= 2 per group)"
        return result

    # 尝试使用 scipy（如果安装了）
    try:
        from scipy import stats as sp_stats
        t_stat, p_val = sp_stats.ttest_ind(values_a, values_b, equal_var=False)
        result["t_statistic"] = round(float(t_stat), 4)
        result["p_value"] = round(float(p_val), 6)
    except ImportError:
        # 手动 Welch's t-test
        mean_a = sum(values_a) / n_a
        mean_b = sum(values_b) / n_b
        var_a = sum((x - mean_a) ** 2 for x in values_a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in values_b) / (n_b - 1)
        se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-10
        t_stat = (mean_a - mean_b) / se
        # 简化 p-value 近似（无 scipy 时）
        df = n_a + n_b - 2
        p_val = 2.0 * (1.0 - min(0.9999, abs(t_stat) / (abs(t_stat) + df)))  # 粗略近似
        result["t_statistic"] = round(t_stat, 4)
        result["p_value"] = round(p_val, 6)

    # Cohen's d
    mean_a = sum(values_a) / n_a
    mean_b = sum(values_b) / n_b
    result["mean_a"] = round(mean_a, 4)
    result["mean_b"] = round(mean_b, 4)

    var_a = sum((x - mean_a) ** 2 for x in values_a) / max(n_a - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in values_b) / max(n_b - 1, 1)
    pooled_std = math.sqrt((var_a * (n_a - 1) + var_b * (n_b - 1)) / max(n_a + n_b - 2, 1))
    result["cohens_d"] = round((mean_a - mean_b) / pooled_std, 4) if pooled_std > 0 else 0.0

    result["is_significant"] = result["p_value"] < 0.05

    return result


# ========== 报告生成 ==========

def generate_ab_report(experiment_name: str) -> Dict[str, Any]:
    """
    生成完整的 A/B 测试分析报告（JSON 格式）

    Returns:
        {
            "experiment": "tutor_strategy",
            "metrics_by_variant": {...},
            "comparisons": [...],
            "recommendation": "..."
        }
    """
    rows = load_experiment_data(experiment_name)
    metrics = calculate_metrics_by_variant(rows)

    variants = list(metrics.keys())
    comparisons = []

    # 两两比较 is_correct 指标
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            va, vb = variants[i], variants[j]
            vals_a = metrics.get(va, {}).get("is_correct", {}).get("values", [])
            vals_b = metrics.get(vb, {}).get("is_correct", {}).get("values", [])
            if vals_a and vals_b:
                test = statistical_significance_test(vals_a, vals_b)
                comparisons.append({
                    "comparison": f"{va} vs {vb}",
                    "metric": "is_correct",
                    **test,
                })

    # 简化 metrics 输出（去掉 values 列表）
    metrics_clean = {}
    for vn, vm in metrics.items():
        metrics_clean[vn] = {}
        for mn, md in vm.items():
            metrics_clean[vn][mn] = {k: v for k, v in md.items() if k != "values"}

    report = {
        "experiment": experiment_name,
        "total_outcome_rows": len(rows),
        "variants": variants,
        "metrics_by_variant": metrics_clean,
        "comparisons": comparisons,
    }

    # 保存到 reports/
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    report_path = os.path.join(PROJECT_ROOT, "reports", f"ab_test_{experiment_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {report_path}")

    return report


# ========== CLI ==========

if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "tutor_strategy"
    print(f"=== A/B Test Analysis: {exp_name} ===\n")

    rows = load_experiment_data(exp_name)
    print(f"Total outcome rows: {len(rows)}")

    if not rows:
        print("No data yet. Run the app and answer some questions first.")
        sys.exit(0)

    metrics = calculate_metrics_by_variant(rows)
    for variant, vm in metrics.items():
        print(f"\n--- {variant} ---")
        for metric, md in vm.items():
            print(f"  {metric}: n={md['count']}, mean={md['mean']:.4f}, std={md['std']:.4f}")

    report = generate_ab_report(exp_name)
    for comp in report.get("comparisons", []):
        print(f"\n{comp['comparison']}:")
        print(f"  t={comp['t_statistic']}, p={comp['p_value']}, d={comp['cohens_d']}, sig={comp['is_significant']}")
