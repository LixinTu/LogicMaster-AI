"""
评分算法模块：IRT (Item Response Theory) 和 GMAT 分数估算

实现三参数逻辑斯蒂模型（3PL）的能力估计、分数映射、信息函数和参数校准。
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 3PL 概率函数
# ---------------------------------------------------------------------------

def probability_3pl(
    theta: float,
    b: float,
    a: float = 1.0,
    c: float = 0.2,
) -> float:
    """
    三参数逻辑斯蒂模型（3PL）的正确回答概率。

    P(θ) = c + (1 - c) / (1 + exp(-a * (θ - b)))

    Args:
        theta: 用户能力值
        b: 题目难度参数（theta 尺度）
        a: 区分度参数（默认 1.0，有效范围 0.5–2.5）
        c: 猜测参数（默认 0.2，5 选 1 GMAT 题目的基线概率）

    Returns:
        正确回答概率，范围 [c, 1.0]
    """
    exponent = -a * (theta - b)
    # 防止溢出
    if exponent > 700:
        return c
    if exponent < -700:
        return 1.0
    return c + (1.0 - c) / (1.0 + math.exp(exponent))


# ---------------------------------------------------------------------------
# Theta 更新
# ---------------------------------------------------------------------------

def calculate_new_theta(
    current_theta: float,
    question_difficulty: float,
    is_correct: bool,
    discrimination: float = 1.0,
    guessing: float = 0.2,
) -> float:
    """
    基于 3PL IRT 模型更新用户能力参数 theta。

    使用期望后验估计（EAP）方法：
    - 计算 3PL 概率 P(θ)
    - theta_new = theta_old + lr * (observed - P(θ))

    Args:
        current_theta: 当前能力估计值（[-3, 3]）
        question_difficulty: 题目难度参数（theta 尺度）
        is_correct: 是否答对
        discrimination: 区分度 a（默认 1.0）
        guessing: 猜测参数 c（默认 0.2）

    Returns:
        更新后的能力估计值，截断至 [-3.0, 3.0]
    """
    p_expect = probability_3pl(
        theta=current_theta,
        b=question_difficulty,
        a=discrimination,
        c=guessing,
    )

    actual_score = 1.0 if is_correct else 0.0

    # EAP 更新：学习率 0.4
    new_theta = current_theta + 0.4 * (actual_score - p_expect)

    return max(-3.0, min(3.0, new_theta))


# ---------------------------------------------------------------------------
# GMAT 分数映射
# ---------------------------------------------------------------------------

def estimate_gmat_score(theta: float) -> int:
    """
    将 IRT 能力参数 theta 映射到 GMAT Critical Reasoning 分数（20–51）。

    线性映射：score = 30 + 7 * theta

    Args:
        theta: IRT 能力参数

    Returns:
        GMAT CR 分数（整数，[20, 51]）
    """
    score = 30.0 + (theta * 7.0)
    score = max(20, min(51, score))
    return int(round(score))


# ---------------------------------------------------------------------------
# 信息函数
# ---------------------------------------------------------------------------

def item_information(
    theta: float,
    b: float,
    a: float = 1.0,
    c: float = 0.2,
) -> float:
    """
    3PL 信息函数：衡量题目在给定能力水平下的信息量。

    I(θ) = a² * (P - c)² * (1 - P) / ((1 - c)² * P)

    信息值越高，该题目对估计该能力水平的考生越有区分价值。
    用于自适应测试中的最优题目选择。

    Args:
        theta: 能力值
        b: 难度参数
        a: 区分度参数
        c: 猜测参数

    Returns:
        信息量（非负浮点数）
    """
    p = probability_3pl(theta, b, a, c)
    # 避免除以零
    if p <= c or p >= 1.0:
        return 0.0
    numerator = (a ** 2) * ((p - c) ** 2) * (1.0 - p)
    denominator = ((1.0 - c) ** 2) * p
    return numerator / denominator


# ---------------------------------------------------------------------------
# 参数校准（MLE）
# ---------------------------------------------------------------------------

def calibrate_item_parameters(
    response_history: List[Dict[str, Any]],
    initial_a: float = 1.0,
    initial_b: float = 0.0,
    initial_c: float = 0.2,
) -> Dict[str, float]:
    """
    使用极大似然估计（MLE）从学生作答数据校准单个题目的 3PL 参数。

    通过 scipy.optimize.minimize 最小化负对数似然函数。

    Args:
        response_history: 作答记录列表，每条记录包含：
            - theta (float): 作答时的学生能力值
            - is_correct (bool): 是否答对
        initial_a: 区分度初始值
        initial_b: 难度初始值
        initial_c: 猜测参数初始值

    Returns:
        校准后的参数字典：{"a": ..., "b": ..., "c": ..., "converged": bool}
    """
    from scipy.optimize import minimize

    if len(response_history) < 5:
        return {
            "a": initial_a,
            "b": initial_b,
            "c": initial_c,
            "converged": False,
        }

    thetas = [r["theta"] for r in response_history]
    responses = [1.0 if r["is_correct"] else 0.0 for r in response_history]

    def neg_log_likelihood(params: List[float]) -> float:
        a_val, b_val, c_val = params
        # 参数边界惩罚
        if a_val <= 0.01 or c_val < 0.0 or c_val >= 1.0:
            return 1e12
        nll = 0.0
        for th, y in zip(thetas, responses):
            p = probability_3pl(th, b_val, a_val, c_val)
            # 钳制概率避免 log(0)
            p = max(1e-10, min(1.0 - 1e-10, p))
            nll -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
        return nll

    result = minimize(
        neg_log_likelihood,
        x0=[initial_a, initial_b, initial_c],
        method="L-BFGS-B",
        bounds=[(0.5, 2.5), (-3.0, 3.0), (0.0, 0.35)],
    )

    a_est, b_est, c_est = result.x
    return {
        "a": round(float(a_est), 4),
        "b": round(float(b_est), 4),
        "c": round(float(c_est), 4),
        "converged": bool(result.success),
    }
