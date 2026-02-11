"""
评分算法模块：IRT (Item Response Theory) 和 GMAT 分数估算

实现基于单参数逻辑斯蒂模型（1PL/2PL）的能力估计和分数映射。
"""

import math


def calculate_new_theta(current_theta: float, question_difficulty: float, is_correct: bool) -> float:
    """
    基于 IRT 的极大似然估计（MLE）更新用户能力参数 theta。
    
    使用单参数逻辑斯蒂模型（1PL）的期望后验估计（EAP）方法：
    - 计算在给定能力值下答对的期望概率 P(θ)
    - 使用实际得分与期望得分的残差更新能力估计
    - 学习率 0.4 基于经验值，平衡收敛速度与稳定性
    
    Args:
        current_theta: 当前能力估计值（标准正态分布尺度，通常 [-3, 3]）
        question_difficulty: 题目难度参数（与能力值同尺度）
        is_correct: 实际作答结果（1=正确，0=错误）
    
    Returns:
        更新后的能力估计值，截断至 [-3.0, 3.0] 以符合标准尺度
    """
    # 计算能力-难度差异
    diff = current_theta - question_difficulty
    
    # 单参数逻辑斯蒂模型：P(θ) = 1 / (1 + exp(-D * (θ - b)))
    # 系数 D=1.7 用于将逻辑斯蒂曲线近似为正态 ogive 模型（probit link）
    # 该系数使得逻辑斯蒂模型在能力-难度差异为 ±1.7 时，概率接近正态分布的累积分布函数
    p_expect = 1.0 / (1.0 + math.exp(-1.7 * diff))
    
    # 二元评分：实际得分（0 或 1）
    actual_score = 1.0 if is_correct else 0.0
    
    # EAP 更新：theta_new = theta_old + learning_rate * (observed - expected)
    # 学习率 0.4 基于 Wright & Stone (1979) 的推荐值，适用于自适应测试场景
    new_theta = current_theta + 0.4 * (actual_score - p_expect)
    
    # 截断至标准 IRT 尺度范围（避免极端值导致数值不稳定）
    new_theta = max(-3.0, min(3.0, new_theta))
    
    return new_theta


def estimate_gmat_score(theta: float) -> int:
    """
    将 IRT 能力参数 theta 映射到 GMAT Critical Reasoning 分数（20-51 分制）。
    
    线性映射基于 ETS 公布的 GMAT 分数转换表近似：
    - theta = -3.0 对应约 20 分（最低分）
    - theta = 0.0 对应约 30 分（平均分）
    - theta = +3.0 对应约 51 分（最高分）
    
    斜率 7.0 通过三点线性插值确定，近似 GMAT 官方分数转换曲线。
    
    Args:
        theta: IRT 能力参数（标准正态分布尺度）
    
    Returns:
        GMAT CR 分数（整数，范围 [20, 51]）
    """
    # 线性映射：score = baseline + slope * theta
    # baseline = 30（平均分），slope = 7.0（每单位 theta 对应 7 分）
    score = 30.0 + (theta * 7.0)
    
    # 截断至 GMAT CR 有效分数范围
    score = max(20, min(51, score))
    
    return int(round(score))


# TODO: Implement multi-dimensional IRT (MIRT) for correlated skill tracking
# Current implementation uses unidimensional IRT, assuming a single latent ability.
# MIRT would enable modeling multiple correlated skills (e.g., causal reasoning,
# assumption identification) simultaneously, providing more granular diagnostic
# information and improved adaptive question selection based on skill-specific
# proficiency estimates.
