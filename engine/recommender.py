"""
推荐算法模块：BKT (Bayesian Knowledge Tracing) 和自适应题目推荐

支持两种最终选择策略：
- legacy: 加权排序（基础分 + 技能加分）
- bandit: Thompson Sampling（explore/exploit 平衡）

间隔重复注入：
- 答过的题目若回忆概率 < 0.5，以 40% 概率插入复习题
"""

import random
import uuid
from typing import Dict, List, Any, Optional
from utils.db_handler import DatabaseManager
from engine.bandit_selector import get_bandit_selector
from engine.spaced_repetition import get_spaced_repetition_model


def _build_question_snapshot(
    next_question: Dict[str, Any],
    question_id: str,
    session_state: Any,
) -> Dict[str, Any]:
    """
    从候选题目构建完整题目快照，并更新 session_state。

    内部辅助函数，供 generate_next_question 的各条选择路径复用。
    """
    current_q: Dict[str, Any] = {
        "question_id": question_id,
        "difficulty": next_question.get("difficulty", "medium"),
        "question_type": next_question.get("question_type", "Weaken"),
        "stimulus": next_question.get("stimulus", ""),
        "question": next_question.get("question", ""),
        "choices": next_question.get("choices", []),
        "correct": next_question.get("correct", ""),
        "correct_choice": next_question.get("correct", ""),
        "explanation": next_question.get("explanation", ""),
        "tags": [],
        "skills": next_question.get("skills", []),
        "label_source": next_question.get("label_source", "Unknown"),
        "skills_rationale": next_question.get("skills_rationale", ""),
        "detailed_explanation": next_question.get("detailed_explanation", ""),
        "diagnoses": next_question.get("diagnoses", {}),
        "elo_difficulty": next_question.get("elo_difficulty", 1500.0),
    }

    session_state.current_q = current_q
    session_state.current_q_id = question_id
    session_state.current_question = current_q
    session_state.radio_key += 1
    session_state.attempt = 0
    session_state.phase = "answering"
    session_state.last_feedback = ""
    session_state.show_explanation = False
    session_state.pending_next_question = False
    session_state.socratic_context = {}
    session_state.chat_history = []

    return current_q


def analyze_weak_skills(questions_log: List[Dict[str, Any]]) -> List[str]:
    """
    分析用户的弱项技能（BKT 分析）
    
    统计每个技能的错误率，返回错误率最高的 3 个技能
    
    Args:
        questions_log: 题目日志列表，每个元素包含 skills 和 is_correct 字段
    
    Returns:
        弱项技能列表，包含错误率最高的 3 个技能名称（按错误率降序）
    """
    try:
        if not questions_log:
            return []  # 没有历史记录，返回空列表
        
        # 统计每个技能的正确率
        skill_stats: Dict[str, Dict[str, int]] = {}  # {skill: {"correct": count, "total": count}}
        
        for log in questions_log:
            skills = log.get("skills", [])
            is_correct = log.get("is_correct", False)
            
            if not isinstance(skills, list):
                continue
            
            # 统计每个技能
            for skill in skills:
                if skill not in skill_stats:
                    skill_stats[skill] = {"correct": 0, "total": 0}
                
                skill_stats[skill]["total"] += 1
                if is_correct:
                    skill_stats[skill]["correct"] += 1
        
        # 计算每个技能的错误率，找出错误率最高的 3 个
        skill_error_rates = []
        for skill, stats in skill_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            if total > 0:
                error_rate = 1.0 - (correct / total)  # 错误率 = 1 - 正确率
                skill_error_rates.append((skill, error_rate, total))
        
        # 按错误率降序排序，取前 3 个
        skill_error_rates.sort(key=lambda x: x[1], reverse=True)
        weak_skills = [skill for skill, _, _ in skill_error_rates[:3]]
        
        return weak_skills
        
    except Exception as e:
        print(f"分析弱项技能失败：{e}")
        return []


def generate_next_question(
    user_theta: float,
    current_q_id: str,
    questions_log: List[Dict[str, Any]],
    session_state: Any,
    history_limit: int = 10,
    db_manager: Optional[DatabaseManager] = None,
    use_bandit: bool = True,
    use_spaced_repetition: bool = True,
    use_dkt: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    使用 IRT + BKT 混合推荐算法选择下一题
    
    算法流程：
    1. 获取候选：调用 get_adaptive_candidates(user_theta)
    2. 计算短板：遍历 questions_log，找出错误率最高的 3 个技能
    3. 过滤历史：排除最近 history_limit 道题中已做过的题目 ID
    4. 加权排序：基础分 + 技能加分
    5. 决策：选最高分题目，返回题目字典
    
    Args:
        user_theta: 用户当前能力值（theta）
        current_q_id: 当前题目 ID（排除）
        questions_log: 题目日志列表
        session_state: Streamlit session_state 对象（用于更新状态）
        history_limit: 历史记录过滤限制（默认 10，即最近 10 道题）
    
    Returns:
        题目字典，包含所有必要字段。如果失败返回 None。
    """
    try:
        # 获取最近 history_limit 道题的 ID 列表（用于过滤重复）
        history_ids: List[str] = []
        if questions_log:
            # 从后往前取最近 history_limit 道题的 ID
            recent_logs: List[Dict[str, Any]] = questions_log[-history_limit:]
            history_ids = [log.get("question_id", "") for log in recent_logs if log.get("question_id")]
        
        # 合并当前题目 ID 和历史 ID
        exclude_ids: set = set([current_q_id] + history_ids)
        
        # 获取候选：调用 DatabaseManager.get_adaptive_candidates(user_theta, exclude_id, limit=20)
        # 注意：get_adaptive_candidates 只支持单个 exclude_id，所以我们需要在后续过滤中处理多个 ID
        # 增加 limit 到 20，以便在过滤历史题目后仍有足够候选
        if db_manager is None:
            from utils.db_handler import get_db_manager
            db_manager = get_db_manager()
        
        candidates = db_manager.get_adaptive_candidates(
            target_difficulty=user_theta, 
            exclude_id=current_q_id, 
            limit=20
        )
        
        if not candidates:
            print(f"❌ 数据库中暂无合适的候选题目（target_difficulty={user_theta:.2f}）")
            return None  # 读取失败，返回 None
        
        # 过滤历史题目：排除最近 history_limit 道题中已做过的题目 ID
        filtered_candidates: List[Dict[str, Any]] = [
            c for c in candidates 
            if c.get("id", "") not in exclude_ids
        ]
        
        # 如果过滤后没有候选，使用原始候选列表（至少保证有题目可选）
        if not filtered_candidates:
            filtered_candidates = candidates
        
        # 计算短板：遍历 questions_log，找出错误率最高的 3 个技能
        weak_skills: List[str] = analyze_weak_skills(questions_log)

        # DKT 技能掌握度（可选）
        dkt_mastery: Optional[Dict[str, float]] = None
        if use_dkt:
            try:
                from engine.dkt_model import get_dkt_model
                dkt_model = get_dkt_model()
                dkt_mastery = dkt_model.predict_mastery(questions_log)
            except Exception:
                dkt_mastery = None  # DKT 失败时降级到 BKT

        # 弱项技能加分：优先包含短板技能的候选题目
        scored_candidates: List[tuple] = []
        for candidate in filtered_candidates:
            score: float = 0.0

            # 基础分 = 1.0 - abs(题目难度 - 用户Theta)
            candidate_elo: float = candidate.get("elo_difficulty", 1500.0)
            candidate_theta: float = (candidate_elo - 1500.0) / 100.0
            difficulty_diff: float = abs(candidate_theta - user_theta)
            base_score: float = 1.0 - difficulty_diff
            score += base_score

            candidate_skills: List[str] = candidate.get("skills", [])
            if isinstance(candidate_skills, list):
                if dkt_mastery is not None:
                    # DKT：连续技能加分 = (1.0 - mastery) * 0.5
                    for skill in candidate_skills:
                        mastery = dkt_mastery.get(skill, 0.5)
                        score += (1.0 - mastery) * 0.5
                else:
                    # BKT 回退：如果题目含短板技能，+0.5
                    for skill in candidate_skills:
                        if skill in weak_skills:
                            score += 0.5
                            break

            scored_candidates.append((candidate, score))

        if not scored_candidates:
            return None

        # ------ 间隔重复注入 ------
        # 如果有需要复习的题目（recall_prob < 0.5），以 40% 概率插入复习题
        if use_spaced_repetition:
            try:
                sr_model = get_spaced_repetition_model()
                review_candidates = sr_model.get_review_candidates(threshold=0.5)
                if review_candidates:
                    # 筛选：复习题必须在当前候选列表中
                    candidate_ids = {c.get("id", "") for c in filtered_candidates}
                    matched_reviews = [
                        r for r in review_candidates
                        if r["question_id"] in candidate_ids
                    ]
                    if matched_reviews and random.random() < 0.4:
                        # 选回忆概率最低的复习题
                        review_q_id = matched_reviews[0]["question_id"]
                        for c in filtered_candidates:
                            if c.get("id", "") == review_q_id:
                                next_question = c
                                # 跳过 bandit/legacy 选择，直接用复习题
                                # 跳转到构建 current_q 的代码
                                question_id = next_question.get("id", "")
                                if not question_id:
                                    question_id = str(uuid.uuid4())[:8]
                                return _build_question_snapshot(
                                    next_question, question_id, session_state
                                )
            except Exception:
                pass  # 间隔重复失败时静默降级

        # ------ 最终选择策略 ------
        if use_bandit and len(filtered_candidates) > 1:
            # Thompson Sampling：用 BKT 过滤后的候选通过 bandit 做最终选择
            bandit = get_bandit_selector()
            next_question = bandit.select_question(
                theta=user_theta,
                candidates=filtered_candidates,
                explore_weight=0.3,
            )
            if next_question is None:
                # 回退到加权排序
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                next_question = scored_candidates[0][0]
        else:
            # legacy：纯加权排序
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            next_question = scored_candidates[0][0]
        
        # 使用数据库中的 question_id
        question_id: str = next_question.get("id", "")
        if not question_id:
            question_id = str(uuid.uuid4())[:8]

        return _build_question_snapshot(next_question, question_id, session_state)
        
    except Exception as e:
        print(f"❌ 读取题目失败：{e}")
        return None
