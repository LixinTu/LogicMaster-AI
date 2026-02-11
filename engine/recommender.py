"""
推荐算法模块：BKT (Bayesian Knowledge Tracing) 和自适应题目推荐
"""

import uuid
from typing import Dict, List, Any, Optional
from utils.db_handler import DatabaseManager


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
    db_manager: Optional[DatabaseManager] = None
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
        
        # 加权排序
        scored_candidates: List[tuple] = []
        for candidate in filtered_candidates:
            score: float = 0.0
            
            # 基础分 = 1.0 - abs(题目难度 - 用户Theta)
            candidate_elo: float = candidate.get("elo_difficulty", 1500.0)
            # 将 elo_difficulty 转换为 theta: theta = (elo - 1500) / 100
            candidate_theta: float = (candidate_elo - 1500.0) / 100.0
            difficulty_diff: float = abs(candidate_theta - user_theta)
            base_score: float = 1.0 - difficulty_diff
            score += base_score
            
            # 加分项：如果题目含短板技能，+0.5
            candidate_skills: List[str] = candidate.get("skills", [])
            if isinstance(candidate_skills, list):
                for skill in candidate_skills:
                    if skill in weak_skills:
                        score += 0.5
                        break  # 每个技能只加一次分
            
            scored_candidates.append((candidate, score))
        
        # 决策：选最高分题目
        if not scored_candidates:
            return None
        
        # 按分数排序，选择最高分
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        next_question: Dict[str, Any]
        best_score: float
        next_question, best_score = scored_candidates[0]
        
        # 使用数据库中的 question_id
        question_id: str = next_question.get("id", "")
        if not question_id:
            question_id = str(uuid.uuid4())[:8]
        
        # 创建完整题目快照（current_q）
        # 确保所有必要字段都存在（包括技能标签相关字段）
        # 保留与旧代码完全兼容的字典结构
        current_q: Dict[str, Any] = {
            "question_id": question_id,
            "difficulty": next_question.get("difficulty", "medium"),
            "question_type": next_question.get("question_type", "Weaken"),
            "stimulus": next_question.get("stimulus", ""),
            "question": next_question.get("question", ""),
            "choices": next_question.get("choices", []),
            "correct": next_question.get("correct", ""),
            "correct_choice": next_question.get("correct", ""),  # 兼容字段
            "explanation": next_question.get("explanation", ""),  # 基础解析，后续会升级
            "tags": [],  # 可选标签
            # 技能标签相关字段（确保存在）
            "skills": next_question.get("skills", []),
            "label_source": next_question.get("label_source", "Unknown"),
            "skills_rationale": next_question.get("skills_rationale", ""),
            # 预生成的详细解析和诊断（从数据库读取）
            "detailed_explanation": next_question.get("detailed_explanation", ""),
            "diagnoses": next_question.get("diagnoses", {}),
            # 添加 elo_difficulty 用于后续 theta 更新
            "elo_difficulty": next_question.get("elo_difficulty", 1500.0)
        }
        
        # 更新锁题状态（只有在 phase == "finished" 时才允许覆盖）
        session_state.current_q = current_q
        session_state.current_q_id = question_id
        session_state.current_question = current_q  # 兼容旧代码
        
        # 重置状态（保留原有逻辑）
        session_state.radio_key += 1  # 增加 radio_key 以重置 radio widget
        session_state.attempt = 0
        session_state.phase = "answering"
        session_state.last_feedback = ""
        session_state.show_explanation = False
        session_state.pending_next_question = False
        session_state.socratic_context = {}  # 清空苏格拉底上下文
        session_state.chat_history = []  # 清空聊天历史
        # 注意：selected_choice 由 radio widget 自动管理，不需要手动重置
        
        return current_q
        
    except Exception as e:
        print(f"❌ 读取题目失败：{e}")
        return None
