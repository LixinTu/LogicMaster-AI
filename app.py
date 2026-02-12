import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import requests as http_requests  # 避免与 FastAPI 的 Request 冲突
from llm_service import generate_question, generate_detailed_explanation, RULE_SKILL_POOL_BY_TYPE
from utils.db_handler import DatabaseManager, get_db_manager
from engine.recommender import analyze_weak_skills
import uuid
import random

# FastAPI 后端地址
API_BASE_URL = "http://localhost:8000"


def _log_ab_outcome(user_id: str, variant: str, metric: str, value: float, metadata: dict = None):
    """Week 4: 向 /api/analytics/log-outcome 提交 A/B 实验结果（fire-and-forget）"""
    try:
        http_requests.post(
            f"{API_BASE_URL}/api/analytics/log-outcome",
            json={
                "user_id": user_id,
                "experiment_name": "tutor_strategy",
                "variant": variant or "socratic_standard",
                "metric": metric,
                "value": value,
                "metadata": metadata,
            },
            timeout=3,
        )
    except Exception:
        pass  # fire-and-forget

# 页面配置
st.set_page_config(page_title="MathQuest Labs — LogicMaster", layout="wide")

# 侧边栏：LLM Configuration
with st.sidebar:
    st.header("LLM Configuration")
    st.text_input(
        "DeepSeek API Key",
        type="password",
        key="DEEPSEEK_API_KEY"
    )
    
    # 读取并显示 API Key 状态
    api_key = st.session_state.get("DEEPSEEK_API_KEY", "").strip()
    if api_key:
        st.success("API Key loaded")

    # API 状态指示器
    st.divider()
    try:
        _health = http_requests.get(f"{API_BASE_URL}/health", timeout=2)
        if _health.ok:
            st.success(f"API Online ({API_BASE_URL})")
        else:
            st.error("API Error")
    except Exception:
        st.warning(f"API Offline ({API_BASE_URL})")
    else:
        st.warning("No API Key")

    # Week 4: Session Analytics
    st.divider()
    st.subheader("Session Stats")
    _s_attempt = st.session_state.get("attempt_count", 0)
    _s_correct = st.session_state.get("correct_count", 0)
    _s_accuracy = (_s_correct / _s_attempt * 100) if _s_attempt > 0 else 0
    _s_theta = st.session_state.get("user_theta", 0.0)
    _s_variant = st.session_state.get("ab_variant")
    col_a, col_b = st.columns(2)
    col_a.metric("Questions", _s_attempt)
    col_b.metric("Accuracy", f"{_s_accuracy:.0f}%")
    st.metric("Current Theta", f"{_s_theta:.2f}")
    if _s_variant:
        st.caption(f"A/B Variant: `{_s_variant}`")

# 初始化 session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 注意：assessor_result 已移除，现在使用 IRT + BKT 驱动的仪表盘

if "score_history" not in st.session_state:
    st.session_state.score_history = []

# 初始化 IRT/Theta 相关状态
if "user_theta" not in st.session_state:
    st.session_state.user_theta = 0.0

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "theta_history" not in st.session_state:
    st.session_state.theta_history = [0.0]

# 初始化锁题机制状态（冷启动优化）
if "current_q" not in st.session_state:
    first_q = None
    
    # 优先从数据库读取题目初始化（带错误处理）
    try:
        db_manager = get_db_manager()
        candidates = db_manager.get_adaptive_candidates(target_difficulty=0.0, limit=1)
        first_q = candidates[0] if candidates and len(candidates) > 0 else None
    except Exception as e:
        # 数据库查询失败，使用默认题目
        print(f"从数据库获取初始题目失败：{e}，使用默认题目")
        first_q = None
    
    if first_q:
        # 成功从数据库读取题目，使用数据库题目初始化
        try:
            # 使用数据库中的 question_id，如果没有则生成一个
            question_id = first_q.get("id")
            if not question_id:
                question_id = str(uuid.uuid4())[:8]
            
            # 构建 current_q 字典（与 _generate_next_question 逻辑保持一致）
            st.session_state.current_q = {
                "question_id": question_id,
                "difficulty": first_q.get("difficulty", "medium"),
                "question_type": first_q.get("question_type", "Weaken"),
                "stimulus": first_q.get("stimulus", ""),
                "question": first_q.get("question", ""),
                "choices": first_q.get("choices", []),
                "correct": first_q.get("correct", ""),
                "correct_choice": first_q.get("correct", ""),  # 兼容字段
                "explanation": first_q.get("explanation", ""),  # 基础解析，后续会升级
                "tags": [],  # 可选标签
                # 技能标签相关字段（确保存在）
                "skills": first_q.get("skills", []),
                "label_source": first_q.get("label_source", "Unknown"),
                "skills_rationale": first_q.get("skills_rationale", ""),
                # 预生成的详细解析和诊断（从数据库读取）
                "detailed_explanation": first_q.get("detailed_explanation", ""),
                "diagnoses": first_q.get("diagnoses", {}),
                # 添加 elo_difficulty 用于后续 theta 更新
                "elo_difficulty": first_q.get("elo_difficulty", 1500.0)
            }
            st.session_state.current_q_id = question_id
            st.session_state.current_question = st.session_state.current_q  # 兼容旧代码
        except Exception as e:
            # 解析数据库题目失败，降级到默认题目
            print(f"解析数据库题目失败：{e}，使用默认题目")
            first_q = None
    
    if not first_q:
        # 数据库为空或读取失败，使用默认题目作为 fallback（冷启动）
        # 只在第一次显示警告，避免重复提示
        if not st.session_state.get("_cold_start_warning_shown", False):
            st.info("ℹ️ **Cold start mode**: Database is empty or has no questions. Using a default question for demo.\n\n"
                   "💡 **Tip**: Run `python generate_pool.py` to generate questions; the app will then use the database.")
            st.session_state._cold_start_warning_shown = True
        
        initial_q_id = str(uuid.uuid4())[:8]
        st.session_state.current_q = {
            "question_id": initial_q_id,
            "difficulty": "medium",
            "question_type": "Weaken",
            "stimulus": "A company plans to launch a new product. Supporters believe it will significantly increase market share. However, competitors are developing similar products, and market research shows limited consumer demand for the new features.",
            "question": "Which of the following most weakens the supporters' argument?",
            "choices": [
                "A. The new product has high development costs",
                "B. The market is highly competitive, making it hard for new products to stand out",
                "C. Consumers have limited interest in the new features",
                "D. The company lacks experience in promoting new products",
                "E. The new product's technology is not yet mature"
            ],
            "correct": "C",
            "correct_choice": "C",
            "explanation": "C directly points to limited consumer demand, weakening the market-share assumption",
            "tags": [],
            "skills": ["Causal Reasoning", "Alternative Explanation"],
            "label_source": "fallback_rule",  # 初始题目使用规则回退
            "skills_rationale": "Initial question with rule-based default skills.",
            # 预生成的详细解析和诊断（默认题目没有，使用空值）
            "detailed_explanation": "",
            "diagnoses": {},
            # 添加 elo_difficulty 用于后续 theta 更新
            "elo_difficulty": 1500.0
        }
        st.session_state.current_q_id = initial_q_id
        st.session_state.current_question = st.session_state.current_q  # 兼容旧代码

if "current_q_id" not in st.session_state:
    st.session_state.current_q_id = st.session_state.current_q.get("question_id", "")

if "socratic_context" not in st.session_state:
    st.session_state.socratic_context = {}

# Week 4: 用户标识（A/B 分组用）
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "ab_variant" not in st.session_state:
    st.session_state.ab_variant = None

# Week 3: LangChain Agent 对话状态
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "tutor_hint_count" not in st.session_state:
    st.session_state.tutor_hint_count = 0

if "tutor_understanding" not in st.session_state:
    st.session_state.tutor_understanding = "confused"

if "show_answer" not in st.session_state:
    st.session_state.show_answer = False

if "radio_key" not in st.session_state:
    st.session_state.radio_key = 0

# 初始化题库缓存和正确性评分
if "question_bank" not in st.session_state:
    st.session_state.question_bank = {
        "easy": [],
        "medium": [],
        "hard": []
    }

if "attempt_count" not in st.session_state:
    st.session_state.attempt_count = 0

if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0

if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []

# 初始化题目标签历史记录（用于统计）
if "questions_log" not in st.session_state:
    st.session_state.questions_log = []  # 存储已完成的题目的标签信息

if "last_answer_result" not in st.session_state:
    st.session_state.last_answer_result = ""

if "last_correct_choice" not in st.session_state:
    st.session_state.last_correct_choice = ""

if "last_user_choice" not in st.session_state:
    st.session_state.last_user_choice = ""

if "show_correctness" not in st.session_state:
    st.session_state.show_correctness = False

# 初始化作答状态管理
# attempt: 0=未作答, 1=第1次作答, 2=第2次作答
if "attempt" not in st.session_state:
    st.session_state.attempt = 0

# phase: "answering"=可作答, "remediation"=苏格拉底问答, "finished"=题目结束
if "phase" not in st.session_state:
    st.session_state.phase = "answering"

if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = ""

if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

if "pending_next_question" not in st.session_state:
    st.session_state.pending_next_question = False

# 注意：pending_next_question 标志在提交答案时设置，在显示解析后通过延迟自动生成下一题

# 两栏布局
col1, col2 = st.columns([0.7, 0.3])

# 左侧聊天区（70%）
with col1:
    st.header("💬 Logical Reasoning")
    
    # 显示当前题目（使用锁题机制：current_q）
    current_q = st.session_state.get("current_q", {})
    current_q_id = st.session_state.get("current_q_id", "")
    
    if current_q:
        # 确保 current_q 中存在所有必要字段（安全读取，避免崩溃）
        question_id = current_q.get("question_id", current_q_id or "N/A")
        difficulty = current_q.get("difficulty", "medium")
        question_type = current_q.get("question_type", "Weaken")
        skills = current_q.get("skills", [])
        label_source = current_q.get("label_source", "Unknown")
        skills_rationale = current_q.get("skills_rationale", "")
        
        # 确保 skills 是列表
        if not isinstance(skills, list):
            skills = []
        
        st.divider()
        st.subheader("📝 Current Question")
        
        phase = st.session_state.get("phase", "answering")
        if phase == "remediation":
            st.caption(f"Question ID: {question_id} (locked — Socratic dialogue applies to this question)")
        
        # 学生可见标签条（题干上方）
        skills_str = ", ".join(skills) if skills else "N/A"
        st.caption(f"**Type:** {question_type} | **Difficulty:** {difficulty} | **Skills:** {skills_str}")
        
        st.markdown(f"**Stimulus:** {current_q.get('stimulus', '')}")
        st.markdown(f"**Question:** {current_q.get('question', '')}")
        
        # 获取当前状态
        attempt = st.session_state.get("attempt", 0)
        phase = st.session_state.get("phase", "answering")
        
        # 判断是否可以作答：phase为"answering"或"remediation"，且attempt < 2
        can_submit = (phase == "answering" or phase == "remediation") and attempt < 2
        
        # 显示选项（只显示 A-E 字母）
        # 注意：使用动态 key 以支持重置，只读，不手动赋值
        choice_options = ["A", "B", "C", "D", "E"]
        selected_choice = st.radio(
            "Select your answer:",
            options=choice_options,
            key=f"selected_choice_{st.session_state.radio_key}",
            label_visibility="visible",
            disabled=not can_submit
        )
        
        # 显示选项内容（锁定显示，使用 current_q）
        choices = current_q.get("choices", [])
        if choices:
            st.markdown("**Choices:**")
            for choice in choices:
                st.markdown(f"- {choice}")
        
        # 显示反馈（在 radio 下方）
        last_feedback = st.session_state.get("last_feedback", "")
        if last_feedback:
            if "Correct" in last_feedback:
                st.success(last_feedback)
            elif "Incorrect" in last_feedback:
                st.error(last_feedback)
        
        # 显示解析（根据规则：第1次答对或第2次答完时显示）
        # 优先调用 RAG API 生成增强解析
        if st.session_state.get("show_explanation", False):
            correct_choice = current_q.get("correct_choice") or current_q.get("correct", "")

            # 尝试从 session_state 获取已缓存的 RAG 结果（避免重复调用）
            rag_result = st.session_state.get("_rag_explanation_result")
            if rag_result is None:
                # 调用 RAG API
                try:
                    rag_resp = http_requests.post(
                        f"{API_BASE_URL}/api/explanations/generate-with-rag",
                        json={
                            "question_id": current_q.get("question_id", ""),
                            "question": current_q,
                            "user_choice": st.session_state.get("last_user_choice", ""),
                            "is_correct": "Correct" in st.session_state.get("last_feedback", ""),
                        },
                        timeout=30,
                    )
                    if rag_resp.ok:
                        rag_result = rag_resp.json()
                        st.session_state._rag_explanation_result = rag_result
                except Exception:
                    rag_result = None

            # 解析内容：优先 RAG 结果，fallback 到 current_q
            if rag_result and rag_result.get("explanation"):
                detailed_explanation = rag_result["explanation"]
                explanation_source = rag_result.get("source", "unknown")
                similar_refs = rag_result.get("similar_references", [])
            else:
                detailed_explanation = current_q.get("detailed_explanation", "") or current_q.get("explanation", "")
                explanation_source = "cached"
                similar_refs = []

            if detailed_explanation:
                st.divider()
                st.subheader("📖 Detailed Explanation")
                if phase == "finished" and attempt == 2 and last_feedback and "Incorrect" in last_feedback:
                    st.markdown(f"**Correct Answer: {correct_choice}**")
                st.caption(f"Source: `{explanation_source}`")
                st.markdown(detailed_explanation)

                # 显示相似题目参考
                if similar_refs:
                    with st.expander("📚 Similar Question References"):
                        for ref in similar_refs:
                            st.caption(f"Question {ref['question_id']} (similarity: {ref['similarity']:.0%})")
                
                # 显示 Next Question 按钮（只有在 finished 阶段才显示）
                if phase == "finished":
                    api_key = st.session_state.get("DEEPSEEK_API_KEY", "").strip()
                    if api_key:
                        if st.button("➡️ Next Question", type="primary", use_container_width=True):
                            # 调用新的推荐函数（带错误处理和冷启动支持）
                            try:
                                user_theta = st.session_state.get("user_theta", 0.0)
                                current_q_id = st.session_state.get("current_q_id", "")
                                questions_log = st.session_state.get("questions_log", [])
                                
                                # 调用 FastAPI 推荐端点
                                try:
                                    api_resp = http_requests.post(
                                        f"{API_BASE_URL}/api/questions/next",
                                        json={
                                            "user_theta": user_theta,
                                            "current_q_id": current_q_id,
                                            "questions_log": [
                                                {"question_id": log.get("question_id", ""),
                                                 "skills": log.get("skills", []),
                                                 "is_correct": log.get("is_correct", False)}
                                                for log in questions_log
                                            ],
                                        },
                                        timeout=10,
                                    )
                                    api_resp.raise_for_status()
                                    api_data = api_resp.json()
                                    # API 不返回 correct，需要从数据库补充完整题目信息
                                    db_manager = get_db_manager()
                                    full_candidates = db_manager.get_adaptive_candidates(
                                        target_difficulty=user_theta, exclude_id=current_q_id, limit=20
                                    )
                                    full_q = next((c for c in full_candidates if c.get("id") == api_data["question_id"]), None)
                                    if full_q:
                                        result = {
                                            "question_id": api_data["question_id"],
                                            "difficulty": api_data["difficulty"],
                                            "question_type": api_data["question_type"],
                                            "stimulus": api_data["stimulus"],
                                            "question": api_data["question"],
                                            "choices": api_data["choices"],
                                            "correct": full_q.get("correct", ""),
                                            "correct_choice": full_q.get("correct", ""),
                                            "explanation": full_q.get("explanation", ""),
                                            "tags": [],
                                            "skills": api_data.get("skills", []),
                                            "label_source": full_q.get("label_source", "Unknown"),
                                            "skills_rationale": full_q.get("skills_rationale", ""),
                                            "detailed_explanation": full_q.get("detailed_explanation", ""),
                                            "diagnoses": full_q.get("diagnoses", {}),
                                            "elo_difficulty": api_data.get("elo_difficulty", 1500.0),
                                        }
                                        # 更新 session_state（原来由 generate_next_question 内部完成）
                                        st.session_state.current_q = result
                                        st.session_state.current_q_id = result["question_id"]
                                        st.session_state.current_question = result
                                        st.session_state.radio_key += 1
                                        st.session_state.attempt = 0
                                        st.session_state.phase = "answering"
                                        st.session_state.last_feedback = ""
                                        st.session_state.show_explanation = False
                                        st.session_state.pending_next_question = False
                                        st.session_state.socratic_context = {}
                                        st.session_state.chat_history = []
                                        st.session_state._rag_explanation_result = None
                                        # Week 3: 清理对话状态
                                        st.session_state.conversation_id = None
                                        st.session_state.tutor_hint_count = 0
                                        st.session_state.tutor_understanding = "confused"
                                    else:
                                        result = None  # 数据库中找不到对应题目，走 fallback
                                except Exception:
                                    result = None  # API 调用失败，走 fallback
                                
                                if result is None:
                                    # 数据库为空或无可用题目，尝试从数据库获取一个默认题目
                                    try:
                                        db_manager = get_db_manager()
                                        fallback_candidates = db_manager.get_adaptive_candidates(target_difficulty=0.0, limit=1)
                                        if fallback_candidates and len(fallback_candidates) > 0:
                                            # 找到了备用题目，直接使用第一个
                                            fallback_q = fallback_candidates[0]
                                            question_id = fallback_q.get("id", str(uuid.uuid4())[:8])
                                            
                                            st.session_state.current_q = {
                                                "question_id": question_id,
                                                "difficulty": fallback_q.get("difficulty", "medium"),
                                                "question_type": fallback_q.get("question_type", "Weaken"),
                                                "stimulus": fallback_q.get("stimulus", ""),
                                                "question": fallback_q.get("question", ""),
                                                "choices": fallback_q.get("choices", []),
                                                "correct": fallback_q.get("correct", ""),
                                                "correct_choice": fallback_q.get("correct", ""),
                                                "explanation": fallback_q.get("explanation", ""),
                                                "tags": [],
                                                "skills": fallback_q.get("skills", []),
                                                "label_source": fallback_q.get("label_source", "Unknown"),
                                                "skills_rationale": fallback_q.get("skills_rationale", ""),
                                                "detailed_explanation": fallback_q.get("detailed_explanation", ""),
                                                "diagnoses": fallback_q.get("diagnoses", {}),
                                                "elo_difficulty": fallback_q.get("elo_difficulty", 1500.0)
                                            }
                                            st.session_state.current_q_id = question_id
                                            st.session_state.current_question = st.session_state.current_q
                                            st.session_state.radio_key += 1
                                            st.session_state.attempt = 0
                                            st.session_state.phase = "answering"
                                            st.session_state.last_feedback = ""
                                            st.session_state.show_explanation = False
                                            st.rerun()
                                        else:
                                            # 数据库为空，显示友好提示并保持当前题目
                                            st.warning("⚠️ No questions in database. Please run `python generate_pool.py` to generate questions.")
                                    except Exception as e:
                                        # 数据库查询失败，显示错误信息
                                        st.error(f"❌ Failed to load questions from database: {e}. Check database connection or run `python generate_pool.py`.")
                                else:
                                    # 成功获取新题目，刷新页面
                                    st.rerun()
                            except Exception as e:
                                # 生成题目时发生未知错误
                                st.error(f"❌ Error loading next question: {e}. Please refresh and try again.")
                                print(f"生成下一题时出错：{e}")
        
        # Submit Answer 按钮
        api_key = st.session_state.get("DEEPSEEK_API_KEY", "").strip()
        if api_key:
            if st.button("Submit Answer", type="primary", use_container_width=True, disabled=not can_submit):
                # 再次检查是否允许提交
                if not can_submit:
                    st.warning("Already submitted. Cannot resubmit.")
                    st.stop()
                
                # 判分逻辑（使用锁题机制：current_q）
                # 注意：只读取 selected_choice，不手动赋值
                user_choice = st.session_state.get(f"selected_choice_{st.session_state.radio_key}")
                if not user_choice:
                    st.warning("Please select an option (A-E) first.")
                else:
                    # 检查 API Key
                    if not api_key:
                        st.info("Enter DeepSeek API Key in the sidebar to enable AI dialogue.")
                        st.stop()
                    
                    # 获取当前题目（使用锁题机制）
                    current_q = st.session_state.get("current_q", {})
                    current_q_id = st.session_state.get("current_q_id", "")
                    
                    if not current_q:
                        st.error("Question data missing. Please refresh the page.")
                        st.stop()
                    
                    # 获取正确答案
                    correct_choice = current_q.get("correct_choice") or current_q.get("correct", "")
                    
                    # 获取当前状态
                    current_attempt = st.session_state.get("attempt", 0)
                    current_phase = st.session_state.get("phase", "answering")
                    
                    # 更新attempt（只在按钮点击事件里更新）
                    new_attempt = current_attempt + 1
                    st.session_state.attempt = new_attempt
                    
                    # 判断对错
                    is_correct = user_choice == correct_choice
                    
                    # === 第1次作答（attempt=1）===
                    if new_attempt == 1:
                        if is_correct:
                            # 第1次答对：显示Correct + 详细解析
                            st.session_state.last_feedback = "Correct ✅"
                            st.session_state.phase = "finished"
                            
                            # 直接从 current_q 读取预生成的详细解析（瞬间显示）
                            # 如果不存在，使用基础 explanation 作为备选
                            if not current_q.get("detailed_explanation"):
                                current_q["detailed_explanation"] = current_q.get("explanation", "")
                                st.session_state.current_q = current_q
                            
                            st.session_state.show_explanation = True
                            
                            # 更新答题统计
                            st.session_state.attempt_count += 1
                            st.session_state.correct_count += 1
                            
                            # 记录题目标签信息到 questions_log（用于统计和BKT分析）
                            # 强制记录：is_correct, user_theta, skills
                            try:
                                questions_log = st.session_state.get("questions_log", [])
                                current_q_id = current_q.get("question_id", "")
                                # 检查是否已记录（避免重复记录）
                                already_logged = any(log.get("question_id") == current_q_id for log in questions_log)
                                if not already_logged:
                                    current_theta = st.session_state.get("user_theta", 0.0)
                                    elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                    question_difficulty = (elo_difficulty - 1500.0) / 100.0  # 转换为 theta
                                    
                                    label_info = {
                                        "question_id": current_q_id,
                                        "question_type": current_q.get("question_type", "Weaken"),
                                        "skills": current_q.get("skills", []),  # 强制记录技能
                                        "label_source": current_q.get("label_source", "Unknown"),
                                        "skills_rationale": current_q.get("skills_rationale", ""),
                                        "is_correct": True,  # 强制记录正确性
                                        "user_theta": current_theta,  # 强制记录能力值
                                        "question_difficulty": question_difficulty  # 记录题目难度（用于后续 theta 更新）
                                    }
                                    questions_log.append(label_info)
                                    st.session_state.questions_log = questions_log
                                    # 只在成功记录 questions_log 时增加 question_count（避免重复）
                                    st.session_state.question_count = len(questions_log)
                            except Exception as e:
                                pass  # 记录失败不影响主流程
                            
                            # 清空聊天历史
                            st.session_state.chat_history = []

                            # 更新 theta（使用 IRT 算法）
                            old_theta = st.session_state.get("user_theta", 0.0)
                            try:
                                elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                question_difficulty = (elo_difficulty - 1500.0) / 100.0
                                theta_resp = http_requests.post(
                                    f"{API_BASE_URL}/api/theta/update",
                                    json={"current_theta": old_theta, "question_difficulty": question_difficulty, "is_correct": True},
                                    timeout=5,
                                )
                                new_theta = theta_resp.json()["new_theta"] if theta_resp.ok else old_theta
                                st.session_state.user_theta = new_theta
                                st.session_state.theta_history.append(new_theta)
                            except Exception as e:
                                new_theta = old_theta

                            # Week 4: 记录 A/B 实验结果
                            _log_ab_outcome(st.session_state.user_id, st.session_state.ab_variant, "is_correct", 1.0, {"question_id": current_q_id, "attempt": 1})
                            _log_ab_outcome(st.session_state.user_id, st.session_state.ab_variant, "theta_change", new_theta - old_theta, {"question_id": current_q_id})

                        else:
                            # 第1次答错：显示Incorrect，进入remediation
                            st.session_state.last_feedback = "Incorrect ❌"
                            st.session_state.phase = "remediation"
                            st.session_state.show_explanation = False  # 先不显示完整解析

                            # Week 3+4: 调用 /api/tutor/start-remediation（A/B 分组 + LangChain Agent 诊断 + 首条提示）
                            try:
                                with st.spinner("🤖 AI is analyzing your answer..."):
                                    rem_resp = http_requests.post(
                                        f"{API_BASE_URL}/api/tutor/start-remediation",
                                        json={
                                            "question_id": current_q_id,
                                            "question": current_q,
                                            "user_choice": user_choice,
                                            "correct_choice": correct_choice,
                                            "user_id": st.session_state.user_id,
                                        },
                                        timeout=30,
                                    )
                                    if rem_resp.ok:
                                        rem_data = rem_resp.json()
                                        st.session_state.conversation_id = rem_data["conversation_id"]
                                        st.session_state.tutor_hint_count = rem_data["hint_count"]
                                        st.session_state.tutor_understanding = rem_data["student_understanding"]
                                        st.session_state.ab_variant = rem_data.get("variant", "socratic_standard")
                                        st.session_state.socratic_context = {
                                            "logic_gap": rem_data["logic_gap"],
                                            "error_type": rem_data["error_type"],
                                        }
                                        # 同步聊天历史到前端显示
                                        st.session_state.chat_history = [
                                            {"role": "user", "content": f"I chose answer: {user_choice}"},
                                            {"role": "assistant", "content": rem_data["first_hint"]},
                                        ]
                                        # direct_explanation 变体：直接进入 finished
                                        if rem_data.get("current_state") == "concluded":
                                            st.session_state.phase = "finished"
                                            st.session_state.show_explanation = True
                                    else:
                                        raise Exception(f"API returned {rem_resp.status_code}")
                            except Exception as e:
                                # 降级：使用默认提示
                                st.session_state.conversation_id = None
                                st.session_state.tutor_hint_count = 1
                                st.session_state.tutor_understanding = "confused"
                                st.session_state.socratic_context = {
                                    "question_id": current_q_id,
                                    "correct_choice": correct_choice,
                                    "user_choice": user_choice,
                                }
                                st.session_state.chat_history = [
                                    {"role": "user", "content": f"I chose answer: {user_choice}"},
                                    {"role": "assistant", "content": "Let's take a step back. What is the main conclusion of the argument?"},
                                ]
                    
                    # === 第2次作答（attempt=2）===
                    elif new_attempt == 2:
                        st.session_state.phase = "finished"
                        
                        # 直接从 current_q 读取预生成的详细解析（瞬间显示）
                        # 如果不存在，使用基础 explanation 作为备选
                        if not current_q.get("detailed_explanation"):
                            current_q["detailed_explanation"] = current_q.get("explanation", "")
                            st.session_state.current_q = current_q
                        
                        st.session_state.show_explanation = True
                        
                        old_theta_2 = st.session_state.get("user_theta", 0.0)
                        if is_correct:
                            # 第2次答对：显示"Correct (after reasoning) ✅" + 解析
                            st.session_state.last_feedback = "Correct (after reasoning) ✅"

                            # 更新答题统计
                            st.session_state.attempt_count += 1
                            st.session_state.correct_count += 1

                            # 更新 theta（使用 IRT 算法）
                            try:
                                elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                question_difficulty = (elo_difficulty - 1500.0) / 100.0
                                theta_resp = http_requests.post(
                                    f"{API_BASE_URL}/api/theta/update",
                                    json={"current_theta": old_theta_2, "question_difficulty": question_difficulty, "is_correct": True},
                                    timeout=5,
                                )
                                new_theta = theta_resp.json()["new_theta"] if theta_resp.ok else old_theta_2
                                st.session_state.user_theta = new_theta
                                st.session_state.theta_history.append(new_theta)
                            except Exception:
                                new_theta = old_theta_2
                        else:
                            # 第2次答错：显示"Incorrect ❌" + 完整解析（包括正确选项）
                            st.session_state.last_feedback = "Incorrect ❌"

                            # 更新答题统计
                            st.session_state.attempt_count += 1

                            # 更新 theta（使用 IRT 算法，答错）
                            try:
                                elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                question_difficulty = (elo_difficulty - 1500.0) / 100.0
                                theta_resp = http_requests.post(
                                    f"{API_BASE_URL}/api/theta/update",
                                    json={"current_theta": old_theta_2, "question_difficulty": question_difficulty, "is_correct": False},
                                    timeout=5,
                                )
                                new_theta = theta_resp.json()["new_theta"] if theta_resp.ok else old_theta_2
                                st.session_state.user_theta = new_theta
                                st.session_state.theta_history.append(new_theta)
                            except Exception:
                                new_theta = old_theta_2

                        # Week 4: 记录 A/B 实验结果（第2次作答）
                        _log_ab_outcome(st.session_state.user_id, st.session_state.ab_variant, "is_correct", 1.0 if is_correct else 0.0, {"question_id": current_q.get("question_id", ""), "attempt": 2})
                        _log_ab_outcome(st.session_state.user_id, st.session_state.ab_variant, "theta_change", new_theta - old_theta_2, {"question_id": current_q.get("question_id", "")})
                        _log_ab_outcome(st.session_state.user_id, st.session_state.ab_variant, "hint_count", float(st.session_state.get("tutor_hint_count", 0)))
                        
                        # 记录题目标签信息到 questions_log（用于统计和BKT分析）
                        # 强制记录：is_correct, user_theta, skills
                        try:
                            questions_log = st.session_state.get("questions_log", [])
                            current_q_id = current_q.get("question_id", "")
                            # 检查是否已记录（避免重复记录）
                            already_logged = any(log.get("question_id") == current_q_id for log in questions_log)
                            if not already_logged:
                                current_theta = st.session_state.get("user_theta", 0.0)
                                elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                question_difficulty = (elo_difficulty - 1500.0) / 100.0  # 转换为 theta
                                
                                label_info = {
                                    "question_id": current_q_id,
                                    "question_type": current_q.get("question_type", "Weaken"),
                                    "skills": current_q.get("skills", []),  # 强制记录技能
                                    "label_source": current_q.get("label_source", "Unknown"),
                                    "skills_rationale": current_q.get("skills_rationale", ""),
                                    "is_correct": is_correct,  # 强制记录正确性
                                    "user_theta": current_theta,  # 强制记录能力值
                                    "question_difficulty": question_difficulty  # 记录题目难度（用于后续 theta 更新）
                                }
                                questions_log.append(label_info)
                                st.session_state.questions_log = questions_log
                                # 只在成功记录 questions_log 时更新 question_count（避免重复）
                                st.session_state.question_count = len(questions_log)
                        except Exception as e:
                            pass  # 记录失败不影响主流程
                        
                        # 清空聊天历史和对话状态
                        st.session_state.chat_history = []
                        st.session_state.conversation_id = None
                        st.session_state.tutor_hint_count = 0
                        st.session_state.tutor_understanding = "confused"
                    
                    st.rerun()
        else:
            st.info("Enter DeepSeek API Key in the sidebar to enable answering.")
        
        # 显示苏格拉底问答模式提示
        phase = st.session_state.get("phase", "answering")
        if phase == "remediation":
            attempt = st.session_state.get("attempt", 0)
            current_q_id = st.session_state.get("current_q_id", "")
            st.info(f"⚠️ There is an issue with your choice. Please answer the follow-up. Attempts: {attempt}/2")
            st.caption(f"Question ID: {current_q_id} (locked)")

            # Week 3: 理解度进度条 + 提示计数器
            hint_count = st.session_state.get("tutor_hint_count", 0)
            understanding = st.session_state.get("tutor_understanding", "confused")
            understanding_map = {"confused": 0.15, "partial": 0.55, "clear": 1.0}
            understanding_label = {"confused": "Confused", "partial": "Partial", "clear": "Clear"}
            prog_val = understanding_map.get(understanding, 0.15)
            prog_label = understanding_label.get(understanding, "Confused")
            col_hint, col_und = st.columns(2)
            with col_hint:
                st.metric("Hints Given", f"{hint_count} / 3")
            with col_und:
                st.caption(f"Understanding: **{prog_label}**")
                st.progress(prog_val)

        st.divider()

    # 显示聊天历史（仅在 remediation 模式下）
    phase = st.session_state.get("phase", "answering")
    if phase == "remediation":
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 聊天输入框（仅在 remediation 模式下显示，强制对齐当前题）
    if phase == "remediation":
        api_key = st.session_state.get("DEEPSEEK_API_KEY", "").strip()
        if api_key:
            if user_input := st.chat_input("Answer the follow-up and reselect your choice..."):
                current_q = st.session_state.get("current_q", {})
                current_q_id = st.session_state.get("current_q_id", "")
                conversation_id = st.session_state.get("conversation_id")

                # Week 3: 调用 /api/tutor/continue（有 conversation_id 时使用 Agent）
                if conversation_id:
                    try:
                        cont_resp = http_requests.post(
                            f"{API_BASE_URL}/api/tutor/continue",
                            json={
                                "conversation_id": conversation_id,
                                "student_message": user_input,
                                "question": current_q,
                                "correct_choice": current_q.get("correct", ""),
                            },
                            timeout=30,
                        )
                        if cont_resp.ok:
                            cont_data = cont_resp.json()
                            st.session_state.tutor_hint_count = cont_data["hint_count"]
                            st.session_state.tutor_understanding = cont_data["student_understanding"]
                            # 同步前端聊天历史
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            st.session_state.chat_history.append({"role": "assistant", "content": cont_data["reply"]})

                            # 判断是否结束 remediation
                            if not cont_data["should_continue"]:
                                st.session_state.phase = "finished"
                                st.session_state.show_explanation = True
                        else:
                            raise Exception(f"API returned {cont_resp.status_code}")
                    except Exception as e:
                        # 降级：回退到旧 /api/tutor/chat
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        try:
                            fallback_resp = http_requests.post(
                                f"{API_BASE_URL}/api/tutor/chat",
                                json={
                                    "message": user_input,
                                    "chat_history": st.session_state.chat_history,
                                    "question_id": current_q_id,
                                    "current_q": current_q,
                                    "socratic_context": st.session_state.get("socratic_context", {}),
                                },
                                timeout=30,
                            )
                            if fallback_resp.ok:
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": fallback_resp.json()["reply"],
                                })
                        except Exception:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "Think about the assumption connecting the premises to the conclusion.",
                            })
                else:
                    # 没有 conversation_id（降级模式），使用旧 /api/tutor/chat
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    try:
                        fallback_resp = http_requests.post(
                            f"{API_BASE_URL}/api/tutor/chat",
                            json={
                                "message": user_input,
                                "chat_history": st.session_state.chat_history,
                                "question_id": current_q_id,
                                "current_q": current_q,
                                "socratic_context": st.session_state.get("socratic_context", {}),
                            },
                            timeout=30,
                        )
                        if fallback_resp.ok:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": fallback_resp.json()["reply"],
                            })
                    except Exception:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Think about the assumption connecting the premises to the conclusion.",
                        })

                st.rerun()
        else:
            st.info("Enter DeepSeek API Key in the sidebar to enable chat.")

# 右侧仪表盘（30%）- IRT + BKT 驱动
with col2:
    st.header("📊 Assessment Dashboard")
    
    # Debug: Question Labels (开发用)
    current_q = st.session_state.get("current_q", {})
    if current_q:
        with st.expander("🔍 Debug: Question Labels", expanded=True):
            # 安全读取字段
            question_id = current_q.get("question_id", "N/A")
            question_type = current_q.get("question_type", "Weaken")
            difficulty = current_q.get("difficulty", "medium")
            skills = current_q.get("skills", [])
            label_source = current_q.get("label_source", "Unknown")
            skills_rationale = current_q.get("skills_rationale", "")
            
            # 确保 skills 是列表
            if not isinstance(skills, list):
                skills = []
            
            st.markdown(f"**Question ID:** `{question_id}`")
            st.markdown(f"**Label Source:** `{label_source}`")
            st.markdown(f"**Question Type:** `{question_type}`")
            st.markdown(f"**Difficulty:** `{difficulty}`")
            st.markdown(f"**Skills:** `{', '.join(skills) if skills else 'N/A'}`")
            
            # 显示 skills_rationale（如果有）
            if skills_rationale:
                st.markdown(f"**Skills Rationale:** {skills_rationale}")
            else:
                st.markdown("**Skills Rationale:** (empty)")
            
            # 检查 skills 是否匹配规则池
            try:
                if question_type in RULE_SKILL_POOL_BY_TYPE:
                    rule_pool = RULE_SKILL_POOL_BY_TYPE[question_type]
                    st.markdown(f"**Rule Pool:** `{', '.join(rule_pool)}`")
                    
                    if skills:
                        # 检查所有技能是否都在规则池内
                        all_match = all(skill in rule_pool for skill in skills)
                        if all_match:
                            st.success("✅ Skills match rule pool.")
                        else:
                            mismatched = [s for s in skills if s not in rule_pool]
                            st.error(f"❌ Skills mismatch: {', '.join(mismatched)} not in rule pool")
                    else:
                        st.warning("⚠️ No skills to check")
                else:
                    st.warning(f"⚠️ Question type '{question_type}' not in rule pool mapping")
            except Exception as e:
                st.warning(f"⚠️ Error checking rule pool: {e}")
    
    # Debug: Label Stats (统计已做过的题目的标签信息)
    questions_log = st.session_state.get("questions_log", [])
    if questions_log:
        with st.expander("📊 Debug: Label Stats", expanded=False):
            try:
                # 1) Label Source Count
                label_source_count = {"llm": 0, "fallback_rule": 0}
                for log in questions_log:
                    source = log.get("label_source", "Unknown")
                    if source == "llm":
                        label_source_count["llm"] += 1
                    elif source == "fallback_rule":
                        label_source_count["fallback_rule"] += 1
                
                st.markdown("**1) Label Source Count:**")
                st.markdown(f"- `llm`: {label_source_count['llm']}")
                st.markdown(f"- `fallback_rule`: {label_source_count['fallback_rule']}")
                
                # 2) Rule Pool Mismatch Count
                mismatch_count = 0
                for log in questions_log:
                    q_type = log.get("question_type", "Weaken")
                    skills = log.get("skills", [])
                    if q_type in RULE_SKILL_POOL_BY_TYPE and skills:
                        rule_pool = RULE_SKILL_POOL_BY_TYPE[q_type]
                        if not all(skill in rule_pool for skill in skills):
                            mismatch_count += 1
                
                st.markdown("**2) Rule Pool Mismatch Count:**")
                st.markdown(f"- `mismatch`: {mismatch_count}")
                
                # 3) Skills Frequency Top 6
                skill_freq = {}
                for log in questions_log:
                    skills = log.get("skills", [])
                    if isinstance(skills, list):
                        for skill in skills:
                            skill_freq[skill] = skill_freq.get(skill, 0) + 1
                
                if skill_freq:
                    # 按出现次数排序，取前6
                    sorted_skills = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:6]
                    st.markdown("**3) Skills Frequency Top 6:**")
                    for skill, count in sorted_skills:
                        st.markdown(f"- `{skill}`: {count}")
                else:
                    st.markdown("**3) Skills Frequency Top 6:**")
                    st.markdown("- (No skills data)")
                    
            except Exception as e:
                st.warning(f"⚠️ Error calculating stats: {e}")
    else:
        with st.expander("📊 Debug: Label Stats", expanded=False):
                st.info("No label stats yet. Complete questions to see stats.")
    
    st.divider()
    
    # ========== 核心指标：GMAT Score ==========
    try:
        current_theta = st.session_state.get("user_theta", 0.0)
        # GMAT 估分：内联计算（与 engine/scoring.py 同公式，纯展示无需走 API）
        gmat_score = int(round(max(20, min(51, 30.0 + current_theta * 7.0))))
        
        # 计算档位
        if current_theta < -1.0:
            level_bucket = "500 band"
        elif current_theta <= 1.0:
            level_bucket = "650 band"
        else:
            level_bucket = "750 band"
        
        st.metric("GMAT CR Estimate", f"V{gmat_score}", delta=f"Theta: {current_theta:.2f}")
        st.caption(f"Current band: {level_bucket}")
    except Exception as e:
        st.metric("GMAT CR Estimate", "V30", delta="Theta: 0.00")
    
    st.divider()
    
    # ========== 能力进度条 ==========
    try:
        current_theta = st.session_state.get("user_theta", 0.0)
        # 归一化 Theta (-3到3) 到 (0.0到1.0)
        normalized_progress = (current_theta + 3.0) / 6.0
        normalized_progress = max(0.0, min(1.0, normalized_progress))
        
        st.subheader("Ability Progress")
        st.progress(normalized_progress)
        
        # 标注当前档位
        if current_theta < -1.0:
            level_label = "500 band (basic)"
        elif current_theta <= 1.0:
            level_label = "650 band (intermediate)"
        else:
            level_label = "750 band (advanced)"
        
        st.caption(f"Current band: {level_label} | Theta: {current_theta:.2f}")
    except Exception as e:
        st.progress(0.5)
        st.caption("Calculating ability progress...")
    
    st.divider()
    
    # ========== 技能掌握度雷达图 (BKT) ==========
    st.subheader("Skill Mastery Radar")
    
    try:
        questions_log = st.session_state.get("questions_log", [])
        
        if not questions_log:
            st.info("📝 Complete questions to build skill profile")
        else:
            # 统计每个 Skill 的 Correct / Total
            skill_stats = {}  # {skill: {"correct": count, "total": count}}
            
            for log in questions_log:
                skills = log.get("skills", [])
                is_correct = log.get("is_correct", False)
                
                if not isinstance(skills, list):
                    continue
                
                for skill in skills:
                    if skill not in skill_stats:
                        skill_stats[skill] = {"correct": 0, "total": 0}
                    
                    skill_stats[skill]["total"] += 1
                    if is_correct:
                        skill_stats[skill]["correct"] += 1
            
            # 只有当 Skill 至少出现 1 次时才纳入图表
            if skill_stats:
                # 计算每个技能的掌握度（正确率 * 100）
                skill_mastery = {}
                for skill, stats in skill_stats.items():
                    total = stats["total"]
                    correct = stats["correct"]
                    if total > 0:
                        mastery = (correct / total) * 100.0
                        skill_mastery[skill] = mastery
                
                if skill_mastery:
                    # 创建雷达图
                    categories = list(skill_mastery.keys())
                    values = [skill_mastery[cat] for cat in categories]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Skill mastery',
                        line_color='rgb(32, 201, 151)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📝 Complete questions to build skill profile")
            else:
                st.info("📝 Complete questions to build skill profile")
                
    except Exception as e:
        st.warning(f"⚠️ Skill radar failed: {e}")
        st.info("📝 Complete questions to build skill profile")
    
    st.divider()
    
    # ========== Theta 历史折线图 ==========
    try:
        theta_history = st.session_state.get("theta_history", [0.0])
        question_count = st.session_state.get("question_count", 0)
        
        if len(theta_history) > 0 and question_count > 0:
            st.subheader("Ability Curve (Theta)")
            
            # 创建折线图数据
            x_data = list(range(len(theta_history)))
            y_data = theta_history
            
            # 使用 plotly 创建折线图
            fig_theta = go.Figure()
            fig_theta.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',
                name='Theta',
                line=dict(color='rgb(32, 201, 151)', width=2),
                marker=dict(size=6)
            ))
            
            fig_theta.update_layout(
                xaxis_title="Question #",
                yaxis_title="Theta",
                height=300,
                showlegend=True,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig_theta, use_container_width=True)
    except Exception as e:
        # 折线图失败不影响主流程
        pass
