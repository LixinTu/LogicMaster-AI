import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import requests as http_requests  # 避免与 FastAPI 的 Request 冲突
from llm_service import generate_question, diagnose_wrong_answer, generate_detailed_explanation, RULE_SKILL_POOL_BY_TYPE
from utils.db_handler import DatabaseManager, get_db_manager
from engine.recommender import analyze_weak_skills
import uuid
import random

# FastAPI 后端地址
API_BASE_URL = "http://localhost:8000"

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
            st.info("ℹ️ **冷启动模式**：系统检测到数据库为空或无可用题目。使用默认题目进行演示。\n\n"
                   "💡 **提示**：运行 `python generate_pool.py` 生成题目后，系统将自动切换到数据库题目。")
            st.session_state._cold_start_warning_shown = True
        
        # 使用默认题目
        initial_q_id = str(uuid.uuid4())[:8]
        st.session_state.current_q = {
            "question_id": initial_q_id,
            "difficulty": "medium",
            "question_type": "Weaken",
            "stimulus": "某公司计划推出新产品。支持者认为新产品将大幅提升市场份额。然而，竞争对手也在研发类似产品，且市场调研显示消费者对新功能需求有限。",
            "question": "以下哪项最能削弱支持者的论证？",
            "choices": [
                "A. 新产品开发成本较高",
                "B. 市场竞争激烈，新产品难以突围",
                "C. 消费者对新功能不感兴趣",
                "D. 公司缺乏新产品推广经验",
                "E. 新产品技术尚未成熟"
            ],
            "correct": "C",
            "correct_choice": "C",
            "explanation": "C 直接指出消费者需求有限，削弱了市场份额提升的假设",
            "tags": [],
            # 技能标签相关字段（默认值）
            "skills": ["因果推理", "替代解释"],  # 默认 skills
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
    st.header("💬 逻辑推理对话")
    
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
        st.subheader("📝 当前题目")
        
        # 在 remediation 阶段显示 question_id（小字）
        phase = st.session_state.get("phase", "answering")
        if phase == "remediation":
            st.caption(f"题目 ID: {question_id}（已锁定，苏格拉底问答针对本题）")
        
        # 学生可见标签条（题干上方）
        skills_str = ", ".join(skills) if skills else "N/A"
        st.caption(f"**Type:** {question_type} | **Difficulty:** {difficulty} | **Skills:** {skills_str}")
        
        st.markdown(f"**题干：** {current_q.get('stimulus', '')}")
        st.markdown(f"**问题：** {current_q.get('question', '')}")
        
        # 获取当前状态
        attempt = st.session_state.get("attempt", 0)
        phase = st.session_state.get("phase", "answering")
        
        # 判断是否可以作答：phase为"answering"或"remediation"，且attempt < 2
        can_submit = (phase == "answering" or phase == "remediation") and attempt < 2
        
        # 显示选项（只显示 A-E 字母）
        # 注意：使用动态 key 以支持重置，只读，不手动赋值
        choice_options = ["A", "B", "C", "D", "E"]
        selected_choice = st.radio(
            "请选择答案：",
            options=choice_options,
            key=f"selected_choice_{st.session_state.radio_key}",
            label_visibility="visible",
            disabled=not can_submit
        )
        
        # 显示选项内容（锁定显示，使用 current_q）
        choices = current_q.get("choices", [])
        if choices:
            st.markdown("**选项内容：**")
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
        # 使用 current_q 中的详细解析（如果已生成）
        if st.session_state.get("show_explanation", False):
            # 优先使用详细解析（如果存在）
            detailed_explanation = current_q.get("detailed_explanation", "")
            if not detailed_explanation:
                detailed_explanation = current_q.get("explanation", "")
            
            correct_choice = current_q.get("correct_choice") or current_q.get("correct", "")
            if detailed_explanation:
                st.divider()
                st.subheader("📖 详细解析")
                if phase == "finished" and attempt == 2 and last_feedback and "Incorrect" in last_feedback:
                    # 第2次答错时显示正确选项
                    st.markdown(f"**正确答案：{correct_choice}**")
                st.markdown(detailed_explanation)
                
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
                                            st.warning("⚠️ 数据库中暂无题目。请先运行 `python generate_pool.py` 生成题目。")
                                    except Exception as e:
                                        # 数据库查询失败，显示错误信息
                                        st.error(f"❌ 无法从数据库获取题目：{e}。请检查数据库连接或运行 `python generate_pool.py` 生成题目。")
                                else:
                                    # 成功获取新题目，刷新页面
                                    st.rerun()
                            except Exception as e:
                                # 生成题目时发生未知错误
                                st.error(f"❌ 生成下一题时出错：{e}。请刷新页面重试。")
                                print(f"生成下一题时出错：{e}")
        
        # Submit Answer 按钮
        api_key = st.session_state.get("DEEPSEEK_API_KEY", "").strip()
        if api_key:
            if st.button("Submit Answer", type="primary", use_container_width=True, disabled=not can_submit):
                # 再次检查是否允许提交
                if not can_submit:
                    st.warning("本题已提交，无法再次提交。")
                    st.stop()
                
                # 判分逻辑（使用锁题机制：current_q）
                # 注意：只读取 selected_choice，不手动赋值
                user_choice = st.session_state.get(f"selected_choice_{st.session_state.radio_key}")
                if not user_choice:
                    st.warning("请先选择一个选项（A-E）")
                else:
                    # 检查 API Key
                    if not api_key:
                        st.info("请在右侧输入 DeepSeek API Key 以启用 AI 对话。")
                        st.stop()
                    
                    # 获取当前题目（使用锁题机制）
                    current_q = st.session_state.get("current_q", {})
                    current_q_id = st.session_state.get("current_q_id", "")
                    
                    if not current_q:
                        st.error("题目数据缺失，请刷新页面。")
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
                            try:
                                current_theta = st.session_state.get("user_theta", 0.0)
                                elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                question_difficulty = (elo_difficulty - 1500.0) / 100.0  # 转换为 theta
                                theta_resp = http_requests.post(
                                    f"{API_BASE_URL}/api/theta/update",
                                    json={"current_theta": current_theta, "question_difficulty": question_difficulty, "is_correct": True},
                                    timeout=5,
                                )
                                new_theta = theta_resp.json()["new_theta"] if theta_resp.ok else current_theta
                                st.session_state.user_theta = new_theta
                                st.session_state.theta_history.append(new_theta)
                            except Exception as e:
                                pass

                        else:
                            # 第1次答错：显示Incorrect，进入remediation
                            st.session_state.last_feedback = "Incorrect ❌"
                            st.session_state.phase = "remediation"
                            st.session_state.show_explanation = False  # 先不显示完整解析
                            
                            # 优先查表：尝试从 current_q 获取预存的诊断信息
                            cached_diagnosis = current_q.get("diagnoses", {}).get(user_choice)
                            
                            # 分支A：命中缓存 - 秒回（不调用任何LLM）
                            if cached_diagnosis:
                                # 直接提取第一句苏格拉底反问
                                first_msg = cached_diagnosis.get("first_socratic_response", "请重新思考这个选项的问题。")
                                
                                # 将 cached_diagnosis（包含 logic_gap 等）存入 socratic_context
                                st.session_state.socratic_context = {
                                    "question_id": current_q_id,
                                    "correct_choice": correct_choice,
                                    "user_choice": user_choice,
                                    "logic_gap": cached_diagnosis.get("logic_gap", ""),
                                    "first_socratic_response": first_msg
                                }
                                
                                # 添加用户选择到聊天历史（首次答错时）
                                if len(st.session_state.chat_history) == 0:
                                    user_message = f"我选择的答案是：{user_choice}"
                                    st.session_state.chat_history.append({
                                        "role": "user",
                                        "content": user_message
                                    })
                                
                                # 直接将 first_msg 添加到聊天历史（role: assistant）
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": first_msg
                                })
                            
                            # 分支B：未命中缓存 - 降级处理（兼容旧题目）
                            else:
                                # 添加用户选择到聊天历史（首次答错时）
                                if len(st.session_state.chat_history) == 0:
                                    user_message = f"我选择的答案是：{user_choice}"
                                    st.session_state.chat_history.append({
                                        "role": "user",
                                        "content": user_message
                                    })
                                
                                # 显示加载提示并调用实时诊断
                                try:
                                    with st.spinner("🤖 AI 正在分析错因..."):
                                        diagnosis = diagnose_wrong_answer(
                                            current_q=current_q,
                                            user_choice=user_choice,
                                            api_key=api_key
                                        )
                                        st.session_state.socratic_context = diagnosis
                                        
                                        # 从诊断结果中提取第一句回复
                                        # diagnose_wrong_answer 返回的格式可能包含 hint_plan，使用第一个作为第一句
                                        first_socratic_response = ""
                                        if diagnosis.get("hint_plan") and len(diagnosis["hint_plan"]) > 0:
                                            first_socratic_response = diagnosis["hint_plan"][0]
                                        elif diagnosis.get("why_user_choice_wrong"):
                                            first_socratic_response = f"让我们分析一下：{diagnosis['why_user_choice_wrong']}"
                                        else:
                                            first_socratic_response = "请重新思考这个选项为什么不对。"
                                        
                                        # 直接将第一句回复添加到聊天历史（不再调用 tutor_reply）
                                        st.session_state.chat_history.append({
                                            "role": "assistant",
                                            "content": first_socratic_response
                                        })
                                        
                                except Exception as e:
                                    # 如果诊断失败，使用默认上下文和回复
                                    st.session_state.socratic_context = {
                                        "question_id": current_q_id,
                                        "correct_choice": correct_choice,
                                        "user_choice": user_choice,
                                        "hint_plan": ["识别结论", "分析假设", "对比选项"]
                                    }
                                    
                                    # 添加默认回复
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": "请重新思考这个选项的问题。"
                                    })
                    
                    # === 第2次作答（attempt=2）===
                    elif new_attempt == 2:
                        st.session_state.phase = "finished"
                        
                        # 直接从 current_q 读取预生成的详细解析（瞬间显示）
                        # 如果不存在，使用基础 explanation 作为备选
                        if not current_q.get("detailed_explanation"):
                            current_q["detailed_explanation"] = current_q.get("explanation", "")
                            st.session_state.current_q = current_q
                        
                        st.session_state.show_explanation = True
                        
                        if is_correct:
                            # 第2次答对：显示"Correct (after reasoning) ✅" + 解析
                            st.session_state.last_feedback = "Correct (after reasoning) ✅"
                            
                            # 更新答题统计
                            st.session_state.attempt_count += 1
                            st.session_state.correct_count += 1
                            
                            # 更新 theta（使用 IRT 算法）
                            try:
                                current_theta = st.session_state.get("user_theta", 0.0)
                                elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                question_difficulty = (elo_difficulty - 1500.0) / 100.0  # 转换为 theta
                                theta_resp = http_requests.post(
                                    f"{API_BASE_URL}/api/theta/update",
                                    json={"current_theta": current_theta, "question_difficulty": question_difficulty, "is_correct": True},
                                    timeout=5,
                                )
                                new_theta = theta_resp.json()["new_theta"] if theta_resp.ok else current_theta
                                st.session_state.user_theta = new_theta
                                # question_count 在 questions_log 记录成功后更新（避免重复）
                                st.session_state.theta_history.append(new_theta)
                            except Exception as e:
                                pass
                        else:
                            # 第2次答错：显示"Incorrect ❌" + 完整解析（包括正确选项）
                            st.session_state.last_feedback = "Incorrect ❌"
                            
                            # 更新答题统计
                            st.session_state.attempt_count += 1
                            
                            # 更新 theta（使用 IRT 算法，答错）
                            try:
                                current_theta = st.session_state.get("user_theta", 0.0)
                                elo_difficulty = current_q.get("elo_difficulty", 1500.0)
                                question_difficulty = (elo_difficulty - 1500.0) / 100.0  # 转换为 theta
                                theta_resp = http_requests.post(
                                    f"{API_BASE_URL}/api/theta/update",
                                    json={"current_theta": current_theta, "question_difficulty": question_difficulty, "is_correct": False},
                                    timeout=5,
                                )
                                new_theta = theta_resp.json()["new_theta"] if theta_resp.ok else current_theta
                                st.session_state.user_theta = new_theta
                                # question_count 在 questions_log 记录成功后更新（避免重复）
                                st.session_state.theta_history.append(new_theta)
                            except Exception as e:
                                pass
                        
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
                        
                        # 清空聊天历史
                        st.session_state.chat_history = []
                    
                    st.rerun()
        else:
            st.info("请在右侧输入 DeepSeek API Key 以启用答题功能。")
        
        # 显示苏格拉底问答模式提示
        phase = st.session_state.get("phase", "answering")
        if phase == "remediation":
            attempt = st.session_state.get("attempt", 0)
            current_q_id = st.session_state.get("current_q_id", "")
            st.info(f"⚠️ 你刚才的选择有问题，请回答下面追问。尝试次数：{attempt}/2")
            st.caption(f"当前题目 ID: {current_q_id}（已锁定）")
        
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
            if user_input := st.chat_input("回答追问并重新选择选项..."):
                # 获取锁定的题目信息
                current_q = st.session_state.get("current_q", {})
                current_q_id = st.session_state.get("current_q_id", "")
                socratic_context = st.session_state.get("socratic_context", {})
                
                # 添加学生回答到聊天历史
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # 调用 Tutor 继续追问（强制对齐当前题）
                try:
                    remediation_prompt = f"学生回答：{user_input}。请继续苏格拉底式追问，不能泄露正确选项。"
                    
                    tutor_resp = http_requests.post(
                        f"{API_BASE_URL}/api/tutor/chat",
                        json={
                            "message": remediation_prompt,
                            "chat_history": [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.chat_history
                                if m.get("role") in ("user", "assistant")
                            ],
                            "question_id": current_q_id,
                            "current_q": current_q,
                            "socratic_context": socratic_context,
                        },
                        timeout=30,
                    )
                    tutor_data = tutor_resp.json() if tutor_resp.ok else None

                    if tutor_data is None or tutor_data.get("is_error"):
                        st.error(tutor_data["reply"] if tutor_data else "Tutor API 调用失败")
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": tutor_data["reply"]
                        })
                except Exception as e:
                    st.warning(f"Tutor 追问出错: {e}")
                
                st.rerun()
        else:
            st.info("请在右侧输入 DeepSeek API Key 以启用对话功能。")

# 右侧仪表盘（30%）- IRT + BKT 驱动
with col2:
    st.header("📊 评估仪表盘")
    
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
                st.markdown("**Skills Rationale:** (空)")
            
            # 检查 skills 是否匹配规则池
            try:
                if question_type in RULE_SKILL_POOL_BY_TYPE:
                    rule_pool = RULE_SKILL_POOL_BY_TYPE[question_type]
                    st.markdown(f"**Rule Pool:** `{', '.join(rule_pool)}`")
                    
                    if skills:
                        # 检查所有技能是否都在规则池内
                        all_match = all(skill in rule_pool for skill in skills)
                        if all_match:
                            st.success("✅ Skills match rule pool")
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
            st.info("No label stats yet.")
    
    st.divider()
    
    # ========== 核心指标：GMAT Score ==========
    try:
        current_theta = st.session_state.get("user_theta", 0.0)
        # GMAT 估分：内联计算（与 engine/scoring.py 同公式，纯展示无需走 API）
        gmat_score = int(round(max(20, min(51, 30.0 + current_theta * 7.0))))
        
        # 计算档位
        if current_theta < -1.0:
            level_bucket = "500档"
        elif current_theta <= 1.0:
            level_bucket = "650档"
        else:
            level_bucket = "750档"
        
        st.metric("GMAT CR 估分", f"V{gmat_score}", delta=f"Theta: {current_theta:.2f}")
        st.caption(f"当前档位：{level_bucket}")
    except Exception as e:
        st.metric("GMAT CR 估分", "V30", delta="Theta: 0.00")
    
    st.divider()
    
    # ========== 能力进度条 ==========
    try:
        current_theta = st.session_state.get("user_theta", 0.0)
        # 归一化 Theta (-3到3) 到 (0.0到1.0)
        normalized_progress = (current_theta + 3.0) / 6.0
        normalized_progress = max(0.0, min(1.0, normalized_progress))
        
        st.subheader("能力进度")
        st.progress(normalized_progress)
        
        # 标注当前档位
        if current_theta < -1.0:
            level_label = "500档（基础）"
        elif current_theta <= 1.0:
            level_label = "650档（中等）"
        else:
            level_label = "750档（高阶）"
        
        st.caption(f"当前档位：{level_label} | Theta: {current_theta:.2f}")
    except Exception as e:
        st.progress(0.5)
        st.caption("能力进度计算中...")
    
    st.divider()
    
    # ========== 技能掌握度雷达图 (BKT) ==========
    st.subheader("技能掌握度雷达图")
    
    try:
        questions_log = st.session_state.get("questions_log", [])
        
        if not questions_log:
            st.info("📝 做题以生成技能画像")
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
                        name='技能掌握度',
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
                    st.info("📝 做题以生成技能画像")
            else:
                st.info("📝 做题以生成技能画像")
                
    except Exception as e:
        st.warning(f"⚠️ 技能雷达图生成失败：{e}")
        st.info("📝 做题以生成技能画像")
    
    st.divider()
    
    # ========== Theta 历史折线图 ==========
    try:
        theta_history = st.session_state.get("theta_history", [0.0])
        question_count = st.session_state.get("question_count", 0)
        
        if len(theta_history) > 0 and question_count > 0:
            st.subheader("能力变化曲线 (Theta)")
            
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
                xaxis_title="题目序号",
                yaxis_title="Theta",
                height=300,
                showlegend=True,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig_theta, use_container_width=True)
    except Exception as e:
        # 折线图失败不影响主流程
        pass
