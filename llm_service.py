"""
LLM 服务模块
使用 DeepSeek API 进行真实调用
"""

import json
from openai import OpenAI
import uuid

# ========== Rule + LLM Hybrid Skill Label System ==========

# Rule mapping: skill pool per question type
RULE_SKILL_POOL_BY_TYPE = {
    "Weaken": ["Causal Reasoning", "Alternative Explanation", "Eliminating Distractors", "Assumption Identification"],
    "Strengthen": ["Causal Reasoning", "Evidence Strength", "Structured Expression", "Assumption Identification"],
    "Assumption": ["Assumption Identification", "Causal Reasoning", "Structured Expression"],
    "Inference": ["Eliminating Distractors", "Evidence Strength", "Structured Expression"],
    "Flaw": ["Structured Expression", "Causal Reasoning", "Assumption Identification", "Evidence Strength"]
}

# Default skills per question type (fallback)
DEFAULT_SKILLS_BY_TYPE = {
    "Weaken": ["Causal Reasoning", "Alternative Explanation"],
    "Strengthen": ["Causal Reasoning", "Evidence Strength"],
    "Assumption": ["Assumption Identification", "Causal Reasoning"],
    "Inference": ["Eliminating Distractors", "Evidence Strength"],
    "Flaw": ["Structured Expression", "Assumption Identification"]
}

SYSTEM_PROMPT = (
    "You are a strict but patient GMAT Critical Reasoning examiner. "
    "You never reveal the correct answer directly. You only use Socratic questioning: "
    "Point out assumptions, logical gaps, causal leaps, sample bias, etc., and use leading questions "
    "to guide the student toward self-correction. Keep responses short (1-3 sentences), one key point at a time."
)


def tutor_reply(user_text: str, api_key: str, chat_history=None, current_q: dict = None, current_q_id: str = None, socratic_context: dict = None) -> str:
    """
    调用 DeepSeek API 获取回复（苏格拉底式追问）
    
    Args:
        user_text: 用户输入的文本
        api_key: DeepSeek API Key
        chat_history: 聊天历史列表，元素为 {role, content}
        current_q: 当前题目完整快照（包含题干、选项、正确答案等）
        current_q_id: 当前题目ID
        socratic_context: 苏格拉底上下文（包含错因诊断、引导计划等）
        
    Returns:
        AI 回复字符串，如果出错则返回以 "[LLM ERROR]" 开头的错误信息
    """
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # 构建增强的 system prompt，强制对齐当前题
        enhanced_system_prompt = SYSTEM_PROMPT
        if current_q and current_q_id:
            enhanced_system_prompt += f"\n\n[IMPORTANT CONSTRAINTS]\n"
            enhanced_system_prompt += f"- You may only discuss question ID: {current_q_id}. Do not switch topics or reference other questions.\n"
            enhanced_system_prompt += f"- Each reply must acknowledge the current question, e.g. 'For this question (ID: {current_q_id}), let us consider...'\n"
            enhanced_system_prompt += f"- You must reference stimulus content ({current_q.get('stimulus', '')[:50]}...) and option letters (A-E).\n"
            enhanced_system_prompt += f"- Never reveal the correct option letter; only guide through questioning.\n"
            
            if socratic_context and socratic_context.get("hint_plan"):
                enhanced_system_prompt += f"- Follow this hint plan step by step: {socratic_context.get('hint_plan', [])}\n"

        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        if current_q:
            question_context = f"[CURRENT QUESTION ID: {current_q_id}]\n"
            question_context += f"Stimulus: {current_q.get('stimulus', '')}\n"
            question_context += f"Question: {current_q.get('question', '')}\n"
            question_context += f"Choices:\n"
            for choice in current_q.get('choices', []):
                question_context += f"  {choice}\n"
            messages.append({"role": "system", "content": question_context})

        # 只带最近几条历史，避免 token 太多
        if chat_history:
            for m in chat_history[-8:]:
                role = m.get("role")
                content = m.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_text})

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"[LLM ERROR] {type(e).__name__}: {e}"


ASSESSOR_SYSTEM_PROMPT = (
    "你是 GMAT Critical Reasoning 逻辑评估员。只评估用户最近一次回答的逻辑质量，不要回答题目本身。"
    "必须输出严格 JSON，不要包含多余文本。"
)


def assessor_eval(user_text: str, api_key: str, chat_history=None) -> dict:
    """
    调用 DeepSeek API 进行逻辑评估，返回结构化 JSON
    
    Args:
        user_text: 用户输入的文本
        api_key: DeepSeek API Key
        chat_history: 聊天历史列表，元素为 {role, content}
        
    Returns:
        评估结果字典，包含：
        - total_score: 0-100 的整数
        - dimensions: 包含 5 个维度分数的字典
        - tags: 长度为 3 的标签数组
        - one_sentence_feedback: 一句话反馈
        如果出错则返回默认值
    """
    # 默认返回值
    default_result = {
        "total_score": 50,
        "dimensions": {
            "论据强度": 50,
            "逻辑连贯性": 50,
            "反驳能力": 50,
            "清晰度": 50,
            "结构化": 50
        },
        "tags": ["证据不足", "因果不清", "假设跳跃"],
        "one_sentence_feedback": "请补充关键假设与证据"
    }
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        messages = [{"role": "system", "content": ASSESSOR_SYSTEM_PROMPT}]

        # 只带最近几条历史，避免 token 太多
        if chat_history:
            for m in chat_history[-8:]:
                role = m.get("role")
                content = m.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})

        # 构建评估 prompt，强调输出 JSON
        assessment_prompt = f"""请评估用户最近一次回答的逻辑质量。

用户回答：{user_text}

请只输出一个严格 JSON 对象，格式如下：
{{
  "total_score": <0-100的整数>,
  "dimensions": {{
    "论据强度": <0-100的整数>,
    "逻辑连贯性": <0-100的整数>,
    "反驳能力": <0-100的整数>,
    "清晰度": <0-100的整数>,
    "结构化": <0-100的整数>
  }},
  "tags": ["标签1", "标签2", "标签3"],
  "one_sentence_feedback": "<不超过25字的中文反馈>"
}}

要求：
1. total_score 是 0-100 的整数，综合评分
2. dimensions 的 5 个维度都是 0-100 的整数
3. tags 是长度为 3 的数组，每个标签是中文短标签（例如：因果跳跃/证据不足/忽略替代解释/偷换概念/样本偏差）
4. one_sentence_feedback 是一句不超过 25 字的中文反馈，指出最关键问题

只输出 JSON，不要包含任何其他文本。"""

        messages.append({"role": "user", "content": assessment_prompt})

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        # 尝试提取 JSON（可能包含 markdown 代码块）
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # 解析 JSON
        result = json.loads(response_text)
        
        # 验证和修复结果格式
        if not isinstance(result, dict):
            return default_result
        
        # 确保所有必需字段存在
        if "total_score" not in result:
            result["total_score"] = 50
        else:
            result["total_score"] = max(0, min(100, int(result["total_score"])))
        
        if "dimensions" not in result or not isinstance(result["dimensions"], dict):
            result["dimensions"] = default_result["dimensions"]
        else:
            # 确保所有维度都存在且为整数
            for key in ["论据强度", "逻辑连贯性", "反驳能力", "清晰度", "结构化"]:
                if key not in result["dimensions"]:
                    result["dimensions"][key] = 50
                else:
                    result["dimensions"][key] = max(0, min(100, int(result["dimensions"][key])))
        
        if "tags" not in result or not isinstance(result["tags"], list) or len(result["tags"]) != 3:
            result["tags"] = default_result["tags"]
        
        if "one_sentence_feedback" not in result:
            result["one_sentence_feedback"] = default_result["one_sentence_feedback"]
        
        return result

    except json.JSONDecodeError:
        # JSON 解析失败，返回默认值
        return default_result
    except Exception as e:
        # 其他异常，返回默认值
        return default_result


def validate_question_labels(question_json: dict) -> dict:
    """
    校验和修复题目技能标签（规则+LLM混合系统）
    
    Args:
        question_json: 题目 JSON 字典
        
    Returns:
        修复后的题目字典，包含 label_source 字段
    """
    try:
        question_type = question_json.get("question_type", "Weaken")
        
        # 检查 question_type 是否合法
        valid_types = ["Weaken", "Strengthen", "Assumption", "Inference", "Flaw"]
        if question_type not in valid_types:
            question_type = "Weaken"  # 默认类型
            question_json["question_type"] = question_type
        
        # 获取该题型的候选技能池和默认技能
        skill_pool = RULE_SKILL_POOL_BY_TYPE.get(question_type, DEFAULT_SKILLS_BY_TYPE.get("Weaken", []))
        default_skills = DEFAULT_SKILLS_BY_TYPE.get(question_type, DEFAULT_SKILLS_BY_TYPE.get("Weaken", []))
        
        # 检查 skills 字段
        skills = question_json.get("skills", [])
        is_valid = False
        
        if isinstance(skills, list) and 2 <= len(skills) <= 3:
            # 检查所有技能是否都在候选池内
            if all(skill in skill_pool for skill in skills):
                is_valid = True
        
        # 如果不合规，使用默认 skills 替换
        if not is_valid:
            question_json["skills"] = default_skills.copy()
            question_json["skills_rationale"] = "Applied rule-based fallback for stability."
            question_json["label_source"] = "fallback_rule"
        else:
            # 确保有 skills_rationale
            if "skills_rationale" not in question_json or not question_json.get("skills_rationale"):
                question_json["skills_rationale"] = "LLM-generated skills based on question content."
            question_json["label_source"] = "llm"
        
        # 确保 label_source 字段存在
        if "label_source" not in question_json:
            question_json["label_source"] = "llm"
        
        return question_json
        
    except Exception as e:
        # 异常时使用默认值
        question_type = question_json.get("question_type", "Weaken")
        default_skills = DEFAULT_SKILLS_BY_TYPE.get(question_type, DEFAULT_SKILLS_BY_TYPE.get("Weaken", []))
        question_json["skills"] = default_skills.copy()
        question_json["skills_rationale"] = "Applied rule-based fallback for stability."
        question_json["label_source"] = "fallback_rule"
        return question_json


def generate_question(theta: float, api_key: str) -> dict:
    """
    根据 theta 生成 GMAT Critical Reasoning 题目
    
    Args:
        theta: 用户当前能力值（IRT theta）
        api_key: DeepSeek API Key
        
    Returns:
        题目字典，包含：
        - difficulty: "easy|medium|hard"
        - question_type: "Weaken|Assumption"
        - stimulus: 题干背景
        - question: 问题
        - choices: 5个选项列表
        - correct: 正确答案 "A|B|C|D|E"
        - explanation: 解释
        如果出错则返回默认题目
    """
    default_question = {
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
        "explanation": "C directly points to limited consumer demand, weakening the market-share assumption",
        "skills": ["Causal Reasoning", "Alternative Explanation"],
        "label_source": "fallback_rule",
        "skills_rationale": "Default question with rule-based fallback skills."
    }
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        if theta < -1.0:
            difficulty = "easy"
            difficulty_desc = "Simple (short text, single causal chain, clear options)"
        elif theta <= 1.0:
            difficulty = "medium"
            difficulty_desc = "Medium (alternative explanations/confounders, closer options)"
        else:
            difficulty = "hard"
            difficulty_desc = "Hard (multiple factors, layered assumptions, strong distractors)"
        
        import random
        question_type = random.choice(["Weaken", "Strengthen", "Assumption", "Inference", "Flaw"])
        
        skill_pool = RULE_SKILL_POOL_BY_TYPE.get(question_type, DEFAULT_SKILLS_BY_TYPE.get("Weaken", []))
        skill_pool_str = ", ".join(skill_pool)
        
        prompt = f"""Generate one GMAT Critical Reasoning question in English.

Requirements:
- Difficulty: {difficulty} ({difficulty_desc})
- Question type: {question_type} (from Weaken/Strengthen/Assumption/Inference/Flaw)
- Stimulus: 2-5 sentences describing a scenario and argument
- Question: one sentence asking the question
- Choices: 5 options labeled A-E
- Correct answer: one of A, B, C, D, or E
- Explanation: one sentence explaining why the correct option is right (<=50 words)

[SKILL LABELS - IMPORTANT]
- Question type is {question_type}, skill pool: {skill_pool_str}
- You MUST select 2-3 skills from the pool for the skills array
- You MUST output skills_rationale (<=80 chars) explaining why this question maps to these skills

Output ONLY a strict JSON object in this format:
{{
  "difficulty": "{difficulty}",
  "question_type": "{question_type}",
  "stimulus": "<2-5 sentence stimulus in English>",
  "question": "<one sentence question in English>",
  "choices": ["A ...", "B ...", "C ...", "D ...", "E ..."],
  "correct": "<A|B|C|D|E>",
  "explanation": "<one sentence explanation in English>",
  "skills": ["<2-3 skills from the pool>"],
  "skills_rationale": "<why this question maps to these skills>"
}}

Output JSON only, no other text."""
        
        messages = [
            {"role": "system", "content": "You are a GMAT Critical Reasoning question generation expert. Output strict JSON only, no extra text."},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        # 尝试提取 JSON（可能包含 markdown 代码块）
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # 解析 JSON
        result = json.loads(response_text)
        
        # 验证和修复结果格式
        if not isinstance(result, dict):
            return default_question
        
        # 确保所有必需字段存在
        if "difficulty" not in result:
            result["difficulty"] = difficulty
        if "question_type" not in result:
            result["question_type"] = question_type
        if "stimulus" not in result:
            result["stimulus"] = default_question["stimulus"]
        if "question" not in result:
            result["question"] = default_question["question"]
        if "choices" not in result or not isinstance(result["choices"], list) or len(result["choices"]) != 5:
            result["choices"] = default_question["choices"]
        if "correct" not in result or result["correct"] not in ["A", "B", "C", "D", "E"]:
            result["correct"] = default_question["correct"]
        if "explanation" not in result:
            result["explanation"] = default_question["explanation"]
        
        # 校验和修复技能标签（关键稳定性）
        result = validate_question_labels(result)
        
        # 确保 default_question 也有 label_source（用于 fallback）
        if result == default_question:
            default_question["skills"] = DEFAULT_SKILLS_BY_TYPE.get("Weaken", [])
            default_question["skills_rationale"] = "Applied rule-based fallback for stability."
            default_question["label_source"] = "fallback_rule"
            return default_question
        
        return result
        
    except json.JSONDecodeError:
        # JSON 解析失败，返回默认题目
        return default_question
    except Exception as e:
        # 其他异常，返回默认题目
        return default_question


def diagnose_wrong_answer(current_q: dict, user_choice: str, api_key: str) -> dict:
    """
    诊断学生错选的原因，生成苏格拉底引导所需的上下文
    
    Args:
        current_q: 当前题目完整快照
        user_choice: 学生选择的选项（A-E）
        api_key: DeepSeek API Key
        
    Returns:
        错因诊断字典，包含：
        - question_id
        - correct_choice
        - user_choice
        - core_conclusion
        - key_premises
        - assumed_link
        - why_user_choice_wrong
        - hint_plan
    """
    default_diagnosis = {
        "question_id": current_q.get("question_id", ""),
        "correct_choice": current_q.get("correct", ""),
        "user_choice": user_choice,
        "core_conclusion": "To be extracted from the stimulus",
        "key_premises": ["Premise 1", "Premise 2"],
        "assumed_link": "Key assumption/gap to identify",
        "why_user_choice_wrong": "This option does not effectively weaken/strengthen the argument",
        "hint_plan": [
            "Step 1: Guide student to identify the conclusion",
            "Step 2: Analyze the gap between premises and conclusion",
            "Step 3: Point out the flaw in the chosen option"
        ]
    }
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        prompt = f"""You are a GMAT Critical Reasoning diagnostic expert. Analyze why the student chose incorrectly and generate a Socratic guidance plan.

Current question:
- Stimulus: {current_q.get('stimulus', '')}
- Question: {current_q.get('question', '')}
- Choices:
{chr(10).join([f"  {choice}" for choice in current_q.get('choices', [])])}
- Correct answer: {current_q.get('correct', '')}
- Student's choice: {user_choice}

Output strict JSON in this format (all text in English):
{{
  "question_id": "{current_q.get('question_id', '')}",
  "correct_choice": "{current_q.get('correct', '')}",
  "user_choice": "{user_choice}",
  "core_conclusion": "<one sentence summarizing the argument's conclusion>",
  "key_premises": ["<premise 1>", "<premise 2>", "<premise 3>"],
  "assumed_link": "<key assumption/causal gap, 1-2 sentences>",
  "why_user_choice_wrong": "<why the chosen option is wrong, 2-3 sentences, must reference option content>",
  "hint_plan": [
    "<Step 1: identify conclusion>",
    "<Step 2: analyze assumption gap>",
    "<Step 3: compare options>"
  ]
}}

Output JSON only, no other text."""
        
        messages = [
            {"role": "system", "content": "You are a GMAT Critical Reasoning diagnostic expert. Output strict JSON only, no extra text."},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        # 提取 JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # 验证和修复
        for key in default_diagnosis:
            if key not in result:
                result[key] = default_diagnosis[key]
        
        return result
        
    except Exception as e:
        return default_diagnosis


def generate_detailed_explanation(current_q: dict, user_choice: str = None, is_correct: bool = False, api_key: str = None) -> str:
    """
    生成详细的 GMAT 标准解析（150-250 中文字）
    
    Args:
        current_q: 当前题目完整快照
        user_choice: 学生选择的选项（可选，用于动态生成错选分析）
        is_correct: 学生是否答对
        api_key: DeepSeek API Key（可选，如果提供则调用 LLM 生成）
        
    Returns:
        详细解析文本（150-250 字）
    """
    # 如果没有 API key 或生成失败，使用模板生成基础解析
    if not api_key:
        return _generate_template_explanation(current_q, user_choice, is_correct)
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        prompt = f"""Generate a detailed explanation (150-250 words in English) for the following GMAT Critical Reasoning question.

Question:
- Type: {current_q.get('question_type', 'Weaken')}
- Stimulus: {current_q.get('stimulus', '')}
- Question: {current_q.get('question', '')}
- Choices:
{chr(10).join([f"  {choice}" for choice in current_q.get('choices', [])])}
- Correct answer: {current_q.get('correct', '')}
"""
        
        if user_choice:
            prompt += f"- Student's choice: {user_choice} ({'correct' if is_correct else 'incorrect'})\n"
        
        prompt += """
Generate explanation following this structure (include all parts, 150-250 words total):

1) Correct answer: X
2) Question type: [Assumption / Weaken / Strengthen / Inference / Flaw]
3) Argument structure:
   - What is the conclusion (one sentence)
   - Key premises (list 2-3)
   - Where is the hidden assumption/causal gap (1-2 sentences)
4) Why the correct option works (cite stimulus; explain how it addresses the gap)
5) Why wrong options fail (if student was wrong, explain their choice; if correct, explain at least 2 distractors)
6) Takeaway: one sentence on how to approach similar questions

Output explanation text only, no headings or numbering."""
        
        messages = [
            {"role": "system", "content": "You are a GMAT Critical Reasoning explanation expert. Generate detailed, clear, helpful explanations."},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.4,
        )
        
        explanation = resp.choices[0].message.content.strip()
        
        # 确保长度足够
        if len(explanation) < 100:
            return _generate_template_explanation(current_q, user_choice, is_correct)
        
        return explanation
        
    except Exception as e:
        return _generate_template_explanation(current_q, user_choice, is_correct)


def _generate_template_explanation(current_q: dict, user_choice: str = None, is_correct: bool = False) -> str:
    """Generate template explanation when LLM call fails."""
    correct_choice = current_q.get("correct", "")
    question_type = current_q.get("question_type", "Weaken")
    stimulus = current_q.get("stimulus", "")
    choices = current_q.get("choices", [])
    
    explanation = f"""[Correct Answer: {correct_choice}]

[Question Type] {question_type}

[Argument Structure]
- Conclusion: Identify the core conclusion from the stimulus
- Premises: List key premises supporting the conclusion
- Hidden assumption: Unstated assumption or causal gap required by the argument

[Why the correct option works]
Option {correct_choice} addresses the gap via [specific mechanism], thus [weakening/strengthening/filling] the argument.

[Why other options are wrong]"""
    
    if user_choice and not is_correct:
        explanation += f"\nYour choice {user_choice} [explain why it is wrong]."
    
    wrong_options = [c for c in ["A", "B", "C", "D", "E"] if c != correct_choice][:2]
    for opt in wrong_options:
        explanation += f"\nOption {opt} [explain why wrong]."
    
    explanation += "\n\n[Takeaway] For similar questions, [key approach summary]."
    
    return explanation


def generate_all_diagnoses(current_q: dict, api_key: str) -> dict:
    """
    一次性分析所有错误选项，生成每个选项的逻辑漏洞和苏格拉底反问
    
    Args:
        current_q: 当前题目完整快照
        api_key: DeepSeek API Key
    
    Returns:
        字典，键是选项字母（'A', 'B', 'C', 'D', 'E'），值是包含以下字段的对象：
        - logic_gap: 逻辑漏洞描述
        - first_socratic_response: 第一句苏格拉底反问
        如果失败则返回空字典
    """
    try:
        # 获取正确答案
        correct_choice = current_q.get("correct", "") or current_q.get("correct_choice", "")
        
        # 识别所有选项字母
        all_options = ["A", "B", "C", "D", "E"]
        
        # 找出 4 个错误选项（排除正确答案）
        wrong_options = [opt for opt in all_options if opt != correct_choice]
        
        if len(wrong_options) != 4:
            print(f"警告：题目选项数量异常，正确答案：{correct_choice}，错误选项：{wrong_options}")
            return {}
        
        # 获取选项内容
        choices = current_q.get("choices", [])
        if len(choices) != 5:
            print(f"警告：选项数量不是5个，实际：{len(choices)}")
            return {}
        
        # 构建错误选项列表（包含字母和内容）
        wrong_options_with_content = []
        for opt in wrong_options:
            # 找到对应的选项内容
            opt_content = ""
            for choice in choices:
                if choice.startswith(f"{opt}.") or choice.startswith(f"{opt} "):
                    opt_content = choice
                    break
            wrong_options_with_content.append(f"{opt}: {opt_content}")
        
        wrong_options_str = "\n".join(wrong_options_with_content)
        
        # 构建 prompt
        prompt = f"""Analyze the 4 wrong options for this GMAT Critical Reasoning question.

Question info:
- Type: {current_q.get('question_type', 'Weaken')}
- Stimulus: {current_q.get('stimulus', '')}
- Question: {current_q.get('question', '')}
- Correct answer: {correct_choice}

Wrong options:
{wrong_options_str}

For each wrong option provide:
1. logic_gap: 1-2 sentences explaining why the option is wrong, the logical flaw
2. first_socratic_response: 1 sentence Socratic question guiding the student to see the problem (do not reveal the answer)

Output strict JSON in this format (all text in English):
{{
  "A": {{
    "logic_gap": "<logic gap description>",
    "first_socratic_response": "<first Socratic question>"
  }},
  "B": {{
    "logic_gap": "<logic gap description>",
    "first_socratic_response": "<first Socratic question>"
  }},
  "C": {{
    "logic_gap": "<logic gap description>",
    "first_socratic_response": "<first Socratic question>"
  }},
  "D": {{
    "logic_gap": "<logic gap description>",
    "first_socratic_response": "<first Socratic question>"
  }},
  "E": {{
    "logic_gap": "<logic gap description>",
    "first_socratic_response": "<first Socratic question>"
  }}
}}

Note: Only output analysis for wrong options (exclude correct answer {correct_choice}). Output JSON only, no other text."""
        
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        messages = [
            {"role": "system", "content": "You are a GMAT Critical Reasoning diagnostic expert. Output strict JSON only, no extra text."},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        # 提取 JSON（可能包含 markdown 代码块）
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # 解析 JSON
        result = json.loads(response_text)
        
        # 验证和修复结果格式
        if not isinstance(result, dict):
            return {}
        
        # 确保只包含错误选项，并且每个选项都有必需的字段
        validated_result = {}
        for opt in wrong_options:
            if opt in result and isinstance(result[opt], dict):
                opt_data = result[opt]
                validated_result[opt] = {
                    "logic_gap": opt_data.get("logic_gap", "Logic gap needs analysis."),
                    "first_socratic_response": opt_data.get("first_socratic_response", "Please reconsider this option.")
                }
        
        return validated_result
        
    except json.JSONDecodeError:
        print("JSON 解析失败：generate_all_diagnoses")
        return {}
    except Exception as e:
        print(f"生成所有诊断失败：{e}")
        return {}
