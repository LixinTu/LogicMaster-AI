"""
LLM 服务模块
使用 DeepSeek API 进行真实调用
"""

import json
from openai import OpenAI
import uuid

# ========== 规则+LLM混合技能标签系统 ==========

# 规则映射：每种题型的候选技能池
RULE_SKILL_POOL_BY_TYPE = {
    "Weaken": ["因果推理", "替代解释", "排除干扰项", "假设识别"],
    "Strengthen": ["因果推理", "证据强度", "结构化表达", "假设识别"],
    "Assumption": ["假设识别", "因果推理", "结构化表达"],
    "Inference": ["排除干扰项", "证据强度", "结构化表达"],
    "Flaw": ["结构化表达", "因果推理", "假设识别", "证据强度"]
}

# 每种题型的默认 skills（fallback）
DEFAULT_SKILLS_BY_TYPE = {
    "Weaken": ["因果推理", "替代解释"],
    "Strengthen": ["因果推理", "证据强度"],
    "Assumption": ["假设识别", "因果推理"],
    "Inference": ["排除干扰项", "证据强度"],
    "Flaw": ["结构化表达", "假设识别"]
}

SYSTEM_PROMPT = (
    "你是一个严厉但有耐心的 GMAT Critical Reasoning 逻辑考官。"
    "你绝不直接给出标准答案。你只做苏格拉底式追问："
    "指出论证中的假设、漏洞、因果跳跃、样本偏差等，并用反问推动用户自我修正。"
    "回复尽量短（1-3 句），每次只追问一个关键点。"
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
            enhanced_system_prompt += f"\n\n【重要约束】\n"
            enhanced_system_prompt += f"- 你只能讨论题目 ID: {current_q_id}，禁止换题或引用其他题目。\n"
            enhanced_system_prompt += f"- 每次回复必须先确认当前题：例如「针对本题（ID: {current_q_id}），我们先看...」\n"
            enhanced_system_prompt += f"- 必须引用题干信息（{current_q.get('stimulus', '')[:50]}...）和选项字母（A-E）。\n"
            enhanced_system_prompt += f"- 禁止直接给出正确选项字母，只能通过追问引导。\n"
            
            if socratic_context and socratic_context.get("hint_plan"):
                enhanced_system_prompt += f"- 按照以下引导计划逐步推进：{socratic_context.get('hint_plan', [])}\n"

        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # 如果有当前题目信息，添加到上下文
        if current_q:
            question_context = f"【当前题目 ID: {current_q_id}】\n"
            question_context += f"题干：{current_q.get('stimulus', '')}\n"
            question_context += f"问题：{current_q.get('question', '')}\n"
            question_context += f"选项：\n"
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
    # 默认题目
    default_question = {
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
        "explanation": "C 直接指出消费者需求有限，削弱了市场份额提升的假设"
    }
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 根据 theta 确定难度
        if theta < -1.0:
            difficulty = "easy"
            difficulty_desc = "简单（短文本、单一因果链、选项更直观）"
        elif theta <= 1.0:
            difficulty = "medium"
            difficulty_desc = "中等（存在替代解释/混杂变量，选项更接近）"
        else:
            difficulty = "hard"
            difficulty_desc = "困难（多因素、多层假设、强干扰项）"
        
        # 随机选择题型（从所有题型中选择）
        import random
        question_type = random.choice(["Weaken", "Strengthen", "Assumption", "Inference", "Flaw"])
        
        # 获取该题型的候选技能池
        skill_pool = RULE_SKILL_POOL_BY_TYPE.get(question_type, DEFAULT_SKILLS_BY_TYPE.get("Weaken", []))
        skill_pool_str = "、".join(skill_pool)
        
        # 构建 prompt（包含技能标签要求）
        prompt = f"""请生成一道 GMAT Critical Reasoning 中文题目。

要求：
- 难度：{difficulty} ({difficulty_desc})
- 题型：{question_type}（从 Weaken/Strengthen/Assumption/Inference/Flaw 中选择）
- 题干（stimulus）：2-5句，描述一个场景和论证
- 问题（question）：一句话提问
- 选项（choices）：5个选项，标记为 A-E
- 正确答案（correct）：A、B、C、D 或 E 中的一个
- 解释（explanation）：一句话解释正确选项为什么对（<=30字）

【技能标签要求（重要）】
- 题型为 {question_type}，候选技能池：{skill_pool_str}
- 必须从候选池中选择 2-3 个技能，组成 skills 数组
- 必须输出 skills_rationale（<=60字），说明为什么这题对应这些技能

请只输出一个严格 JSON 对象，格式如下：
{{
  "difficulty": "{difficulty}",
  "question_type": "{question_type}",
  "stimulus": "<题干背景，2-5句>",
  "question": "<问题一句话>",
  "choices": ["A ...", "B ...", "C ...", "D ...", "E ..."],
  "correct": "<A|B|C|D|E>",
  "explanation": "<一句话解释，<=30字>",
  "skills": ["<从候选池中选择2-3个技能>"],
  "skills_rationale": "<说明为什么这题对应这些技能，<=60字>"
}}

只输出 JSON，不要包含任何其他文本。"""
        
        messages = [
            {"role": "system", "content": "你是 GMAT Critical Reasoning 题目生成专家。只输出严格 JSON，不要包含多余文本。"},
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
        "core_conclusion": "需要从题干中提取",
        "key_premises": ["前提1", "前提2"],
        "assumed_link": "关键假设需要识别",
        "why_user_choice_wrong": "该选项未能有效削弱/加强论证",
        "hint_plan": [
            "第一步：引导学生识别结论",
            "第二步：分析前提与结论的缺口",
            "第三步：指出错选选项的问题"
        ]
    }
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        prompt = f"""你是 GMAT Critical Reasoning 错因诊断专家。分析学生错选的原因，生成苏格拉底引导计划。

当前题目：
- 题干：{current_q.get('stimulus', '')}
- 问题：{current_q.get('question', '')}
- 选项：
{chr(10).join([f"  {choice}" for choice in current_q.get('choices', [])])}
- 正确答案：{current_q.get('correct', '')}
- 学生选择：{user_choice}

请输出严格 JSON，格式如下：
{{
  "question_id": "{current_q.get('question_id', '')}",
  "correct_choice": "{current_q.get('correct', '')}",
  "user_choice": "{user_choice}",
  "core_conclusion": "<用一句话总结论证的结论>",
  "key_premises": ["<前提1>", "<前提2>", "<前提3>"],
  "assumed_link": "<最关键假设/因果链缺口，1-2句话>",
  "why_user_choice_wrong": "<为什么学生选择的选项不对，2-3句话，必须引用选项内容>",
  "hint_plan": [
    "<第一步引导：识别结论>",
    "<第二步引导：分析假设缺口>",
    "<第三步引导：对比选项>"
  ]
}}

只输出 JSON，不要包含任何其他文本。"""
        
        messages = [
            {"role": "system", "content": "你是 GMAT Critical Reasoning 错因诊断专家。只输出严格 JSON，不要包含多余文本。"},
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
        
        prompt = f"""请为以下 GMAT Critical Reasoning 题目生成详细解析（150-250 中文字）。

题目：
- 题型：{current_q.get('question_type', 'Weaken')}
- 题干：{current_q.get('stimulus', '')}
- 问题：{current_q.get('question', '')}
- 选项：
{chr(10).join([f"  {choice}" for choice in current_q.get('choices', [])])}
- 正确答案：{current_q.get('correct', '')}
"""
        
        if user_choice:
            prompt += f"- 学生选择：{user_choice}（{'正确' if is_correct else '错误'}）\n"
        
        prompt += """
请按照以下结构生成解析（必须包含全部部分，总计 150-250 字）：

1) 正确答案：X
2) 题型：[Assumption / Weaken / Strengthen / Inference / Flaw]
3) 论证结构拆解：
   - 结论是什么（用一句话复述）
   - 前提有哪些（列2-3条）
   - 隐含假设/因果链缺口在哪里（1-2句）
4) 为什么正确选项对（必须引用题干，并解释它如何作用在"缺口"上）
5) 为什么你选的选项错（如果学生答错，针对其错选解释；如果学生答对，至少提2个干扰项为什么不对）
6) Takeaway：一句话总结"下次遇到这类题怎么做"

只输出解析文本，不要包含标题或编号。"""
        
        messages = [
            {"role": "system", "content": "你是 GMAT Critical Reasoning 解析专家。生成详细、清晰、对学生有帮助的解析。"},
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
    """生成模板解析（当 LLM 调用失败时使用）"""
    correct_choice = current_q.get("correct", "")
    question_type = current_q.get("question_type", "Weaken")
    stimulus = current_q.get("stimulus", "")
    choices = current_q.get("choices", [])
    
    explanation = f"""【正确答案：{correct_choice}】

【题型】{question_type}

【论证结构拆解】
- 结论：需要从题干中识别核心结论
- 前提：列出支持结论的关键前提
- 隐含假设：论证中未明说但必需的假设或因果链缺口

【为什么正确选项对】
正确选项 {correct_choice} 通过[具体机制]作用在论证的[缺口位置]，从而[削弱/加强/填补]了论证。

【为什么其他选项不对】"""
    
    if user_choice and not is_correct:
        explanation += f"\n你选择的选项 {user_choice} [具体说明为什么不对]。"
    
    # 至少提2个干扰项
    wrong_options = [c for c in ["A", "B", "C", "D", "E"] if c != correct_choice][:2]
    for opt in wrong_options:
        explanation += f"\n选项 {opt} [说明为什么不对]。"
    
    explanation += "\n\n【Takeaway】遇到这类题目时，要[关键方法总结]。"
    
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
        prompt = f"""请分析这道 GMAT Critical Reasoning 题目的 4 个错误选项。

题目信息：
- 题型：{current_q.get('question_type', 'Weaken')}
- 题干：{current_q.get('stimulus', '')}
- 问题：{current_q.get('question', '')}
- 正确答案：{correct_choice}

错误选项：
{wrong_options_str}

请为每个错误选项提供：
1. logic_gap（逻辑漏洞）：1-2句话说明这个选项为什么不对，指出其逻辑漏洞
2. first_socratic_response（第一句苏格拉底反问）：1句话，用反问的方式引导学生思考这个选项的问题，不能直接给答案

请输出严格 JSON 对象，格式如下：
{{
  "A": {{
    "logic_gap": "<逻辑漏洞描述>",
    "first_socratic_response": "<第一句苏格拉底反问>"
  }},
  "B": {{
    "logic_gap": "<逻辑漏洞描述>",
    "first_socratic_response": "<第一句苏格拉底反问>"
  }},
  "C": {{
    "logic_gap": "<逻辑漏洞描述>",
    "first_socratic_response": "<第一句苏格拉底反问>"
  }},
  "D": {{
    "logic_gap": "<逻辑漏洞描述>",
    "first_socratic_response": "<第一句苏格拉底反问>"
  }},
  "E": {{
    "logic_gap": "<逻辑漏洞描述>",
    "first_socratic_response": "<第一句苏格拉底反问>"
  }}
}}

注意：只输出错误选项（排除正确答案 {correct_choice}）的分析。只输出 JSON，不要包含任何其他文本。"""
        
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        messages = [
            {"role": "system", "content": "你是 GMAT Critical Reasoning 错因分析专家。只输出严格 JSON，不要包含多余文本。"},
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
                    "logic_gap": opt_data.get("logic_gap", "逻辑漏洞需要分析"),
                    "first_socratic_response": opt_data.get("first_socratic_response", "请重新思考这个选项。")
                }
        
        return validated_result
        
    except json.JSONDecodeError:
        print("JSON 解析失败：generate_all_diagnoses")
        return {}
    except Exception as e:
        print(f"生成所有诊断失败：{e}")
        return {}
