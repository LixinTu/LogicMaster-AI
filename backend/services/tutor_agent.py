"""
LangChain 驱动的苏格拉底 Tutor Agent
使用 ChatOpenAI + ChatPromptTemplate 实现结构化 prompt 和链式调用
"""

import json
import logging
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------- JSON 提取工具函数 ----------

def _extract_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON（兼容 markdown 代码块）"""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


class SocraticTutorAgent:
    """
    苏格拉底 Tutor Agent，使用 LangChain 框架。
    通过 ChatOpenAI 连接 DeepSeek API，使用结构化 prompt 模板。
    """

    def __init__(self, api_key: Optional[str] = None):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key or settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=0.4,
        )
        self.str_parser = StrOutputParser()

        # ---------- Prompt 模板 ----------

        self.diagnosis_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a GMAT Critical Reasoning diagnostic expert. "
             "Analyze the student's error and output strict JSON only, no extra text."),
            ("human", """\
Analyze why the student chose incorrectly and identify the logic gap.

Question info:
- Type: {question_type}
- Stimulus: {stimulus}
- Question: {question_stem}
- Choices:
{choices_text}
- Correct answer: {correct_choice}
- Student's choice: {user_choice}

Output strict JSON:
{{
  "logic_gap": "<1-2 sentences: the specific logical flaw in the student's reasoning>",
  "error_type": "<one of: causal_confusion | correlation_causation | scope_shift | overlooked_alternative | hasty_generalization | false_analogy | sampling_bias | other>",
  "core_conclusion": "<one sentence: the argument's conclusion>",
  "key_assumption": "<one sentence: the hidden assumption the student missed>",
  "why_wrong": "<2-3 sentences: why the student's choice fails, referencing the option content>"
}}

Output JSON only."""),
        ])

        self.hint_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict but patient GMAT Socratic tutor. "
             "Never reveal the correct answer. Only use questions to guide the student. "
             "Keep your response to 1-3 sentences, focusing on one key point."),
            ("human", """\
Context:
- The student chose {user_choice} (incorrect). The correct answer is hidden.
- Logic gap identified: {logic_gap}
- Error type: {error_type}
- This is hint #{hint_number} of 3.

Hint strength instructions:
{strength_instruction}

Question context:
- Stimulus: {stimulus}
- Question: {question_stem}

Chat history:
{chat_history_text}

Generate a Socratic question that guides the student toward recognizing their error.
Do NOT reveal the correct answer. Output the hint text only."""),
        ])

        self.understanding_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an educational assessment expert. "
             "Evaluate the student's understanding level and output strict JSON only."),
            ("human", """\
The student was asked about this GMAT question:
- Logic gap: {logic_gap}
- Key assumption: {key_assumption}

The student's latest response:
"{student_response}"

Recent chat context:
{chat_history_text}

Evaluate the student's understanding. Output strict JSON:
{{
  "understanding": "<confused | partial | clear>",
  "reasoning": "<one sentence explaining your judgment>"
}}

Criteria:
- "confused": Student shows no grasp of the logic gap or repeats the same mistake
- "partial": Student recognizes part of the issue but hasn't fully connected the reasoning
- "clear": Student demonstrates understanding of the assumption/gap and why their choice was wrong

Output JSON only."""),
        ])

        self.blooms_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an educational assessment expert using Bloom's Taxonomy. "
             "Classify the student's cognitive level and output strict JSON only."),
            ("human", """\
The student was asked about a GMAT Critical Reasoning question:
- Logic gap: {logic_gap}
- Key assumption: {key_assumption}

The student's latest response:
"{student_response}"

Recent chat context:
{chat_history_text}

Classify the student's response into one of 6 Bloom's Taxonomy levels:
1-Remember: student just recalls facts ("the answer is C")
2-Understand: student can explain in own words ("the argument assumes correlation means causation")
3-Apply: student applies the concept to the specific question ("here the author assumes that because revenue rose after the campaign, the campaign caused it")
4-Analyze: student breaks down the logical structure ("the premise is X, the conclusion is Y, the gap is the unstated assumption that Z")
5-Evaluate: student judges the argument quality ("this is a weak argument because it ignores alternative explanations like seasonal trends")
6-Create: student generates counterexamples or new arguments ("a better study would control for other variables like price changes")

Output strict JSON:
{{
  "level": <integer 1-6>,
  "reasoning": "<one sentence explaining your classification>"
}}

Output JSON only."""),
        ])

    # ---------- 核心方法 ----------

    def diagnose_error(
        self,
        question: Dict[str, Any],
        user_choice: str,
        correct_choice: str,
    ) -> Dict[str, Any]:
        """
        诊断学生错误原因

        Args:
            question: 完整题目字典（含 stimulus, question, choices 等）
            user_choice: 学生选择 (A-E)
            correct_choice: 正确答案 (A-E)

        Returns:
            {
                "logic_gap": str,
                "error_type": str,
                "core_conclusion": str,
                "key_assumption": str,
                "why_wrong": str,
            }
        """
        default = {
            "logic_gap": "The student may have confused correlation with causation or missed a key assumption.",
            "error_type": "other",
            "core_conclusion": "To be determined from the stimulus.",
            "key_assumption": "A hidden assumption connects the premises to the conclusion.",
            "why_wrong": "The chosen option does not address the core logical gap in the argument.",
        }

        try:
            choices_text = "\n".join(
                f"  {c}" for c in question.get("choices", [])
            )
            chain = self.diagnosis_prompt | self.llm | self.str_parser
            raw = chain.invoke({
                "question_type": question.get("question_type", "Weaken"),
                "stimulus": question.get("stimulus", ""),
                "question_stem": question.get("question", ""),
                "choices_text": choices_text,
                "correct_choice": correct_choice,
                "user_choice": user_choice,
            })
            result = _extract_json(raw)
            # 确保所有字段存在
            for key in default:
                if key not in result:
                    result[key] = default[key]
            return result

        except Exception as e:
            logger.warning("diagnose_error failed, using defaults: %s", e)
            return default

    def generate_socratic_hint(
        self,
        question: Dict[str, Any],
        user_choice: str,
        logic_gap: str,
        error_type: str,
        hint_count: int,
        chat_history: Optional[List[Dict[str, str]]] = None,
        blooms_level: Optional[int] = None,
    ) -> str:
        """
        生成苏格拉底式提示，强度随 hint_count 递增，并根据 Bloom's level 调整策略

        Args:
            question: 完整题目字典
            user_choice: 学生选择
            logic_gap: 已诊断的逻辑漏洞
            error_type: 错误类型
            hint_count: 当前是第几次提示 (0-based → 显示为 1-based)
            chat_history: 对话历史
            blooms_level: 学生当前 Bloom's 层级 (1-6)，用于调整策略

        Returns:
            提示文本字符串
        """
        # 根据 hint_count 调整基础提示强度
        hint_number = hint_count + 1  # 1-based for display

        # Bloom's level 调整策略
        blooms_adjustment = ""
        if blooms_level is not None:
            if blooms_level <= 2:
                # Remember/Understand → 更多脚手架
                blooms_adjustment = (
                    " The student is at a basic cognitive level (Remember/Understand). "
                    "Provide more scaffolding: break the problem into smaller steps, "
                    "define key terms, and ask simpler questions."
                )
            elif blooms_level <= 4:
                # Apply/Analyze → 推动分析
                blooms_adjustment = (
                    " The student shows some analytical ability (Apply/Analyze). "
                    "Push them toward deeper analysis: ask them to identify the premise, "
                    "conclusion, and the gap between them."
                )
            # 5-6 (Evaluate/Create) → 不需要额外调整，会在 /continue 中触发结束

        if hint_number == 1:
            strength = (
                "Hint #1 (gentle): Ask a broad, open-ended question that nudges the student "
                "to re-examine the argument's structure. Do NOT mention the specific flaw directly."
                + blooms_adjustment
            )
        elif hint_number == 2:
            strength = (
                "Hint #2 (moderate): Point toward the specific area where the logic breaks down. "
                "You may reference the type of reasoning error (e.g., 'Have you considered whether "
                "this is a causal claim?') but do NOT name the correct answer."
                + blooms_adjustment
            )
        else:
            strength = (
                "Hint #3 (direct): Clearly describe the logical flaw and ask the student to "
                "reconsider which option addresses it. Be specific about the assumption gap. "
                "Still do NOT reveal the correct answer letter."
                + blooms_adjustment
            )

        # 格式化对话历史
        history_text = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role = "Student" if msg["role"] == "user" else "Tutor"
                history_text += f"{role}: {msg['content']}\n"
        if not history_text:
            history_text = "(no prior conversation)"

        try:
            chain = self.hint_prompt | self.llm | self.str_parser
            hint = chain.invoke({
                "user_choice": user_choice,
                "logic_gap": logic_gap,
                "error_type": error_type,
                "hint_number": hint_number,
                "strength_instruction": strength,
                "stimulus": question.get("stimulus", ""),
                "question_stem": question.get("question", ""),
                "chat_history_text": history_text,
            })
            return hint.strip()

        except Exception as e:
            logger.warning("generate_socratic_hint failed: %s", e)
            # 按 hint 强度返回不同的默认回复
            if hint_number == 1:
                return "Let's take a step back. What is the main conclusion of the argument?"
            elif hint_number == 2:
                return "Think about the assumption connecting the premises to the conclusion. Is it valid?"
            else:
                return "Consider whether your chosen option truly addresses the core logical gap in the argument."

    def evaluate_blooms_level(
        self,
        student_response: str,
        logic_gap: str,
        key_assumption: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        使用 Bloom's Taxonomy 评估学生认知水平。

        6 个层级：
        1-Remember, 2-Understand, 3-Apply, 4-Analyze, 5-Evaluate, 6-Create

        Args:
            student_response: 学生最新回复
            logic_gap: 已识别的逻辑漏洞
            key_assumption: 关键假设
            chat_history: 对话历史

        Returns:
            {
                "level": int (1-6),
                "level_name": str,
                "reasoning": str,
                "mapped_understanding": str  ("confused"|"partial"|"clear")
            }
        """
        default = {
            "level": 1,
            "level_name": "Remember",
            "reasoning": "Unable to evaluate.",
            "mapped_understanding": "confused",
        }

        history_text = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role = "Student" if msg["role"] == "user" else "Tutor"
                history_text += f"{role}: {msg['content']}\n"
        if not history_text:
            history_text = "(no prior conversation)"

        try:
            chain = self.blooms_prompt | self.llm | self.str_parser
            raw = chain.invoke({
                "logic_gap": logic_gap,
                "key_assumption": key_assumption,
                "student_response": student_response,
                "chat_history_text": history_text,
            })
            result = _extract_json(raw)

            level = int(result.get("level", 1))
            level = max(1, min(6, level))

            level_names = {
                1: "Remember", 2: "Understand", 3: "Apply",
                4: "Analyze", 5: "Evaluate", 6: "Create",
            }
            level_name = level_names.get(level, "Remember")

            # 映射到旧的三级制（向后兼容）
            if level <= 2:
                mapped = "confused"
            elif level <= 4:
                mapped = "partial"
            else:
                mapped = "clear"

            return {
                "level": level,
                "level_name": level_name,
                "reasoning": result.get("reasoning", ""),
                "mapped_understanding": mapped,
            }

        except Exception as e:
            logger.warning("evaluate_blooms_level failed: %s", e)
            return default

    def evaluate_understanding(
        self,
        student_response: str,
        logic_gap: str,
        key_assumption: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, str]:
        """
        评估学生的理解程度（向后兼容接口）。

        内部调用 evaluate_blooms_level() 并映射到 confused/partial/clear。

        Returns:
            {"understanding": "confused|partial|clear", "reasoning": str}
        """
        blooms = self.evaluate_blooms_level(
            student_response=student_response,
            logic_gap=logic_gap,
            key_assumption=key_assumption,
            chat_history=chat_history,
        )
        return {
            "understanding": blooms["mapped_understanding"],
            "reasoning": blooms["reasoning"],
        }

    def generate_conclusion(
        self,
        question: Dict[str, Any],
        correct_choice: str,
        logic_gap: str,
        student_understanding: str,
    ) -> str:
        """
        生成对话结束时的总结消息

        Args:
            question: 题目字典
            correct_choice: 正确答案
            logic_gap: 逻辑漏洞
            student_understanding: 最终理解程度

        Returns:
            结束消息文本
        """
        conclusion_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a GMAT Socratic tutor wrapping up a remediation session. "
             "Now you MAY reveal the correct answer. Be encouraging and concise (3-5 sentences)."),
            ("human", """\
The student worked through a remediation session for this question.

- Question type: {question_type}
- Stimulus: {stimulus}
- Correct answer: {correct_choice}
- Logic gap they struggled with: {logic_gap}
- Final understanding level: {understanding}

Generate a warm, encouraging conclusion that:
1. Reveals the correct answer is {correct_choice}
2. Briefly explains the key takeaway
3. Encourages the student based on their understanding level

Output the conclusion text only."""),
        ])

        try:
            chain = conclusion_prompt | self.llm | self.str_parser
            return chain.invoke({
                "question_type": question.get("question_type", "Weaken"),
                "stimulus": question.get("stimulus", "")[:300],
                "correct_choice": correct_choice,
                "logic_gap": logic_gap,
                "understanding": student_understanding,
            }).strip()

        except Exception as e:
            logger.warning("generate_conclusion failed: %s", e)
            if student_understanding == "clear":
                return f"Great work! The correct answer is {correct_choice}. You identified the key logical gap. Keep this approach in mind for similar questions."
            else:
                return f"The correct answer is {correct_choice}. The key insight here is the hidden assumption in the argument. Review this type of reasoning gap for future questions."


# 模块级单例
_tutor_agent: Optional[SocraticTutorAgent] = None


def get_tutor_agent() -> SocraticTutorAgent:
    """获取 SocraticTutorAgent 单例"""
    global _tutor_agent
    if _tutor_agent is None:
        _tutor_agent = SocraticTutorAgent()
    return _tutor_agent
