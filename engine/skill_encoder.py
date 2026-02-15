"""
技能编码器（Skill Encoder）

将技能名称映射为数值向量，供 DKT 模型使用。

核心功能：
- build_vocab: 从数据库扫描所有唯一技能，构建词表
- encode_interaction: 将 (技能列表, 是否正确) 编码为 2K 维向量
  - 前 K 维 = 正确通道（答对时对应技能位为 1）
  - 后 K 维 = 错误通道（答错时对应技能位为 1）
- decode_predictions: K 维输出向量 → {技能名: 概率}
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional

import numpy as np


def _get_default_db_path() -> str:
    """项目根目录下的 logicmaster.db"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "logicmaster.db")


class SkillEncoder:
    """
    技能编码器：技能名称 ↔ 数值 ID 映射 + 向量编码。

    词表按字母序排列，保证确定性。
    """

    def __init__(self):
        self.skill_to_id: Dict[str, int] = {}
        self.id_to_skill: Dict[int, str] = {}

    @property
    def num_skills(self) -> int:
        """词表大小 K"""
        return len(self.skill_to_id)

    def build_vocab(self, db_path: Optional[str] = None) -> int:
        """
        从数据库 questions.content JSON 扫描所有唯一技能，构建词表。

        Args:
            db_path: 数据库路径，默认 logicmaster.db

        Returns:
            词表大小 K
        """
        db_path = db_path or _get_default_db_path()
        skills_set: set = set()

        try:
            conn = sqlite3.connect(db_path, timeout=10)
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM questions")
            rows = cursor.fetchall()
            conn.close()

            for (content_json,) in rows:
                try:
                    content = json.loads(content_json)
                    skills = content.get("skills", [])
                    if isinstance(skills, list):
                        for s in skills:
                            if isinstance(s, str) and s.strip():
                                skills_set.add(s.strip())
                except (json.JSONDecodeError, AttributeError):
                    continue
        except Exception as e:
            print(f"build_vocab failed: {e}")

        # 按字母序排列，确保确定性映射
        sorted_skills = sorted(skills_set)
        self.skill_to_id = {skill: idx for idx, skill in enumerate(sorted_skills)}
        self.id_to_skill = {idx: skill for idx, skill in enumerate(sorted_skills)}

        return self.num_skills

    def encode_interaction(
        self, skill_names: List[str], is_correct: bool
    ) -> np.ndarray:
        """
        将单次交互编码为 2K 维向量。

        前 K 维 = 正确通道，后 K 维 = 错误通道。
        答对时，对应技能在前 K 维置 1；答错时，在后 K 维置 1。

        Args:
            skill_names: 涉及的技能名称列表
            is_correct: 是否答对

        Returns:
            (2K,) 向量
        """
        k = self.num_skills
        vec = np.zeros(2 * k, dtype=np.float32)

        for skill in skill_names:
            idx = self.skill_to_id.get(skill)
            if idx is not None:
                if is_correct:
                    vec[idx] = 1.0
                else:
                    vec[k + idx] = 1.0

        return vec

    def decode_predictions(self, output_vector: np.ndarray) -> Dict[str, float]:
        """
        将 K 维输出向量解码为 {技能名: 掌握概率}。

        Args:
            output_vector: (K,) 概率向量

        Returns:
            {skill_name: probability}
        """
        result: Dict[str, float] = {}
        for idx in range(min(len(output_vector), self.num_skills)):
            skill = self.id_to_skill.get(idx)
            if skill is not None:
                result[skill] = float(output_vector[idx])
        return result

    def save_vocab(self, filepath: str) -> None:
        """将词表保存为 JSON 文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = {
            "skill_to_id": self.skill_to_id,
            "id_to_skill": {str(k): v for k, v in self.id_to_skill.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath: str) -> int:
        """
        从 JSON 文件加载词表。

        Returns:
            词表大小 K
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.skill_to_id = data["skill_to_id"]
        self.id_to_skill = {int(k): v for k, v in data["id_to_skill"].items()}
        return self.num_skills


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_encoder: Optional[SkillEncoder] = None


def get_skill_encoder(db_path: Optional[str] = None) -> SkillEncoder:
    """获取全局 SkillEncoder 实例（首次调用时自动构建词表）"""
    global _encoder
    if _encoder is None:
        _encoder = SkillEncoder()
        _encoder.build_vocab(db_path)
    return _encoder
