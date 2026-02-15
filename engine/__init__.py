"""
Engine 模块：核心算法引擎
包含评分算法和推荐算法
"""

from typing import TYPE_CHECKING

from .scoring import calculate_new_theta, estimate_gmat_score
from .recommender import analyze_weak_skills, generate_next_question
from .skill_encoder import SkillEncoder, get_skill_encoder
from .dkt_model import get_dkt_model

if TYPE_CHECKING:
    from utils.db_handler import DatabaseManager

__all__ = [
    "calculate_new_theta",
    "estimate_gmat_score",
    "analyze_weak_skills",
    "generate_next_question",
    "SkillEncoder",
    "get_skill_encoder",
    "get_dkt_model",
]
