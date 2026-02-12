"""
Batch generate questions and save to database.
"""

import os
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dotenv import load_dotenv
from llm_service import generate_question, generate_detailed_explanation, generate_all_diagnoses
from utils.db_handler import DatabaseManager, get_db_manager

# API Rate Limit 重试配置
MAX_RETRY_ATTEMPTS = 5
INITIAL_RETRY_DELAY = 2  # 初始重试延迟（秒）
MAX_RETRY_DELAY = 120  # 最大重试延迟（秒）
BASE_SLEEP_TIME = 1  # 基础睡眠时间（秒），用于避免频率限制

# 加载环境变量
load_dotenv()

# 从环境变量中读取 API Key
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    print("Error: DEEPSEEK_API_KEY not found in environment")
    print("Create .env and add: DEEPSEEK_API_KEY=your_api_key_here")
    raise ValueError("DEEPSEEK_API_KEY not set")


def call_with_retry(func: Callable, *args: Any, **kwargs: Any) -> Optional[Any]:
    """
    带重试机制的函数调用，处理 API Rate Limit 错误
    
    Args:
        func: 要调用的函数
        *args, **kwargs: 传递给函数的参数
    
    Returns:
        函数返回值，失败返回 None
    """
    retry_delay: float = INITIAL_RETRY_DELAY
    
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检测 Rate Limit 错误（HTTP 429 或相关错误信息）
            is_rate_limit = (
                "429" in error_msg or
                "rate limit" in error_msg or
                "too many requests" in error_msg or
                "quota" in error_msg or
                "limit" in error_msg
            )
            
            if is_rate_limit and attempt < MAX_RETRY_ATTEMPTS - 1:
                # 指数退避策略
                actual_delay = min(retry_delay, MAX_RETRY_DELAY)
                print(f"  ⚠ API Rate Limit. Waiting {actual_delay}s before retry... (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})")
                time.sleep(actual_delay)
                retry_delay *= 2  # 指数退避：每次重试延迟翻倍
                continue
            else:
                # 非 Rate Limit 错误或已达到最大重试次数
                raise
    
    return None


def worker(count: int) -> None:
    """
    批量生成题目并存入数据库
    
    Args:
        count: 要生成的题目数量
    """
    if not API_KEY:
        print("Error: API_KEY not set")
        return
    
    db_manager: DatabaseManager = get_db_manager()
    theta_values: List[float] = [-2.0, 0.0, 2.0]
    success_count: int = 0
    fail_count: int = 0
    
    print(f"Generating {count} questions...")
    
    for i in range(count):
        # 轮换 theta 值
        theta = theta_values[i % len(theta_values)]
        
        print(f"\n[{i+1}/{count}] Generating with theta={theta}...")
        
        try:
            # 调用 generate_question 生成题目（带重试机制）
            question_data = call_with_retry(generate_question, theta, API_KEY)
            
            if not question_data:
                print(f"  Failed: returned empty data")
                fail_count += 1
                time.sleep(1)
                continue
            
            # 如果没有 question_id，生成一个 UUID
            if "id" not in question_data or not question_data.get("id"):
                question_id = str(uuid.uuid4())[:8]
                question_data["id"] = question_id
                print(f"  Generated question_id: {question_id}")
            
            # 生成详细解析（作为标准解析模板）
            print(f"  Generating detailed explanation...")
            try:
                detailed_explanation = call_with_retry(
                    generate_detailed_explanation,
                    current_q=question_data,
                    user_choice=None,
                    is_correct=True,
                    api_key=API_KEY
                )
                if detailed_explanation:
                    question_data["detailed_explanation"] = detailed_explanation
                    print(f"  ✓ Detailed explanation generated")
                else:
                    print(f"  ⚠ Failed to generate detailed explanation, using basic")
                    question_data["detailed_explanation"] = question_data.get("explanation", "")
            except Exception as e:
                print(f"  ⚠ Failed: {e}, using basic explanation")
                question_data["detailed_explanation"] = question_data.get("explanation", "")
            
            # 生成所有错误选项的诊断（预生成，用于苏格拉底引导）
            print(f"  Generating diagnoses for 4 distractors...")
            try:
                all_diagnoses = call_with_retry(generate_all_diagnoses, question_data, API_KEY)
                if all_diagnoses:
                    question_data["diagnoses"] = all_diagnoses
                    print(f"  ✓ Diagnoses for 4 distractors generated")
                else:
                    print(f"  ⚠ Failed to generate diagnoses, using empty dict")
                    question_data["diagnoses"] = {}
            except Exception as e:
                print(f"  ⚠ Failed: {e}, using empty diagnoses")
                question_data["diagnoses"] = {}
            
            # 准备存入数据库的数据格式
            # 将题目内容（除了 id, question_type, difficulty 之外的字段）放入 content
            db_data = {
                "id": question_data["id"],
                "question_type": question_data.get("question_type", "Weaken"),
                "difficulty": question_data.get("difficulty", "medium"),
                "content": {
                    k: v for k, v in question_data.items()
                    if k not in ["id", "question_type", "difficulty", "elo_difficulty", "is_verified"]
                },
                "elo_difficulty": 1500.0,  # 默认 ELO 值
                "is_verified": False
            }
            
            # 存入数据库
            if db_manager.add_question(db_data):
                success_count += 1
                print(f"  ✓ Question {db_data['id']} ({db_data['difficulty']}, {db_data['question_type']}) saved")
            else:
                fail_count += 1
                print(f"  ✗ Failed to save question {db_data['id']}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            fail_count += 1
        
        # 避免 API 速率限制（基础延迟）
        if i < count - 1:  # 最后一次不需要等待
            time.sleep(BASE_SLEEP_TIME)
    
    print(f"\nDone. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    worker(50)
