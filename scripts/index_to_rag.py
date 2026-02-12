"""
批量索引脚本：将 SQLite 中的题目索引到 Qdrant 向量数据库
支持 --force 重新索引所有题目，默认跳过已索引的题目（upsert 幂等）

用法:
    python scripts/index_to_rag.py          # 索引所有题目
    python scripts/index_to_rag.py --force  # 强制重新索引
"""

import sys
import os
import argparse
import json
import sqlite3

# 将项目根目录加入 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.rag_service import get_rag_service


def load_questions_from_db(db_path: str) -> list:
    """从 SQLite 数据库加载所有已验证的题目"""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, question_type, difficulty, content, elo_difficulty
        FROM questions
        WHERE is_verified != 0
    """)
    rows = cursor.fetchall()
    conn.close()

    questions = []
    for row in rows:
        qid, qtype, difficulty, content_json, elo = row
        try:
            content = json.loads(content_json)
            questions.append({
                "id": qid,
                "question_type": qtype,
                "difficulty": difficulty,
                "elo_difficulty": elo,
                **content,
            })
        except json.JSONDecodeError:
            print(f"  Warning: Failed to parse content for question {qid}, skipping")
    return questions


def main():
    parser = argparse.ArgumentParser(description="Index questions to Qdrant RAG")
    parser.add_argument("--force", action="store_true", help="Force re-index all questions")
    parser.add_argument("--db", default=os.path.join(PROJECT_ROOT, "logicmaster.db"),
                        help="Path to SQLite database")
    args = parser.parse_args()

    print("=" * 50)
    print("LogicMaster RAG Indexing")
    print("=" * 50)

    # 加载题目
    questions = load_questions_from_db(args.db)
    if not questions:
        print("No questions found in database. Run generate_pool.py first.")
        return

    print(f"Found {len(questions)} verified questions in database")

    # 获取 RAG 服务
    rag = get_rag_service()

    success_count = 0
    fail_count = 0
    failed_ids = []

    for i, q in enumerate(questions):
        qid = q["id"]
        stimulus = q.get("stimulus", "")
        question = q.get("question", "")
        explanation = q.get("explanation", "")
        detailed_explanation = q.get("detailed_explanation", "")
        qtype = q.get("question_type", "")
        skills = q.get("skills", [])
        difficulty = q.get("difficulty", "")

        # 构建题目文本
        question_text = f"{stimulus}\n{question}"
        # 优先使用详细解析
        expl = detailed_explanation if detailed_explanation else explanation

        print(f"  [{i+1}/{len(questions)}] Indexing {qid} ({qtype}, {difficulty})...", end=" ")

        ok = rag.index_question(
            question_id=qid,
            question_text=question_text,
            explanation=expl,
            question_type=qtype,
            skills=skills,
            difficulty=difficulty,
        )

        if ok:
            success_count += 1
            print("OK")
        else:
            fail_count += 1
            failed_ids.append(qid)
            print("FAILED")

    # 汇总
    print()
    print("=" * 50)
    print(f"Indexing complete: {success_count}/{len(questions)} succeeded")
    if fail_count > 0:
        print(f"Failed: {fail_count} questions")
        for fid in failed_ids:
            print(f"  - {fid}")

        # 写入日志文件
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "index_errors.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        print(f"Failed IDs saved to {log_path}")


if __name__ == "__main__":
    main()
