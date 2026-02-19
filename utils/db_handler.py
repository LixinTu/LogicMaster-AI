"""
数据库管理模块：SQLite 数据库操作封装
用于 LogicMaster 应用的题目存储和用户日志记录
"""

import sqlite3
import json
import os
import time
from typing import Dict, List, Optional, Any


class DatabaseManager:
    """
    数据库管理器类，封装所有数据库操作
    
    提供题目存储、查询、用户日志记录等功能
    """
    
    def __init__(self, db_path: str = "logicmaster.db", max_retry_attempts: int = 3, retry_delay: float = 0.5):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
            max_retry_attempts: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.db_path: str = db_path
        self.max_retry_attempts: int = max_retry_attempts
        self.retry_delay: float = retry_delay
    
    def init_db(self) -> bool:
        """
        初始化数据库，创建 questions 和 user_logs 两个表
        
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        for attempt in range(self.max_retry_attempts):
            try:
                # 检查数据库文件是否存在，如果不存在则创建
                db_exists = os.path.exists(self.db_path)
                
                # 设置超时，避免数据库锁定问题
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                cursor = conn.cursor()
                
                # 创建 questions 表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS questions (
                        id TEXT PRIMARY KEY,
                        question_type TEXT NOT NULL,
                        difficulty TEXT NOT NULL,
                        content TEXT NOT NULL,
                        elo_difficulty REAL DEFAULT 1500.0,
                        is_verified INTEGER DEFAULT 0
                    )
                """)
                
                # 创建 user_logs 表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        action_type TEXT NOT NULL,
                        question_id TEXT,
                        details TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 创建 experiment_logs 表（Week 4: A/B Testing）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiment_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        experiment_name TEXT NOT NULL,
                        variant TEXT NOT NULL,
                        event_type TEXT NOT NULL DEFAULT 'exposure',
                        outcome_metric TEXT,
                        outcome_value REAL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # 索引：按实验+变体查询、按用户+实验查询
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_exp_variant
                    ON experiment_logs (experiment_name, variant)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_exp
                    ON experiment_logs (user_id, experiment_name)
                """)

                # 间隔重复统计表（Half-Life Regression）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS spaced_repetition_stats (
                        user_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        half_life REAL NOT NULL DEFAULT 1.0,
                        last_practiced TIMESTAMP NOT NULL,
                        n_correct INTEGER NOT NULL DEFAULT 0,
                        n_attempts INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (user_id, question_id)
                    )
                """)

                # 用户账户表（Auth）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        display_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_email
                    ON users (email)
                """)

                # DKT 答题历史表（Deep Knowledge Tracing）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS answer_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL DEFAULT 'default',
                        question_id TEXT NOT NULL,
                        skill_ids TEXT NOT NULL,
                        is_correct INTEGER NOT NULL,
                        theta_at_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_answer_history_user_time
                    ON answer_history (user_id, created_at)
                """)

                # 收藏/错题本表（Bookmarks）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS bookmarks (
                        user_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        bookmark_type TEXT NOT NULL CHECK(bookmark_type IN ('favorite', 'wrong')),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, question_id, bookmark_type)
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bookmarks_user_type
                    ON bookmarks (user_id, bookmark_type)
                """)

                # 学习目标表（Learning Goals）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_goals (
                        user_id TEXT PRIMARY KEY,
                        target_gmat_score INTEGER DEFAULT 40,
                        daily_question_goal INTEGER DEFAULT 5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 邮件发送日志表（Email Logs）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS email_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        email_type TEXT NOT NULL,
                        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_email_logs_user
                    ON email_logs (user_id, sent_at)
                """)

                # 3PL IRT 参数列（向后兼容：仅在列不存在时添加）
                existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(questions)").fetchall()}
                if "discrimination" not in existing_cols:
                    cursor.execute("ALTER TABLE questions ADD COLUMN discrimination REAL DEFAULT 1.0")
                if "guessing" not in existing_cols:
                    cursor.execute("ALTER TABLE questions ADD COLUMN guessing REAL DEFAULT 0.2")

                conn.commit()
                conn.close()
                
                if not db_exists:
                    print(f"数据库文件已创建：{self.db_path}")
                print(f"数据库初始化成功：{self.db_path}")
                return True
                
            except sqlite3.OperationalError as e:
                error_msg = str(e)
                if "database is locked" in error_msg.lower():
                    if attempt < self.max_retry_attempts - 1:
                        print(f"数据库被锁定，等待 {self.retry_delay} 秒后重试... (尝试 {attempt + 1}/{self.max_retry_attempts})")
                        time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                        continue
                    else:
                        print(f"数据库初始化失败：数据库被锁定，已达到最大重试次数")
                        return False
                elif "database disk image is malformed" in error_msg.lower():
                    print(f"数据库初始化失败：数据库文件损坏，请检查 {self.db_path}")
                    return False
                else:
                    print(f"数据库初始化失败（操作错误）：{e}")
                    return False
            except sqlite3.DatabaseError as e:
                print(f"数据库初始化失败（数据库错误）：{e}")
                return False
            except PermissionError as e:
                print(f"数据库初始化失败（权限错误）：无法访问 {self.db_path}，请检查文件权限")
                return False
            except Exception as e:
                print(f"数据库初始化失败（未知错误）：{e}")
                return False
        
        return False
    
    def add_question(self, q_data: Dict[str, Any]) -> bool:
        """
        将题目数据添加到数据库
        
        Args:
            q_data: 题目字典，必须包含：
                - id: 题目ID（主键）
                - question_type: 题型
                - difficulty: 难度
                - content: 题目内容（字典，会被转为JSON）
                其他字段可选（elo_difficulty, is_verified）
        
        Returns:
            bool: 成功返回 True，失败或ID重复返回 False
        """
        # 检查必需字段
        if not isinstance(q_data, dict):
            print("错误：题目数据必须是字典类型")
            return False
        
        if "id" not in q_data:
            print("错误：题目数据缺少 'id' 字段")
            return False
        
        for attempt in range(self.max_retry_attempts):
            conn = None
            try:
                # 检查数据库文件是否存在
                if not os.path.exists(self.db_path):
                    print(f"警告：数据库文件 {self.db_path} 不存在，尝试初始化...")
                    if not self.init_db():
                        print(f"错误：无法初始化数据库 {self.db_path}")
                        return False
                
                question_id: str = q_data["id"]
                question_type: str = q_data.get("question_type", "Weaken")
                difficulty: str = q_data.get("difficulty", "medium")
                
                # 将 content 转为 JSON 字符串
                # 如果 q_data 中已经有 content 字段，直接使用；否则将整个 q_data 作为 content
                if "content" in q_data:
                    content_json: str = json.dumps(q_data["content"], ensure_ascii=False)
                else:
                    # 如果没有 content 字段，将除了 id, question_type, difficulty, elo_difficulty, is_verified 之外的字段作为 content
                    content_dict: Dict[str, Any] = {k: v for k, v in q_data.items() 
                                  if k not in ["id", "question_type", "difficulty", "elo_difficulty", "is_verified"]}
                    content_json = json.dumps(content_dict, ensure_ascii=False)
                
                elo_difficulty: float = q_data.get("elo_difficulty", 1500.0)
                is_verified: bool = q_data.get("is_verified", False)
                
                # 设置超时，避免数据库锁定问题
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                cursor = conn.cursor()
                
                # 检查 ID 是否已存在
                cursor.execute("SELECT id FROM questions WHERE id = ?", (question_id,))
                if cursor.fetchone():
                    print(f"题目 ID {question_id} 已存在，跳过插入")
                    conn.close()
                    return False
                
                # 插入新题目
                cursor.execute("""
                    INSERT INTO questions (id, question_type, difficulty, content, elo_difficulty, is_verified)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (question_id, question_type, difficulty, content_json, elo_difficulty, 1 if is_verified else 0))
                
                conn.commit()
                conn.close()
                print(f"题目 {question_id} 已添加到数据库")
                return True
                
            except sqlite3.IntegrityError:
                print(f"题目 ID {q_data.get('id', 'unknown')} 已存在，跳过插入")
                if conn:
                    conn.close()
                return False
            except sqlite3.OperationalError as e:
                error_msg = str(e)
                if conn:
                    conn.close()
                
                if "database is locked" in error_msg.lower():
                    if attempt < self.max_retry_attempts - 1:
                        print(f"数据库被锁定，等待 {self.retry_delay * (attempt + 1)} 秒后重试... (尝试 {attempt + 1}/{self.max_retry_attempts})")
                        time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                        continue
                    else:
                        print(f"添加题目失败：数据库被锁定，已达到最大重试次数")
                        return False
                elif "database disk image is malformed" in error_msg.lower():
                    print(f"添加题目失败：数据库文件损坏")
                    return False
                else:
                    print(f"添加题目失败（操作错误）：{e}")
                    return False
            except sqlite3.DatabaseError as e:
                if conn:
                    conn.close()
                print(f"添加题目失败（数据库错误）：{e}")
                return False
            except PermissionError as e:
                if conn:
                    conn.close()
                print(f"添加题目失败（权限错误）：无法访问 {self.db_path}")
                return False
            except Exception as e:
                if conn:
                    conn.close()
                print(f"添加题目到数据库失败（未知错误）：{e}")
                return False
        
        return False
    
    def get_adaptive_candidates(
        self, 
        target_difficulty: float, 
        exclude_id: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        根据目标难度（theta）获取自适应候选题目列表（IRT + BKT 驱动）
        
        Args:
            target_difficulty: 目标能力值（theta，通常在 -3.0 到 3.0 之间）
            exclude_id: 要排除的题目 ID（避免重复推荐，默认 None）
            limit: 返回的最大候选题目数量（默认 10）
        
        Returns:
            题目字典列表，每个字典包含所有字段，content 字段已从 JSON 解析回字典
            特别注意：包含 skills 字段（BKT 需要）
            如果未找到题目，返回空列表 []
        """
        # 参数验证
        if not isinstance(target_difficulty, (int, float)):
            print(f"错误：target_difficulty 必须是数字，收到：{type(target_difficulty)}")
            return []
        
        if limit <= 0:
            print(f"错误：limit 必须大于 0，收到：{limit}")
            return []
        
        # 检查数据库文件是否存在
        if not os.path.exists(self.db_path):
            print(f"警告：数据库文件 {self.db_path} 不存在")
            return []
        
        for attempt in range(self.max_retry_attempts):
            conn = None
            try:
                # 映射 Elo：target_elo = 1500 + target_difficulty * 100
                target_elo: float = 1500.0 + target_difficulty * 100.0
                
                # SQL 筛选：elo_difficulty 在 target_elo ± 200 之间
                elo_min: float = target_elo - 200.0
                elo_max: float = target_elo + 200.0
                
                # 设置超时，避免数据库锁定问题
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                cursor = conn.cursor()
                
                # 构建查询条件
                query_params: List[Any] = [elo_min, elo_max]
                exclude_condition: str = ""
                if exclude_id:
                    exclude_condition = "AND id != ?"
                    query_params.append(exclude_id)
                
                # 查询 ELO 范围内的题目，排除 is_verified=0 的题目和 exclude_id
                query: str = f"""
                    SELECT id, question_type, difficulty, content, elo_difficulty, is_verified
                    FROM questions
                    WHERE elo_difficulty >= ? 
                      AND elo_difficulty <= ?
                      AND is_verified != 0
                      {exclude_condition}
                    ORDER BY RANDOM()
                    LIMIT ?
                """
                query_params.append(limit)
                
                cursor.execute(query, tuple(query_params))
                rows: List[tuple] = cursor.fetchall()
                conn.close()
                conn = None  # 标记已关闭
                
                # 兜底：若题目不足，放宽范围重试
                if len(rows) < limit:
                    # 放宽范围到 ±400
                    elo_min_fallback: float = target_elo - 400.0
                    elo_max_fallback: float = target_elo + 400.0
                    
                    conn = sqlite3.connect(self.db_path, timeout=10.0)
                    cursor = conn.cursor()
                    
                    query_params_fallback: List[Any] = [elo_min_fallback, elo_max_fallback]
                    if exclude_id:
                        exclude_condition = "AND id != ?"
                        query_params_fallback.append(exclude_id)
                    else:
                        exclude_condition = ""
                    
                    query_fallback: str = f"""
                        SELECT id, question_type, difficulty, content, elo_difficulty, is_verified
                        FROM questions
                        WHERE elo_difficulty >= ? 
                          AND elo_difficulty <= ?
                          AND is_verified != 0
                          {exclude_condition}
                        ORDER BY RANDOM()
                        LIMIT ?
                    """
                    query_params_fallback.append(limit)
                    
                    cursor.execute(query_fallback, tuple(query_params_fallback))
                    rows = cursor.fetchall()
                    conn.close()
                    conn = None  # 标记已关闭
                
                if not rows:
                    # 第一次查询没有结果时，直接返回空列表，不再重试
                    return []
                
                # 解析所有候选题目
                candidates: List[Dict[str, Any]] = []
                for row in rows:
                    question_id, question_type, difficulty_val, content_json, elo_difficulty, is_verified = row
                    
                    try:
                        # 将 content JSON 字符串解析回字典
                        content_dict: Dict[str, Any] = json.loads(content_json)
                        
                        # 构建题目字典（确保包含所有字段，特别是 skills）
                        question_dict: Dict[str, Any] = {
                            "id": question_id,
                            "question_type": question_type,
                            "difficulty": difficulty_val,
                            "elo_difficulty": elo_difficulty,
                            "is_verified": bool(is_verified),
                            **content_dict  # 将 content 中的字段展开到顶层（包含 skills、diagnoses 等）
                        }
                        
                        candidates.append(question_dict)
                        
                    except json.JSONDecodeError as e:
                        print(f"解析题目 {question_id} 的 content JSON 失败：{e}")
                        continue
                    except Exception as e:
                        print(f"处理题目 {question_id} 时出错：{e}")
                        continue
                
                if candidates:
                    print(f"找到 {len(candidates)} 道候选题目（ELO: {elo_min:.1f}-{elo_max:.1f}，排除ID: {exclude_id if exclude_id else '无'}）")
                return candidates
                
            except sqlite3.OperationalError as e:
                error_msg = str(e)
                if conn:
                    conn.close()
                    conn = None
                
                if "database is locked" in error_msg.lower():
                    if attempt < self.max_retry_attempts - 1:
                        print(f"数据库被锁定，等待 {self.retry_delay * (attempt + 1)} 秒后重试... (尝试 {attempt + 1}/{self.max_retry_attempts})")
                        time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                        continue
                    else:
                        print(f"从数据库获取候选题目失败：数据库被锁定，已达到最大重试次数")
                        return []
                elif "no such table" in error_msg.lower():
                    print(f"从数据库获取候选题目失败：表不存在，请先运行 `python -m utils.db_handler` 初始化数据库")
                    return []
                elif "database disk image is malformed" in error_msg.lower():
                    print(f"从数据库获取候选题目失败：数据库文件损坏")
                    return []
                else:
                    print(f"从数据库获取候选题目失败（操作错误）：{e}")
                    return []
            except sqlite3.DatabaseError as e:
                if conn:
                    conn.close()
                print(f"从数据库获取候选题目失败（数据库错误）：{e}")
                return []
            except PermissionError as e:
                if conn:
                    conn.close()
                print(f"从数据库获取候选题目失败（权限错误）：无法访问 {self.db_path}")
                return []
            except Exception as e:
                if conn:
                    conn.close()
                print(f"从数据库获取自适应候选题目失败（未知错误）：{e}")
                return []
        
        return []


    # ========== DKT: answer_history ==========

    def insert_answer_history(
        self,
        question_id: str,
        skill_ids: List[str],
        is_correct: bool,
        theta_at_time: Optional[float] = None,
        user_id: str = "default",
    ) -> bool:
        """
        插入一条答题历史记录（供 DKT 训练使用）

        Args:
            question_id: 题目 ID
            skill_ids: 涉及的技能列表
            is_correct: 是否答对
            theta_at_time: 答题时的能力值
            user_id: 用户标识

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            skill_json = json.dumps(skill_ids, ensure_ascii=False)
            cursor.execute("""
                INSERT INTO answer_history
                    (user_id, question_id, skill_ids, is_correct, theta_at_time)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, question_id, skill_json, 1 if is_correct else 0, theta_at_time))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"insert_answer_history failed: {e}")
            return False

    def query_answer_history(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询答题历史（按 created_at 升序）

        Args:
            user_id: 指定用户，None 返回所有
            limit: 最大返回数量

        Returns:
            记录字典列表，skill_ids 已解析为 list
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if user_id:
                query = "SELECT * FROM answer_history WHERE user_id = ? ORDER BY created_at ASC"
                params: tuple = (user_id,)
            else:
                query = "SELECT * FROM answer_history ORDER BY created_at ASC"
                params = ()

            if limit:
                query += f" LIMIT {int(limit)}"

            cursor.execute(query, params)
            rows = []
            for row in cursor.fetchall():
                d = dict(row)
                # 解析 skill_ids JSON
                try:
                    d["skill_ids"] = json.loads(d["skill_ids"])
                except (json.JSONDecodeError, TypeError):
                    d["skill_ids"] = []
                rows.append(d)
            conn.close()
            return rows
        except Exception as e:
            if conn:
                conn.close()
            print(f"query_answer_history failed: {e}")
            return []

    def count_answer_history(self, user_id: Optional[str] = None) -> int:
        """
        统计答题历史总数

        Args:
            user_id: 指定用户，None 返回全部

        Returns:
            记录数
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM answer_history WHERE user_id = ?",
                    (user_id,),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM answer_history")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            if conn:
                conn.close()
            print(f"count_answer_history failed: {e}")
            return 0

    # ========== A/B Testing: experiment_logs ==========

    def insert_experiment_log(
        self,
        user_id: str,
        experiment_name: str,
        variant: str,
        event_type: str = "exposure",
        outcome_metric: Optional[str] = None,
        outcome_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        插入一条实验日志（曝光或结果）

        Args:
            user_id: 用户标识
            experiment_name: 实验名称
            variant: 分配的变体
            event_type: "exposure" 或 "outcome"
            outcome_metric: 指标名称（如 "is_correct", "theta_gain"）
            outcome_value: 指标数值
            metadata: 额外元数据（JSON 序列化存储）

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
            cursor.execute("""
                INSERT INTO experiment_logs
                    (user_id, experiment_name, variant, event_type,
                     outcome_metric, outcome_value, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, experiment_name, variant, event_type,
                  outcome_metric, outcome_value, metadata_json))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"insert_experiment_log failed: {e}")
            return False

    def query_logs_by_experiment(
        self,
        experiment_name: str,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询某个实验的所有日志

        Args:
            experiment_name: 实验名称
            event_type: 可选过滤（"exposure" 或 "outcome"）

        Returns:
            日志字典列表
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if event_type:
                cursor.execute(
                    "SELECT * FROM experiment_logs WHERE experiment_name = ? AND event_type = ? ORDER BY created_at",
                    (experiment_name, event_type),
                )
            else:
                cursor.execute(
                    "SELECT * FROM experiment_logs WHERE experiment_name = ? ORDER BY created_at",
                    (experiment_name,),
                )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            if conn:
                conn.close()
            print(f"query_logs_by_experiment failed: {e}")
            return []

    def query_logs_by_user(
        self,
        user_id: str,
        experiment_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询某个用户的实验日志

        Args:
            user_id: 用户标识
            experiment_name: 可选：仅查某个实验

        Returns:
            日志字典列表
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if experiment_name:
                cursor.execute(
                    "SELECT * FROM experiment_logs WHERE user_id = ? AND experiment_name = ? ORDER BY created_at",
                    (user_id, experiment_name),
                )
            else:
                cursor.execute(
                    "SELECT * FROM experiment_logs WHERE user_id = ? ORDER BY created_at",
                    (user_id,),
                )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            if conn:
                conn.close()
            print(f"query_logs_by_user failed: {e}")
            return []


    # ========== Dashboard: answer_history 统计 ==========

    def count_today_answers(self, user_id: str = "default") -> int:
        """
        统计用户今天的答题数量（按 UTC 日期）

        Returns:
            今日答题总数
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM answer_history WHERE user_id = ? AND DATE(created_at) = DATE('now')",
                (user_id,),
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            if conn:
                conn.close()
            print(f"count_today_answers failed: {e}")
            return 0

    def calculate_streak(self, user_id: str = "default") -> int:
        """
        计算用户连续答题天数（从今天往前，每天至少1题）

        Returns:
            连续天数。若今天无记录，从昨天开始往前数（允许当天未答题）
        """
        from datetime import datetime, timedelta, timezone

        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT DISTINCT DATE(created_at) as practice_date
                   FROM answer_history WHERE user_id = ?
                   ORDER BY practice_date DESC""",
                (user_id,),
            )
            dates = {row[0] for row in cursor.fetchall()}
            conn.close()

            if not dates:
                return 0

            # Use UTC date to match DATE(created_at) which extracts from stored +00:00 timestamps
            today = datetime.now(timezone.utc).date()
            streak = 0
            # Start from today; if today has no record, check if yesterday starts a streak
            check_day = today
            if today.isoformat() not in dates:
                check_day = today - timedelta(days=1)

            while check_day.isoformat() in dates:
                streak += 1
                check_day -= timedelta(days=1)
            return streak
        except Exception as e:
            if conn:
                conn.close()
            print(f"calculate_streak failed: {e}")
            return 0

    def get_skill_error_rates(
        self, user_id: str = "default", limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        按技能计算错误率，返回最弱的技能列表（用于 Dashboard weak_skills）。
        使用 SQLite json_each 展开 skill_ids JSON 数组。

        Returns:
            [{"skill_name": str, "error_rate": float, "mastery": float}]
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT je.value AS skill_name,
                          COUNT(*) AS total,
                          SUM(CASE WHEN ah.is_correct = 0 THEN 1 ELSE 0 END) AS wrong_count
                   FROM answer_history ah, json_each(ah.skill_ids) je
                   WHERE ah.user_id = ?
                   GROUP BY je.value
                   HAVING total > 0
                   ORDER BY (wrong_count * 1.0 / total) DESC
                   LIMIT ?""",
                (user_id, limit),
            )
            rows = cursor.fetchall()
            conn.close()
            result = []
            for skill_name, total, wrong_count in rows:
                error_rate = wrong_count / total if total > 0 else 0.0
                result.append({
                    "skill_name": skill_name,
                    "error_rate": round(error_rate, 3),
                    "mastery": round(1.0 - error_rate, 3),
                })
            return result
        except Exception as e:
            if conn:
                conn.close()
            print(f"get_skill_error_rates failed: {e}")
            return []

    def get_latest_theta(self, user_id: str = "default") -> Optional[float]:
        """
        从 answer_history 获取用户最新的 theta 值

        Returns:
            最新 theta，若无记录返回 None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT theta_at_time FROM answer_history
                   WHERE user_id = ? AND theta_at_time IS NOT NULL
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id,),
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            if conn:
                conn.close()
            return None

    def get_last_practiced_time(self, user_id: str = "default") -> Optional[str]:
        """
        获取用户最后一次答题的时间戳

        Returns:
            ISO 格式时间字符串，或 None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(created_at) FROM answer_history WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception as e:
            if conn:
                conn.close()
            return None

    # ========== Bookmarks: 收藏/错题本 ==========

    def insert_bookmark(
        self, user_id: str, question_id: str, bookmark_type: str
    ) -> bool:
        """
        添加书签（收藏或错题）。重复插入静默忽略（INSERT OR IGNORE）。

        Args:
            user_id: 用户标识
            question_id: 题目 ID
            bookmark_type: "favorite" 或 "wrong"

        Returns:
            成功（含已存在）返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                """INSERT OR IGNORE INTO bookmarks (user_id, question_id, bookmark_type)
                   VALUES (?, ?, ?)""",
                (user_id, question_id, bookmark_type),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"insert_bookmark failed: {e}")
            return False

    def remove_bookmark(
        self, user_id: str, question_id: str, bookmark_type: str
    ) -> bool:
        """
        删除书签

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                """DELETE FROM bookmarks WHERE user_id = ? AND question_id = ? AND bookmark_type = ?""",
                (user_id, question_id, bookmark_type),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"remove_bookmark failed: {e}")
            return False

    def query_bookmarks(
        self,
        user_id: str,
        bookmark_type: Optional[str] = None,
        skill_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询用户书签，JOIN questions 获取题目内容

        Args:
            user_id: 用户标识
            bookmark_type: 可选过滤 "favorite" 或 "wrong"
            skill_filter: 可选技能名过滤（如 "Causal Reasoning"）

        Returns:
            书签列表，含 stimulus_preview 和 skills
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT b.question_id, b.bookmark_type, b.created_at,
                       q.question_type, q.difficulty, q.content
                FROM bookmarks b
                LEFT JOIN questions q ON b.question_id = q.id
                WHERE b.user_id = ?
            """
            params: List[Any] = [user_id]
            if bookmark_type:
                query += " AND b.bookmark_type = ?"
                params.append(bookmark_type)
            query += " ORDER BY b.created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            result = []
            for row in rows:
                content: Dict[str, Any] = {}
                try:
                    if row["content"]:
                        content = json.loads(row["content"])
                except (json.JSONDecodeError, TypeError):
                    pass

                skills: List[str] = content.get("skills", [])
                if skill_filter and skill_filter not in skills:
                    continue

                stimulus = content.get("stimulus", "")
                preview = stimulus[:150] + "..." if len(stimulus) > 150 else stimulus

                result.append({
                    "question_id": row["question_id"],
                    "question_type": row["question_type"] or "",
                    "difficulty": row["difficulty"] or "",
                    "stimulus_preview": preview,
                    "skills": skills,
                    "bookmark_type": row["bookmark_type"],
                    "created_at": row["created_at"],
                })
            return result
        except Exception as e:
            if conn:
                conn.close()
            print(f"query_bookmarks failed: {e}")
            return []

    def get_wrong_stats(self, user_id: str) -> Dict[str, Any]:
        """
        统计错题本分布：按技能和题型

        Returns:
            {"total_wrong": int, "by_skill": [...], "by_type": [...]}
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()

            # 总错题数
            cursor.execute(
                "SELECT COUNT(*) FROM bookmarks WHERE user_id = ? AND bookmark_type = 'wrong'",
                (user_id,),
            )
            total_wrong: int = cursor.fetchone()[0]

            # 按题型统计（JOIN questions）
            cursor.execute(
                """SELECT q.question_type, COUNT(*) as cnt
                   FROM bookmarks b
                   LEFT JOIN questions q ON b.question_id = q.id
                   WHERE b.user_id = ? AND b.bookmark_type = 'wrong'
                   GROUP BY q.question_type
                   ORDER BY cnt DESC""",
                (user_id,),
            )
            by_type = [
                {"question_type": t or "Unknown", "count": c}
                for t, c in cursor.fetchall()
            ]

            # 按技能统计（从 questions.content JSON 解析）
            cursor.execute(
                """SELECT q.content FROM bookmarks b
                   LEFT JOIN questions q ON b.question_id = q.id
                   WHERE b.user_id = ? AND b.bookmark_type = 'wrong'""",
                (user_id,),
            )
            skill_counts: Dict[str, int] = {}
            for (content_json,) in cursor.fetchall():
                try:
                    if content_json:
                        skills = json.loads(content_json).get("skills", [])
                        for skill in skills:
                            skill_counts[skill] = skill_counts.get(skill, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

            conn.close()
            by_skill = sorted(
                [{"skill_name": k, "count": v} for k, v in skill_counts.items()],
                key=lambda x: -x["count"],
            )
            return {"total_wrong": total_wrong, "by_skill": by_skill, "by_type": by_type}
        except Exception as e:
            if conn:
                conn.close()
            print(f"get_wrong_stats failed: {e}")
            return {"total_wrong": 0, "by_skill": [], "by_type": []}

    # ========== Learning Goals: 学习目标 ==========

    def upsert_learning_goal(
        self,
        user_id: str,
        target_gmat_score: int,
        daily_question_goal: int,
    ) -> bool:
        """
        插入或更新用户学习目标（UPSERT）

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                """INSERT INTO learning_goals (user_id, target_gmat_score, daily_question_goal, created_at, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                   ON CONFLICT(user_id) DO UPDATE SET
                       target_gmat_score = excluded.target_gmat_score,
                       daily_question_goal = excluded.daily_question_goal,
                       updated_at = CURRENT_TIMESTAMP""",
                (user_id, target_gmat_score, daily_question_goal),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"upsert_learning_goal failed: {e}")
            return False

    def get_learning_goal(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户学习目标。若无记录，返回默认值。

        Returns:
            {"target_gmat_score": int, "daily_question_goal": int, "updated_at": str}
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT target_gmat_score, daily_question_goal, updated_at FROM learning_goals WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
            # 默认值
            return {"target_gmat_score": 40, "daily_question_goal": 5, "updated_at": None}
        except Exception as e:
            if conn:
                conn.close()
            print(f"get_learning_goal failed: {e}")
            return {"target_gmat_score": 40, "daily_question_goal": 5, "updated_at": None}

    # ========== Email Logs: 邮件发送日志 ==========

    def insert_email_log(self, user_id: str, email_type: str) -> bool:
        """
        记录一次邮件发送

        Args:
            user_id: 用户标识
            email_type: 邮件类型（如 "review_reminder"）

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                "INSERT INTO email_logs (user_id, email_type) VALUES (?, ?)",
                (user_id, email_type),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"insert_email_log failed: {e}")
            return False

    def get_last_reminder_time(self, user_id: str) -> Optional[str]:
        """
        获取用户最近一次 review_reminder 的发送时间

        Returns:
            ISO 格式时间字符串，或 None（从未发送过）
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT MAX(sent_at) FROM email_logs
                   WHERE user_id = ? AND email_type = 'review_reminder'""",
                (user_id,),
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception as e:
            if conn:
                conn.close()
            print(f"get_last_reminder_time failed: {e}")
            return None

    # ========== Auth: users ==========

    def insert_user(
        self,
        user_id: str,
        email: str,
        password_hash: str,
        display_name: Optional[str] = None,
    ) -> bool:
        """
        插入新用户记录

        Returns:
            成功返回 True；email 已存在返回 False
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (?, ?, ?, ?)
            """, (user_id, email.lower().strip(), password_hash, display_name))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            if conn:
                conn.close()
            return False
        except Exception as e:
            if conn:
                conn.close()
            print(f"insert_user failed: {e}")
            return False

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        按 email 查询用户

        Returns:
            用户字典，或 None（不存在时）
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, email, password_hash, display_name, created_at FROM users WHERE email = ?",
                (email.lower().strip(),),
            )
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            if conn:
                conn.close()
            print(f"get_user_by_email failed: {e}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        按 ID 查询用户

        Returns:
            用户字典（不含 password_hash），或 None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, email, display_name, created_at FROM users WHERE id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            if conn:
                conn.close()
            print(f"get_user_by_id failed: {e}")
            return None

    def update_user_display_name(self, user_id: str, display_name: str) -> bool:
        """
        更新用户显示名称

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                "UPDATE users SET display_name = ? WHERE id = ?",
                (display_name, user_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"update_user_display_name failed: {e}")
            return False

    def update_user_password(self, user_id: str, new_password_hash: str) -> bool:
        """
        更新用户密码哈希

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_password_hash, user_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"update_user_password failed: {e}")
            return False

    def delete_user_and_data(self, user_id: str) -> bool:
        """
        删除用户账户及其所有关联数据（级联删除）

        删除顺序：answer_history, spaced_repetition_stats, bookmarks,
        learning_goals, email_logs, experiment_logs, users

        注意：bandit_stats 表无 user_id 列，跳过。

        Returns:
            成功返回 True
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            related_tables = [
                ("answer_history", "user_id"),
                ("spaced_repetition_stats", "user_id"),
                ("bookmarks", "user_id"),
                ("learning_goals", "user_id"),
                ("email_logs", "user_id"),
                ("experiment_logs", "user_id"),
            ]
            for table, col in related_tables:
                conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if conn:
                conn.close()
            print(f"delete_user_and_data failed: {e}")
            return False

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户学习统计数据

        Returns:
            {total_questions, total_correct, accuracy_pct, best_streak,
             member_since, current_theta, favorite_question_type}
        """
        _default: Dict[str, Any] = {
            "total_questions": 0,
            "total_correct": 0,
            "accuracy_pct": 0.0,
            "best_streak": 0,
            "member_since": None,
            "current_theta": None,
            "favorite_question_type": None,
        }
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()

            # 总答题数 & 正确数
            cursor.execute(
                "SELECT COUNT(*), SUM(is_correct) FROM answer_history WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            total_questions: int = row[0] or 0
            total_correct: int = int(row[1] or 0)
            accuracy_pct: float = (
                round(total_correct / total_questions * 100, 1) if total_questions > 0 else 0.0
            )

            # 最新 theta（按 created_at 降序）
            cursor.execute(
                """SELECT theta_at_time FROM answer_history
                   WHERE user_id = ? AND theta_at_time IS NOT NULL
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id,),
            )
            theta_row = cursor.fetchone()
            current_theta = theta_row[0] if theta_row else None

            # 最常见题型
            cursor.execute(
                """SELECT q.question_type, COUNT(*) as cnt
                   FROM answer_history ah
                   LEFT JOIN questions q ON ah.question_id = q.id
                   WHERE ah.user_id = ?
                   GROUP BY q.question_type
                   ORDER BY cnt DESC LIMIT 1""",
                (user_id,),
            )
            fav_row = cursor.fetchone()
            favorite_question_type = fav_row[0] if fav_row and fav_row[0] else None

            # 所有练习日期（用于计算最长连续天数）
            cursor.execute(
                """SELECT DISTINCT DATE(created_at) FROM answer_history
                   WHERE user_id = ? ORDER BY 1 ASC""",
                (user_id,),
            )
            date_strs = [r[0] for r in cursor.fetchall()]

            # 注册时间
            cursor.execute("SELECT created_at FROM users WHERE id = ?", (user_id,))
            user_row = cursor.fetchone()
            member_since = str(user_row[0]) if user_row and user_row[0] else None

            conn.close()

            # 计算历史最长连续天数
            best_streak: int = 0
            if date_strs:
                from datetime import date as _date, timedelta
                dates = [_date.fromisoformat(d) for d in date_strs]
                current_run = 1
                best_streak = 1
                for i in range(1, len(dates)):
                    if dates[i] == dates[i - 1] + timedelta(days=1):
                        current_run += 1
                        if current_run > best_streak:
                            best_streak = current_run
                    else:
                        current_run = 1

            return {
                "total_questions": total_questions,
                "total_correct": total_correct,
                "accuracy_pct": accuracy_pct,
                "best_streak": best_streak,
                "member_since": member_since,
                "current_theta": current_theta,
                "favorite_question_type": favorite_question_type,
            }
        except Exception as e:
            if conn:
                conn.close()
            print(f"get_user_stats failed: {e}")
            return _default


# 为了向后兼容，创建全局实例和函数包装器
_default_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取默认数据库管理器实例（单例模式）"""
    global _default_db_manager
    if _default_db_manager is None:
        _default_db_manager = DatabaseManager()
    return _default_db_manager


# 向后兼容的函数接口
def init_db() -> bool:
    """初始化数据库（向后兼容函数）"""
    return get_db_manager().init_db()


def add_question_to_db(q_data: Dict[str, Any]) -> bool:
    """添加题目到数据库（向后兼容函数）"""
    return get_db_manager().add_question(q_data)


def get_adaptive_candidates(
    target_difficulty: float, 
    exclude_id: Optional[str] = None, 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """获取自适应候选题目（向后兼容函数）"""
    return get_db_manager().get_adaptive_candidates(target_difficulty, exclude_id, limit)


if __name__ == "__main__":
    # 运行数据库初始化
    init_db()
