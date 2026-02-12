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
