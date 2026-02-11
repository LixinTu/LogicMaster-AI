"""
配置管理：从环境变量和 .env 文件加载配置
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DeepSeek LLM
    DEEPSEEK_API_KEY: str = ""

    # OpenAI (后续 RAG 使用)
    OPENAI_API_KEY: str = ""

    # 数据库 (Week 1 先继续用 SQLite，后续迁移 PostgreSQL)
    DATABASE_URL: str = "sqlite:///logicmaster.db"

    # 应用配置
    APP_ENV: str = "development"
    DEBUG: bool = True

    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
