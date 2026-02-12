"""
配置管理：从环境变量和 .env 文件加载配置
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DeepSeek LLM
    DEEPSEEK_API_KEY: str = ""

    # OpenAI (RAG embedding + 后续评估)
    OPENAI_API_KEY: str = ""

    # 数据库 (Week 1 先继续用 SQLite，后续迁移 PostgreSQL)
    DATABASE_URL: str = "sqlite:///logicmaster.db"

    # Qdrant 向量数据库
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "gmat_explanations"

    # OpenAI Embedding 配置
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMS: int = 1536

    # 应用配置
    APP_ENV: str = "development"
    DEBUG: bool = True

    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
