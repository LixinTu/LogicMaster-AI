"""
FastAPI 应用入口
"""

import sys
import os

# Windows 环境下确保 stdout/stderr 使用 UTF-8，避免 engine/ 中的 emoji print 导致 GBK 编码错误
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # 如果 reconfigure 失败（如 stdout 被 uvicorn 重定向），忽略

# 将项目根目录加入 Python 路径，以便导入 engine/ 和 llm_service
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers.theta import router as theta_router
from backend.routers.questions import router as questions_router
from backend.routers.tutor import router as tutor_router
from backend.routers.explanations import router as explanations_router
from backend.routers.analytics import router as analytics_router

app = FastAPI(
    title="LogicMaster AI API",
    description="GMAT Critical Reasoning 自适应学习平台后端",
    version="0.1.0",
)

# CORS: 允许 Streamlit (localhost:8501) 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 注册路由 ----------

app.include_router(theta_router)
app.include_router(questions_router)
app.include_router(tutor_router)
app.include_router(explanations_router)
app.include_router(analytics_router)


# ---------- Health Check ----------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "env": settings.APP_ENV,
    }
