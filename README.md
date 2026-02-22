# LogicMaster AI

> AI-Native GMAT Critical Reasoning Tutor — A GlitchMind Product

**Live:** [gmat.glitchmind.io](https://gmat.glitchmind.io) &nbsp;|&nbsp; **API:** [api.glitchmind.io](https://api.glitchmind.io) &nbsp;|&nbsp; **API Docs:** [api.glitchmind.io/docs](https://api.glitchmind.io/docs)

---

## 1. 项目概述

LogicMaster AI 是一个面向 GMAT Critical Reasoning 备考的**全栈自适应学习平台**。

**解决的核心问题：**
- 传统刷题平台题目固定、难度不自适应，效率低
- 答错后缺乏个性化引导，只给答案不解释逻辑
- 无法追踪学生真实的认知薄弱点和遗忘规律

**解决方案：**
- 基于 IRT 3PL 模型实时估算学生能力值（θ），动态调整题目难度
- 答错时触发 Socratic AI 导师，通过苏格拉底式对话引导学生发现错误逻辑，永不直接给答案
- Deep Knowledge Tracing（DKT LSTM）追踪每个技能维度的掌握程度
- Half-Life Regression 遗忘曲线计算最佳复习时机

---

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | React 18, TypeScript, Vite, Tailwind CSS, Zustand, Framer Motion, Recharts, Lucide |
| **后端** | FastAPI, Python 3.12, Pydantic v2, LangChain, Uvicorn |
| **AI / LLM** | DeepSeek API（LLM 推理）, LangChain-OpenAI 封装 |
| **向量检索** | OpenAI `text-embedding-3-small` (1536 dims), Qdrant 向量数据库 |
| **ML 算法** | IRT 3PL, Thompson Sampling, DKT LSTM (PyTorch), Half-Life Regression, BKT, Bloom's Taxonomy |
| **数据库** | SQLite（业务数据）, Qdrant（向量索引） |
| **认证** | JWT（PyJWT 7天有效期）, bcrypt 密码哈希 |
| **测试** | pytest — 297 个测试用例，100% 通过 |
| **部署** | AWS EC2（后端）, AWS S3 + CloudFront（前端静态）, Nginx 反向代理, systemd |
| **容器** | Docker Compose（本地开发：Qdrant + Backend + Frontend） |

---

## 3. 项目目录结构

```
mathquest_logicmaster/
│
├── frontend/                        # React + TypeScript 前端
│   ├── src/
│   │   ├── pages/                   # 页面组件
│   │   │   ├── Login.tsx            # 登录/注册页
│   │   │   ├── Dashboard.tsx        # 学习仪表盘
│   │   │   ├── Practice.tsx         # 刷题页（核心交互）
│   │   │   ├── WrongBook.tsx        # 错题本
│   │   │   ├── Review.tsx           # 间隔重复复习页
│   │   │   ├── Analytics.tsx        # 学习分析图表
│   │   │   ├── Profile.tsx          # 用户资料
│   │   │   └── Settings.tsx         # 设置页
│   │   ├── components/              # 可复用 UI 组件
│   │   ├── lib/
│   │   │   ├── api.ts               # API 客户端（封装所有 HTTP 请求）
│   │   │   └── config.ts            # API_BASE_URL 配置
│   │   ├── store/
│   │   │   ├── useAppStore.ts       # Zustand 全局状态（题目/答题进度）
│   │   │   └── useAuthStore.ts      # Zustand 认证状态（JWT token）
│   │   └── hooks/                   # 自定义 React Hooks
│   ├── .env.production              # 生产环境变量（VITE_API_BASE_URL）
│   ├── vite.config.ts               # Vite 配置
│   └── package.json
│
├── backend/                         # FastAPI 后端
│   ├── main.py                      # 应用入口：CORS、路由注册、/health
│   ├── config.py                    # pydantic-settings 配置（读取 .env）
│   ├── Dockerfile                   # 后端容器镜像
│   ├── pytest.ini                   # pytest 配置
│   ├── routers/
│   │   ├── auth.py                  # /api/auth — 注册/登录/资料/统计
│   │   ├── questions.py             # /api/questions — 自适应选题/bandit/复习
│   │   ├── tutor.py                 # /api/tutor — Socratic 导师对话
│   │   ├── explanations.py          # /api/explanations — RAG 解析生成
│   │   ├── analytics.py             # /api/analytics — A/B 测试/学习分析
│   │   ├── dashboard.py             # /api/dashboard — 仪表盘统计
│   │   ├── bookmarks.py             # /api/bookmarks — 收藏/错题本
│   │   ├── goals.py                 # /api/goals — 学习目标
│   │   └── theta.py                 # /api/theta — IRT 能力值更新
│   ├── services/
│   │   ├── auth_service.py          # JWT 签发/验证，bcrypt 哈希
│   │   ├── rag_service.py           # RAGService：Qdrant 嵌入检索
│   │   ├── explanation_service.py   # 3-tier fallback：缓存→RAG→纯LLM
│   │   ├── tutor_agent.py           # SocraticTutorAgent（LangChain Agent）
│   │   ├── conversation_manager.py  # 多轮对话状态管理 + Bloom's 追踪
│   │   ├── ab_testing.py            # A/B 测试框架（确定性分组 + 日志）
│   │   └── email_service.py         # SMTP 每日提醒邮件服务
│   ├── ml/
│   │   ├── rag_evaluator.py         # RAG 评估：Precision@K, Recall@K, MRR, F1@K
│   │   └── llm_evaluator.py         # LLM-as-Judge 解析质量评分
│   └── tests/                       # 297 个 pytest 测试
│       ├── test_api.py              # 基础 API 测试（16个）
│       ├── test_rag.py              # RAG 系统测试（21个）
│       ├── test_tutor_agent.py      # 导师 Agent 测试（26个）
│       ├── test_blooms_taxonomy.py  # Bloom's 分类测试（15个）
│       ├── test_ab_testing.py       # A/B 测试框架测试（20个）
│       ├── test_bandit.py           # Thompson Sampling 测试（24个）
│       ├── test_spaced_repetition.py # 间隔重复测试（20个）
│       ├── test_dkt.py              # DKT 模型测试（39个）
│       ├── test_irt_3pl.py          # IRT 3PL 测试（18个）
│       ├── test_features.py         # 功能测试（51个）：仪表盘/书签/目标/邮件
│       └── test_auth.py             # 认证测试（44个）
│
├── engine/                          # 核心 ML 引擎（无 FastAPI 依赖）
│   ├── scoring.py                   # IRT 3PL：probability_3pl, calculate_new_theta, estimate_gmat_score
│   ├── bandit_selector.py           # Thompson Sampling 选题器（Beta 分布 + SQLite）
│   ├── spaced_repetition.py         # Half-Life Regression 遗忘曲线
│   ├── skill_encoder.py             # SkillEncoder：技能↔向量编码（DKT 输入）
│   ├── dkt_model.py                 # DKTModelNumpy（冷启动）+ DKTModelLSTM（PyTorch）
│   ├── recommender.py               # 推荐 pipeline：BKT/DKT → SR注入 → Bandit选题
│   └── __init__.py                  # SkillEncoder/get_dkt_model 导出
│
├── utils/
│   └── db_handler.py                # DatabaseManager：SQLite schema + 所有 DB 操作
│
├── scripts/
│   ├── train_dkt.py                 # DKT 模型训练（argparse，用户级分割，早停）
│   ├── index_to_rag.py              # 批量索引 SQLite → Qdrant
│   ├── analyze_ab_tests.py          # A/B 统计分析（t-test，Cohen's d）
│   ├── evaluate_llm_quality.py      # 批量 LLM 解析质量评估
│   └── send_reminders.py            # 每日提醒邮件 cron 脚本
│
├── docs/
│   ├── api.md                       # 完整 API 端点参考文档
│   ├── demo_script.md               # 5 分钟 Demo 演示脚本
│   └── resume_bullet_points.md      # 简历要点
│
├── app.py                           # 旧版 Streamlit 前端（保留，已被 React 替代）
├── llm_service.py                   # DeepSeek LLM 封装（tutor_reply 向后兼容）
├── logicmaster.db                   # SQLite 数据库（~50题，生产约 421KB）
├── docker-compose.yml               # 本地开发：Qdrant + Backend + Frontend
├── Dockerfile.streamlit             # Streamlit 前端容器（旧版）
├── .env                             # 本地环境变量（不提交 git）
├── .env.example                     # 环境变量模板
├── requirements-backend.txt         # Python 后端依赖
└── requirements.txt                 # 旧版 Streamlit 依赖
```

---

## 4. 环境变量

复制 `.env.example` 为 `.env` 并填写：

```dotenv
# ===== 必填：LLM API Keys =====
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
OPENAI_API_KEY=sk-your-openai-key-here

# ===== Qdrant 向量数据库（Docker 默认值可不改）=====
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=gmat_explanations

# ===== OpenAI Embedding 配置 =====
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMS=1536

# ===== JWT 认证（生产环境必须修改 SECRET！）=====
JWT_SECRET_KEY=glitchmind-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

# ===== SMTP 邮件提醒（可选，留空则不发送）=====
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# ===== 应用配置 =====
APP_ENV=development        # development | production
DEBUG=True
DAILY_QUESTION_GOAL=5      # 默认每日目标题数
```

前端生产环境变量（`frontend/.env.production`）：

```dotenv
VITE_API_BASE_URL=https://api.glitchmind.io
```

---

## 5. 本地开发启动步骤

### 前置条件

- Python 3.11+
- Node.js 18+
- Docker（用于 Qdrant）
- `DEEPSEEK_API_KEY` + `OPENAI_API_KEY`

### 后端启动

```bash
# 1. 克隆项目
git clone <repo-url>
cd mathquest_logicmaster

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API keys

# 3. 安装 Python 依赖（建议使用虚拟环境）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-backend.txt

# 4. 启动 Qdrant（Docker）
docker-compose up -d qdrant

# 5. 索引题目到向量数据库（首次运行）
python scripts/index_to_rag.py

# 6. 启动 FastAPI 后端
cd backend
uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 前端启动

```bash
cd frontend
npm install
npm run dev
# 访问: http://localhost:5173
```

### 完整 Docker 启动（本地一键）

```bash
cp .env.example .env
# 编辑 .env
docker-compose up -d
# 后端: http://localhost:8000
# 前端 (Streamlit旧版): http://localhost:8501
```

### 运行测试

```bash
cd backend
pytest tests/ -v
# 预期：297 个测试全部通过
```

---

## 6. 生产环境部署架构

```
用户浏览器
    │ HTTPS
    ▼
AWS CloudFront + S3
  (gmat.glitchmind.io)
  静态文件：React build dist/
    │
    │ HTTPS API 请求
    ▼
Nginx（EC2 上，api.glitchmind.io）
  ┌─────────────────────────────────┐
  │  server { listen 443 ssl; }     │
  │    proxy_pass → 127.0.0.1:8000 │
  │  server { listen 80; }          │
  │    OPTIONS → 直接 204           │
  │    其他 → 301 → HTTPS           │
  └─────────────────────────────────┘
    │
    ▼
FastAPI（uvicorn，systemd 服务：logicmaster）
  端口 8000，进程守护
    │
    ├── SQLite (logicmaster.db)
    └── Qdrant（Docker，端口 6333/6334）
```

### systemd 服务管理

```bash
# 服务文件位置：/etc/systemd/system/logicmaster.service
sudo systemctl start logicmaster
sudo systemctl stop logicmaster
sudo systemctl restart logicmaster
sudo systemctl status logicmaster

# 查看实时日志
journalctl -u logicmaster -f
```

### Nginx 配置（关键：CORS + OPTIONS 处理）

```nginx
server {
    listen 80;
    server_name api.glitchmind.io;

    # OPTIONS preflight 不经过重定向，直接响应
    if ($request_method = OPTIONS) {
        return 204;
    }
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name api.glitchmind.io;

    ssl_certificate     /etc/letsencrypt/live/api.glitchmind.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.glitchmind.io/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # 注意：不在 Nginx 层添加 CORS 头，由 FastAPI CORSMiddleware 处理
    }
}
```

### 更新部署流程

```bash
# 在服务器上执行
cd /path/to/mathquest_logicmaster
git pull
sudo systemctl restart logicmaster

# 前端更新（本地构建后上传 S3）
cd frontend
npm run build
aws s3 sync dist/ s3://your-bucket-name/ --delete
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

---

## 7. API 端点完整列表

**Base URL（生产）：** `https://api.glitchmind.io`
**Swagger UI：** `https://api.glitchmind.io/docs`

### Health

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康检查（DB + Qdrant 状态） |

### 认证 `/api/auth`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| POST | `/api/auth/register` | 无 | 注册（邮箱+密码），返回 JWT |
| POST | `/api/auth/login` | 无 | 登录，返回 JWT |
| GET | `/api/auth/me` | Bearer | 获取当前用户资料 |
| PUT | `/api/auth/profile` | Bearer | 更新显示名称 |
| PUT | `/api/auth/change-password` | Bearer | 修改密码 |
| DELETE | `/api/auth/account` | Bearer | 删除账户及所有数据 |
| GET | `/api/auth/stats` | Bearer | 获取学习统计（总题数/正确率/最长连击/GMAT估分） |

### 题目 `/api/questions`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/questions/next` | 获取下一道自适应推荐题（支持 `strategy: bandit\|legacy`） |
| POST | `/api/questions/bandit-update` | 答题后更新 Bandit/SR/DKT/错题本 |
| GET | `/api/questions/review-schedule` | 获取间隔重复待复习队列 |
| GET | `/api/questions/{question_id}` | 按 ID 获取指定题目 |

### Socratic 导师 `/api/tutor`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/tutor/start-remediation` | 开始导师对话（诊断错误 + 首条提示 + A/B分组） |
| POST | `/api/tutor/continue` | 继续对话（Bloom's评估 → 下一提示或结论） |
| POST | `/api/tutor/conclude` | 结束对话，返回总结 |
| POST | `/api/tutor/chat` | 旧版无状态对话（向后兼容） |

### 解析 `/api/explanations`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/explanations/generate-with-rag` | 生成 RAG 增强解析（3层fallback） |
| POST | `/api/explanations/search-similar` | 向量搜索相似题目 |

### 仪表盘 `/api/dashboard`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/dashboard/summary` | 仪表盘汇总（今日进度/连击/theta/GMAT估分/薄弱技能/待复习数） |

### 收藏/错题本 `/api/bookmarks`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/bookmarks/add` | 添加收藏/错题（幂等） |
| DELETE | `/api/bookmarks/remove` | 删除书签 |
| GET | `/api/bookmarks/list` | 查询列表（支持 type/skill 过滤） |
| GET | `/api/bookmarks/wrong-stats` | 错题统计（按技能/题型分布） |

### 学习目标 `/api/goals`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/goals/set` | 设置目标 GMAT 分数 + 每日题数 |
| GET | `/api/goals/progress` | 获取目标进度（分差/估计剩余题数/是否达标） |

### 数据分析 `/api/analytics`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/analytics/log-outcome` | 记录 A/B 实验结果 |
| GET | `/api/analytics/ab-test-results` | 获取 A/B 测试聚合统计 |
| GET | `/api/analytics/summary` | 学习分析汇总（答题历史/错题分析/技能掌握度） |
| GET | `/api/analytics/rag-performance` | RAG 系统性能指标 |

### IRT 能力值 `/api/theta`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/theta/update` | 更新 theta（3PL IRT），返回新 theta + GMAT 估分 |

---

## 8. 已实现功能

### 自适应学习引擎

- **IRT 3PL 评分** — 三参数逻辑斯蒂模型（区分度 *a*、难度 *b*、猜测 *c*），MLE 校准（L-BFGS-B），映射到 GMAT Verbal 量表（V20–V51）
- **Thompson Sampling 选题** — Multi-armed bandit，Beta(α, β) 先验，每次答题后更新；探索与利用自动平衡
- **Deep Knowledge Tracing** — PyTorch LSTM（Piech et al. 2015）预测各技能掌握度；冷启动（<50交互）自动降级为 numpy logistic regression
- **Spaced Repetition** — Half-Life Regression（Settles & Meeder 2016）建模遗忘曲线；回忆概率 <0.5 时注入复习题（40%概率）
- **BKT 技能追踪** — 贝叶斯知识追踪，覆盖 10+ 认知技能（Causal Reasoning, Assumption Identification 等）
- **混合推荐 Pipeline** — BKT/DKT 薄弱评分 → SR 注入 → Thompson Sampling 最终选题

### AI 导师系统

- **Socratic Tutor** — LangChain Agent（DeepSeek LLM），渐进式提示：gentle（提示1）→ moderate（提示2）→ direct（提示3），永不直接揭示答案
- **Bloom's Taxonomy 认知评估** — LLM 实时评估学生回答的认知层级（1-6级：Remember → Create），据此调整提示策略
- **RAG 解析** — Qdrant + OpenAI 嵌入，3层 fallback：缓存命中 → RAG增强 → 纯LLM

### 用户功能

- **JWT 认证** — 注册/登录/资料管理/密码修改/账户注销（7天 token，bcrypt 哈希）
- **Dashboard** — 实时统计：GMAT 估分、连击天数、theta、薄弱技能、待复习数、7天日历
- **错题本（Wrong Book）** — 答错自动加入，支持按技能/题型筛选，统计分析
- **学习目标** — 设置目标 GMAT 分数和每日刷题量，追踪进度和达成情况
- **学习分析** — 答题历史曲线、错题分布饼图、技能掌握度雷达

### 数据科学基础设施

- **A/B 测试框架** — 确定性变体分配（哈希）；`tutor_strategy`（socratic_standard / socratic_aggressive / direct_explanation）和 `explanation_source`（rag_enhanced / baseline）两个实验；t-test 显著性 + Cohen's d 效应量分析
- **LLM-as-Judge** — DeepSeek 自动评估解析质量（正确性/清晰度/完整性/教学价值，0-10分）

---

## 9. 已知问题 / TODO

### 已知问题

| 问题 | 状态 | 描述 |
|------|------|------|
| CORS 跨域 | 修复中 | Nginx HTTP→HTTPS 重定向拦截 OPTIONS preflight；`backend/main.py` CORS 配置已修正（显式域名白名单），待服务器 `git pull + restart` |
| `config.ts` fallback URL | 已修复 | 旧版 fallback 使用 `http://`（非 HTTPS），已改为 `https://api.glitchmind.io` |
| SQLite 并发 | 已知限制 | SQLite 不适合高并发写入；生产规模化需迁移 PostgreSQL |
| DKT 冷启动 | 已处理 | <50 条交互时自动降级 numpy 模型，但初始推荐质量有限 |
| Bandit stats 无 user_id | 已知 | `bandit_stats` 表不含 `user_id` 列，Bandit 统计全局共享（非用户级） |

### TODO

- [ ] PostgreSQL 迁移（替代 SQLite）
- [ ] 题库扩充（当前约 50 题）
- [ ] 用户级 Bandit 统计（当前全局共享）
- [ ] 题目生成 Pipeline（LLM 自动出题 + 人工审核）
- [ ] 移动端适配优化
- [ ] DKT 模型定期在线学习（目前离线训练）
- [ ] Qdrant 持久化备份方案
- [ ] Rate limiting（当前无请求限制）

---

## 10. 数据库结构概述

数据库文件：`logicmaster.db`（SQLite）

### `questions` — 题库

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | TEXT PK | 题目唯一 ID |
| `question_type` | TEXT | 题型（Weaken/Strengthen/Assumption/Inference/Flaw/Evaluate/Boldface） |
| `difficulty` | TEXT | 难度（easy/medium/hard） |
| `content` | TEXT | 题目 JSON（stimulus, question, choices, correct, skills, explanation） |
| `elo_difficulty` | REAL | IRT 难度参数 b（默认 1500.0） |
| `is_verified` | INTEGER | 是否审核通过（1=通过，仅已验证题目参与推荐） |
| `discrimination` | REAL | 3PL 区分度参数 a（默认 1.0） |
| `guessing` | REAL | 3PL 猜测参数 c（默认 0.2） |

### `users` — 用户账户

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | TEXT PK | UUID |
| `email` | TEXT UNIQUE | 邮箱（唯一索引） |
| `password_hash` | TEXT | bcrypt 哈希 |
| `display_name` | TEXT | 显示名称（可选） |
| `created_at` | TIMESTAMP | 注册时间 |

### `answer_history` — 答题历史（DKT 训练数据）

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | INTEGER PK | 自增 |
| `user_id` | TEXT | 用户 ID（默认 "default"） |
| `question_id` | TEXT | 题目 ID |
| `skill_ids` | TEXT | 技能列表 JSON |
| `is_correct` | INTEGER | 0/1 |
| `theta_at_time` | REAL | 答题时的 theta 值 |
| `created_at` | TIMESTAMP | 答题时间（索引：user_id + created_at） |

### `spaced_repetition_stats` — 间隔重复统计

| 列 | 类型 | 说明 |
|----|------|------|
| `user_id` | TEXT | 用户 ID（联合主键） |
| `question_id` | TEXT | 题目 ID（联合主键） |
| `half_life` | REAL | 记忆半衰期（天数） |
| `last_practiced` | TIMESTAMP | 最后练习时间 |
| `n_correct` | INTEGER | 答对次数 |
| `n_attempts` | INTEGER | 总尝试次数 |

### `bookmarks` — 收藏/错题本

| 列 | 类型 | 说明 |
|----|------|------|
| `user_id` | TEXT | 用户 ID（联合主键） |
| `question_id` | TEXT | 题目 ID（联合主键） |
| `bookmark_type` | TEXT | "favorite" 或 "wrong"（联合主键） |
| `created_at` | TIMESTAMP | 添加时间 |

### `learning_goals` — 学习目标

| 列 | 类型 | 说明 |
|----|------|------|
| `user_id` | TEXT PK | 用户 ID |
| `target_gmat_score` | INTEGER | 目标 GMAT 分（20-51） |
| `daily_question_goal` | INTEGER | 每日目标题数 |
| `created_at` / `updated_at` | TIMESTAMP | 创建/更新时间 |

### `experiment_logs` — A/B 测试日志

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | INTEGER PK | 自增 |
| `user_id` | TEXT | 用户 ID |
| `experiment_name` | TEXT | 实验名（tutor_strategy / explanation_source） |
| `variant` | TEXT | 变体名 |
| `event_type` | TEXT | "exposure" 或 "outcome" |
| `outcome_metric` | TEXT | 指标名（is_correct, theta_gain 等） |
| `outcome_value` | REAL | 指标数值 |
| `created_at` | TIMESTAMP | 记录时间 |

### `email_logs` — 邮件发送记录

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | INTEGER PK | 自增 |
| `user_id` | TEXT | 用户 ID |
| `email_type` | TEXT | 邮件类型 |
| `sent_at` | TIMESTAMP | 发送时间（24h 去重依据） |

### `bandit_stats`（engine SQLite，独立于 users）— Thompson Sampling 统计

| 列 | 类型 | 说明 |
|----|------|------|
| `question_id` | TEXT PK | 题目 ID |
| `alpha` | REAL | Beta 分布 α 参数（成功次数 + 1） |
| `beta` | REAL | Beta 分布 β 参数（失败次数 + 1） |
| `n_trials` | INTEGER | 总选取次数 |
| `n_successes` | INTEGER | 答对次数 |

---

## 评估脚本

```bash
# 训练 DKT 模型（自动选择 LSTM 或 numpy）
python scripts/train_dkt.py --epochs 10 --compare

# 重新索引题目到 Qdrant
python scripts/index_to_rag.py --force

# A/B 测试统计分析（显著性 + 效应量）
python scripts/analyze_ab_tests.py

# LLM 解析质量评估
python scripts/evaluate_llm_quality.py

# 手动发送每日提醒邮件
python scripts/send_reminders.py
```

---

## License

MIT — © 2025 GlitchMind
