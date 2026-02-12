FROM python:3.11-slim

WORKDIR /app

COPY ../requirements-backend.txt ./requirements-backend.txt
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy engine modules (IRT scoring, BKT recommender)
COPY ../engine/ ./engine/
# Copy LLM service
COPY ../llm_service.py ./llm_service.py
# Copy utils
COPY ../utils/ ./utils/
# Copy SQLite database
COPY ../logicmaster.db ./logicmaster.db
# Copy backend package
COPY . ./backend/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
