FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY lib .
COPY api.py .
COPY best_model.pkl .

LABEL authors="admin"

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "6000"]