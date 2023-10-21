FROM python:3.10-slim
WORKDIR /
COPY requirements.txt .


RUN pip install -r requirements.txt


COPY api.py .
COPY best_model.pkl .


LABEL authors="admin"

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]