FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi>=0.110.0 httpx>=0.27.0 uvicorn>=0.29.0

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
