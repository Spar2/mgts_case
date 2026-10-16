FROM python:3.14-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY fiber ./fiber

ENV APP_PORT=8082 PYTHONUNBUFFERED=1
EXPOSE 8082
CMD ["uvicorn", "fiber.api:app", "--host", "0.0.0.0", "--port", "8082"]
