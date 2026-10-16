FROM python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends age && rm -rf /var/lib/apt/lists/*
COPY services/web-api/requirements.lock.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/web-api/workbench ./workbench
COPY services/web-api/public ./public
COPY services/web-api/research ./research
ENV DTS_PUBLIC_PACKAGE=/app/public/demo-v1
RUN useradd --uid 10001 --create-home workbench && mkdir /runtime && chown workbench:workbench /runtime
USER workbench
CMD ["python", "-m", "uvicorn", "workbench.api:app", "--host", "0.0.0.0", "--port", "8091", "--no-access-log"]
