FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qec_gnn ./qec_gnn
COPY tests ./tests
COPY scripts ./scripts

RUN chmod +x scripts/run_pipeline.sh

ENTRYPOINT ["scripts/run_pipeline.sh"]
