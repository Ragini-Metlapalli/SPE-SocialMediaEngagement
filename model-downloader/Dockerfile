FROM python:3.9-slim

WORKDIR /app

# Only deps required for downloading HF models
RUN pip install --no-cache-dir transformers torch detoxify sentencepiece protobuf

COPY download_models_2.py .

ENV HF_CACHE=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models

CMD ["python", "download_models_2.py"]
