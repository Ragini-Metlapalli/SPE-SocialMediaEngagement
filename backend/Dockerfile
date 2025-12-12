FROM raginimetlapalli/social-media-backend-base:latest

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

