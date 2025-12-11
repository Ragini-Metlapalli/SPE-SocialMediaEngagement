# Stage 1: Base (Dependencies)
FROM python:3.9-slim AS base
WORKDIR /app

# Install system dependencies if any (usually not needed for pure python, but good practice)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# This layer is CACHED unless requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Test (Runs tests)
FROM base AS test
COPY . .
# We can run pytest directly using this stage
RUN python -m pytest test_main.py

# Stage 3: Production (Lean, runs app)
FROM base AS production
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
