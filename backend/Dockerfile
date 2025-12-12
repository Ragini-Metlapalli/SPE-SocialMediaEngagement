FROM raginimetlapalli/social-media-backend-base-cpu:latest

WORKDIR /app

# FIX scikit-learn model compatibility
RUN pip uninstall -y scikit-learn && pip install scikit-learn==1.2.2

# FIX NumPy 2.0 incompatibility (VERY IMPORTANT)
RUN pip uninstall -y numpy && pip install numpy==1.26.4

RUN curl -L https://github.com/storj/storj/releases/latest/download/uplink_linux_amd64.zip -o uplink.zip \
    && unzip uplink.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/uplink \
    && rm uplink.zip

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

