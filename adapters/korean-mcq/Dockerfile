# Korean MCQ Evaluation Adapter for EvalHub
FROM registry.access.redhat.com/ubi9/python-311:latest

USER 0

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy adapter source code
COPY main.py prompts.py parser.py datasets_config.py ./
COPY mapping/ ./mapping/

# Writable directories for HuggingFace cache and tmp
RUN mkdir -p /tmp/.cache/huggingface && \
    chmod -R 777 /tmp && \
    chown -R 1001:0 /app && \
    chmod -R g+rwX /app

# Environment
ENV PYTHONUNBUFFERED=1
ENV EVALHUB_MODE=k8s
ENV HOME=/tmp
ENV HF_HOME=/tmp/.cache/huggingface

USER 1001

ENTRYPOINT ["python", "main.py"]
