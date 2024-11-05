# Dockerfile
FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src  # Copy the source code to /app/src

ENTRYPOINT ["python", "src/train.py"]
