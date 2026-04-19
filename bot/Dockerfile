FROM python:3.11-slim

WORKDIR /bot

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "app.py"]
