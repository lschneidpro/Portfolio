FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .

CMD ["python", "./churn_script_logging_and_tests.py"]