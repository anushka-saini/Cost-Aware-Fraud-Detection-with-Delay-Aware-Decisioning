FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/main.py api/dashboard_routes.py api/validation_metrics.json api/rf_model.pkl ./
# dashboard_routes.py looks for the demo sample one level above the API folder
COPY demo_sample.csv /demo_sample.csv
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
