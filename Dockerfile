FROM python:3.14-slim

# LightGBM and XGBoost need OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# The app does not need the notebook tooling.
RUN grep -vE '^(jupyter|nbconvert)==' requirements.txt > req-app.txt && pip install --no-cache-dir -r req-app.txt

COPY vehicle_model.py app.py ./
COPY model/ model/
COPY results/metrics.json results/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
