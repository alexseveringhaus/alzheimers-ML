FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Application code
COPY api/ ./api/

# Shared data extraction module (provides amy_rois / tau_rois)
COPY data/data_extraction.py ./data/data_extraction.py
RUN touch ./data/__init__.py

# Model artifacts
COPY checkpoints/logreg_combined.joblib ./checkpoints/logreg_combined.joblib
COPY checkpoints/shap_background.npy   ./checkpoints/shap_background.npy

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
