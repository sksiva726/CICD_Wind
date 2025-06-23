<<<<<<< HEAD
# Wind Turbine MLOps Project

This project implements a local, modular, end-to-end machine learning pipeline for wind turbine sensor data using MongoDB/MySQL, MLflow, FastAPI, and Streamlit.

## Project Structure

- `data/` - Raw and cleaned data CSVs
- `notebooks/` - Jupyter notebooks for EDA
- `scripts/` - Modular Python scripts for each pipeline stage
- `api/` - FastAPI app for model serving
- `dashboard/` - Streamlit dashboard for monitoring
- `mlruns/` - MLflow experiment logs
- `requirements.txt` - Python dependencies

## Local Run Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start MLflow server:
   ```bash
   mlflow server \
     --backend-store-uri mysql+pymysql://root:password@localhost/mlflow_db \
     --default-artifact-root ./mlruns \
     --host 127.0.0.1 --port 5000
   ```
3. Ingest data:
   ```bash
   python scripts/ingest_data.py
   ```
4. Train and log model:
   ```bash
   python scripts/train_model.py
   ```
5. Register model in MLflow UI (http://localhost:5000) as "WindPowerPredictor" and set to Production.
6. Start FastAPI:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
7. Start Streamlit dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

---

- All data and logs are stored locally in MongoDB and MySQL.
- No cloud, GitHub, or CI/CD is used at this stage.
=======
# CICD_Wind
CICD_MLflow_Wind
>>>>>>> 45300cb51de4d4c981d589761637ded68176350f
