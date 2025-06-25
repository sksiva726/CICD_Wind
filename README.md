# Wind Turbine MLOps Project

This project implements a modular, end-to-end machine learning pipeline for wind turbine sensor data using MongoDB/MySQL, MLflow, FastAPI, Streamlit, and DagsHub for remote experiment tracking.

## Project Structure

- `data/` - Raw and cleaned data CSVs
- `notebooks/` - Jupyter notebooks for EDA
- `scripts/` - Modular Python scripts for each pipeline stage
- `api/` - FastAPI app for model serving
- `dashboard/` - Streamlit dashboard for monitoring
- `mlruns/` - MLflow experiment logs (local, ignored when using DagsHub)
- `requirements.txt` - Python dependencies

## Local Run Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ingest data:
   ```bash
   python scripts/ingest_data.py
   ```
3. Train and log model:
   ```bash
   python scripts/train_model.py
   ```
4. Register model in MLflow UI (http://localhost:5000) as "WindPowerPredictor" and set to Production.
5. Start FastAPI:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
6. Start Streamlit dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```



"token 71c649ea582f516dac5d5446057c572d58efe844

"
---

## Using DagsHub as Remote MLflow Tracking Server

1. **Update MLflow tracking URI in your code:**
   ```python
   import mlflow
   mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")
   # Optionally, set artifact location if needed
   # mlflow.set_artifact_uri("https://dagshub.com/<username>/<repo>.mlflow/artifacts")
   ```
   Or use DagsHub's helper:
   ```python
   import dagshub
   dagshub.init(repo_owner='<username>', repo_name='<repo>', mlflow=True)
   ```
2. **Set DagsHub credentials as environment variables:**

   ```Command Prompt(Windows)
      set MLFLOW_TRACKING_URI=https://dagshub.com/sksivakumar726/CICD_Wind.mlflow
      set MLFLOW_TRACKING_USERNAME=sksiva726
      set MLFLOW_TRACKING_PASSWORD=your-dagshub-
      
      ```
   - On Windows (PowerShell):
     ```powershell
     $env:MLFLOW_TRACKING_USERNAME="<your-dagshub-username>"
     $env:MLFLOW_TRACKING_PASSWORD="<your-dagshub-token>"
     ```
   - On Linux/macOS:
     ```bash
     export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
     export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
     ```
3. **Ignore local experiment logs and environments in `.gitignore`:**
   ```
   mlruns/
   .venv/
   __pycache__/
   *.pkl
   *.log
   data/
   models/
   !data/.gitkeep
   !models/.gitkeep
   ```
4. **(Optional) Use DVC for data/model versioning:**
   - Initialize DVC:
     ```bash
     dvc init
     ```
   - Add data/model files:
     ```bash
     dvc add data/wind_turbine_Data.csv
     dvc add models/trained_model.pkl
     ```
   - Commit DVC metafiles:
     ```bash
     git add data/wind_turbine_Data.csv.dvc models/trained_model.pkl.dvc dvc.yaml .gitignore
     git commit -m "Track data and model with DVC"
     ```
   - Set up DagsHub DVC remote (see DagsHub docs for details).

---

## Best Practices
- Always pull remote changes before pushing to avoid merge conflicts.
- Do not commit large data or model files directly; use DVC and DagsHub storage.
- Keep your MLflow tracking URI and credentials secure (do not hardcode tokens in code).
- Use `.gitignore` to avoid pushing local experiment logs and environments.

---

## Example MLflow Logging (with DagsHub)
```python
import dagshub
import mlflow

dagshub.init(repo_owner='<username>', repo_name='<repo>', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")

with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
```

---

- All experiment tracking and artifacts are now stored remotely on DagsHub.
- Local `mlruns/` is ignored when using DagsHub as the tracking server.
- For more, see [DagsHub MLflow Docs](https://dagshub.com/docs/integrations/mlflow/).