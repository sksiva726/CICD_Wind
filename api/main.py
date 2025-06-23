# from fastapi import FastAPI
# from pydantic import BaseModel
# import mlflow.sklearn
# from pymongo import MongoClient
# import pandas as pd

# app = FastAPI()
# model = mlflow.sklearn.load_model("models:/WindPowerPredictor/Production")
# client = MongoClient("mongodb://localhost:27017/")
# logs = client["wind_monitoring"]["logs"]

# class InputData(BaseModel):
#     Wind_Speed: float
#     Wind_direction: float
#     Ambient_Air_temp: float
#     Bearing_Temp: float
#     GearTemp: float
#     GeneratorTemp: float
#     GearBoxSumpTemp: float
#     BladePitchAngle: float
#     Hub_Speed: float
#     Generator_Speed: float
#     TurbineName: int

# @app.post("/predict")
# def predict(data: InputData):
#     input_df = pd.DataFrame([data.dict()])
#     prediction = float(model.predict(input_df)[0])
#     logs.insert_one({"input": data.dict(), "prediction": prediction})
#     return {"prediction": prediction}
 
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime

app = FastAPI()

def load_scaler(path="models/scaler.pkl"):
    import joblib
    return joblib.load(path)

# Load model and scaler
model = joblib.load("models/trained_model.pkl")
scaler = load_scaler()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
log_collection = client["wind_monitoring"]["logs"]

class InputData(BaseModel):
    Wind_direction: float
    Ambient_Air_temp: float
    Bearing_Temp: float
    BladePitchAngle: float
    GearBoxSumpTemp: float
    Generator_Speed: float
    Hub_Speed: float
    Wind_Speed: float
    GearTemp: float
    GeneratorTemp: float
    TurbineName: int  # encoded as integer

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    input_df = np.array([list(input_dict.values())])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Save to MongoDB
    log_collection.insert_one({
        "input": input_dict,
        "prediction": float(prediction),
        "timestamp": datetime.now()
    })

    return {"prediction": prediction}
