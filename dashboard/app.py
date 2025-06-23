import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Load model and scaler once
model = joblib.load("models/trained_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
log_collection = client["wind_monitoring"]["logs"]

st.title("🌀 Wind Turbine Prediction Dashboard")

# # User Input Form
# with st.form("prediction_form"):
#     Wind_direction = st.number_input("Wind Direction", value=100.0)
#     Ambient_Air_temp = st.number_input("Ambient Air Temperature", value=25.5)
#     Bearing_Temp = st.number_input("Bearing Temperature", value=60.0)
#     BladePitchAngle = st.number_input("Blade Pitch Angle", value=20.0)
#     GearBoxSumpTemp = st.number_input("Gear Box Sump Temp", value=80.0)
#     Generator_Speed = st.number_input("Generator Speed", value=45.0)
#     Hub_Speed = st.number_input("Hub Speed", value=3.5)
#     Wind_Speed = st.number_input("Wind Speed", value=10.5)
#     GearTemp = st.number_input("Gear Temperature", value=85.0)
#     GeneratorTemp = st.number_input("Generator Temp", value=75.0)
#     TurbineName = st.number_input("Turbine Name (Encoded)", value=1)
#     submit = st.form_submit_button("Predict")

with st.form("prediction_form"):
    Nacelle_Position = st.number_input("Nacelle Position", value=0.0)
    Wind_direction = st.number_input("Wind Direction", value=100.0)
    Ambient_Air_temp = st.number_input("Ambient Air Temperature", value=25.5)
    Bearing_Temp = st.number_input("Bearing Temperature", value=60.0)
    BladePitchAngle = st.number_input("Blade Pitch Angle", value=20.0)
    GearBoxSumpTemp = st.number_input("Gear Box Sump Temp", value=80.0)
    Generator_Speed = st.number_input("Generator Speed", value=45.0)
    Hub_Speed = st.number_input("Hub Speed", value=3.5)
    Wind_Speed = st.number_input("Wind Speed", value=10.5)
    GearTemp = st.number_input("Gear Temperature", value=85.0)
    GeneratorTemp = st.number_input("Generator Temp", value=75.0)
    TurbineName = st.number_input("Turbine Name (Encoded)", value=1)
    submit = st.form_submit_button("Predict")
    
if submit:
    input_dict = {
    "Nacelle_Position": Nacelle_Position,
    "Wind_direction": Wind_direction,
    "Ambient_Air_temp": Ambient_Air_temp,
    "Bearing_Temp": Bearing_Temp,
    "BladePitchAngle": BladePitchAngle,
    "GearBoxSumpTemp": GearBoxSumpTemp,
    "Generator_Speed": Generator_Speed,
    "Hub_Speed": Hub_Speed,
    "Wind_Speed": Wind_Speed,
    "GearTemp": GearTemp,
    "GeneratorTemp": GeneratorTemp,
    "TurbineName": TurbineName
}
    # input_dict = {
    #     "Wind_direction": Wind_direction,
    #     "Ambient_Air_temp": Ambient_Air_temp,
    #     "Bearing_Temp": Bearing_Temp,
    #     "BladePitchAngle": BladePitchAngle,
    #     "GearBoxSumpTemp": GearBoxSumpTemp,
    #     "Generator_Speed": Generator_Speed,
    #     "Hub_Speed": Hub_Speed,
    #     "Wind_Speed": Wind_Speed,
    #     "GearTemp": GearTemp,
    #     "GeneratorTemp": GeneratorTemp,
    #     "TurbineName": TurbineName
    # }
    try:
        input_df = np.array([list(input_dict.values())])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        # Log to MongoDB
        log_collection.insert_one({
            "input": input_dict,
            "prediction": float(prediction),
            "timestamp": datetime.now()
        })
        st.success(f"Predicted Power: ⚡ {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")

# Show logs from MongoDB
try:
    data = list(log_collection.find())
    if data:
        df = pd.DataFrame(data)
        st.subheader("📈 Recent Prediction Logs")
        st.write(df.tail(10)[["input", "prediction"]])
    else:
        st.info("No predictions logged yet.")
except Exception as e:
    st.error(f"Could not load logs: {e}")
