# app/model.py
import joblib
import numpy as np
import pandas as pd

model1 = joblib.load("models/model1.pkl")
model2 = joblib.load("models/model2.pkl")
X1_scaler = joblib.load("models/scaler_X1.pkl")
y1_scaler = joblib.load("models/scaler_y1.pkl")
X2_scaler = joblib.load("models/scaler_X2.pkl")
y2_scaler = joblib.load("models/scaler_y2.pkl")

def predict_engine_outputs(throttle: float, gear: float):
    X1_input = pd.DataFrame([[throttle, gear]], columns=['THROTTLE_POS', 'Gear'])
    X1_scaled = X1_scaler.transform(X1_input)

    y1_pred_scaled = model1.predict(X1_scaled)
    y1_pred = y1_scaler.inverse_transform(y1_pred_scaled)

    X2_input = pd.DataFrame([y1_pred[0]], columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr'])
    X2_scaled = X2_scaler.transform(X2_input)

    y2_pred_scaled = model2.predict(X2_scaled)
    y2_pred = y2_scaler.inverse_transform(y2_pred_scaled)

    efficiency = (y2_pred[0][2] / y2_pred[0][3]) * 100
    return {
        "engine_rpm": y1_pred[0][0],
        "intake_flow": y1_pred[0][1],
        "fuel_flow": y1_pred[0][2],
        "afr": y1_pred[0][3],
        "torque": y2_pred[0][0],
        "bsfc": y2_pred[0][1],
        "power_transferred": y2_pred[0][2],
        "fuel_power": y2_pred[0][3],
        "efficiency": efficiency
    }
