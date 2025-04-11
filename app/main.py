from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and scalers
model = joblib.load("random_forest_model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

app = FastAPI()

class InputData(BaseModel):
    throttle_pos: float
    gear: float

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([[data.throttle_pos, data.gear]], columns=["THROTTLE_POS", "Gear"])
    input_scaled = x_scaler.transform(input_df)

    pred_scaled = model.predict(input_scaled)
    pred = y_scaler.inverse_transform(pred_scaled)

    return {
        "ENGINE_RPM": round(pred[0][0], 2),
        "Intake_Gas_Mass_Flow": round(pred[0][1], 6),
        "Fuel_Mass_Flow": round(pred[0][2], 6),
        "Air_Fuel_Ratio": round(pred[0][3], 2)
    }
