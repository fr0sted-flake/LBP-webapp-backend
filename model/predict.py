import joblib
import numpy as np
import pandas as pd

model1 = joblib.load("models/model1.pkl")
model2 = joblib.load("models/model2.pkl")
scaler_X1 = joblib.load("models/scaler_X1.pkl")
scaler_y1 = joblib.load("models/scaler_y1.pkl")
scaler_X2 = joblib.load("models/scaler_X2.pkl")
scaler_y2 = joblib.load("models/scaler_y2.pkl")

def predict_all(throttle: float, gear: float):
    # Step 1
    X1_input = pd.DataFrame([[throttle, gear]], columns=["THROTTLE_POS", "Gear"])
    X1_scaled_input = scaler_X1.transform(X1_input)
    y1_scaled_pred = model1.predict(X1_scaled_input)
    engine_params = scaler_y1.inverse_transform(y1_scaled_pred)

    # Step 2
    X2_input = pd.DataFrame([engine_params[0]], columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr'])
    X2_scaled_input = scaler_X2.transform(X2_input)
    y2_scaled_pred = model2.predict(X2_scaled_input)
    final_outputs = scaler_y2.inverse_transform(y2_scaled_pred.reshape(1, -1))

    efficiency = (final_outputs[0][2] / final_outputs[0][3]) * 100

    return {
        "engine_parameters": {
            "ENGINE_RPM": float(engine_params[0][0]),
            "IntakeGasMassFlow": float(engine_params[0][1]),
            "FuelMassFlow": float(engine_params[0][2]),
            "AirFuelRatio": float(engine_params[0][3]),
        },
        "final_outputs": {
            "EngineTorque": float(final_outputs[0][0]),
            "BSFC": float(final_outputs[0][1]),
            "PowerTransferred": float(final_outputs[0][2]),
            "PowerFromFuel": float(final_outputs[0][3]),
            "Efficiency": float(efficiency)
        }
    }