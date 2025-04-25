import joblib
import numpy as np
import pandas as pd
import os

# Model paths
MODELS_PATH = "models/"

# Load scalers
scaler_X1 = joblib.load(f"{MODELS_PATH}scaler_X1.pkl")
scaler_y1 = joblib.load(f"{MODELS_PATH}scaler_y1.pkl")
scaler_X2 = joblib.load(f"{MODELS_PATH}scaler_X2.pkl")
scaler_y2 = joblib.load(f"{MODELS_PATH}scaler_y2.pkl")

# Load Step 1 models
model1_rpm = joblib.load(f"{MODELS_PATH}model1_ENGINE_RPM.pkl")
model1_intake = joblib.load(f"{MODELS_PATH}model1_IntkGasMassFlw.pkl")
model1_fuel = joblib.load(f"{MODELS_PATH}model1_FuelMassFlw.pkl")

# Load Step 2 models
model2_torque = joblib.load(f"{MODELS_PATH}model2_EngTrq.pkl")
model2_bsfc = joblib.load(f"{MODELS_PATH}model2_Bsfc.pkl")
model2_power = joblib.load(f"{MODELS_PATH}model2_PwrTrnsfrd.pkl")

def predict_all(throttle: float, gear: float):
    """Make engine performance predictions based on throttle position and gear"""
    # Step 1: Predict initial engine parameters
    X1_input = pd.DataFrame([[throttle, gear]], columns=["THROTTLE_POS", "Gear"])
    X1_scaled_input = scaler_X1.transform(X1_input)
    
    # Make predictions with each Step 1 model
    rpm_pred = model1_rpm.predict(X1_scaled_input)
    intake_pred = model1_intake.predict(X1_scaled_input)
    fuel_pred = model1_fuel.predict(X1_scaled_input)
    
    # Combine predictions into a single array
    y1_scaled_pred = np.column_stack([rpm_pred, intake_pred, fuel_pred])
    
    # Transform back to original scale
    engine_params = scaler_y1.inverse_transform(y1_scaled_pred)
    
    # Step 2: Use engine parameters to predict performance metrics
    X2_input = pd.DataFrame([engine_params[0]], columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw'])
    X2_scaled_input = scaler_X2.transform(X2_input)
    
    # Make predictions with each Step 2 model
    torque_pred = model2_torque.predict(X2_scaled_input)
    bsfc_pred = model2_bsfc.predict(X2_scaled_input)
    power_pred = model2_power.predict(X2_scaled_input)
    
    # Combine predictions into a single array
    y2_scaled_pred = np.column_stack([torque_pred, bsfc_pred, power_pred])
    
    # Transform back to original scale
    final_outputs = scaler_y2.inverse_transform(y2_scaled_pred)
    
    # Calculate fuel power (no longer directly predicted)
    # Using the relationship: Power (kW) = Fuel flow (g/s) * Lower Heating Value (kJ/g) / 1000
    # Assuming a typical LHV for gasoline of about 44 kJ/g
    LHV = 38000000 # Lower Heating Value in J/kg for gasoline
    fuel_power = engine_params[0][2] * LHV 
    
    # Calculate efficiency
    efficiency = (final_outputs[0][2] / fuel_power) * 100

    afr=engine_params[0][1]/engine_params[0][2]
    # Round the values to 2 decimal places for final output
    
    return {
        "engine_parameters": {
            "ENGINE_RPM": round(float(engine_params[0][0]), 2),
            "IntakeGasMassFlow": round(float(engine_params[0][1]), 6),
            "FuelMassFlow": round(float(engine_params[0][2]), 6),
            "AirFuelRatio": round(float(afr), 2),
        },
        "final_outputs": {
            "EngineTorque": round(float(final_outputs[0][0]), 2),
            "BSFC": round(float(final_outputs[0][1]), 2),
            "PowerTransferred": round(float(final_outputs[0][2]), 2),
            "PowerFromFuel": round(float(fuel_power), 2),
            "Efficiency": round(float(efficiency), 2)
        }
    }