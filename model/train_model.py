import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

DATA_PATH = "data/output_engine_data_2.csv"
SAVE_PATH = "models/"

def train_models():
    # Load and clean data
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Step 1: Predict ENGINE_RPM, IntkGasMassFlw, FuelMassFlw, Afr
    X1 = df[['THROTTLE_POS', 'Gear']]
    y1 = df[['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr']]

    X1_scaler = MinMaxScaler()
    X1_scaled = X1_scaler.fit_transform(X1)

    y1_scaler = MinMaxScaler()
    y1_scaled = y1_scaler.fit_transform(y1)

    # Actually keep the test sets for validation
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1_scaled, test_size=0.2, random_state=42)

    # Using Random Forest Regressor
    model1 = RandomForestRegressor(
        n_estimators=1000,  # Number of trees
        max_depth=15,  # Increased depth for better learning
        min_samples_split=3,  # Minimum samples to split a node
        min_samples_leaf=2,  # Minimum samples per leaf
        random_state=42,  # Ensures reproducibility
        n_jobs=-1  # Utilize all CPU cores
    )
    model1.fit(X1_train, y1_train)
    
    # Evaluate model1
    y1_pred = model1.predict(X1_test)
    print("Model 1 (Random Forest) Performance:")
    print(f"MSE: {mean_squared_error(y1_test, y1_pred)}")
    print(f"R²: {r2_score(y1_test, y1_pred)}")

    # Step 2: Use predictions from Step 1 to train XGBoost
    # Get predictions for the entire dataset (inverse transform to get real values)
    df_pred1 = model1.predict(X1_scaled)
    df_pred1 = y1_scaler.inverse_transform(df_pred1)
    
    X2 = pd.DataFrame(df_pred1, columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr'])
    y2 = df[['EngTrq', 'Bsfc', 'PwrTrnsfrd', 'PwrFuel']]

    X2_scaler = MinMaxScaler()
    X2_scaled = X2_scaler.fit_transform(X2)

    y2_scaler = MinMaxScaler()
    y2_scaled = y2_scaler.fit_transform(y2)

    # Actually keep the test sets for validation
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2_scaled, test_size=0.2, random_state=42)

    model2 = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model2.fit(X2_train, y2_train)
    
    # Evaluate XGBoost model
    y2_pred = model2.predict(X2_test)
    print("\nModel 2 (XGBoost) Performance:")
    print(f"MSE: {mean_squared_error(y2_test, y2_pred)}")
    print(f"R²: {r2_score(y2_test, y2_pred)}")

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Save everything
    joblib.dump(model1, SAVE_PATH + "model1.pkl")
    joblib.dump(model2, SAVE_PATH + "model2.pkl")
    joblib.dump(X1_scaler, SAVE_PATH + "scaler_X1.pkl")
    joblib.dump(y1_scaler, SAVE_PATH + "scaler_y1.pkl")
    joblib.dump(X2_scaler, SAVE_PATH + "scaler_X2.pkl")
    joblib.dump(y2_scaler, SAVE_PATH + "scaler_y2.pkl")
    
    print("\nModels trained and saved successfully!")

if __name__ == "__main__":
    train_models()