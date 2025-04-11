import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

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

    X1_train, _, y1_train, _ = train_test_split(X1_scaled, y1_scaled, test_size=0.2, random_state=42)

    model1 = RandomForestRegressor(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model1.fit(X1_train, y1_train)

    # Step 2: Use predictions from Step 1 to train XGBoost
    df_pred1 = model1.predict(X1_scaled)
    X2 = pd.DataFrame(df_pred1, columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr'])
    y2 = df[['EngTrq', 'Bsfc', 'PwrTrnsfrd', 'PwrFuel']]

    X2_scaler = MinMaxScaler()
    X2_scaled = X2_scaler.fit_transform(X2)

    y2_scaler = MinMaxScaler()
    y2_scaled = y2_scaler.fit_transform(y2)

    X2_train, _, y2_train, _ = train_test_split(X2_scaled, y2_scaled, test_size=0.2, random_state=42)

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

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Save everything
    joblib.dump(model1, SAVE_PATH + "model1.pkl")
    joblib.dump(model2, SAVE_PATH + "model2.pkl")
    joblib.dump(X1_scaler, SAVE_PATH + "scaler_X1.pkl")
    joblib.dump(y1_scaler, SAVE_PATH + "scaler_y1.pkl")
    joblib.dump(X2_scaler, SAVE_PATH + "scaler_X2.pkl")
    joblib.dump(y2_scaler, SAVE_PATH + "scaler_y2.pkl")

if __name__ == "__main__":
    train_models()
