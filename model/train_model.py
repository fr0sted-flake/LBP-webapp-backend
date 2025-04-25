import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import time

DATA_PATH = "data/output_engine_data_2.csv"
SAVE_PATH = "models/"

def train_models():
    # Load and clean data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Step 1: Predict ENGINE_RPM, IntkGasMassFlw, FuelMassFlw (removed Afr)
    X1 = df[['THROTTLE_POS', 'Gear']]
    y1 = df[['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw']]  # Removed Afr

    X1_scaler = MinMaxScaler()
    X1_scaled = X1_scaler.fit_transform(X1)

    y1_scaler = MinMaxScaler()
    y1_scaled = y1_scaler.fit_transform(y1)

    # Test phase - train with split data to evaluate performance
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1_scaled, test_size=0.2, random_state=42)

    # Updated: Using XGBoost with GridSearchCV for individual targets
    # Define parameter grid for optimization
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    # Base XGBoost model
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # Train individual models for each target in step 1
    best_models_step1 = []
    best_params_step1 = []
    y1_pred_scaled = np.zeros_like(y1_test)

    print("\n🔧 Step 1: Training individual XGBoost models for each target")
    for i, target in enumerate(y1.columns):
        print(f"\nTarget: {target}")
        
        # Use GridSearchCV to find best hyperparameters
        grid = GridSearchCV(
            estimator=xgb_regressor, 
            param_grid=param_grid, 
            cv=3, 
            scoring='r2', 
            n_jobs=-1, 
            verbose=1
        )

        start_time = time.time()
        grid.fit(X1_train, y1_train[:, i])
        end_time = time.time()

        best_model = grid.best_estimator_
        best_models_step1.append(best_model)
        best_params_step1.append(grid.best_params_)

        y1_pred_scaled[:, i] = best_model.predict(X1_test)

        print(f"Best Parameters: {grid.best_params_}")
        print(f"Training Time: {end_time - start_time:.2f}s")
        print(f"MSE: {mean_squared_error(y1_test[:, i], y1_pred_scaled[:, i]):.4f}")
        print(f"R²: {r2_score(y1_test[:, i], y1_pred_scaled[:, i]):.4f}")

    # Step 2: Get predictions for the entire dataset using the best models
    print("\nGenerating predictions for Stage 2...")
    y1_pred_all = np.column_stack([model.predict(X1_scaled) for model in best_models_step1])
    df_pred1 = y1_scaler.inverse_transform(y1_pred_all)
    
    # Update outputs for stage 2 (removed PwrFuel)
    X2 = pd.DataFrame(df_pred1, columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw'])
    y2 = df[['EngTrq', 'Bsfc', 'PwrTrnsfrd']]

    X2_scaler = MinMaxScaler()
    X2_scaled = X2_scaler.fit_transform(X2)

    y2_scaler = MinMaxScaler()
    y2_scaled = y2_scaler.fit_transform(y2)

    # Test phase for model2
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2_scaled, test_size=0.2, random_state=42)

    # Train individual models for each target in step 2
    best_models_step2 = []
    best_params_step2 = []
    y2_pred_scaled = np.zeros_like(y2_test)

    print("\n⚙️ Step 2: Training individual XGBoost models for each target")
    for i, target in enumerate(y2.columns):
        print(f"\nTarget: {target}")
        
        # Use GridSearchCV to find best hyperparameters
        grid = GridSearchCV(
            estimator=xgb_regressor, 
            param_grid=param_grid, 
            cv=3, 
            scoring='r2', 
            n_jobs=-1, 
            verbose=1
        )

        start_time = time.time()
        grid.fit(X2_train, y2_train[:, i])
        end_time = time.time()

        best_model = grid.best_estimator_
        best_models_step2.append(best_model)
        best_params_step2.append(grid.best_params_)

        y2_pred_scaled[:, i] = best_model.predict(X2_test)

        print(f"Best Parameters: {grid.best_params_}")
        print(f"Training Time: {end_time - start_time:.2f}s")
        print(f"MSE: {mean_squared_error(y2_test[:, i], y2_pred_scaled[:, i]):.4f}")
        print(f"R²: {r2_score(y2_test[:, i], y2_pred_scaled[:, i]):.4f}")

    print("\n--- Now training final models on FULL dataset for maximum accuracy ---")
    
    # RETRAIN ALL MODELS ON FULL DATASET
    print("Retraining all models on full dataset...")
    
    # Step 1 models (full dataset)
    final_models_step1 = []
    for i, target in enumerate(y1.columns):
        print(f"Retraining Step 1 model for {target}...")
        model = xgb.XGBRegressor(**best_params_step1[i], random_state=42)
        model.fit(X1_scaled, y1_scaled[:, i])
        final_models_step1.append(model)
    
    # Generate predictions for Step 2 using the full dataset models
    y1_pred_all = np.column_stack([model.predict(X1_scaled) for model in final_models_step1])
    df_pred1 = y1_scaler.inverse_transform(y1_pred_all)
    
    # Create features for model2
    X2 = pd.DataFrame(df_pred1, columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw'])
    X2_scaled = X2_scaler.transform(X2)
    
    # Step 2 models (full dataset)
    final_models_step2 = []
    for i, target in enumerate(y2.columns):
        print(f"Retraining Step 2 model for {target}...")
        model = xgb.XGBRegressor(**best_params_step2[i], random_state=42)
        model.fit(X2_scaled, y2_scaled[:, i])
        final_models_step2.append(model)

    # Create directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Save all models and scalers
    print("Saving models and scalers...")
    
    # Save Step 1 models
    for i, target in enumerate(y1.columns):
        joblib.dump(final_models_step1[i], f"{SAVE_PATH}model1_{target}.pkl")
    
    # Save Step 2 models
    for i, target in enumerate(y2.columns):
        joblib.dump(final_models_step2[i], f"{SAVE_PATH}model2_{target}.pkl")
    
    # Save scalers
    joblib.dump(X1_scaler, f"{SAVE_PATH}scaler_X1.pkl")
    joblib.dump(y1_scaler, f"{SAVE_PATH}scaler_y1.pkl")
    joblib.dump(X2_scaler, f"{SAVE_PATH}scaler_X2.pkl")
    joblib.dump(y2_scaler, f"{SAVE_PATH}scaler_y2.pkl")
    
    # Also save best parameters for reference
    joblib.dump(best_params_step1, f"{SAVE_PATH}best_params_step1.pkl")
    joblib.dump(best_params_step2, f"{SAVE_PATH}best_params_step2.pkl")
    
    print("\nAll models trained on FULL dataset and saved successfully!")
    
    # Return information about the trained models
    return {
        "step1_targets": list(y1.columns),
        "step2_targets": list(y2.columns),
        "step1_params": best_params_step1,
        "step2_params": best_params_step2
    }

if __name__ == "__main__":
    train_models()