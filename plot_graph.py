import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

# Set paths
DATA_PATH = "data/output_engine_data_2.csv"
MODELS_PATH = "models/"
PLOTS_PATH = "plots/"

# Create plots directory
os.makedirs(PLOTS_PATH, exist_ok=True)

def load_models_and_data():
    """Load trained models, scalers, and original data"""
    # Load models and scalers
    model1 = joblib.load(MODELS_PATH + "model1.pkl")
    model2 = joblib.load(MODELS_PATH + "model2.pkl")
    X1_scaler = joblib.load(MODELS_PATH + "scaler_X1.pkl")
    y1_scaler = joblib.load(MODELS_PATH + "scaler_y1.pkl")
    X2_scaler = joblib.load(MODELS_PATH + "scaler_X2.pkl")
    y2_scaler = joblib.load(MODELS_PATH + "scaler_y2.pkl")
    
    # Load original data
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    
    return {
        'model1': model1,
        'model2': model2,
        'X1_scaler': X1_scaler,
        'y1_scaler': y1_scaler,
        'X2_scaler': X2_scaler,
        'y2_scaler': y2_scaler,
        'data': df
    }

def generate_predictions(components):
    """Generate predictions using the trained models"""
    df = components['data']
    model1 = components['model1']
    model2 = components['model2']
    X1_scaler = components['X1_scaler']
    y1_scaler = components['y1_scaler']
    X2_scaler = components['X2_scaler']
    y2_scaler = components['y2_scaler']
    
    # Prepare inputs for model1
    X1 = df[['THROTTLE_POS', 'Gear']]
    X1_scaled = X1_scaler.transform(X1)
    
    # Get ground truth for model1
    y1_true = df[['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr']]
    
    # Generate predictions from model1
    y1_scaled_pred = model1.predict(X1_scaled)
    y1_pred = y1_scaler.inverse_transform(y1_scaled_pred)
    
    # Use model1 predictions as input for model2
    X2 = pd.DataFrame(y1_pred, columns=['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr'])
    X2_scaled = X2_scaler.transform(X2)
    
    # Get ground truth for model2
    y2_true = df[['EngTrq', 'Bsfc', 'PwrTrnsfrd', 'PwrFuel']]
    
    # Generate predictions from model2
    y2_scaled_pred = model2.predict(X2_scaled)
    y2_pred = y2_scaler.inverse_transform(y2_scaled_pred)
    
    # Calculate metrics
    r2_model1 = r2_score(y1_true, y1_pred)
    r2_model2 = r2_score(y2_true, y2_pred)
    
    print(f"Model 1 (Random Forest) R² score: {r2_model1:.4f}")
    print(f"Model 2 (XGBoost) R² score: {r2_model2:.4f}")
    
    return {
        'y1_true': y1_true,
        'y1_pred': y1_pred,
        'y2_true': y2_true,
        'y2_pred': y2_pred,
        'r2_model1': r2_model1,
        'r2_model2': r2_model2
    }

def plot_comparison_line(true_values, predicted_values, feature_name, sample_count=100):
    """Create line plot comparing predicted vs actual values"""
    plt.figure(figsize=(12, 6))
    
    # Get a subset of data points for clarity
    indices = np.arange(min(sample_count, len(true_values)))
    
    plt.plot(indices, true_values.iloc[indices], 'b-', label='Actual', linewidth=2)
    plt.plot(indices, predicted_values[indices], 'r--', label='Predicted', linewidth=2)
    
    plt.title(f'{feature_name} - Actual vs Predicted Values', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate R² for this feature
    r2 = r2_score(true_values, predicted_values)
    plt.figtext(0.15, 0.85, f'R² = {r2:.4f}', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}{feature_name}_line_comparison.png", dpi=300)
    plt.close()

def plot_correlation_scatter(true_values, predicted_values, feature_name):
    """Create scatter plot showing correlation between predicted and actual values"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.5, s=30)
    
    # Add perfect prediction line
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    plt.title(f'{feature_name} - Correlation Plot', fontsize=16)
    plt.xlabel(f'Actual {feature_name}', fontsize=14)
    plt.ylabel(f'Predicted {feature_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Calculate and display R²
    r2 = r2_score(true_values, predicted_values)
    plt.figtext(0.15, 0.85, f'R² = {r2:.4f}', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}{feature_name}_scatter_correlation.png", dpi=300)
    plt.close()

def plot_error_distribution(true_values, predicted_values, feature_name):
    """Create histogram of prediction errors"""
    errors = predicted_values - true_values
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30)
    
    plt.title(f'{feature_name} - Prediction Error Distribution', fontsize=16)
    plt.xlabel('Prediction Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    plt.figtext(0.15, 0.85, f'RMSE = {rmse:.4f}', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}{feature_name}_error_distribution.png", dpi=300)
    plt.close()

def plot_model_performance_comparison(r2_scores):
    """Create bar chart comparing R² scores across models"""
    plt.figure(figsize=(12, 7))
    
    models = ['Random Forest', 'XGBoost']
    scores = [r2_scores['r2_model1'], r2_scores['r2_model2']]
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(models, scores, color=colors, width=0.6)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', fontsize=14)
    
    plt.title('Model Performance Comparison (R² Score)', fontsize=18)
    plt.ylabel('R² Score', fontsize=16)
    plt.ylim(0, 1.05)  # R² score is usually between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}model_performance_comparison.png", dpi=300)
    plt.close()

def create_feature_importance_plot(model, feature_names, title, filename):
    """Create feature importance plot for a model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title(title, fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

def main():
    print("Loading models and data...")
    components = load_models_and_data()
    
    print("Generating predictions...")
    results = generate_predictions(components)
    
    print("Creating visualizations...")
    
    # Plot model1 predictions (first model)
    for i, feature in enumerate(['ENGINE_RPM', 'IntkGasMassFlw', 'FuelMassFlw', 'Afr']):
        true_values = results['y1_true'][feature]
        predicted_values = results['y1_pred'][:, i]
        
        print(f"Creating plots for {feature}...")
        plot_comparison_line(true_values, predicted_values, feature)
        plot_correlation_scatter(true_values, predicted_values, feature)
        plot_error_distribution(true_values, predicted_values, feature)
    
    # Plot model2 predictions (second model)
    for i, feature in enumerate(['EngTrq', 'Bsfc', 'PwrTrnsfrd', 'PwrFuel']):
        true_values = results['y2_true'][feature]
        predicted_values = results['y2_pred'][:, i]
        
        print(f"Creating plots for {feature}...")
        plot_comparison_line(true_values, predicted_values, feature)
        plot_correlation_scatter(true_values, predicted_values, feature)
        plot_error_distribution(true_values, predicted_values, feature)
    
    # Plot overall model performance comparison
    plot_model_performance_comparison({
        'r2_model1': results['r2_model1'],
        'r2_model2': results['r2_model2']
    })
    
    # Create feature importance plots
    create_feature_importance_plot(
        components['model1'],
        ['THROTTLE_POS', 'Gear'],
        'Random Forest Feature Importance',
        f"{PLOTS_PATH}rf_feature_importance.png"
    )
    
    print(f"All visualizations saved to {PLOTS_PATH}")
    print("Done!")

if __name__ == "__main__":
    main()