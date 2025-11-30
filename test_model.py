#!/usr/bin/env python3
"""
Test script for validating ML model predictions and performance
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train import preprocess, get_models, infer_problem_type
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_test_model(data_path, target_col, test_fraction=0.2):
    """
    Load data, train model, and test on holdout set
    """
    # Load data
    df = pd.read_csv(data_path)
    X, y = preprocess(df, target_col)
    
    # Determine problem type
    problem_type = infer_problem_type(y)
    print(f"Problem type detected: {problem_type}")
    
    # Split data chronologically for time series
    split_idx = int(len(df) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Get and train models
    rf, xgb = get_models(problem_type, X.shape[1])
    
    # Train XGBoost (typically better performer)
    xgb.fit(X_train, y_train)
    
    # Make predictions
    y_pred = xgb.predict(X_test)
    
    # Calculate metrics
    if problem_type == "regression":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }
        
        print(f"\nRegression Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    else:
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nClassification Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig('outputs/model_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return xgb, metrics if problem_type == "regression" else {'accuracy': accuracy}

def test_feature_importance(model, feature_names, top_n=10):
    """
    Test and visualize feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop {top_n} Most Important Features:")
        for i in range(min(top_n, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(min(top_n, len(feature_names))), 
                importances[indices][:min(top_n, len(feature_names))])
        plt.xticks(range(min(top_n, len(feature_names))), 
                  [feature_names[i] for i in indices[:min(top_n, len(feature_names))]], 
                  rotation=45, ha='right')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance_test.png', dpi=300, bbox_inches='tight')
        plt.show()

def stress_test_model(model, X, y, problem_type, noise_levels=[0.1, 0.2, 0.5]):
    """
    Test model robustness by adding noise to input features
    """
    print("\nStress Testing Model with Noise:")
    
    original_pred = model.predict(X)
    
    if problem_type == "regression":
        original_score = r2_score(y, original_pred)
        print(f"Original R² Score: {original_score:.4f}")
    else:
        from sklearn.metrics import accuracy_score
        original_score = accuracy_score(y, original_pred)
        print(f"Original Accuracy: {original_score:.4f}")
    
    for noise_level in noise_levels:
        # Add Gaussian noise to features
        X_noisy = X + np.random.normal(0, noise_level * X.std(), X.shape)
        noisy_pred = model.predict(X_noisy)
        
        if problem_type == "regression":
            noisy_score = r2_score(y, noisy_pred)
            score_drop = original_score - noisy_score
            print(f"Noise Level {noise_level:.1f}: R² = {noisy_score:.4f} (drop: {score_drop:.4f})")
        else:
            noisy_score = accuracy_score(y, noisy_pred)
            score_drop = original_score - noisy_score
            print(f"Noise Level {noise_level:.1f}: Accuracy = {noisy_score:.4f} (drop: {score_drop:.4f})")

def main():
    """
    Run comprehensive model tests
    """
    print("=== ML Model Testing Suite ===\n")
    
    # Test 1: Profit prediction (regression)
    print("Test 1: Daily Profit Prediction")
    print("-" * 40)
    
    try:
        model, metrics = load_and_test_model(
            'single_company_timeseries_data.csv', 
            'daily_profit_thousands'
        )
        
        # Load data for additional tests
        df = pd.read_csv('single_company_timeseries_data.csv')
        X, y = preprocess(df, 'daily_profit_thousands')
        
        test_feature_importance(model, X.columns.tolist())
        stress_test_model(model, X, y, 'regression')
        
    except Exception as e:
        print(f"Error in profit prediction test: {e}")
    
    print("\n" + "="*50)
    
    # Test 2: Revenue prediction (regression)
    print("Test 2: Daily Revenue Prediction")
    print("-" * 40)
    
    try:
        model, metrics = load_and_test_model(
            'single_company_timeseries_data.csv', 
            'daily_revenue_thousands'
        )
        
    except Exception as e:
        print(f"Error in revenue prediction test: {e}")
    
    print("\nTesting completed. Check outputs/ folder for visualizations.")

if __name__ == "__main__":
    main()