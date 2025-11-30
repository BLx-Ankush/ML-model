"""
Model Loading and Testing Utility

This script provides utilities to load trained models and make predictions on new data.
"""

import os
import json
import pandas as pd
import numpy as np
from train import load_trained_model, preprocess

def predict_with_saved_model(data, model_type="xgb", return_proba=False):
    """
    Make predictions using a saved model.
    
    Args:
        data: DataFrame with features (without target column)
        model_type: "xgb" or "rf" for XGBoost or Random Forest
        return_proba: If True, return prediction probabilities (classification only)
    
    Returns:
        predictions: Array of predictions
    """
    try:
        model, encoder, feature_names = load_trained_model(model_type)
        
        # Ensure data has the same features as training data
        if feature_names:
            # Add missing columns with default values
            for col in feature_names:
                if col not in data.columns:
                    if data.select_dtypes(include=['object', 'category']).columns.size > 0:
                        data[col] = "Unknown"  # Default for categorical
                    else:
                        data[col] = 0  # Default for numerical
            
            # Reorder columns to match training data
            data = data[feature_names]
        
        # Apply preprocessing (encoding for categorical variables)
        if encoder is not None:
            cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                data = data.copy()
                data[cat_cols] = encoder.transform(data[cat_cols])
        
        # Make predictions
        if return_proba and hasattr(model, 'predict_proba'):
            return model.predict_proba(data)
        else:
            return model.predict(data)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to create the models.")
        return None
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def get_model_info():
    """Get information about saved models."""
    metrics_path = os.path.join("outputs", "metrics.json")
    
    if not os.path.exists(metrics_path):
        print("No model metrics found. Please train the model first.")
        return None
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    print("=== Model Information ===")
    print(f"Problem Type: {metrics.get('problem_type', 'Unknown')}")
    
    # XGBoost CV scores
    if 'xgb_cv' in metrics:
        xgb_scores = metrics['xgb_cv']
        if metrics.get('problem_type') == 'classification':
            avg_acc = np.mean([s['accuracy'] for s in xgb_scores])
            avg_auc = np.mean([s['roc_auc'] for s in xgb_scores])
            print(f"XGBoost - Average Accuracy: {avg_acc:.3f}, Average ROC-AUC: {avg_auc:.3f}")
        else:
            avg_rmse = np.mean([s['rmse'] for s in xgb_scores])
            avg_r2 = np.mean([s['r2'] for s in xgb_scores])
            print(f"XGBoost - Average RMSE: {avg_rmse:.3f}, Average R²: {avg_r2:.3f}")
    
    # Random Forest CV scores
    if 'rf_cv' in metrics:
        rf_scores = metrics['rf_cv']
        if metrics.get('problem_type') == 'classification':
            avg_acc = np.mean([s['accuracy'] for s in rf_scores])
            avg_auc = np.mean([s['roc_auc'] for s in rf_scores])
            print(f"Random Forest - Average Accuracy: {avg_acc:.3f}, Average ROC-AUC: {avg_auc:.3f}")
        else:
            avg_rmse = np.mean([s['rmse'] for s in rf_scores])
            avg_r2 = np.mean([s['r2'] for s in rf_scores])
            print(f"Random Forest - Average RMSE: {avg_rmse:.3f}, Average R²: {avg_r2:.3f}")
    
    # Model files
    model_paths = metrics.get('model_paths', {})
    print(f"\nModel Files:")
    for model_name, path in model_paths.items():
        if path and isinstance(path, str) and path.endswith('.joblib'):
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {model_name}: {exists} {path}")
    
    return metrics

def test_model_on_sample():
    """Test the saved model on a small sample of data."""
    # Load original data to test on
    if os.path.exists("single_company_timeseries_data.csv"):
        df = pd.read_csv("single_company_timeseries_data.csv")
        
        # Get model info to determine target and problem type
        metrics_path = os.path.join("outputs", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                feature_names = metrics.get('model_paths', {}).get('feature_names', [])
                problem_type = metrics.get('problem_type', 'classification')
        else:
            print("No model metrics found.")
            return
        
        # Use first 5 rows for testing (excluding target)
        test_data = df[feature_names].head(5)
        
        print("\n=== Testing Saved Models ===")
        print("Test data shape:", test_data.shape)
        
        # Test XGBoost
        print("\nXGBoost Predictions:")
        xgb_pred = predict_with_saved_model(test_data, model_type="xgb")
        if xgb_pred is not None:
            print(xgb_pred)
            
            if problem_type == "classification":
                xgb_proba = predict_with_saved_model(test_data, model_type="xgb", return_proba=True)
                if xgb_proba is not None:
                    print("XGBoost Probabilities:")
                    print(xgb_proba)
        
        # Test Random Forest
        print("\nRandom Forest Predictions:")
        rf_pred = predict_with_saved_model(test_data, model_type="rf")
        if rf_pred is not None:
            print(rf_pred)
            
            if problem_type == "classification":
                rf_proba = predict_with_saved_model(test_data, model_type="rf", return_proba=True)
                if rf_proba is not None:
                    print("Random Forest Probabilities:")
                    print(rf_proba)
    else:
        print("No test data file found.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and test trained models")
    parser.add_argument("--info", action="store_true", help="Show model information")
    parser.add_argument("--test", action="store_true", help="Test models on sample data")
    parser.add_argument("--predict", type=str, help="CSV file to make predictions on")
    parser.add_argument("--model", type=str, choices=["xgb", "rf"], default="xgb", help="Model to use for prediction")
    parser.add_argument("--output", type=str, help="Output file for predictions")
    
    args = parser.parse_args()
    
    if args.info:
        get_model_info()
    
    if args.test:
        test_model_on_sample()
    
    if args.predict:
        if os.path.exists(args.predict):
            data = pd.read_csv(args.predict)
            predictions = predict_with_saved_model(data, model_type=args.model)
            
            if predictions is not None:
                output_df = data.copy()
                output_df['predictions'] = predictions
                
                if args.output:
                    output_df.to_csv(args.output, index=False)
                    print(f"Predictions saved to {args.output}")
                else:
                    print("Predictions:")
                    print(predictions)
        else:
            print(f"File {args.predict} not found.")
    
    if not any([args.info, args.test, args.predict]):
        # Default: show info and test
        get_model_info()
        test_model_on_sample()