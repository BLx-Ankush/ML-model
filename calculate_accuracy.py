#!/usr/bin/env python3
"""
Calculate accuracy metrics for ML model predictions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_regression_accuracy(predictions_file, actual_col='profit_margin_percent', pred_col='predictions'):
    """
    Calculate comprehensive accuracy metrics for regression model
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Extract actual and predicted values
    actual = df[actual_col]
    predicted = df[pred_col]
    
    print("=== REGRESSION MODEL ACCURACY METRICS ===\n")
    
    # 1. Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, predicted)
    print(f"📊 Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   → Average error: ±{mae:.2f} percentage points")
    
    # 2. Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"📊 Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"   → Typical error: ±{rmse:.2f} percentage points")
    
    # 3. R² Score (Coefficient of Determination)
    r2 = r2_score(actual, predicted)
    print(f"📊 R² Score: {r2:.4f}")
    print(f"   → Model explains {r2*100:.1f}% of variance")
    
    # 4. Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"📊 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"   → Average relative error: {mape:.1f}%")
    
    # 5. Custom accuracy metrics for business context
    print(f"\n=== BUSINESS-FRIENDLY METRICS ===")
    
    # Percentage of predictions within different error ranges
    errors = np.abs(actual - predicted)
    
    within_1pct = (errors <= 1.0).sum() / len(errors) * 100
    within_2pct = (errors <= 2.0).sum() / len(errors) * 100
    within_5pct = (errors <= 5.0).sum() / len(errors) * 100
    
    print(f"🎯 Predictions within ±1%:  {within_1pct:.1f}% of cases")
    print(f"🎯 Predictions within ±2%:  {within_2pct:.1f}% of cases")
    print(f"🎯 Predictions within ±5%:  {within_5pct:.1f}% of cases")
    
    # Model performance interpretation
    print(f"\n=== MODEL PERFORMANCE RATING ===")
    if r2 >= 0.95:
        rating = "EXCELLENT"
        emoji = "🟢"
    elif r2 >= 0.90:
        rating = "VERY GOOD"  
        emoji = "🔵"
    elif r2 >= 0.80:
        rating = "GOOD"
        emoji = "🟡"
    elif r2 >= 0.70:
        rating = "FAIR"
        emoji = "🟠"
    else:
        rating = "POOR"
        emoji = "🔴"
    
    print(f"{emoji} Overall Model Rating: {rating}")
    
    # Statistical summary
    print(f"\n=== STATISTICAL SUMMARY ===")
    print(f"📈 Actual values range: {actual.min():.2f}% to {actual.max():.2f}%")
    print(f"📈 Predicted values range: {predicted.min():.2f}% to {predicted.max():.2f}%")
    print(f"📈 Actual mean: {actual.mean():.2f}%")
    print(f"📈 Predicted mean: {predicted.mean():.2f}%")
    print(f"📈 Prediction bias: {(predicted.mean() - actual.mean()):.2f}% {'(overestimate)' if predicted.mean() > actual.mean() else '(underestimate)'}")
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'within_1pct': within_1pct,
        'within_2pct': within_2pct,
        'within_5pct': within_5pct,
        'rating': rating
    }

def plot_accuracy_visualization(predictions_file, actual_col='profit_margin_percent', pred_col='predictions'):
    """
    Create visualizations to assess model accuracy
    """
    df = pd.read_csv(predictions_file)
    actual = df[actual_col]
    predicted = df[pred_col]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted Scatter Plot
    axes[0,0].scatter(actual, predicted, alpha=0.6, color='blue')
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0,0].set_xlabel('Actual Profit Margin %')
    axes[0,0].set_ylabel('Predicted Profit Margin %')
    axes[0,0].set_title('Predictions vs Actual Values')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Residuals Plot
    residuals = actual - predicted
    axes[0,1].scatter(predicted, residuals, alpha=0.6, color='green')
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Predicted Profit Margin %')
    axes[0,1].set_ylabel('Residuals (Actual - Predicted)')
    axes[0,1].set_title('Residual Plot')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Error Distribution
    errors = np.abs(residuals)
    axes[1,0].hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].axvline(errors.mean(), color='red', linestyle='--', 
                     label=f'Mean Error: {errors.mean():.2f}')
    axes[1,0].set_xlabel('Absolute Error')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Error Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Accuracy Over Time (if date column exists)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['error'] = np.abs(actual - predicted)
        
        # Rolling average of errors
        df = df.sort_values('date')
        df['rolling_error'] = df['error'].rolling(window=10, center=True).mean()
        
        axes[1,1].plot(df['date'], df['rolling_error'], color='purple', linewidth=2)
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Rolling Average Error (10-day)')
        axes[1,1].set_title('Model Accuracy Over Time')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
    else:
        # Alternative: Show prediction confidence intervals
        sorted_pred = np.sort(predicted)
        sorted_actual = actual[predicted.argsort()]
        axes[1,1].plot(sorted_pred, sorted_actual, 'b-', alpha=0.6)
        axes[1,1].plot(sorted_pred, sorted_pred, 'r--', label='Perfect Prediction')
        axes[1,1].set_xlabel('Predicted (Sorted)')
        axes[1,1].set_ylabel('Actual (Corresponding)')
        axes[1,1].set_title('Sorted Predictions vs Actual')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Accuracy visualization saved as 'outputs/accuracy_analysis.png'")

def compare_with_baseline(predictions_file, actual_col='profit_margin_percent', pred_col='predictions'):
    """
    Compare model accuracy with simple baseline predictions
    """
    df = pd.read_csv(predictions_file)
    actual = df[actual_col]
    predicted = df[pred_col]
    
    print("\n=== BASELINE COMPARISON ===")
    
    # Baseline 1: Always predict the mean
    mean_baseline = np.full_like(actual, actual.mean())
    mean_r2 = r2_score(actual, mean_baseline)
    
    # Baseline 2: Always predict previous value (naive forecast)
    prev_baseline = np.roll(actual, 1)
    prev_baseline[0] = actual[0]  # Handle first value
    prev_r2 = r2_score(actual, prev_baseline)
    
    model_r2 = r2_score(actual, predicted)
    
    print(f"🔄 Mean Baseline R²: {mean_r2:.4f}")
    print(f"🔄 Previous Value Baseline R²: {prev_r2:.4f}")
    print(f"🤖 Your Model R²: {model_r2:.4f}")
    
    print(f"\n📈 Model vs Mean Baseline: {((model_r2 - mean_r2) / abs(mean_r2) * 100):+.1f}% improvement")
    print(f"📈 Model vs Previous Value: {((model_r2 - prev_r2) / abs(prev_r2) * 100):+.1f}% improvement")

if __name__ == "__main__":
    print("🚀 Calculating Model Accuracy...\n")
    
    # Calculate all accuracy metrics
    metrics = calculate_regression_accuracy('Book1.csv')
    
    # Create accuracy visualizations
    plot_accuracy_visualization('Book1.csv')
    
    # Compare with baselines
    compare_with_baseline('Book1.csv')
    
    print(f"\n✅ Accuracy analysis complete!")
    print(f"📁 Check 'outputs/accuracy_analysis.png' for detailed visualizations")