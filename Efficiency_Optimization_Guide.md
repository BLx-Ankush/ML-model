# ML Model Efficiency Optimization Guide

## Quick Commands

### 1. Automatic Optimization (Recommended)
```bash
# Quick optimize with defaults
python efficiency_optimizer.py data.csv target_column

# Custom optimization
python efficiency_optimizer.py retailhub_company_data.csv profit_margin_percent my_model
```

### 2. Manual Feature Selection
```python
from efficiency_optimizer import ModelOptimizer

optimizer = ModelOptimizer("data.csv", "target_column")
X, y = optimizer.load_and_prepare_data()

# Try different methods
optimizer.optimize_features('selectk', 15)     # Statistical selection
optimizer.optimize_features('rfe', 15)         # Recursive elimination
optimizer.optimize_features('importance', 15)  # Tree importance
```

### 3. Model Configuration Tuning
```python
# XGBoost configurations
optimizer.optimize_hyperparameters('xgb')

# Available configs: 'fast', 'balanced', 'accurate'
```

## Efficiency Techniques

### Feature Selection Methods

#### 1. SelectKBest (Statistical)
- **Best for**: Linear relationships, continuous targets
- **Speed**: Fast ⚡
- **Accuracy**: Good for regression

#### 2. Recursive Feature Elimination (RFE)
- **Best for**: Complex relationships, any model type
- **Speed**: Moderate 🔄
- **Accuracy**: High accuracy

#### 3. Tree-based Importance
- **Best for**: Non-linear relationships, mixed data types
- **Speed**: Fast ⚡
- **Accuracy**: Good for tree models

### Model Efficiency Configurations

#### Fast Configuration
```python
# Quick predictions, lower accuracy
XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.15,
    reg_lambda=2.0
)
```

#### Balanced Configuration (Recommended)
```python
# Good balance of speed and accuracy
XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    reg_lambda=2.0
)
```

#### Accurate Configuration
```python
# Best accuracy, slower training
XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    reg_lambda=1.5
)
```

## Performance Benchmarks

### Efficiency Grades

- **EXCELLENT**: >40% feature reduction, >95% performance retention
- **GOOD**: >25% feature reduction, >90% performance retention
- **MODERATE**: >15% feature reduction, decent performance
- **MINIMAL**: <15% feature reduction

### Target Metrics

- **Training Speed**: <30 seconds for 10K samples
- **Prediction Speed**: <100ms for 1K predictions
- **Memory Usage**: <500MB for typical datasets
- **Accuracy Retention**: >90% of original R²

## Advanced Optimization

### 1. Cross-Validation Optimization
```python
# Find optimal number of features
for k in [10, 15, 20, 25]:
    optimizer.optimize_features('selectk', k)
    score = cross_val_score(model, X_selected, y, cv=5)
    print(f"k={k}: CV Score = {score.mean():.4f}")
```

### 2. Custom Feature Engineering
```python
# Add polynomial features for efficiency
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X_selected)
```

### 3. Pipeline Optimization
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=15)),
    ('model', XGBRegressor(n_estimators=100))
])
```

## Monitoring Efficiency

### 1. Training Time Tracking
```python
import time

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"Training time: {training_time:.2f} seconds")
```

### 2. Prediction Speed Testing
```python
start_time = time.time()
predictions = model.predict(X_test)
prediction_time = time.time() - start_time

print(f"Prediction time: {prediction_time:.4f} seconds for {len(X_test)} samples")
print(f"Speed: {len(X_test)/prediction_time:.0f} predictions/second")
```

### 3. Memory Usage Monitoring
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_usage:.1f} MB")
```

## Production Optimization Checklist

### Before Deployment
- [ ] Feature selection completed (>25% reduction target)
- [ ] Model configuration optimized
- [ ] Cross-validation scores acceptable
- [ ] Training time < 5 minutes
- [ ] Prediction speed > 100 pred/sec
- [ ] Memory usage < 1GB

### Performance Thresholds
```python
# Set performance alerts
performance_thresholds = {
    'min_r2': 0.85,           # Minimum R² score
    'max_training_time': 300,  # Max 5 minutes
    'min_pred_speed': 100,     # Min 100 pred/sec
    'max_memory_mb': 1000      # Max 1GB memory
}
```

### Monitoring in Production
```python
# Performance monitoring function
def monitor_model_performance(model, X_test, y_test):
    start_time = time.time()
    predictions = model.predict(X_test)
    pred_time = time.time() - start_time
    
    r2 = r2_score(y_test, predictions)
    speed = len(X_test) / pred_time
    
    alerts = []
    if r2 < 0.85:
        alerts.append(f"Low R²: {r2:.3f}")
    if speed < 100:
        alerts.append(f"Slow predictions: {speed:.0f}/sec")
    
    return {'r2': r2, 'speed': speed, 'alerts': alerts}
```

## Common Efficiency Issues & Solutions

### Issue: Slow Training
**Solutions:**
- Reduce n_estimators (100-200 range)
- Use subsample=0.8
- Reduce max_depth (3-5 range)
- Enable early stopping

### Issue: Large Model Size
**Solutions:**
- Feature selection (SelectKBest)
- Model compression techniques
- Prune low-importance features
- Use regularization (L1/L2)

### Issue: Slow Predictions
**Solutions:**
- Reduce model complexity
- Optimize feature pipeline
- Use faster algorithms (XGBoost over RF)
- Batch predictions

### Issue: High Memory Usage
**Solutions:**
- Process data in chunks
- Use sparse matrices
- Feature selection
- Model compression

## Example Optimization Workflow

```python
# Complete optimization workflow
def optimize_complete_workflow(data_path, target_col):
    # 1. Initialize optimizer
    optimizer = ModelOptimizer(data_path, target_col)
    
    # 2. Load and prepare data
    X, y = optimizer.load_and_prepare_data()
    
    # 3. Find optimal feature count
    best_k = None
    best_score = 0
    
    for k in [10, 15, 20, 25]:
        optimizer.optimize_features('selectk', k)
        model = XGBRegressor(n_estimators=100, random_state=42)
        score = cross_val_score(model, optimizer.X_selected, y, cv=5).mean()
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # 4. Optimize with best k
    optimizer.optimize_features('selectk', best_k)
    
    # 5. Optimize model configuration
    optimizer.optimize_hyperparameters('xgb')
    
    # 6. Evaluate and save
    results = optimizer.evaluate_efficiency()
    optimizer.save_optimized_model()
    
    return optimizer, results

# Run complete optimization
optimizer, results = optimize_complete_workflow('data.csv', 'target')
```