# 🚀 ML Model Training & Testing - Complete Cheat Sheet

## 📋 1. EXPLORE NEW DATASET
```powershell
# See what's in your dataset and get target suggestions
python model_assistant.py explore YOUR_DATASET.csv
```

## 🎯 2. TRAIN MODEL

### Auto-detect target (easiest):
```powershell
python dynamic_model.py --data YOUR_DATASET.csv --target auto --model-name "YOUR_MODEL_NAME"
```

### Specify target column:
```powershell
python dynamic_model.py --data YOUR_DATASET.csv --target COLUMN_NAME --model-name "YOUR_MODEL_NAME"
```

### With ID columns to exclude:
```powershell
python dynamic_model.py --data YOUR_DATASET.csv --target COLUMN_NAME --id-cols "date,id,name" --model-name "YOUR_MODEL_NAME"
```

## 📊 3. CHECK TRAINING RESULTS

### List all models:
```powershell
python model_assistant.py list
```

### Detailed performance analysis:
```powershell
python -c "
import json
with open('outputs/latest_model_metadata.json', 'r') as f:
    metadata = json.load(f)
    
print('Model:', metadata['model_name'])
print('Target:', metadata['target_name'])
results = metadata['results']['xgb_cv']
avg_r2 = sum(s['r2'] for s in results) / len(results)
avg_rmse = sum(s['rmse'] for s in results) / len(results)
print(f'Performance: R²={avg_r2:.3f}, RMSE={avg_rmse:.3f}')
"
```

## 🧪 4. TEST MODEL

### Test on same dataset (validation):
```powershell
python model_assistant.py predict YOUR_DATASET.csv YOUR_MODEL_NAME combined
```

### Test on different dataset:
```powershell
python model_assistant.py predict NEW_DATASET.csv YOUR_MODEL_NAME combined
```

### Different output formats:
```powershell
# Combined (original + predictions)
python model_assistant.py predict DATA.csv MODEL_NAME combined

# Predictions only
python model_assistant.py predict DATA.csv MODEL_NAME predictions_only

# Separate files
python model_assistant.py predict DATA.csv MODEL_NAME separate
```

## 📈 5. CALCULATE ACCURACY
```powershell
python -c "
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv('YOUR_PREDICTIONS_FILE.csv')
actual = df['ACTUAL_COLUMN']
predicted = df['PREDICTED_COLUMN']

r2 = r2_score(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = np.mean(np.abs(actual - predicted))

print(f'R² Score: {r2:.4f}')
print(f'RMSE: {rmse:.3f}')
print(f'MAE: {mae:.3f}')
"
```

## 📊 6. GENERATE GRAPHS
```powershell
python -c "
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv('YOUR_PREDICTIONS_FILE.csv')
actual = df['ACTUAL_COLUMN']
predicted = df['PREDICTED_COLUMN']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(actual, predicted, alpha=0.6)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'R² = {r2_score(actual, predicted):.3f}')

plt.subplot(1, 2, 2)
residuals = actual - predicted
plt.scatter(predicted, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

plt.tight_layout()
plt.savefig('model_results.png', dpi=300)
plt.show()
print('Graph saved: model_results.png')
"
```

## 🔄 7. COMPLETE WORKFLOWS

### New Dataset Workflow:
```powershell
# 1. Explore
python model_assistant.py explore new_data.csv

# 2. Train
python dynamic_model.py --data new_data.csv --target TARGET_COLUMN --model-name "new_model"

# 3. Test
python model_assistant.py predict new_data.csv new_model combined

# 4. Check accuracy
python -c "
import pandas as pd; from sklearn.metrics import r2_score
df = pd.read_csv('new_data_predictions.csv')
print('R²:', r2_score(df['TARGET_COLUMN'], df['predicted_TARGET_COLUMN']))
"
```

### Quick Test on Multiple Datasets:
```powershell
python model_assistant.py predict dataset1.csv YOUR_MODEL combined
python model_assistant.py predict dataset2.csv YOUR_MODEL combined  
python model_assistant.py predict dataset3.csv YOUR_MODEL combined
```

## 📝 8. REPLACE THESE PLACEHOLDERS

- `YOUR_DATASET.csv` → Your data file name
- `YOUR_MODEL_NAME` → Choose a model name (e.g., "sales_model", "profit_model")
- `COLUMN_NAME` → Your target column name
- `TARGET_COLUMN` → Actual target column name
- `PREDICTED_COLUMN` → Usually `predicted_COLUMN_NAME`
- `YOUR_PREDICTIONS_FILE.csv` → File with predictions (usually `DATASET_predictions.csv`)

## ⚡ 9. ONE-LINERS FOR SPEED

### Quick train + test:
```powershell
python dynamic_model.py --data data.csv --target auto --model-name "quick_model" && python model_assistant.py predict data.csv quick_model combined
```

### Quick accuracy check:
```powershell
python -c "import pandas as pd; from sklearn.metrics import r2_score; df=pd.read_csv('PREDICTIONS.csv'); print('Accuracy:', r2_score(df['actual'], df['predicted']))"
```

### List all models:
```powershell
python model_assistant.py list
```

### Open graphs:
```powershell
start *.png
```

## 🎯 10. REAL EXAMPLES (Copy-Paste Ready)

### Sales prediction:
```powershell
python dynamic_model.py --data sales_data.csv --target monthly_sales --model-name "sales_predictor"
python model_assistant.py predict sales_data.csv sales_predictor combined
```

### Customer analysis:
```powershell
python dynamic_model.py --data customer_data.csv --target satisfaction_score --id-cols "customer_id,date" --model-name "satisfaction_model"
python model_assistant.py predict customer_data.csv satisfaction_model predictions_only
```

### Auto-everything:
```powershell
python dynamic_model.py --data any_data.csv --target auto --model-name "auto_model"
python model_assistant.py predict any_data.csv auto_model combined
```

## 📊 11. COMPREHENSIVE MODEL ANALYSIS

### Full performance analysis:
```powershell
python -c "
import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os

print('=== MODEL PERFORMANCE ANALYSIS ===')

# Training performance
if os.path.exists('outputs/latest_model_metadata.json'):
    with open('outputs/latest_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print('Model:', metadata['model_name'])
    print('Target:', metadata['target_name'])
    
    results = metadata['results']
    if 'xgb_cv' in results:
        xgb_scores = results['xgb_cv']
        avg_r2 = sum(s['r2'] for s in xgb_scores) / len(xgb_scores)
        avg_rmse = sum(s['rmse'] for s in xgb_scores) / len(xgb_scores)
        print(f'XGBoost - R²: {avg_r2:.4f}, RMSE: {avg_rmse:.3f}')

# Test performance
test_files = ['retailhub_company_data_predictions.csv', 'medicare_solutions_data_predictions.csv']
for test_file in test_files:
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        if 'profit_margin_percent' in df.columns and 'predicted_profit_margin_percent' in df.columns:
            actual = df['profit_margin_percent']
            predicted = df['predicted_profit_margin_percent']
            r2 = r2_score(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            print(f'{test_file}: R²={r2:.4f}, RMSE={rmse:.3f}')

print('Analysis complete!')
"
```

## 🎖️ 12. PERFORMANCE BENCHMARKS

### R² Score (Coefficient of Determination):
- **0.98+**: Outstanding ⭐⭐⭐⭐⭐
- **0.95+**: Excellent ⭐⭐⭐⭐
- **0.90+**: Very Good ⭐⭐⭐
- **0.80+**: Good ⭐⭐
- **<0.80**: Needs Improvement ⭐

### RMSE (Root Mean Square Error):
- Lower is better
- Should be small relative to your target range
- Compare across different models

## 🚀 Quick Start Template

Replace the placeholders and run:

```powershell
# STEP 1: Explore your data
python model_assistant.py explore YOUR_DATA.csv

# STEP 2: Train model  
python dynamic_model.py --data YOUR_DATA.csv --target YOUR_TARGET --model-name "YOUR_MODEL"

# STEP 3: Test model
python model_assistant.py predict YOUR_DATA.csv YOUR_MODEL combined

# STEP 4: Check accuracy
python -c "
import pandas as pd; from sklearn.metrics import r2_score
df = pd.read_csv('YOUR_DATA_predictions.csv') 
print('R²:', r2_score(df['YOUR_TARGET'], df['predicted_YOUR_TARGET']))
"
```

---
**💾 Save this file and reference it anytime you need to work with ML models!**

**📧 Generated on:** November 30, 2025  
**🎯 For:** Dynamic ML Model Training & Testing  
**📂 Location:** D:\ML Model\ML_Model_Cheat_Sheet.md