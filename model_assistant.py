"""
Easy Dynamic Model Interface
Simple commands to work with dynamic models on any dataset
"""

import os
import pandas as pd
from dynamic_model import DynamicMLModel

def quick_train(data_path, target_column="auto", model_name=None):
    """
    Quick training on any dataset
    """
    print(f"🚀 Quick training on: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"📊 Dataset loaded: {df.shape}")
    
    # Create and train model
    model = DynamicMLModel(model_name)
    results = model.train(df, target_column)
    
    return model, results

def quick_predict(data_path, model_name=None, output_path=None, output_mode='combined'):
    """
    Quick prediction on new data
    
    Args:
        data_path: Path to input data
        model_name: Name of trained model
        output_path: Custom output path
        output_mode: 'combined' (original + predictions), 'predictions_only', 'separate'
    """
    print(f"🔮 Making predictions on: {data_path}")
    
    # Load model
    model = DynamicMLModel.load_model(model_name)
    
    # Load new data
    new_data = pd.read_csv(data_path)
    print(f"📊 New data loaded: {new_data.shape}")
    
    # Make predictions
    predictions = model.predict(new_data)
    
    # Prepare output based on mode
    base_name = os.path.splitext(data_path)[0]
    
    if output_mode == 'combined':
        # Add predictions to original data
        result_df = new_data.copy()
        result_df[f'predicted_{model.target_name}'] = predictions
        
        output_file = output_path or f"{base_name}_predictions.csv"
        result_df.to_csv(output_file, index=False)
        print(f"💾 Combined data + predictions saved to: {output_file}")
        return result_df, predictions
        
    elif output_mode == 'predictions_only':
        # Save only predictions with minimal identifiers
        pred_df = pd.DataFrame({'predictions': predictions})
        
        # Add ID columns if they exist
        id_cols = ['date', 'id', 'index', 'company_name', 'name']
        for col in id_cols:
            if col in new_data.columns:
                pred_df[col] = new_data[col]
        
        # Reorder columns (IDs first, then predictions)
        id_columns = [col for col in pred_df.columns if col != 'predictions']
        pred_df = pred_df[id_columns + ['predictions']]
        
        output_file = output_path or f"{base_name}_predictions_only.csv"
        pred_df.to_csv(output_file, index=False)
        print(f"💾 Predictions only saved to: {output_file}")
        return pred_df, predictions
        
    elif output_mode == 'separate':
        # Save original data and predictions in separate files
        pred_df = pd.DataFrame({'predictions': predictions})
        
        # Add ID columns for matching
        id_cols = ['date', 'id', 'index', 'company_name', 'name']
        for col in id_cols:
            if col in new_data.columns:
                pred_df[col] = new_data[col]
        
        # Save original data (unchanged)
        original_output = output_path or f"{base_name}_original.csv"
        new_data.to_csv(original_output, index=False)
        
        # Save predictions
        pred_output = output_path.replace('.csv', '_predictions.csv') if output_path else f"{base_name}_predictions.csv"
        pred_df.to_csv(pred_output, index=False)
        
        print(f"💾 Original data saved to: {original_output}")
        print(f"💾 Predictions saved to: {pred_output}")
        return new_data, predictions
    
    else:
        raise ValueError("output_mode must be 'combined', 'predictions_only', or 'separate'")

def list_models():
    """
    List all available trained models
    """
    models_dir = "outputs/models"
    if not os.path.exists(models_dir):
        print("No models found.")
        return []
    
    models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path):
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                models.append(item)
    
    print("📋 Available Models:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    return models

def compare_targets(data_path, targets_to_try=None):
    """
    Compare different target columns on the same dataset
    """
    print(f"🔬 Comparing targets on: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if not targets_to_try:
        # Auto-detect potential targets (numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        targets_to_try = numeric_cols[-3:]  # Try last 3 numeric columns
    
    results = {}
    
    for target in targets_to_try:
        if target in df.columns:
            print(f"\n🎯 Testing target: {target}")
            try:
                model = DynamicMLModel(f"test_{target}")
                result = model.train(df, target, cv_folds=3)  # Quick 3-fold CV
                results[target] = {
                    'model': model,
                    'results': result,
                    'problem_type': model.problem_type
                }
            except Exception as e:
                print(f"   ❌ Failed: {e}")
    
    # Summary
    print(f"\n📈 Target Comparison Summary:")
    print("=" * 50)
    for target, data in results.items():
        model = data['model']
        if model.problem_type == 'regression':
            cv_scores = data['results']['xgb_cv']
            avg_r2 = sum(s['r2'] for s in cv_scores) / len(cv_scores)
            avg_rmse = sum(s['rmse'] for s in cv_scores) / len(cv_scores)
            print(f"{target:25} | R²: {avg_r2:.3f} | RMSE: {avg_rmse:.3f}")
        else:
            cv_scores = data['results']['xgb_cv']
            avg_acc = sum(s['accuracy'] for s in cv_scores) / len(cv_scores)
            print(f"{target:25} | Accuracy: {avg_acc:.3f}")
    
    return results

def auto_explore_dataset(data_path):
    """
    Automatically explore and suggest best targets for a dataset
    """
    print(f"🔍 Auto-exploring dataset: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"📊 Dataset info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Analyze columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"   Numeric: {len(numeric_cols)}")
    print(f"   Categorical: {len(categorical_cols)}")
    
    # Suggest potential targets
    target_suggestions = []
    
    # Look for common target patterns
    target_patterns = ['profit', 'revenue', 'sales', 'price', 'target', 'label', 'output', 'y']
    for pattern in target_patterns:
        matches = [col for col in df.columns if pattern in col.lower()]
        target_suggestions.extend(matches)
    
    # Add high-variance numeric columns
    for col in numeric_cols:
        if df[col].std() / (df[col].mean() + 1e-8) > 0.1:  # High coefficient of variation
            target_suggestions.append(col)
    
    # Remove duplicates and limit
    target_suggestions = list(set(target_suggestions))[:5]
    
    print(f"\n🎯 Suggested targets to try:")
    for i, target in enumerate(target_suggestions, 1):
        print(f"   {i}. {target}")
    
    # Ask user which target to use
    print(f"\n💡 You can now train models with:")
    print(f"   python dynamic_model.py --data {data_path} --target <target_name>")
    print(f"   Or use auto-detection: --target auto")
    
    return target_suggestions

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("🤖 Dynamic Model Assistant")
        print("=" * 30)
        print("Usage:")
        print("  python model_assistant.py train <data.csv> [target_column] [model_name]")
        print("  python model_assistant.py predict <data.csv> [model_name] [output_mode]")
        print("     Output modes: 'combined' (default), 'predictions_only', 'separate'")
        print("  python model_assistant.py list")
        print("  python model_assistant.py explore <data.csv>")
        print("  python model_assistant.py compare <data.csv>")
        print("")
        print("Prediction Output Options:")
        print("  combined: Original data + predictions in one file")
        print("  predictions_only: Just predictions with ID columns")
        print("  separate: Original and predictions in separate files")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "train":
        data_path = sys.argv[2]
        target = sys.argv[3] if len(sys.argv) > 3 else "auto"
        model_name = sys.argv[4] if len(sys.argv) > 4 else None
        quick_train(data_path, target, model_name)
        
    elif command == "predict":
        data_path = sys.argv[2]
        model_name = sys.argv[3] if len(sys.argv) > 3 else None
        output_mode = sys.argv[4] if len(sys.argv) > 4 else 'combined'
        quick_predict(data_path, model_name, output_mode=output_mode)
        
    elif command == "list":
        list_models()
        
    elif command == "explore":
        data_path = sys.argv[2]
        auto_explore_dataset(data_path)
        
    elif command == "compare":
        data_path = sys.argv[2]
        compare_targets(data_path)
        
    else:
        print(f"Unknown command: {command}")