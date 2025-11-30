"""
Dynamic ML Model System
Automatically adapts to different datasets and target columns
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

class DynamicMLModel:
    """
    A dynamic ML model that can adapt to different datasets and targets
    """
    
    def __init__(self, model_name=None):
        self.model_name = model_name or f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_dir = os.path.join("outputs", "models", self.model_name)
        self.config = {}
        self.encoder = None
        self.scaler = None
        self.models = {}
        self.feature_names = []
        self.target_name = ""
        self.problem_type = ""
        
    def auto_detect_config(self, df: pd.DataFrame, target: str, id_cols=None):
        """
        Automatically detect dataset configuration
        """
        id_cols = id_cols or []
        
        # Auto-detect ID columns if not specified
        if not id_cols:
            potential_id_cols = []
            for col in df.columns:
                if col.lower() in ['id', 'index', 'date', 'time', 'timestamp', 'company_name', 'name']:
                    potential_id_cols.append(col)
                elif df[col].dtype == 'object' and df[col].nunique() == len(df):
                    potential_id_cols.append(col)  # Unique string columns
            id_cols = potential_id_cols
            
        # Auto-detect target if not specified
        if target == "auto":
            # Look for common target patterns
            target_patterns = ['target', 'label', 'y', 'output', 'profit', 'revenue', 'sales', 'price', 'margin']
            found_target = None
            for pattern in target_patterns:
                matching_cols = [col for col in df.columns if pattern in col.lower() and col not in id_cols]
                if matching_cols:
                    found_target = matching_cols[0]
                    break
            
            if found_target:
                target = found_target
            else:
                # Default to last numeric column that's not an ID
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col not in id_cols]
                target = numeric_cols[-1] if numeric_cols else None
                
            if not target:
                raise ValueError("Could not auto-detect target column. Please specify --target explicitly.")
        
        # Detect problem type
        y = df[target]
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() > 20:
                problem_type = "regression"
            else:
                problem_type = "classification"
        else:
            problem_type = "classification"
        
        self.config = {
            'target': target,
            'id_cols': id_cols,
            'problem_type': problem_type,
            'dataset_shape': df.shape,
            'target_stats': {
                'unique_values': y.nunique(),
                'min': y.min() if pd.api.types.is_numeric_dtype(y) else None,
                'max': y.max() if pd.api.types.is_numeric_dtype(y) else None,
                'mean': y.mean() if pd.api.types.is_numeric_dtype(y) else None,
                'null_count': y.isnull().sum()
            }
        }
        
        print(f"✅ Auto-detected configuration:")
        print(f"   Target: {target}")
        print(f"   Problem Type: {problem_type}")
        print(f"   ID Columns: {id_cols}")
        print(f"   Dataset Shape: {df.shape}")
        
        return self.config
    
    def preprocess_data(self, df: pd.DataFrame, is_training=True):
        """
        Preprocess data with automatic feature engineering
        """
        df = df.copy()
        target = self.config['target']
        id_cols = self.config['id_cols']
        
        # Remove ID columns
        df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        # Separate features and target
        if target in df.columns:
            y = df[target]
            X = df.drop(columns=[target])
        else:
            y = None
            X = df
        
        # Handle categorical variables
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            if is_training:
                self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X[cat_cols] = self.encoder.fit_transform(X[cat_cols])
            elif self.encoder:
                X[cat_cols] = self.encoder.transform(X[cat_cols])
        
        # Handle missing values
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Feature scaling for certain algorithms (optional)
        # if is_training:
        #     self.scaler = StandardScaler()
        #     X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        # elif self.scaler:
        #     X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        
        if is_training:
            self.feature_names = X.columns.tolist()
            self.target_name = target
            self.problem_type = self.config['problem_type']
        
        return X, y
    
    def get_models(self, n_features: int):
        """
        Get appropriate models based on problem type
        """
        if self.problem_type == "classification":
            models = {
                'rf': RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
                'xgb': XGBClassifier(
                    n_estimators=600, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=42, n_jobs=-1, tree_method="hist",
                    eval_metric="logloss", missing=np.nan
                )
            }
        else:
            models = {
                'rf': RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1),
                'xgb': XGBRegressor(
                    n_estimators=800, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=42, n_jobs=-1, tree_method="hist", missing=np.nan
                )
            }
        return models
    
    def cross_validate(self, X, y, model, cv_folds=5):
        """
        Perform cross-validation
        """
        scores = []
        
        if self.problem_type == "classification":
            kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            splitter = kf.split(X, y)
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            splitter = kf.split(X)
        
        for train_idx, val_idx in splitter:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            try:
                if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
                else:
                    model.fit(X_train, y_train)
            except:
                model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            if self.problem_type == "classification":
                acc = accuracy_score(y_val, y_pred)
                try:
                    y_proba = model.predict_proba(X_val)
                    if y_proba.shape[1] == 2:
                        auc = roc_auc_score(y_val, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_val, y_proba, multi_class="ovr")
                except:
                    auc = np.nan
                f1 = f1_score(y_val, y_pred, average="weighted")
                scores.append({"accuracy": acc, "roc_auc": auc, "f1": f1})
            else:
                rmse = mean_squared_error(y_val, y_pred, squared=False)
                r2 = r2_score(y_val, y_pred)
                mae = np.mean(np.abs(y_val - y_pred))
                scores.append({"rmse": rmse, "r2": r2, "mae": mae})
        
        return scores
    
    def train(self, df: pd.DataFrame, target: str = "auto", id_cols=None, cv_folds=5):
        """
        Train the dynamic model
        """
        print(f"🚀 Training dynamic model: {self.model_name}")
        
        # Auto-detect configuration
        self.auto_detect_config(df, target, id_cols)
        
        # Preprocess data
        X, y = self.preprocess_data(df, is_training=True)
        
        # Get models
        models = self.get_models(X.shape[1])
        
        # Train and evaluate models
        results = {}
        for model_name, model in models.items():
            print(f"📊 Training {model_name.upper()}...")
            
            # Cross-validation
            cv_scores = self.cross_validate(X, y, model, cv_folds)
            
            # Final training
            model.fit(X, y)
            self.models[model_name] = model
            
            results[f"{model_name}_cv"] = cv_scores
            
            # Print results
            if self.problem_type == "classification":
                avg_acc = np.mean([s['accuracy'] for s in cv_scores])
                avg_auc = np.mean([s['roc_auc'] for s in cv_scores if not np.isnan(s['roc_auc'])])
                print(f"   ✅ {model_name.upper()} - Accuracy: {avg_acc:.3f}, ROC-AUC: {avg_auc:.3f}")
            else:
                avg_rmse = np.mean([s['rmse'] for s in cv_scores])
                avg_r2 = np.mean([s['r2'] for s in cv_scores])
                print(f"   ✅ {model_name.upper()} - RMSE: {avg_rmse:.3f}, R²: {avg_r2:.3f}")
        
        # Save model
        self.save_model(results)
        
        return results
    
    def predict(self, df: pd.DataFrame, model_type="xgb"):
        """
        Make predictions with the trained model
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Available: {list(self.models.keys())}")
        
        X, _ = self.preprocess_data(df, is_training=False)
        
        # Ensure same features as training
        if set(X.columns) != set(self.feature_names):
            # Add missing columns
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            # Reorder columns
            X = X[self.feature_names]
        
        return self.models[model_type].predict(X)
    
    def save_model(self, results):
        """
        Save the trained model and metadata
        """
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save models
        model_paths = {}
        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f"{name}_model.joblib")
            joblib.dump(model, path)
            model_paths[f"{name}_model"] = os.path.abspath(path)
        
        # Save preprocessors
        if self.encoder:
            encoder_path = os.path.join(self.model_dir, "encoder.joblib")
            joblib.dump(self.encoder, encoder_path)
            model_paths["encoder"] = os.path.abspath(encoder_path)
        
        if self.scaler:
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            model_paths["scaler"] = os.path.abspath(scaler_path)
        
        # Convert results to JSON-serializable format
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "config": convert_to_json_serializable(self.config),
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "problem_type": self.problem_type,
            "model_paths": model_paths,
            "results": convert_to_json_serializable(results),
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Also save to main outputs for compatibility
        main_metadata_path = os.path.join("outputs", "latest_model_metadata.json")
        with open(main_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to: {self.model_dir}")
        return metadata_path
    
    @classmethod
    def load_model(cls, model_name=None, model_dir=None):
        """
        Load a previously trained dynamic model
        """
        if model_dir:
            metadata_path = os.path.join(model_dir, "metadata.json")
        elif model_name:
            metadata_path = os.path.join("outputs", "models", model_name, "metadata.json")
        else:
            # Load latest model
            metadata_path = os.path.join("outputs", "latest_model_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create model instance
        model = cls(metadata["model_name"])
        model.config = metadata["config"]
        model.feature_names = metadata["feature_names"]
        model.target_name = metadata["target_name"]
        model.problem_type = metadata["problem_type"]
        
        # Load trained models
        model_paths = metadata["model_paths"]
        for key, path in model_paths.items():
            if key.endswith("_model"):
                model_type = key.replace("_model", "")
                model.models[model_type] = joblib.load(path)
            elif key == "encoder":
                model.encoder = joblib.load(path)
            elif key == "scaler":
                model.scaler = joblib.load(path)
        
        print(f"📂 Loaded model: {model.model_name}")
        print(f"   Target: {model.target_name}")
        print(f"   Problem Type: {model.problem_type}")
        print(f"   Features: {len(model.feature_names)}")
        
        return model

def main():
    parser = argparse.ArgumentParser(description="Dynamic ML Model System")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--target", default="auto", help="Target column name or 'auto' for auto-detection")
    parser.add_argument("--id-cols", help="Comma-separated ID columns to exclude")
    parser.add_argument("--model-name", help="Custom model name")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--predict", help="Path to new data for prediction")
    parser.add_argument("--load-model", help="Load existing model by name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("outputs/models", exist_ok=True)
    
    if args.predict and args.load_model:
        # Load model and make predictions
        model = DynamicMLModel.load_model(args.load_model)
        pred_df = pd.read_csv(args.predict)
        predictions = model.predict(pred_df)
        
        # Save predictions
        pred_df['predictions'] = predictions
        output_path = args.predict.replace('.csv', '_predictions.csv')
        pred_df.to_csv(output_path, index=False)
        print(f"💾 Predictions saved to: {output_path}")
        
    else:
        # Train new model
        df = pd.read_csv(args.data)
        id_cols = [col.strip() for col in args.id_cols.split(",")] if args.id_cols else None
        
        model = DynamicMLModel(args.model_name)
        results = model.train(df, args.target, id_cols, args.cv)
        
        print("\n🎉 Training completed!")
        print(f"Model saved as: {model.model_name}")

if __name__ == "__main__":
    main()