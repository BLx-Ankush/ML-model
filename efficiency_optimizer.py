"""
ML Model Efficiency Optimizer
Comprehensive toolkit for improving model performance and efficiency
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import json
import os
from datetime import datetime

class ModelOptimizer:
    """
    Comprehensive model efficiency optimization toolkit
    """
    
    def __init__(self, data_path, target_column, model_name="optimized_model"):
        self.data_path = data_path
        self.target_column = target_column
        self.model_name = model_name
        self.results = {}
        
    def load_and_prepare_data(self, exclude_cols=None):
        """Load and prepare data for optimization"""
        exclude_cols = exclude_cols or ['date', 'id', 'index', 'company_name', 'name']
        
        df = pd.read_csv(self.data_path)
        
        # Remove ID columns
        for col in exclude_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        self.original_features = X.columns.tolist()
        self.X = X
        self.y = y
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def optimize_features(self, method='selectk', k=15):
        """
        Optimize feature selection
        
        Methods:
        - 'selectk': SelectKBest using statistical tests
        - 'rfe': Recursive Feature Elimination
        - 'importance': Tree-based feature importance
        """
        print(f"\nOptimizing features using {method}...")
        
        if method == 'selectk':
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(self.X, self.y)
            selected_features = self.X.columns[selector.get_support()]
            
        elif method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(self.X, self.y)
            selected_features = self.X.columns[selector.get_support()]
            
        elif method == 'importance':
            # Use feature importance from Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(self.X, self.y)
            
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(k)['feature'].tolist()
            X_selected = self.X[selected_features]
            selector = None
        
        self.selector = selector
        self.selected_features = selected_features
        self.X_selected = X_selected
        
        feature_reduction = (len(self.original_features) - len(selected_features)) / len(self.original_features) * 100
        
        print(f"Features reduced from {len(self.original_features)} to {len(selected_features)}")
        print(f"Feature reduction: {feature_reduction:.1f}%")
        print(f"Top 5 selected features: {list(selected_features[:5])}")
        
        return X_selected, selected_features
    
    def optimize_hyperparameters(self, model_type='xgb'):
        """
        Optimize model hyperparameters for efficiency
        """
        print(f"\nOptimizing {model_type} hyperparameters...")
        
        if model_type == 'xgb':
            # Efficient XGBoost configuration
            models = {
                'fast': XGBRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.15,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=2.0, reg_alpha=0.5,
                    random_state=42, n_jobs=-1
                ),
                'balanced': XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=2.0, reg_alpha=0.5,
                    random_state=42, n_jobs=-1
                ),
                'accurate': XGBRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.5, reg_alpha=0.3,
                    random_state=42, n_jobs=-1
                )
            }
        else:
            # Efficient Random Forest configuration
            models = {
                'fast': RandomForestRegressor(
                    n_estimators=50, max_depth=10,
                    random_state=42, n_jobs=-1
                ),
                'balanced': RandomForestRegressor(
                    n_estimators=100, max_depth=15,
                    random_state=42, n_jobs=-1
                ),
                'accurate': RandomForestRegressor(
                    n_estimators=200, max_depth=None,
                    random_state=42, n_jobs=-1
                )
            }
        
        # Test different configurations
        best_model = None
        best_score = -np.inf
        best_config = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_selected, self.y, test_size=0.2, random_state=42
        )
        
        for config_name, model in models.items():
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"  {config_name}: R² = {r2:.4f}, RMSE = {rmse:.3f}")
            
            # Score based on R² (you can adjust this)
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_config = config_name
        
        self.optimized_model = best_model
        self.best_config = best_config
        
        print(f"Best configuration: {best_config} (R² = {best_score:.4f})")
        
        return best_model, best_config
    
    def evaluate_efficiency(self, original_performance=None):
        """
        Evaluate efficiency improvements
        """
        print(f"\nEvaluating efficiency improvements...")
        
        # Test optimized model
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_selected, self.y, test_size=0.2, random_state=42
        )
        
        y_pred = self.optimized_model.predict(X_test)
        optimized_r2 = r2_score(y_test, y_pred)
        optimized_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Calculate efficiency metrics
        feature_reduction = (len(self.original_features) - len(self.selected_features)) / len(self.original_features) * 100
        
        efficiency_report = {
            'optimized_r2': optimized_r2,
            'optimized_rmse': optimized_rmse,
            'feature_reduction_pct': feature_reduction,
            'selected_features': len(self.selected_features),
            'original_features': len(self.original_features),
            'model_config': self.best_config
        }
        
        if original_performance:
            performance_retention = (optimized_r2 / original_performance) * 100
            efficiency_report['performance_retention_pct'] = performance_retention
            
            print(f"Original R²: {original_performance:.4f}")
            print(f"Optimized R²: {optimized_r2:.4f}")
            print(f"Performance retention: {performance_retention:.1f}%")
        else:
            print(f"Optimized R²: {optimized_r2:.4f}")
        
        print(f"Feature reduction: {feature_reduction:.1f}%")
        print(f"Model configuration: {self.best_config}")
        
        # Efficiency grade
        if feature_reduction > 40 and (not original_performance or (optimized_r2 / original_performance) > 0.95):
            grade = "EXCELLENT"
        elif feature_reduction > 25 and (not original_performance or (optimized_r2 / original_performance) > 0.90):
            grade = "GOOD"
        elif feature_reduction > 15:
            grade = "MODERATE"
        else:
            grade = "MINIMAL"
        
        efficiency_report['efficiency_grade'] = grade
        print(f"Efficiency Grade: {grade}")
        
        self.results = efficiency_report
        return efficiency_report
    
    def save_optimized_model(self):
        """
        Save the optimized model and components
        """
        print(f"\nSaving optimized model...")
        
        output_dir = f"outputs/optimized_models/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(self.optimized_model, model_path)
        
        # Save feature selector
        if self.selector:
            selector_path = os.path.join(output_dir, "selector.joblib")
            joblib.dump(self.selector, selector_path)
        
        # Save selected features list
        features_path = os.path.join(output_dir, "selected_features.joblib")
        joblib.dump(list(self.selected_features), features_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat(),
            'original_features': self.original_features,
            'selected_features': list(self.selected_features),
            'target_column': self.target_column,
            'results': self.results,
            'model_config': self.best_config
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Optimized model saved to: {output_dir}")
        print(f"   - Model: {model_path}")
        print(f"   - Metadata: {metadata_path}")
        
        return output_dir

def quick_optimize(data_path, target_column, model_name="quick_optimized"):
    """
    Quick optimization with default settings
    """
    optimizer = ModelOptimizer(data_path, target_column, model_name)
    
    # Load data
    X, y = optimizer.load_and_prepare_data()
    
    # Optimize features (top 15)
    X_selected, selected_features = optimizer.optimize_features('selectk', 15)
    
    # Optimize model (balanced configuration)
    model, config = optimizer.optimize_hyperparameters('xgb')
    
    # Evaluate efficiency
    results = optimizer.evaluate_efficiency()
    
    # Save model
    output_dir = optimizer.save_optimized_model()
    
    return optimizer, results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python efficiency_optimizer.py <data_file> <target_column> [model_name]")
        print("Example: python efficiency_optimizer.py data.csv profit_margin_percent my_optimized_model")
        sys.exit(1)
    
    data_file = sys.argv[1]
    target_col = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "optimized_model"
    
    # Run quick optimization
    optimizer, results = quick_optimize(data_file, target_col, model_name)
    
    print("\n🎉 Optimization complete!")
    print(f"Efficiency Grade: {results['efficiency_grade']}")
    print(f"Feature Reduction: {results['feature_reduction_pct']:.1f}%")