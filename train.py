import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def infer_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > 20:
            return "regression"
        return "classification"
    return "classification"

def preprocess(df: pd.DataFrame, target: str, id_cols=None, encoder=None):
    id_cols = id_cols or []
    df = df.drop(columns=[c for c in id_cols if c in df.columns])
    y = df[target]
    X = df.drop(columns=[target])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        if encoder is None:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[cat_cols] = enc.fit_transform(X[cat_cols])
            # Store encoder for saving
            preprocess._encoder = enc
        else:
            X[cat_cols] = encoder.transform(X[cat_cols])
    return X, y

def get_models(problem_type: str, n_features: int):
    if problem_type == "classification":
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            missing=np.nan,
        )
    else:
        rf = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
        xgb = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            missing=np.nan,
        )
    return rf, xgb

def cross_validate(model, X, y, problem_type: str, n_splits: int = 5, timeseries: bool = False):
    scores = []
    if timeseries:
        kf = TimeSeriesSplit(n_splits=n_splits)
    elif problem_type == "classification":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splitter = kf.split(X) if timeseries else kf.split(X, y if problem_type == "classification" else None)
    for train_idx, val_idx in splitter:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
        except Exception:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if problem_type == "classification":
            acc = accuracy_score(y_val, y_pred)
            try:
                y_proba = model.predict_proba(X_val)
                if y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_val, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_val, y_proba, multi_class="ovr")
            except Exception:
                auc = np.nan
            f1 = f1_score(y_val, y_pred, average="weighted")
            scores.append({"accuracy": acc, "roc_auc": auc, "f1": f1})
        else:
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val, y_pred)
            scores.append({"rmse": rmse, "r2": r2})
    return scores

def feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        items = [(k, v) for k, v in score.items()]
        df = pd.DataFrame(items, columns=["feature", "importance"]).sort_values("importance", ascending=False)
        return df
    except Exception:
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})

def shap_summary(model, X, out_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def parse_crisis_adjust(s: str):
    adj = {}
    if not s:
        return adj
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            col, val = p.split(":", 1)
            try:
                adj[col] = float(val)
            except Exception:
                pass
    return adj

def apply_crisis(X: pd.DataFrame, adjustments: dict):
    X = X.copy()
    for col, factor in adjustments.items():
        if col in X.columns:
            X[col] = X[col].astype(float) * factor
    return X

def load_trained_model(model_type="xgb", model_dir="outputs/models"):
    """
    Load a previously trained model with its encoder.
    
    Args:
        model_type: "xgb" or "rf" for XGBoost or Random Forest
        model_dir: Directory containing saved models
    
    Returns:
        tuple: (model, encoder, feature_names)
    """
    if model_type == "xgb":
        model_path = os.path.join(model_dir, "xgb_model.joblib")
    elif model_type == "rf":
        model_path = os.path.join(model_dir, "rf_model.joblib")
    else:
        raise ValueError("model_type must be 'xgb' or 'rf'")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    
    # Load encoder if it exists
    encoder_path = os.path.join(model_dir, "encoder.joblib")
    encoder = None
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
    
    # Load feature names from metrics
    metrics_path = os.path.join("outputs", "metrics.json")
    feature_names = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            feature_names = metrics.get("model_paths", {}).get("feature_names")
    
    return model, encoder, feature_names

def generate_sample(problem_type: str, n_samples: int = 1000, n_features: int = 20, n_classes: int = 2):
    rng = np.random.RandomState(42)
    if problem_type == "classification":
        X = rng.randn(n_samples, n_features)
        w = rng.randn(n_features)
        logits = X.dot(w)
        probs = 1 / (1 + np.exp(-logits))
        y = (probs > np.quantile(probs, 1 - 1 / n_classes)).astype(int)
    else:
        X = rng.randn(n_samples, n_features)
        w = rng.randn(n_features)
        y = X.dot(w) + rng.randn(n_samples) * 0.5
    cols = [f"num_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["cat_a"] = rng.choice(["A", "B", "C"], size=n_samples)
    df["cat_b"] = rng.choice(["X", "Y"], size=n_samples)
    df["target"] = y
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--id-cols", type=str, default=None)
    parser.add_argument("--crisis-adjust", type=str, default=None)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--generate-sample", action="store_true")
    parser.add_argument("--problem-type", type=str, choices=["classification", "regression"], default=None)
    parser.add_argument("--timeseries", action="store_true")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if args.generate_sample:
        df = generate_sample(args.problem_type or "classification")
        args.target = "target"
    else:
        if not args.data or not args.target:
            print("Provide --data path and --target column or use --generate-sample")
            sys.exit(1)
        df = pd.read_csv(args.data)

    id_cols = [c.strip() for c in args.id_cols.split(",")] if args.id_cols else []
    X, y = preprocess(df, args.target, id_cols)
    problem_type = args.problem_type or infer_problem_type(y)

    rf, xgb = get_models(problem_type, X.shape[1])

    rf_scores = cross_validate(rf, X, y, problem_type, n_splits=args.cv, timeseries=args.timeseries)
    xgb_scores = cross_validate(xgb, X, y, problem_type, n_splits=args.cv, timeseries=args.timeseries)

    xgb.fit(X, y)
    rf.fit(X, y)

    # Save trained models
    model_dir = os.path.join("outputs", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    xgb_model_path = os.path.join(model_dir, "xgb_model.joblib")
    rf_model_path = os.path.join(model_dir, "rf_model.joblib")
    
    joblib.dump(xgb, xgb_model_path)
    joblib.dump(rf, rf_model_path)
    
    # Save preprocessing encoder if categorical columns exist
    encoder_path = None
    if hasattr(preprocess, '_encoder'):
        encoder_path = os.path.join(model_dir, "encoder.joblib")
        joblib.dump(preprocess._encoder, encoder_path)

    fi = feature_importance(xgb, X.columns.tolist())
    fi_path = os.path.join("outputs", "feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    plt.figure(figsize=(10, 6))
    topn = fi.head(25)
    plt.barh(topn["feature"][::-1], topn["importance"][::-1])
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "feature_importance.png"))
    plt.close()

    try:
        shap_summary(xgb, X, os.path.join("outputs", "shap_summary.png"))
    except Exception:
        pass

    metrics = {
        "problem_type": problem_type,
        "rf_cv": rf_scores,
        "xgb_cv": xgb_scores,
        "feature_importance_path": fi_path,
        "outputs_dir": os.path.abspath("outputs"),
        "model_paths": {
            "xgb_model": os.path.abspath(xgb_model_path),
            "rf_model": os.path.abspath(rf_model_path),
            "encoder": os.path.abspath(encoder_path) if encoder_path else None,
            "feature_names": X.columns.tolist()
        }
    }

    if args.crisis_adjust:
        adjustments = parse_crisis_adjust(args.crisis_adjust)
        X_crisis = apply_crisis(X, adjustments)
        y_pred_base = xgb.predict(X)
        y_pred_crisis = xgb.predict(X_crisis)
        if problem_type == "classification":
            base_acc = accuracy_score(y, y_pred_base)
            crisis_acc = accuracy_score(y, y_pred_crisis)
            metrics["crisis"] = {"adjustments": adjustments, "base_acc": base_acc, "crisis_acc": crisis_acc, "delta_acc": crisis_acc - base_acc}
        else:
            base_rmse = mean_squared_error(y, y_pred_base)
            crisis_rmse = mean_squared_error(y, y_pred_crisis, squared=False)
            metrics["crisis"] = {"adjustments": adjustments, "base_rmse": base_rmse, "crisis_rmse": crisis_rmse, "delta_rmse": crisis_rmse - base_rmse}

    with open(os.path.join("outputs", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
