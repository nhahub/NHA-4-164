import sys, os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import sklearn

ARTIFACTS_DIR = "app/models"
print(f"scikit-learn version: {sklearn.__version__}")

X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv")).iloc[:, 0]
X_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).iloc[:, 0]
print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")

print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
joblib.dump(xgb, os.path.join(ARTIFACTS_DIR, "xgb_churn_model.pkl"))

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "rf_churn_model.pkl"))

best_model = rf if rf_auc >= xgb_auc else xgb
best_name = "Random Forest" if rf_auc >= xgb_auc else "XGBoost"
joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_churn_model.pkl"))
print(f"Best model: {best_name}")
print(f"scikit-learn version used: {sklearn.__version__}")