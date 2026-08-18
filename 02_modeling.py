"""
Telco Customer Churn — Predictive Modeling
Author: Rithvik Ponugoti

Goal: Predict which customers are likely to churn so the business can
target retention offers, and identify the strongest churn drivers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------------
df = pd.read_csv("data/telco_churn_clean.csv")
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")

# ---------------------------------------------------------------
# 2. Train / test split (stratified because of class imbalance)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.1%}, Test churn rate: {y_test.mean():.1%}")

# ---------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
])

# ---------------------------------------------------------------
# 4. Models to compare
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
}

results = {}
fig, ax = plt.subplots(figsize=(6, 6))

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {"roc_auc": auc, "f1": f1, "report": report}
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.3f} | F1 (churn class): {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # keep the fitted pipeline for the best model later
    results[name]["pipe"] = pipe

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/04_roc_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Pick best model by ROC-AUC, show confusion matrix
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_pipe = results[best_name]["pipe"]
print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.3f})")

y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("outputs/05_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Feature importance (from Random Forest / XGBoost, whichever won,
#    or coefficients if Logistic Regression won)
# ---------------------------------------------------------------
feature_names = (
    numeric_cols
    + list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_cols))
)
model_step = best_pipe.named_steps["model"]

if hasattr(model_step, "feature_importances_"):
    importances = model_step.feature_importances_
elif hasattr(model_step, "coef_"):
    importances = np.abs(model_step.coef_[0])
else:
    importances = None

if importances is not None:
    imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    imp_df.sort_values().plot(kind="barh", color="#4C72B0")
    plt.title(f"Top 15 Feature Importances — {best_name}")
    plt.tight_layout()
    plt.savefig("outputs/06_feature_importance.png", dpi=150)
    plt.close()
    print("\nTop 10 features driving churn:")
    print(imp_df.head(10))

# ---------------------------------------------------------------
# 7. Save a summary of results for the README / report
# ---------------------------------------------------------------
summary = {
    name: {"roc_auc": round(r["roc_auc"], 4), "f1_churn": round(r["f1"], 4)}
    for name, r in results.items()
}
summary["best_model"] = best_name
with open("outputs/model_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved model comparison chart, confusion matrix, feature importance, and results JSON -> outputs/")
