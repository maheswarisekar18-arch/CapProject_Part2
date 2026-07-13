import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE


df = pd.read_csv("cleaned_data.csv")


X = df.drop(columns=["marks", "name"])   # drop target + non-predictive name
y_reg = df["marks"]

y_clf = (y_reg > y_reg.median()).astype(int)

print("Regression label: marks")
print("Classification label: 1 if marks > median else 0")


X = pd.get_dummies(X, columns=["grade"], drop_first=True)

print("Encoded feature matrix:\n", X.head())


X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

print("Linear Regression MSE:", mean_squared_error(y_reg_test, y_pred_reg))
print("Linear Regression R²:", r2_score(y_reg_test, y_pred_reg))

coeffs = pd.Series(lin_reg.coef_, index=X.columns)
print("Coefficients:\n", coeffs)
print("Top 3 features by absolute coefficient:\n", coeffs.abs().nlargest(3))


ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("Ridge Regression MSE:", mean_squared_error(y_reg_test, y_pred_ridge))
print("Ridge Regression R²:", r2_score(y_reg_test, y_pred_ridge))


print("Class balance:\n", y_clf_train.value_counts())

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_clf_train)

y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_pred_clf))
print("Classification Report:\n", classification_report(y_clf_test, y_pred_clf))

fpr, tpr, _ = roc_curve(y_clf_test, y_proba_clf)
auc = roc_auc_score(y_clf_test, y_proba_clf)

plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
results = []
for th in thresholds:
    preds = (y_proba_clf >= th).astype(int)
    prec = precision_score(y_clf_test, preds)
    rec = recall_score(y_clf_test, preds)
    f1 = f1_score(y_clf_test, preds)
    results.append([th, prec, rec, f1])

threshold_table = pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1"])
print(threshold_table)


log_reg_strong = LogisticRegression(max_iter=1000, C=0.01, class_weight="balanced")
log_reg_strong.fit(X_train_scaled, y_clf_train)

y_proba_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]
y_pred_strong = log_reg_strong.predict(X_test_scaled)

prec1 = precision_score(y_clf_test, y_pred_clf)
rec1 = recall_score(y_clf_test, y_pred_clf)
auc1 = roc_auc_score(y_clf_test, y_proba_clf)

prec2 = precision_score(y_clf_test, y_pred_strong)
rec2 = recall_score(y_clf_test, y_pred_strong)
auc2 = roc_auc_score(y_clf_test, y_proba_strong)

comparison = pd.DataFrame({
    "Model": ["LogReg C=1.0", "LogReg C=0.01"],
    "Precision": [prec1, prec2],
    "Recall": [rec1, rec2],
    "AUC": [auc1, auc2]
})
print(comparison)

n_boot = 500
diffs = []
for _ in range(n_boot):
    idx = np.random.choice(len(y_clf_test), size=len(y_clf_test), replace=True)
    y_true = y_clf_test.iloc[idx]
    proba1 = y_proba_clf[idx]
    proba2 = y_proba_strong[idx]
    auc_diff = roc_auc_score(y_true, proba1) - roc_auc_score(y_true, proba2)
    diffs.append(auc_diff)

mean_diff = np.mean(diffs)
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

print("Bootstrap mean AUC difference:", mean_diff)
print("95% CI:", (ci_low, ci_high))