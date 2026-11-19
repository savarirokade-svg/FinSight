
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report
)

print("=" * 55)
print("FINSIGHT - DAY 5: MODEL EVALUATION")
print("=" * 55)

# 1. Load testing datasets
X_test = pd.read_csv("../dataset/X_test.csv")
X_test_scaled = pd.read_csv("../dataset/X_test_scaled.csv")
y_test = pd.read_csv("../dataset/y_test.csv").iloc[:, 0]

# 2. Load scaled and unscaled models
model_files = {
    "Random Forest": "../models/random_forest.pkl",
    "XGBoost": "../models/xgboost.pkl",
    "SVM": "../models/svm.pkl",
    "Gradient Boosting": "../models/gradient_boosting.pkl"
}

models = {}

for name, path in model_files.items():
    models[name] = joblib.load(path)

print("\nAll four saved models loaded successfully!")

# 3. Prepare evaluation
results = []
roc_data = {}
pr_data = {}

for name, model in models.items():

    # SVM was trained using scaled data
    if name == "SVM":
        X = X_test_scaled
        scores = model.decision_function(X)
    else:
        X = X_test
        scores = model.predict_proba(X)[:, 1]

    predictions = model.predict(X)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    # ROC curve values
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_data[name] = (fpr, tpr, roc_auc)

    # Precision-Recall curve values
    precision_values, recall_values, _ = precision_recall_curve(
        y_test, scores
    )
    pr_data[name] = (
        recall_values,
        precision_values,
        pr_auc
    )

    print(f"\n{name}")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# 4. Save detailed evaluation report
results_df = pd.DataFrame(results)
results_df.to_csv("../reports/day5_evaluation_report.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# 5. Create ROC curve
plt.figure(figsize=(9, 6))

for name, (fpr, tpr, auc_value) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("FinSight - ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/charts/day5_roc_curve.png", dpi=300)
plt.close()

# 6. Create Precision-Recall curve
plt.figure(figsize=(9, 6))

for name, (recall_values, precision_values, pr_auc) in pr_data.items():
    plt.plot(
        recall_values,
        precision_values,
        label=f"{name} (PR-AUC={pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("FinSight - Precision-Recall Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/charts/day5_precision_recall_curve.png", dpi=300)
plt.close()

# 7. Create metric comparison chart
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("FinSight - Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/charts/day5_metric_comparison.png", dpi=300)
plt.close()

print("\nEvaluation report and charts saved successfully!")
print("DAY 5 MODEL EVALUATION COMPLETED!")
