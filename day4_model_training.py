
import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.utils.class_weight import compute_sample_weight

print("=" * 55)
print("FINSIGHT - DAY 4: MODEL TRAINING")
print("=" * 55)

# 1. Load prepared data
X_train = pd.read_csv("../dataset/X_train.csv")
X_test = pd.read_csv("../dataset/X_test.csv")

X_train_scaled = pd.read_csv("../dataset/X_train_scaled.csv")
X_test_scaled = pd.read_csv("../dataset/X_test_scaled.csv")

y_train = pd.read_csv("../dataset/y_train.csv").squeeze()
y_test = pd.read_csv("../dataset/y_test.csv").squeeze()

os.makedirs("../models", exist_ok=True)
os.makedirs("../reports/charts", exist_ok=True)

# Handle class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# 2. Define four models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []

# 3. Train and evaluate
for name, model in models.items():

    print("\n" + "=" * 45)
    print("Training:", name)

    start = time.time()

    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        scores = model.decision_function(X_test_scaled)

    elif name == "Gradient Boosting":
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )
        predictions = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]

    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - start

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, scores)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(
        f"../reports/charts/{safe_name}_confusion_matrix.png"
    )
    plt.close()

    # Save trained model
    joblib.dump(
        model,
        f"../models/{safe_name}.pkl"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "Training Time": elapsed
    })

# 4. Compare all models
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

comparison.to_csv(
    "../reports/model_comparison.csv",
    index=False
)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)
print(comparison.to_string(index=False))

print("\nAll four models trained successfully!")
print("Models saved inside models/")
print("Comparison saved inside reports/")
print("DAY 4 MODEL TRAINING COMPLETED!")
