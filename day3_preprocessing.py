
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("FINSIGHT - DAY 3: DATA PREPROCESSING")
print("=" * 50)

# 1. Load the cleaned dataset
df = pd.read_csv("../dataset/credit_default_clean.csv")

print("\nOriginal dataset shape:", df.shape)

# 2. Remove unnecessary ID column
df = df.drop(columns=["ID"])

# 3. Handle unknown categorical values
# Education: 0, 5 and 6 are grouped as Other
df["EDUCATION"] = df["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4
})

# Marriage: 0 is grouped as Other
df["MARRIAGE"] = df["MARRIAGE"].replace({
    0: 3
})

# 4. Separate features and target
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# 5. Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["SEX", "EDUCATION", "MARRIAGE"],
    dtype=int
)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Scale features
# Fit the scaler only on training data to prevent data leakage
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Create output folders
os.makedirs("../models", exist_ok=True)

# 9. Save original (unscaled) training and testing data
X_train.to_csv("../dataset/X_train.csv", index=False)
X_test.to_csv("../dataset/X_test.csv", index=False)

# 10. Save scaled data
pd.DataFrame(
    X_train_scaled, columns=X_train.columns
).to_csv("../dataset/X_train_scaled.csv", index=False)

pd.DataFrame(
    X_test_scaled, columns=X_test.columns
).to_csv("../dataset/X_test_scaled.csv", index=False)

# 11. Save target values
y_train.to_csv("../dataset/y_train.csv", index=False)
y_test.to_csv("../dataset/y_test.csv", index=False)

# 12. Save the scaler for future predictions
joblib.dump(scaler, "../models/scaler.pkl")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nAll preprocessing files saved successfully!")
print("DAY 3 DATA PREPROCESSING COMPLETED!")
