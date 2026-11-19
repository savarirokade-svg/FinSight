
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("../dataset/credit_default_clean.csv")

# Create chart folder
os.makedirs("../reports/charts", exist_ok=True)

target = "default payment next month"

sns.set_theme(style="whitegrid")

print("=" * 45)
print("          FINSIGHT - DATA ANALYSIS")
print("=" * 45)

# 1. Basic information
print("\n1. DATASET INFORMATION")
print("Total customers:", len(df))
print("Total columns:", len(df.columns))
print("\nDataset information:")
print(df.info())

# 2. Statistical summary
print("\n2. STATISTICAL SUMMARY")
print(df.describe())

# 3. Default distribution
print("\n3. DEFAULT DISTRIBUTION")
print(df[target].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=target)
plt.title("Default vs Non-Default Customers")
plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.savefig("../reports/charts/01_default_distribution.png")
plt.close()

# 4. Customer age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="AGE", bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.savefig("../reports/charts/02_age_distribution.png")
plt.close()

# 5. Credit limit analysis
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=target, y="LIMIT_BAL")
plt.title("Credit Limit vs Default Status")
plt.xlabel("Default Status")
plt.ylabel("Credit Limit")
plt.savefig("../reports/charts/03_credit_limit.png")
plt.close()

# 6. Payment history analysis
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="PAY_0", hue=target)
plt.title("Payment Status vs Default")
plt.xlabel("Most Recent Payment Status")
plt.ylabel("Number of Customers")
plt.savefig("../reports/charts/04_payment_history.png")
plt.close()

# 7. Correlation heatmap
plt.figure(figsize=(12, 9))

correlation = df.select_dtypes(include="number").corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)

plt.title("Financial Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig("../reports/charts/05_correlation_heatmap.png")
plt.close()

# 8. Default percentage
default_percentage = df[target].mean() * 100

print("\n4. DEFAULT PERCENTAGE")
print(f"Default customers: {default_percentage:.2f}%")

print("\n5. CHARTS SAVED SUCCESSFULLY!")
print("Location: reports/charts/")

print("\nDAY 2 EDA COMPLETED!")
