
import pandas as pd

file_path = "../dataset/creditcarclients.xls"

# Load dataset with correct headers
df = pd.read_excel(file_path, engine="xlrd", header=1)

# Save cleaned dataset
df.to_csv("../dataset/credit_default_clean.csv", index=False)

print("Cleaned dataset saved successfully!")
print("Records:", len(df))
print("Columns:", len(df.columns))

# Check target distribution
print("\nDefault payment distribution:")
print(df["default payment next month"].value_counts())
