"""
Telco Customer Churn — Data Cleaning & EDA
Author: Rithvik Ponugoti

Goal: Understand the drivers of customer churn for a telecom provider and
prepare a clean dataset for modeling.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. Load and inspect
# ---------------------------------------------------------------
df = pd.read_csv("data/telco_churn.csv")
print(f"Shape: {df.shape}")
print(df.info())

# ---------------------------------------------------------------
# 2. Clean TotalCharges (loaded as text because of blank strings)
# ---------------------------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"\nRows with missing TotalCharges: {df['TotalCharges'].isna().sum()}")

# These are all customers with tenure == 0 (brand new, haven't been billed yet)
print(df.loc[df["TotalCharges"].isna(), "tenure"].unique())

# Fill with 0 rather than drop — tenure=0 customers are a real, meaningful group
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# ---------------------------------------------------------------
# 3. Target distribution (class imbalance check)
# ---------------------------------------------------------------
churn_rate = (df["Churn"] == "Yes").mean()
print(f"\nOverall churn rate: {churn_rate:.1%}")

plt.figure(figsize=(5, 4))
df["Churn"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Class balance: Churn")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("outputs/01_class_balance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Churn rate by key categorical drivers
# ---------------------------------------------------------------
drivers = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, drivers):
    rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    rate.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Churn rate by {col}")
    ax.set_xlabel("Churn rate")
plt.tight_layout()
plt.savefig("outputs/02_churn_by_driver.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Tenure and MonthlyCharges vs churn
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[0], palette=["#4C72B0", "#DD8452"])
axes[0].set_title("Tenure (months) vs Churn")
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1], palette=["#4C72B0", "#DD8452"])
axes[1].set_title("Monthly charges vs Churn")
plt.tight_layout()
plt.savefig("outputs/03_tenure_charges.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Save cleaned dataset for modeling
# ---------------------------------------------------------------
df.to_csv("data/telco_churn_clean.csv", index=False)
print("\nSaved cleaned dataset -> data/telco_churn_clean.csv")
print("Saved 3 EDA charts -> outputs/")
