# Telco Customer Churn Prediction

Predicting which telecom customers are likely to cancel their service, and identifying
the strongest drivers of churn, so the business can target retention offers where they
matter most.

## Business Problem

Customer acquisition costs 5–25x more than retention. This project uses historical
customer data (demographics, account details, and subscribed services) to:
1. Identify the segments and behaviors most associated with churn
2. Build a classifier that flags high-risk customers before they leave
3. Translate model output into a short list of actionable retention levers

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) —
7,043 customers, 21 features (demographics, contract type, services subscribed,
billing details), 26.5% churn rate.

## Approach

**1. Cleaning & EDA** (`01_eda_cleaning.py`)
- Fixed `TotalCharges` (loaded as text due to 11 blank values for customers with
  0 months tenure — filled with 0 rather than dropped, since these are a real,
  meaningful customer segment, not bad data)
- Explored churn rate by contract type, internet service, payment method, and
  tenure/spend distributions

**2. Modeling** (`02_modeling.py`)
- Stratified 80/20 train/test split to preserve the 26.5% churn rate in both sets
- `ColumnTransformer` pipeline: standard-scaled numeric features + one-hot encoded
  categoricals, so preprocessing is never leaked across the train/test boundary
- Compared three classifiers, all class-weighted to handle the imbalance:
  Logistic Regression, Random Forest, XGBoost
- Evaluated with ROC-AUC and F1 on the churn class specifically — accuracy alone
  is misleading here, since a model that always predicts "no churn" would already
  score 73.5% accuracy while catching zero at-risk customers

## Results

| Model | ROC-AUC | F1 (churn class) |
|---|---|---|
| **Logistic Regression** | **0.842** | **0.614** |
| XGBoost | 0.839 | 0.627 |
| Random Forest | 0.825 | 0.557 |

Logistic Regression edged out the more complex models on ROC-AUC — a reminder that
a simpler, more interpretable model is a perfectly legitimate (and sometimes
preferable) choice when it performs comparably, especially since stakeholders can
read its coefficients directly as churn drivers.

### Top churn drivers (Logistic Regression coefficients)
1. **Contract length** — two-year and one-year contracts sharply reduce churn risk
   vs. month-to-month
2. **Fiber optic internet** — associated with higher churn, likely a pricing or
   service-quality signal worth investigating with the product team
3. **Tenure** — newer customers churn more; the first several months are the
   highest-risk retention window
4. **Monthly/total charges** — higher bills correlate with higher churn risk
5. **Electronic check payment** — the highest-churn payment method, possibly a
   proxy for lower engagement with autopay

### Business recommendation
Prioritize retention outreach (loyalty offers, contract-upgrade incentives) at
**month-to-month, fiber-optic customers in their first 6 months** — this segment
combines the three strongest churn signals the model found.

## Files
```
churn_project/
├── data/
│   ├── telco_churn.csv          # raw data
│   └── telco_churn_clean.csv    # cleaned data
├── outputs/                     # charts + results
├── 01_eda_cleaning.py
├── 02_modeling.py
└── README.md
```

## What I'd add with more time
- Hyperparameter tuning (GridSearchCV) on the shortlisted model
- SHAP values for per-customer explanations, not just global feature importance
- A cost-based threshold: tune the classification cutoff against the actual
  $-cost of a false negative (lost customer) vs. false positive (wasted retention offer)
"# Telco_Churn_Prediction" 
