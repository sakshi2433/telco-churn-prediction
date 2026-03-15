# Telco Customer Churn Prediction

An end-to-end machine learning project predicting customer churn for a telecom company. Covers the full pipeline — business hypothesis formation, EDA, preprocessing, multi-model training with hyperparameter tuning, and evaluation with business-relevant metrics.

---

## Business problem

A telecom company wants to identify customers at risk of cancelling their subscription before they leave. Retaining an existing customer costs significantly less than acquiring a new one — so early churn detection directly impacts revenue.

---

## Hypotheses tested

Before looking at any charts, three hypotheses were formed based on business logic:

| Hypothesis | Result |
|---|---|
| Month-to-month contract customers churn more | ✅ Strongly supported |
| Senior citizens are more likely to churn | ⚠️ Partially supported |
| Customers with lower tenure churn more | ✅ Strongly supported |

---

## Key findings

- **Contract type** is the strongest predictor — month-to-month customers churn at a dramatically higher rate than one or two-year contract holders
- **Tenure** is inversely correlated with churn — newer customers are far more likely to leave
- **Monthly charges** are higher for churned customers on average
- **Senior citizens** show slightly elevated churn but it is not the dominant factor
- **Gradient Boosting** achieved the highest ROC-AUC of **0.84**, outperforming Logistic Regression (0.78) and Random Forest (0.82)

---

## ML pipeline

```
Raw data (7,043 rows · 21 columns)
   ↓ Data cleaning — TotalCharges blank strings, drop nulls, remove customerID
   ↓ EDA — churn distribution, categorical countplots, numerical histograms
   ↓ Preprocessing — LabelEncoder, OneHotEncoding, StandardScaler
   ↓ Train/test split — 80/20, stratified on target
   ↓ 8 models trained with GridSearchCV
   ↓ Evaluation — classification report, confusion matrix, ROC-AUC
   ↓ Feature importance — top 10 predictors visualised
```

---

## Models compared

| Model | ROC-AUC |
|---|---|
| Gradient Boosting | **0.84** |
| Random Forest | 0.82 |
| AdaBoost | 0.81 |
| XGBoost | 0.81 |
| Logistic Regression | 0.78 |
| Decision Tree | 0.74 |
| KNN | 0.73 |
| SVM | 0.72 |

---

## Evaluation approach

Accuracy alone is misleading for churn (26% positive class). This project uses:
- **ROC-AUC** as the primary comparison metric
- **Recall for churn class** as the business-critical metric — missing a churner is more costly than a false alarm
- **Confusion matrix** to understand false negative rate
- **Classification report** for full precision/recall/F1 breakdown

---

## Project structure

```
telco-churn-prediction/
├── Telco_Customer_Churn.ipynb    # Full notebook — EDA + modelling + evaluation
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

---

## How to run

```bash
git clone https://github.com/sakshi2433/telco-churn-prediction
cd telco-churn-prediction
pip install -r requirements.txt
jupyter notebook Telco_Customer_Churn.ipynb
```

**Dependencies:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers · 21 features · binary churn target

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189fdd?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

> **Author:** Sakshi · [GitHub](https://github.com/sakshi2433) · [Email](mailto:sakshi240905@gmail.com)
