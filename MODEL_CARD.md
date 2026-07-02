# ChurnSense Model Card

## Model

Tuned XGBoost Classifier saved as `models/churn_prediction_model.pkl`.

## Business Problem

Predict whether a telecom customer is likely to churn so the business can prioritize retention actions.

## Dataset

IBM Telco Customer Churn dataset with 7,043 customer records.

## Target

`Churn`

- `0`: retained customer
- `1`: churned customer

## Final Test Metrics

| Metric | Score |
| --- | ---: |
| Accuracy | 0.7672 |
| Precision | 0.5513 |
| Recall | 0.6600 |
| F1 Score | 0.6009 |
| ROC AUC | 0.8280 |
| Mean 5-Fold CV ROC AUC | 0.9174 |
| CV Standard Deviation | 0.0296 |

## Important Churn Signals

- Contract type
- Tenure
- Internet service type
- Payment method
- Monthly charges
- Senior citizen status

## Intended Use

This model is intended for internship project demonstration, churn-risk scoring, and retention analytics. It should be validated with current business data before production use.

## Limitations

- The model uses historical Telco data and may not reflect new market behavior.
- Predictions are probabilities, not guarantees.
- The saved model expects the exact feature engineering implemented in `src/preprocessing.py`.
