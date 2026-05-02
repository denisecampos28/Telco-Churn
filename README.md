# Telco Customer Churn Prediction

## Project Overview
This project develops a machine learning system to predict customer churn and help businesses identify customers at risk of leaving. The goal is to support proactive retention decisions using interpretable machine learning.

## Business Problem
Customer churn is expensive because retaining existing customers is often more cost-effective than acquiring new ones. This project predicts which customers are likely to churn and explains the main drivers behind those predictions.

## Dataset
The final dataset is the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/data

The dataset contains 7,043 fictional telco customers from California in Q3, including demographics, services, billing information, satisfaction scores, churn labels, churn score, and CLTV. :contentReference[oaicite:1]{index=1}

## Important Data Leakage Decisions
Several columns were removed because they directly reveal churn or are calculated after churn occurs:
- Customer Status
- Churn Label
- Churn Score
- Churn Score Category
- Churn Category
- Churn Reason
- CLTV
- CLTV Category

The final target variable is `Churn Value`. :contentReference[oaicite:2]{index=2}

## Methods
The project compares multiple modeling approaches, with the final model selected for interpretability and business usability.

Final model:
- Logistic Regression
- L1 Regularization
- SMOTE for class imbalance
- Stratified train/test split
- Pipeline-based preprocessing to prevent leakage

Benchmark model:
- XGBoost
- GridSearchCV tuning
- scale_pos_weight for class imbalance
- SHAP explainability

## Feature Engineering
Key engineered features include:
- Tenure_Bucket
- Revenue_per_Month

These features were created to represent customer lifecycle stage and normalized spending behavior. :contentReference[oaicite:3]{index=3}

## Final Model Performance
The final Logistic Regression model achieved approximately:
- Accuracy: 0.84
- Precision: 0.68
- Recall: 0.73
- F1-score: 0.70
- PR-AUC: 0.77

These metrics show the model can identify churners while maintaining a reasonable balance between false positives and false negatives. :contentReference[oaicite:4]{index=4}

## Key Findings
The strongest churn drivers were:
- Monthly Charge
- Revenue per Month
- Tenure Bucket

Higher monthly charges and higher spending intensity increased churn risk, while longer tenure and stronger customer commitment reduced churn risk. :contentReference[oaicite:5]{index=5}

## Model Interpretability
The project uses:
- Logistic Regression coefficients
- SHAP summary plots
- SHAP local explanations
- LIME explanations

These methods help explain both global model behavior and individual customer predictions.

## Deployment
A Streamlit app allows users to input customer information and receive:
- Churn prediction
- Churn probability
- Risk level
- Reason codes explaining the prediction

## Limitations
The model relies heavily on spending and lifecycle features. It may miss churn caused by external factors such as poor customer service, moving, outages, or competitor offers. :contentReference[oaicite:6]{index=6}

## Future Work
Future improvements could include:
- Adding customer service interaction data
- Adding time-series behavior
- Testing retention offer simulations
- Monitoring model drift after deployment
