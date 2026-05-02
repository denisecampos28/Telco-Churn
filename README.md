# Telco Customer Churn Prediction

## Project Overview
This project develops a machine learning system to predict customer churn and identify customers at risk of leaving. The goal is to support proactive retention decisions using interpretable machine learning.

## Business Problem
Customer churn is costly, as retaining existing customers is typically more efficient than acquiring new ones. This project focuses on predicting which customers are likely to churn and identifying the key factors driving those decisions.

## Dataset
The final dataset used is the Telco Customer Churn dataset from Kaggle:

https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/data

The dataset contains 7,043 fictional telco customers from California in Q3, including demographic information, service usage, billing data, satisfaction scores, and churn indicators.

---

## Data Cleaning and Preprocessing

The project began with the raw Telco dataset. While the dataset contains valuable information, it also includes variables that introduce noise, redundancy, and data leakage. These issues can artificially inflate model performance and reduce real-world reliability.

A structured preprocessing pipeline was implemented to ensure the model produces realistic and deployable predictions.

### Raw vs Clean Data

- Raw dataset: `telco.csv`  
- Cleaned dataset: `telco_churn_encoded.csv`  

The cleaned dataset is fully processed and model-ready.

---

### Data Leakage Removal

Data leakage variables were removed to ensure the model only uses information available at prediction time.

Removed variables included:

- Direct churn indicators:
  - Customer Status  
  - Churn Label  
  - Churn Score  
  - Churn Score Category  
  - Churn Category  
  - Churn Reason  

- Post-churn metrics:
  - CLTV  
  - CLTV Category  

- Highly correlated cumulative features:
  - Total Charges  
  - Total Revenue  



<img width="631" height="539" alt="Unknown-3" src="https://github.com/user-attachments/assets/d7614748-9551-4c5d-bb61-2d692e6a9852" />




---

### Feature Engineering

Two key features were created to better capture customer behavior:

- `Revenue_per_Month`  
  - Total Revenue divided by Tenure  
  - Captures normalized spending behavior  

- `Tenure_Bucket`  
  - Groups customers into lifecycle stages  
  - Distinguishes new customers from long-term customers  

---

### Handling Missing Values

- Numeric features were imputed using the median  
- Categorical features were imputed using the most frequent value  

---

### Encoding and Scaling

- One-hot encoding was used for categorical variables  
- Binary encoding was used for yes/no features  
- StandardScaler was applied to numeric features  

All preprocessing steps were implemented within a pipeline to prevent leakage.

---

### Class Imbalance

SMOTE was applied to the training data to address class imbalance and improve the model’s ability to learn churn patterns.


<img width="580" height="455" alt="Unknown-2" src="https://github.com/user-attachments/assets/a98d78ee-b088-4f3a-8c72-b718dbaca607" />


---

### Train-Test Split

A 70/30 stratified split was used to maintain consistent churn distribution across training and testing sets.

---

## Methods

Two models were developed and compared:

### Final Model
- Logistic Regression  
- L1 Regularization  
- SMOTE  
- Pipeline-based preprocessing  

### Benchmark Model
- XGBoost  
- GridSearchCV tuning  
- Class weighting  
- SHAP explainability



<img width="646" height="213" alt="Screenshot 2026-05-01 at 8 53 15 PM" src="https://github.com/user-attachments/assets/d15b0298-4132-454b-b78a-c478b306bcff" />

Ultimately, the logistic regression model was selected because it is better suited for business decision-making.

---

## Model Performance

The final Logistic Regression model achieved:

- Accuracy: 0.84  
- Precision: 0.68  
- Recall: 0.73  
- F1-score: 0.70  
- PR-AUC: 0.77  

The model is designed to prioritize recall while maintaining reasonable precision.


<img width="846" height="547" alt="Unknown" src="https://github.com/user-attachments/assets/cbc5969f-67e5-40fa-bbd4-119e8ff13ce4" />





<img width="846" height="547" alt="Unknown2" src="https://github.com/user-attachments/assets/0cec18e8-88c6-4388-80b8-5c53f63b79a5" />


---

## Model Interpretability

SHAP values were used to provide both global and local explanations of model behavior.

### Global Drivers of Churn

Global explanations describe how features influence predictions across the entire dataset.



<img width="791" height="540" alt="Unknown3" src="https://github.com/user-attachments/assets/f6b49a99-74d8-44c5-813c-19d77407fcc6" />




Key patterns observed:

- Higher Monthly Charge increases churn risk  
- Longer contract lengths reduce churn  
- Higher customer engagement (referrals) improves retention  
- Newer customers are more likely to churn  

The model relies on a small number of strong predictors, reflecting the effect of L1 regularization in removing less important features.

---

### Local Drivers (Customer-Level Insights)

Local explanations describe why a specific customer is predicted to churn.




<img width="776" height="455" alt="Unknown4" src="https://github.com/user-attachments/assets/93b0ba4c-8b88-4cb3-b00b-ea02106e6ebb" />




For an individual prediction:

- High Monthly Charge pushes the prediction toward churn  
- High service usage (such as long-distance charges) pulls the prediction toward retention  

This creates a trade-off between pricing pressure and perceived value.

These explanations make it possible to understand not just the prediction, but the reasoning behind it for each customer.

---

## Overall Insights

- Price sensitivity is the strongest driver of churn  
- Early-stage customers are more likely to leave  
- Long-term commitment reduces churn risk  
- Customer engagement improves retention  

---

## Deployment

A Streamlit application was developed to:

- Input customer data  
- Generate churn predictions  
- Output churn probability  
- Provide interpretable explanations  

---

## Limitations

The model primarily captures financial and lifecycle patterns. It may not detect churn driven by external factors such as customer service issues, relocation, or competitor actions.

---

## Future Work

Future improvements could include:

- Incorporating customer service interaction data  
- Adding time-based behavioral features  
- Testing retention strategies  
- Monitoring model performance over time


## Tools and Libraries Used

The project was developed using the following tools:

- **Python** – primary programming language  
- **pandas** – data manipulation and preprocessing  
- **numpy** – numerical operations  
- **scikit-learn** – modeling, preprocessing pipelines, and evaluation  
- **imblearn (SMOTE)** – handling class imbalance  
- **XGBoost** – gradient boosting model for comparison  
- **SHAP** – model interpretability and feature contribution analysis  
- **matplotlib / seaborn** – data visualization  
- **joblib** – model saving and loading  
- **Jupyter Notebook** – development environment

  ### Notebooks

- `01_data_understanding.ipynb`  
  Initial exploration of the dataset, including structure, variable types, and basic summaries to understand the data.

- `02_data_cleaning.ipynb`  
  Data preprocessing steps, including handling missing values, removing leakage variables, feature selection, and encoding.

- `03_baseline_modeling.ipynb`  
  Development of the baseline Logistic Regression model, including preprocessing pipeline, SMOTE implementation, and initial evaluation.

- `xgboost_telcodata.ipynb`  
  Preparation of the dataset specifically for tree-based modeling, including encoding and feature setup for XGBoost.

- `xgboost_featureengineering.ipynb`  
  Additional feature engineering for the XGBoost model, focusing on improving predictive performance through behavioral features.

- `xgboost_model_deployment.ipynb`  
  XGBoost deployment procedures.

- `app.py`  
  Streamlit app.

 - `churn_xgb_model.joblib`  
  Final XGBoost model training, tuning, evaluation, and preparation for deployment.

---

## Workflow Summary

The project follows a structured pipeline:

1. Data understanding  
2. Data cleaning and preprocessing  
3. Feature engineering  
4. Model development (Logistic Regression and XGBoost)  
5. Model evaluation  
6. Model interpretation (SHAP)  
7. Deployment 

This structure ensures the project is reproducible, interpretable, and aligned with real-world machine learning practices.
