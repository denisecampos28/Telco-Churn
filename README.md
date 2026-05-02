# Telco Customer Churn Prediction

## Project Overview
This project develops a machine learning system to predict customer churn and help businesses identify customers at risk of leaving. The goal is to support proactive retention decisions using interpretable machine learning.

## Business Problem
Customer churn is expensive because retaining existing customers is often more cost-effective than acquiring new ones. This project predicts which customers are likely to churn and explains the main drivers behind those predictions.

## Dataset
The final dataset is the Telco Customer Churn dataset from Kaggle. 
Link: https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/data

The dataset contains 7,043 fictional telco customers from California in Q3, including demographics, services, billing information, satisfaction scores, churn labels, churn score, and CLTV. 

## Data Cleaning & Preprocessing

The project began with the raw Telco Customer Churn dataset obtained from Kaggle. While the dataset contains rich customer information, it also includes several variables that introduce noise, redundancy, and data leakage, which can artificially inflate model performance.

To ensure the model produces realistic, deployable predictions, a structured data cleaning and preprocessing pipeline was implemented.

---

## Raw vs Clean Data

- **Raw dataset:** `telco.csv`
- **Cleaned dataset:** `telco_churn_encoded.csv`

The cleaned dataset represents a fully processed, model-ready version of the data, with:
- Leakage removed
- Features engineered
- Variables encoded
- Data standardized for modeling

---

## Data Leakage Removal

One of the most important steps in this project was identifying and removing data leakage variables; features that contain information about the target (`Churn Value`) that would not be available at prediction time.

### Removed variables:

#### Direct churn indicators (leakage)
- Customer Status  
- Churn Label  
- Churn Score  
- Churn Score Category  
- Churn Category  
- Churn Reason  

These variables directly reveal whether a customer churned or are derived from post-churn analysis.

#### Post-churn / derived business metrics
- CLTV  
- CLTV Category  

These are calculated using internal churn models and would not be known beforehand.

#### Highly correlated cumulative features
- Total Charges  
- Total Revenue  

These variables are strongly correlated with tenure and can allow the model to indirectly infer churn timing rather than true behavior.

---

## Feature Selection & Noise Reduction

To improve model generalization and interpretability, non-informative or redundant features were removed.

### Dropped features:
- CustomerID (identifier only)
- Count (reporting artifact)
- Geographic features:
  - Country, State, City, Zip Code
  - Latitude, Longitude, Population

These variables added noise without improving predictive power and could introduce overfitting.

---

## Feature Engineering

To move beyond raw variables and better capture customer behavior, new features were created:

### Engineered features:

#### `Revenue_per_Month`
- Defined as:  
  `Total Revenue / Tenure`
- Purpose: captures normalized spending intensity, not just cumulative value

#### `Tenure_Bucket`
- Groups customers into lifecycle stages:
  - Early-stage
  - Mid-term
  - Long-term
- Purpose: allows the model to distinguish between **new customers vs loyal customers**

These transformations allow the model to learn behavioral patterns, not just static values.

---

## Handling Missing Values

- Missing values were identified and handled using:
  - Imputation (for numeric features)
  - Encoding strategies (for categorical variables)

This ensured no loss of data while maintaining dataset integrity.

---

## Encoding Categorical Variables

Categorical variables were transformed into numeric format using:

- One-Hot Encoding for nominal variables
- Binary encoding (0/1) for yes/no variables

This step ensures compatibility with machine learning algorithms such as Logistic Regression and XGBoost.

---

## Scaling & Preprocessing Pipeline

For models sensitive to feature scale (e.g., Logistic Regression):

- StandardScaler was applied to numeric features

To prevent data leakage during preprocessing:

- All transformations (scaling + SMOTE) were applied using an **`imblearn Pipeline`**
- The pipeline was fit only on the training data

---

## Handling Class Imbalance

The dataset is imbalanced, with significantly fewer churn cases.

To address this:

- **SMOTE (Synthetic Minority Oversampling Technique)** was applied
- Synthetic churn samples were generated to balance the dataset

This allowed the model to better learn patterns associated with churn instead of defaulting to the majority class.

---

## 8. Train-Test Split Strategy

- **70/30 Stratified Split** was used
- Maintains the same churn distribution in both training and test sets

This prevents evaluation bias and ensures reliable model performance.

---

## 9. Final Dataset Characteristics

After preprocessing:

- ~40+ features retained
- Fully numeric and model-ready
- Balanced training data (via SMOTE)
- No leakage variables
- Behavior-focused features instead of raw cumulative metrics


---


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
