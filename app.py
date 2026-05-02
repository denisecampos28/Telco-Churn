import streamlit as st
import joblib
import pandas as pd

st.markdown(
    """
    <style>
    /* Remove white container + make full dark UI */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: white;
    }

    /* Main content area */
    .block-container {
        background-color: rgba(255, 255, 255, 0.02);
        padding: 2rem;
        border-radius: 15px;
    }

    /* Headers */
    h1 {
        color: #e2e8f0;
        font-size: 2.8rem;
        font-weight: 700;
    }

    h2, h3 {
        color: #38bdf8;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8, #22c55e);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }

    /* Inputs */
    .stNumberInput, .stSelectbox {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
    }

    /* Divider line */
    hr {
        border: 1px solid rgba(255,255,255,0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = joblib.load("churn_xgb_model.joblib")

# Page config
st.set_page_config(page_title="Churn Predictor", layout="centered")

col1, col2 = st.columns([1,4])

with col1:
    st.image("image.jpeg", width=100)

with col2:
    st.title("Telco Customer Retention Intelligence")
    st.markdown("For Telco internal use only. This tool predicts a customer's likelihood of continuing business with the company.")

st.write("---")

# Section: Inputs
st.markdown("### 📥 Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 40)
    tenure = st.number_input("Tenure (Months)", 0, 100, 12)
    referrals = st.number_input("Number of Referrals", 0, 20, 0)

with col2:
    monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 200.0, 75.0)
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])

# Convert contract
contract_one_year = 1 if contract == "One Year" else 0
contract_two_year = 1 if contract == "Two Year" else 0

# Create dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "Tenure in Months": tenure,
    "Monthly Charge": monthly_charge,
    "Number of Referrals": referrals,
    "Contract_One Year": contract_one_year,
    "Contract_Two Year": contract_two_year
}])

# Fill missing columns
model_columns = model.get_booster().feature_names
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_columns]

st.write("---")

# Prediction section
st.header("Prediction")

if st.button("Predict Churn Risk"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader(f"Churn Probability: {probability:.2%}")

    # Risk display
    if probability > 0.7:
        st.error("🔴 High Risk of Churn")
    elif probability > 0.4:
        st.warning("🟠 Moderate Risk of Churn")
    else:
        st.success("🟢 Low Risk of Churn")

    st.write("---")

    # Insights section
    st.header("Key Insights")

    insights = []

    if tenure < 6:
        insights.append("Customer is relatively new → higher churn risk")

    if monthly_charge > 90:
        insights.append("High monthly cost → may increase churn risk")

    if referrals > 2:
        insights.append("Customer has multiple referrals → strong engagement (lower risk)")

    if contract == "Month-to-Month":
        insights.append("No long-term contract → higher likelihood to leave")

    if contract == "Two Year":
        insights.append("Long-term contract → strong retention signal")

    if age < 30:
        insights.append("Younger customers may be more likely to switch providers")

    if len(insights) == 0:
        st.write("No major risk factors detected.")
    else:
        for i in insights:
            st.write(f"- {i}")

    st.write("---")

    # Business recommendation
    st.header("Recommended Action")

    if probability > 0.7:
        st.write("🚨 Immediate retention action recommended: Offer discounts or incentives.")
    elif probability > 0.4:
        st.write("⚠️ Monitor customer and consider targeted engagement.")
    else:
        st.write("✅ Customer is stable. Maintain current experience.")