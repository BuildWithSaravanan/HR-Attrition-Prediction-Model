import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('hr_attrition_model.pkl')

# App title
st.title("💼 HR Employee Attrition Prediction App")

st.write("Enter employee details below to predict whether they are likely to leave or stay.")

# --- Employee Input Form ---
# You can modify or add fields based on your dataset
Age = st.number_input("Age", min_value=18, max_value=60, value=30)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
DistanceFromHome = st.slider("Distance From Home (in km)", 0, 50, 10)
JobSatisfaction = st.slider("Job Satisfaction (1-Low to 4-High)", 1, 4, 3)
EnvironmentSatisfaction = st.slider("Environment Satisfaction (1-Low to 4-High)", 1, 4, 3)
WorkLifeBalance = st.slider("Work Life Balance (1-Low to 4-High)", 1, 4, 3)
OverTime = st.selectbox("OverTime", ["Yes", "No"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Education = st.slider("Education Level (1–5)", 1, 5, 3)
JobLevel = st.slider("Job Level (1–5)", 1, 5, 2)
Department = st.selectbox(
    "Department",
    ["Sales", "Research & Development", "Human Resources"]
)

BusinessTravel = st.selectbox(
    "Business Travel",
    ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
)


# --- Convert Inputs to DataFrame ---
input_data = pd.DataFrame({
    'Age': [Age],
    'MonthlyIncome': [MonthlyIncome],
    'DistanceFromHome': [DistanceFromHome],
    'JobSatisfaction': [JobSatisfaction],
    'EnvironmentSatisfaction': [EnvironmentSatisfaction],
    'WorkLifeBalance': [WorkLifeBalance],
    'OverTime': [OverTime],
    'MaritalStatus': [MaritalStatus],
    'Gender': [Gender],
    'Education': [Education],
    'JobLevel': [JobLevel],
'Department': [Department],
    'BusinessTravel': [BusinessTravel]
})

# --- Prediction Button ---
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Employee is likely to leave. (Attrition Probability: {proba:.2f})")
    else:
        st.success(f"✅ Employee is likely to stay. (Attrition Probability: {proba:.2f})")
