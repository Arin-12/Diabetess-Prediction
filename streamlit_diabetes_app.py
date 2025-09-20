# streamlit_diabetes_app.py
import streamlit as st
import pandas as pd
import pickle

# Load trained models
with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# Title
st.title("🩺 Diabetes Prediction App")
st.markdown("Predict diabetes using **Support Vector Machine (SVM)** and **Random Forest (RF)** with risk factor interpretation.")

# Sidebar for user input
st.sidebar.header("Enter Patient Details")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale input
input_scaled = scaler.transform(input_df)

# Function for risk interpretation
def feature_warnings(input_df):
    messages = []

    if input_df["Glucose"].values[0] > 125:
        messages.append("🔴 High Glucose: Possible diabetes risk (consider blood sugar test).")
    elif input_df["Glucose"].values[0] < 70:
        messages.append("🟡 Low Glucose: Possible hypoglycemia risk.")
    else:
        messages.append("🟢 Glucose is in normal range.")

    if input_df["BloodPressure"].values[0] > 130:
        messages.append("🔴 High Blood Pressure: Increases diabetes & heart disease risk.")
    elif input_df["BloodPressure"].values[0] < 60:
        messages.append("🟡 Low Blood Pressure: May indicate underlying health issues.")
    else:
        messages.append("🟢 Blood Pressure is in normal range.")

    if input_df["BMI"].values[0] >= 30:
        messages.append("🔴 High BMI (Obesity): Strongly linked with type 2 diabetes.")
    elif input_df["BMI"].values[0] < 18.5:
        messages.append("🟡 Low BMI: Possible undernutrition.")
    else:
        messages.append("🟢 BMI is in healthy range.")

    if input_df["Age"].values[0] > 45:
        messages.append("🟡 Age above 45: Higher diabetes risk group.")
    else:
        messages.append("🟢 Age is within lower risk group.")

    if input_df["Insulin"].values[0] > 200:
        messages.append("🟡 High Insulin: May indicate insulin resistance.")
    else:
        messages.append("🟢 Insulin is in normal range.")

    if input_df["SkinThickness"].values[0] > 40:
        messages.append("🟡 High Skin Thickness: Often linked with obesity & insulin resistance.")
    else:
        messages.append("🟢 Skin Thickness is in normal range.")

    if input_df["DiabetesPedigreeFunction"].values[0] > 1:
        messages.append("🟡 Family History (DPF high): Greater genetic risk of diabetes.")
    else:
        messages.append("🟢 DPF is in safe range.")

    if input_df["Pregnancies"].values[0] >= 6:
        messages.append("🟡 Multiple Pregnancies: Slightly higher diabetes risk in women.")
    else:
        messages.append("🟢 Pregnancy count is within typical range.")

    return messages


# Prediction button
if st.button("Predict"):
    # --- SVM Prediction ---
    svm_pred = svm.predict(input_scaled)
    svm_prob = svm.predict_proba(input_scaled)[0][1] if hasattr(svm, "predict_proba") else None
    svm_result = "🟢 No Diabetes" if svm_pred[0] == 0 else "🔴 Diabetes Detected"

    # --- Random Forest Prediction ---
    rf_pred = clf.predict(input_scaled)
    rf_prob = clf.predict_proba(input_scaled)[0][1]
    rf_result = "🟢 No Diabetes" if rf_pred[0] == 0 else "🔴 Diabetes Detected"

    # Show comparison
    st.subheader("📊 Model Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔹 SVM Model")
        st.write(svm_result)
        if svm_prob is not None:
            st.write(f"**Risk Probability:** {svm_prob*100:.2f}%")

    with col2:
        st.markdown("### 🔹 Random Forest Model")
        st.write(rf_result)
        st.write(f"**Risk Probability:** {rf_prob*100:.2f}%")

    # Risk factor interpretation
    st.subheader("📌 Risk Factor Analysis")
    warnings = feature_warnings(input_df)
    for msg in warnings:
        st.write(msg)
