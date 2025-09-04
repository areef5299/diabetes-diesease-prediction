# ================================
# Diabetes Disease Prediction (UI)
# ================================
# Run with:
# python3.12 -m streamlit run C:\Users\dell\Documents\mlproject2.py
# --------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Disease Prediction", page_icon="🩺", layout="centered")

# ---------- Load dataset ----------
@st.cache_data
def load_data():
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    # Try local first, then fallback to URL
    try:
        df = pd.read_csv("diabetes.csv")
        # Ensure expected column names
        if df.columns.tolist() != cols:
            df.columns = cols
    except Exception:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        df = pd.read_csv(url, names=cols)
    return df

data = load_data()

# ---------- Train a baseline model ----------
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, scaler, acc

model, scaler, acc = train_model(data)

# ---------- UI ----------
st.title("A Machine Learning based system to predict whether a person is **Diabetic** or **Not Diabetic**.")

st.markdown("---")
st.subheader("Patient Data")

with st.form("patient_form", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)

    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=72, step=1)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)

    with c2:
        glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.0, step=0.1)

    with c3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)

    submitted = st.form_submit_button("Predict")

st.markdown("---")
st.subheader("Prediction Result")

if submitted:
    # Build input frame
    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    st.write("**Entered Patient Data**")
    st.dataframe(input_df, use_container_width=True)

    # Scale and predict
    X_scaled = scaler.transform(input_df)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    label = "Diabetic" if pred == 1 else "Not Diabetic"
    confidence = proba[1] if pred == 1 else proba[0]

    if pred == 1:
        st.error(f"🔴 **{label}**  — Confidence: **{confidence*100:.2f}%**")
    else:
        st.success(f"🟢 **{label}**  — Confidence: **{confidence*100:.2f}%**")

    with st.expander("Model details"):
        st.write(f"Validation Accuracy: **{acc*100:.2f}%** (RandomForest, scaled features)")
        st.caption("Note: This is a screening tool and not a medical diagnosis. Please consult a healthcare professional for clinical decisions.")
else:
    st.info("Fill in the patient data above and click **Predict** to see the result.")

# Footer
st.markdown("---")
st.caption("Diabetes Disease Prediction • Streamlit • scikit-learn")
