import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Heart Disease App", layout="centered")

# Load model
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("❤️ Heart Disease Prediction System")
st.markdown("### Predict heart disease risk using clinical data")

st.markdown("---")

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80)
    sex = st.selectbox("Sex", ["Female", "Male"])
    chest_pain = st.selectbox("Chest Pain Type", [1,2,3,4])
    bp = st.slider("Resting Blood Pressure", 90, 200)
    chol = st.slider("Cholesterol", 100, 400)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
    rest_ecg = st.selectbox("Rest ECG", [0,1,2])

with col2:
    max_hr = st.slider("Max Heart Rate", 70, 200)
    ex_angina = st.selectbox("Exercise Angina", [0,1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0)
    slope = st.selectbox("Slope", [1,2,3])
    ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encoding
sex = 1 if sex == "Male" else 0
thal_fixed = 1 if thal == "Fixed Defect" else 0
thal_normal = 1 if thal == "Normal" else 0
thal_reversible = 1 if thal == "Reversible Defect" else 0

st.markdown("---")

# Prediction
if st.button("🔍 Predict"):

    sample = pd.DataFrame([[slope, bp, chest_pain, ca, fbs, rest_ecg,
                            chol, oldpeak, sex, age, max_hr, ex_angina,
                            thal_fixed, thal_normal, thal_reversible]],
    columns=['slope_of_peak_exercise_st_segment',
             'resting_blood_pressure',
             'chest_pain_type',
             'num_major_vessels',
             'fasting_blood_sugar_gt_120_mg_per_dl',
             'resting_ekg_results',
             'serum_cholesterol_mg_per_dl',
             'oldpeak_eq_st_depression',
             'sex',
             'age',
             'max_heart_rate_achieved',
             'exercise_induced_angina',
             'thal_fixed_defect',
             'thal_normal',
             'thal_reversible_defect'])

    cols_to_scale = ['age','resting_blood_pressure',
                     'serum_cholesterol_mg_per_dl',
                     'max_heart_rate_achieved',
                     'oldpeak_eq_st_depression']

    sample[cols_to_scale] = scaler.transform(sample[cols_to_scale])

    prediction = model.predict(sample)
    prob = model.predict_proba(sample)

    # Result
    st.markdown("## 📊 Prediction Result")

    if prediction[0] == 1:
        result_text = "Heart Disease Detected"
        confidence = prob[0][1]*100
        st.error(f"⚠️ {result_text}\n\nConfidence: {confidence:.2f}%")
    else:
        result_text = "No Heart Disease"
        confidence = prob[0][0]*100
        st.success(f"✅ {result_text}\n\nConfidence: {confidence:.2f}%")

    # Create report
    report = sample.copy()
    report["Prediction"] = result_text
    report["Confidence"] = f"{confidence:.2f}%"

    # Download button
    csv = report.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="heart_disease_report.csv",
        mime="text/csv"
    )