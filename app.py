import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Risk Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- LOAD MODELS ---
@st.cache_resource
def load_heart_model():
    try:
        return joblib.load('heart_model.joblib')
    except Exception as e:
        st.error(f"Error loading heart model: {e}")
        return None

@st.cache_resource
def load_stress_model():
    try:
        return joblib.load('stress_model.joblib')
    except Exception as e:
        st.error(f"Error loading stress model: {e}")
        return None

heart_model = load_heart_model()
stress_model = load_stress_model()

# --- HELPER FUNCTIONS FOR HEART ---
def get_heart_risk_category(prob):
    if prob < 0.3:
        return "Low", "green"
    elif prob < 0.6:
        return "Moderate", "orange"
    else:
        return "High", "red"

def generate_heart_explanation(inputs, risk_category):
    factors = []
    if inputs['trestbps'] > 130:
        factors.append("elevated blood pressure")
    if inputs['chol'] > 200:
        factors.append("high cholesterol")
    if inputs['oldpeak'] > 1.0:
        factors.append("ST depression during exercise")
    if inputs['fbs'] == 1:
        factors.append("high fasting blood sugar")
        
    if factors:
        factor_str = ", ".join(factors[:-1]) + (" and " + factors[-1] if len(factors) > 1 else factors[0])
        return f"Your risk is **{risk_category}**. This may be influenced by {factor_str}."
    else:
        if risk_category == "Low":
            return "Your risk is **Low**. Your entered metrics appear to be in healthy ranges."
        return f"Your risk is **{risk_category}** based on the combined pattern of your health metrics."

# --- HELPER FUNCTIONS FOR STRESS ---
def get_stress_color(stress_level):
    if stress_level == 'Low Stress':
        return "green"
    elif stress_level == 'Moderate Stress':
        return "orange"
    else:
        return "red"

# --- UI DESIGN ---
st.title("🩺 Health Risk Prediction System")
st.markdown("Enter your health metrics below to predict your estimated probability of heart disease or stress levels using Machine Learning models.")
st.markdown("---")

tab1, tab2 = st.tabs(["🫀 Heart Risk", "🧠 Stress Level"])

# ==========================================
# TAB 1: HEART RISK PREDICTION
# ==========================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Info")
        heart_age = st.slider("Age", 20, 100, 50, key="h_age")
        sex_label = st.selectbox("Gender", ["Male", "Female"], key="h_sex")
        sex = 1 if sex_label == "Male" else 0
        
        st.subheader("Blood Vitals")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120, key="h_trestbps")
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, key="h_chol")
        fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], key="h_fbs")
        fbs = 1 if fbs_label == "Yes" else 0

    with col2:
        st.subheader("Exercise Metrics")
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150, key="h_thalach")
        exang_label = st.selectbox("Exercise Induced Angina?", ["No", "Yes"], key="h_exang")
        exang = 1 if exang_label == "Yes" else 0
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 0.0, step=0.1, key="h_oldpeak")

    st.markdown("---")

    if st.button("Predict Heart Risk", type="primary", use_container_width=True, key="btn_heart"):
        if heart_model is None:
            st.error("Heart Model is not available.")
        else:
            input_data = pd.DataFrame([{
                'age': heart_age,
                'trestbps': trestbps,
                'chol': chol,
                'thalach': thalach,
                'oldpeak': oldpeak,
                'sex': sex,
                'fbs': fbs,
                'exang': exang
            }])
            
            probs = heart_model.predict_proba(input_data)[0]
            disease_prob = probs[list(heart_model.classes_).index(1)] if 1 in heart_model.classes_ else probs[1]
            risk_category, color = get_heart_risk_category(disease_prob)
            
            st.subheader("Prediction Results")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(label="Risk Probability", value=f"{disease_prob * 100:.1f}%", delta=risk_category, delta_color="inverse" if risk_category == "High" else "normal")
            with res_col2:
                st.progress(float(disease_prob))
                st.markdown(f"<h3 style='color: {color}; text-align: center;'>{risk_category} Risk</h3>", unsafe_allow_html=True)
                
            st.info(generate_heart_explanation(input_data.iloc[0], risk_category))
            
            with st.expander("View Feature Importance"):
                classifier = heart_model.named_steps['classifier']
                preprocessor = heart_model.named_steps['preprocessor']
                num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                cat_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(['sex', 'fbs', 'exang'])
                feature_names = num_features + list(cat_features_out)
                importances = classifier.feature_importances_
                imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(5)
                st.bar_chart(imp_df.set_index('Feature'))

# ==========================================
# TAB 2: STRESS PREDICTION
# ==========================================
with tab2:
    scol1, scol2 = st.columns(2)

    with scol1:
        st.subheader("Personal & Lifestyle")
        stress_age = st.slider("Age", 18, 100, 30, key="s_age")
        stress_gender = st.selectbox("Gender", ["Male", "Female"], key="s_gender")
        occupation = st.selectbox("Occupation", [
            "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
            "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", "Salesperson", "Manager"
        ], key="s_occ")
        bmi_cat = st.selectbox("BMI Category", ["Normal Weight", "Normal", "Overweight", "Obese"], key="s_bmi")
        
    with scol2:
        st.subheader("Vitals & Activity")
        sleep_dur = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, step=0.1, key="s_sleep_dur")
        sleep_qual = st.slider("Quality of Sleep (1-10)", 1, 10, 7, key="s_sleep_qual")
        phys_act = st.slider("Physical Activity Level", 0, 100, 50, key="s_phys_act")
        heart_rate = st.number_input("Resting Heart Rate", 40, 120, 70, key="s_hr")
        daily_steps = st.number_input("Daily Steps", 0, 30000, 8000, step=500, key="s_steps")
        sys_bp = st.number_input("Systolic BP (e.g., 120)", 80, 200, 120, key="s_sys_bp")
        dia_bp = st.number_input("Diastolic BP (e.g., 80)", 40, 130, 80, key="s_dia_bp")

    st.markdown("---")

    if st.button("Predict Stress Level", type="primary", use_container_width=True, key="btn_stress"):
        if stress_model is None:
            st.error("Stress Model is not available. Please ensure it's copied to the app directory.")
        else:
            input_data = pd.DataFrame([{
                'Age': stress_age,
                'Sleep Duration': sleep_dur,
                'Quality of Sleep': sleep_qual,
                'Physical Activity Level': phys_act,
                'Heart Rate': heart_rate,
                'Daily Steps': daily_steps,
                'Systolic BP': sys_bp,
                'Diastolic BP': dia_bp,
                'Gender': stress_gender,
                'Occupation': occupation,
                'BMI Category': bmi_cat
            }])
            
            try:
                prediction_array = stress_model.predict(input_data)
                stress_level = prediction_array[0]
                color = get_stress_color(stress_level)
                
                st.subheader("Prediction Results")
                st.markdown(f"<h2 style='color: {color}; text-align: center;'>{stress_level}</h2>", unsafe_allow_html=True)
                
                st.info(f"Your predicted stress level is **{stress_level}**. Consistent sleep and physical activity can significantly impact your overall stress.")
                
            except Exception as e:
                st.error(f"Error predicting stress: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This application is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician.")
