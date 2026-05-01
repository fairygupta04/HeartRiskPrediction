import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediPredict",
    page_icon="🩺",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* REMOVE SIDEBAR */
    [data-testid="stSidebar"] {
        display: none;
    }

    .hero-banner {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    .hero-banner h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
    }

    .hero-banner p {
        color: #ccc;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_heart_model():
    return joblib.load('heart_model.joblib')

@st.cache_resource
def load_stress_model():
    return joblib.load('stress_model.joblib')

heart_model = load_heart_model()
stress_model = load_stress_model()

# --- HERO SECTION ---
st.markdown("""
<div class='hero-banner'>
    <h1>Health Risk Prediction</h1>
    <p>Enter your health details to check your health risk.</p>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["Heart Risk", "Stress Level"])

# ================= HEART =================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 0

        trestbps = st.number_input("Blood Pressure", 90, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)

    with col2:
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0)

        fbs = st.selectbox("High Blood Sugar?", ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0

        exang = st.selectbox("Exercise Angina?", ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Risk"):
        data = pd.DataFrame([{
            'age': age,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'oldpeak': oldpeak,
            'sex': sex,
            'fbs': fbs,
            'exang': exang
        }])

        prob = heart_model.predict_proba(data)[0][1]
        st.success(f"Risk Probability: {prob*100:.2f}%")

# ================= STRESS =================
with tab2:
    age = st.slider("Age", 18, 100, 30)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    activity = st.slider("Activity Level", 0, 100, 50)

    if st.button("Predict Stress"):
        data = pd.DataFrame([{
            'Age': age,
            'Sleep Duration': sleep,
            'Physical Activity Level': activity
        }])

        result = stress_model.predict(data)[0]
        st.success(f"Stress Level: {result}")