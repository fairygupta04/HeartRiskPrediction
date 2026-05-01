import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ── Main Content Text ── */
    .stApp, .stMarkdown, .stText, label, p, div {
        color: #e0e0e0;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(255,77,106,0.15), rgba(99,102,241,0.15));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .hero-banner h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4d6a, #a855f7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-banner p {
        font-size: 1rem;
        color: #b0b0c8 !important;
        max-width: 600px;
        margin: 0 auto;
    }

    /* ── Card Containers ── */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #a78bfa !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Section Divider ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        margin: 1.5rem 0;
    }

    /* ── Result Cards ── */
    .result-card {
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .result-low  { background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1)); border-color: rgba(16,185,129,0.4); }
    .result-mod  { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(217,119,6,0.1));  border-color: rgba(245,158,11,0.4); }
    .result-high { background: linear-gradient(135deg, rgba(239,68,68,0.2),  rgba(185,28,28,0.1));  border-color: rgba(239,68,68,0.4);  }

    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.4rem 0;
    }
    .result-sub {
        font-size: 0.9rem;
        color: #b0b0c8 !important;
    }

    /* ── Metric Pills ── */
    .metric-pill {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 50px;
        padding: 0.3rem 1rem;
        font-size: 1.6rem;
        font-weight: 700;
        color: #a5b4fc !important;
        margin-bottom: 0.5rem;
    }

    /* ── Progress Bar Override ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ff4d6a) !important;
        border-radius: 10px !important;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        height: 10px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.08);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9ca3af !important;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"]    { display: none; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(168,85,247,0.5);
        background: linear-gradient(135deg, #4f46e5, #9333ea);
    }

    /* ── Inputs / Sliders / Selects ── */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e0e0e0 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        color: #a78bfa !important;
        font-weight: 600;
    }

    /* ── Info / Warning boxes ── */
    .stAlert {
        background: rgba(99,102,241,0.12) !important;
        border-left: 4px solid #6366f1 !important;
        border-radius: 10px !important;
        color: #c4b5fd !important;
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        color: #6b7280 !important;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
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

heart_model  = load_heart_model()
stress_model = load_stress_model()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS  (ML logic unchanged)
# ─────────────────────────────────────────────
def get_heart_risk_category(prob):
    if prob < 0.3:
        return "Low", "low"
    elif prob < 0.6:
        return "Moderate", "mod"
    else:
        return "High", "high"

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

def get_stress_color_class(stress_level):
    mapping = {'Low Stress': 'low', 'Moderate Stress': 'mod', 'High Stress': 'high'}
    return mapping.get(stress_level, 'mod')

RISK_EMOJI = {"low": "💚", "mod": "🟡", "high": "❤️"}
RISK_COLOR = {"low": "#10b981", "mod": "#f59e0b", "high": "#ef4444"}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:3rem;'>🩺</div>
        <div style='font-size:1.2rem; font-weight:700; color:#a78bfa;'>Health Risk AI</div>
        <div style='font-size:0.8rem; color:#6b7280; margin-top:0.3rem;'>Powered by Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='padding: 0.5rem 0;'>
        <div style='color:#a78bfa; font-weight:600; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;'>📋 Features</div>
        <div style='color:#c4b5fd; font-size:0.88rem; line-height:2;'>
            🫀 Heart Disease Risk<br>
            🧠 Stress Level Analysis<br>
            📊 Feature Importance<br>
            🎯 Real-time Predictions
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='padding: 0.5rem 0;'>
        <div style='color:#a78bfa; font-weight:600; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;'>⚙️ Models</div>
        <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.5rem; font-size:0.82rem; color:#6ee7b7;'>
            ✅ Random Forest — Heart
        </div>
        <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:0.5rem 0.8rem; font-size:0.82rem; color:#6ee7b7;'>
            ✅ Random Forest — Stress
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#4b5563; text-align:center; padding-top:0.5rem;'>
        ⚠️ For educational use only.<br>Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <h1>🩺 Health Risk Prediction System</h1>
    <p>Enter your health metrics to receive an AI-powered assessment of your heart disease risk and stress levels using trained Machine Learning models.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🫀  Heart Risk Prediction", "🧠  Stress Level Prediction"])


# ══════════════════════════════════════════════
# TAB 1 — HEART RISK PREDICTION
# ══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='card'><div class='card-title'>👤 Basic Information</div>", unsafe_allow_html=True)
        heart_age  = st.slider("Age", 20, 100, 50, key="h_age")
        sex_label  = st.selectbox("Gender", ["Male", "Female"], key="h_sex")
        sex        = 1 if sex_label == "Male" else 0
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>🩸 Blood Vitals</div>", unsafe_allow_html=True)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120, key="h_trestbps")
        chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, key="h_chol")
        fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], key="h_fbs")
        fbs = 1 if fbs_label == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><div class='card-title'>🏃 Exercise Metrics</div>", unsafe_allow_html=True)
        thalach    = st.slider("Max Heart Rate Achieved", 60, 220, 150, key="h_thalach")
        exang_label = st.selectbox("Exercise Induced Angina?", ["No", "Yes"], key="h_exang")
        exang      = 1 if exang_label == "Yes" else 0
        oldpeak    = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 0.0, step=0.1, key="h_oldpeak")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict_heart = st.button("🔍  Predict Heart Risk", type="primary", use_container_width=True, key="btn_heart")

    if predict_heart:
        if heart_model is None:
            st.error("Heart Model is not available.")
        else:
            input_data = pd.DataFrame([{
                'age': heart_age, 'trestbps': trestbps, 'chol': chol,
                'thalach': thalach, 'oldpeak': oldpeak,
                'sex': sex, 'fbs': fbs, 'exang': exang
            }])
            probs        = heart_model.predict_proba(input_data)[0]
            disease_prob = probs[list(heart_model.classes_).index(1)] if 1 in heart_model.classes_ else probs[1]
            risk_category, css_cls = get_heart_risk_category(disease_prob)

            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("#### 📊 Prediction Results")

            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                st.markdown(f"""
                <div style='text-align:center;'>
                    <div style='font-size:0.8rem; color:#9ca3af; margin-bottom:0.3rem;'>Risk Probability</div>
                    <div class='metric-pill'>{disease_prob * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                emoji = RISK_EMOJI[css_cls]
                color = RISK_COLOR[css_cls]
                st.markdown(f"""
                <div style='text-align:center;'>
                    <div style='font-size:0.8rem; color:#9ca3af; margin-bottom:0.3rem;'>Risk Category</div>
                    <div style='font-size:1.6rem; font-weight:700; color:{color};'>{emoji} {risk_category}</div>
                </div>
                """, unsafe_allow_html=True)
            with r3:
                st.markdown("<div style='padding-top:1.5rem;'>", unsafe_allow_html=True)
                st.progress(float(disease_prob))
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='result-card result-{css_cls}'>
                <div class='result-sub'>Analysis Summary</div>
                <div class='result-label' style='color:{RISK_COLOR[css_cls]};'>{RISK_EMOJI[css_cls]} {risk_category} Risk</div>
                <div class='result-sub'>{generate_heart_explanation(input_data.iloc[0], risk_category)}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📊 View Feature Importance"):
                classifier  = heart_model.named_steps['classifier']
                preprocessor = heart_model.named_steps['preprocessor']
                num_features     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                cat_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(['sex', 'fbs', 'exang'])
                feature_names    = num_features + list(cat_features_out)
                importances      = classifier.feature_importances_
                imp_df = (pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                          .sort_values(by='Importance', ascending=False).head(5))
                st.bar_chart(imp_df.set_index('Feature'))


# ══════════════════════════════════════════════
# TAB 2 — STRESS LEVEL PREDICTION
# ══════════════════════════════════════════════
with tab2:
    scol1, scol2 = st.columns(2, gap="large")

    with scol1:
        st.markdown("<div class='card'><div class='card-title'>👤 Personal & Lifestyle</div>", unsafe_allow_html=True)
        stress_age    = st.slider("Age", 18, 100, 30, key="s_age")
        stress_gender = st.selectbox("Gender", ["Male", "Female"], key="s_gender")
        occupation    = st.selectbox("Occupation", [
            "Software Engineer", "Doctor", "Sales Representative", "Teacher",
            "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", "Salesperson", "Manager"
        ], key="s_occ")
        bmi_cat = st.selectbox("BMI Category", ["Normal Weight", "Normal", "Overweight", "Obese"], key="s_bmi")
        st.markdown("</div>", unsafe_allow_html=True)

    with scol2:
        st.markdown("<div class='card'><div class='card-title'>💓 Vitals & Activity</div>", unsafe_allow_html=True)
        sleep_dur  = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, step=0.1, key="s_sleep_dur")
        sleep_qual = st.slider("Quality of Sleep (1–10)", 1, 10, 7, key="s_sleep_qual")
        phys_act   = st.slider("Physical Activity Level (0–100)", 0, 100, 50, key="s_phys_act")
        heart_rate = st.number_input("Resting Heart Rate (bpm)", 40, 120, 70, key="s_hr")
        daily_steps = st.number_input("Daily Steps", 0, 30000, 8000, step=500, key="s_steps")
        sys_bp  = st.number_input("Systolic BP (e.g. 120)", 80, 200, 120, key="s_sys_bp")
        dia_bp  = st.number_input("Diastolic BP (e.g. 80)", 40, 130, 80, key="s_dia_bp")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    btn_col2, _ = st.columns([1, 2])
    with btn_col2:
        predict_stress = st.button("🔍  Predict Stress Level", type="primary", use_container_width=True, key="btn_stress")

    if predict_stress:
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
                stress_level = stress_model.predict(input_data)[0]
                css_cls      = get_stress_color_class(stress_level)

                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                st.markdown("#### 📊 Prediction Results")

                st.markdown(f"""
                <div class='result-card result-{css_cls}'>
                    <div class='result-sub'>Stress Assessment</div>
                    <div class='result-label' style='color:{RISK_COLOR[css_cls]};'>{RISK_EMOJI[css_cls]} {stress_level}</div>
                    <div class='result-sub'>Consistent sleep and physical activity can significantly impact your overall stress.</div>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"Your predicted stress level is **{stress_level}**. Consistent sleep and physical activity can significantly impact your overall stress.")
            except Exception as e:
                st.error(f"Error predicting stress: {e}")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='footer-text'>
    ⚠️ <strong>Disclaimer:</strong> This application is for educational purposes only and is not a substitute
    for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician.
</div>
""", unsafe_allow_html=True)