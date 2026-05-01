# 🩺 Health Risk Prediction System (Streamlit)

A fast, clean, and production-ready Machine Learning web application built with Streamlit. This application allows users to predict their **Heart Disease Risk** and **Stress Levels** using custom-trained Random Forest models.

## 🚀 Features

- **🫀 Heart Risk Prediction:** Predicts the probability of heart disease based on 8 key health metrics (Age, Gender, Blood Pressure, Cholesterol, Max Heart Rate, Fasting Blood Sugar, Exercise-Induced Angina, and ST Depression). Features interactive risk gauges and dynamic rule-based health interpretations.
- **🧠 Stress Level Prediction:** Classifies user stress into Low, Moderate, or High based on 11 lifestyle and biometric inputs including sleep patterns, physical activity, and daily steps.
- **📊 Feature Importance:** Explains *why* the model made a decision by visualizing the most impactful features for the prediction.

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (Frontend/UI)
- **Scikit-Learn** (Machine Learning Pipeline)
- **Pandas & Numpy** (Data Manipulation)
- **Joblib** (Model Serialization)

## ⚙️ Local Setup

Follow these instructions to run the application locally:

### 1. Clone the repository
```bash
git clone https://github.com/ShallyKaushik/HeartRiskPrediction.git
cd HeartRiskPrediction
```

### 2. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Launch the Streamlit server:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

## 📂 Project Structure

```text
.
├── app.py                  # Main Streamlit application
├── train_model.py          # Script used to train the Heart Risk model
├── heart_model.joblib      # Serialized ML pipeline for Heart Risk
├── stress_model.joblib     # Serialized ML pipeline for Stress Levels
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚠️ Disclaimer
This application is for educational and portfolio showcase purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult a healthcare professional for medical concerns.
