import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

print("--- Starting Streamlit App ML Training ---")

# 1. Load Data
data_path = '../heart_data.csv'
if not os.path.exists(data_path):
    print(f"❌ Error: '{data_path}' not found.")
    exit(1)

data = pd.read_csv(data_path)
print("✅ Data loaded successfully.")

# 2. Define Features exactly as requested for the Streamlit app
# User requested: Age, Gender, Resting Blood Pressure, Cholesterol, Max Heart Rate, 
# Fasting Blood Sugar, Exercise Induced Angina, ST Depression
NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'fbs', 'exang']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'target'

X = data[FEATURES]
y = data[TARGET]

# 3. Create Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ]
)

# 4. Create Full Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy on these 8 features: {accuracy * 100:.2f}%")

# 7. Save Model
model_file = 'heart_model.joblib'
joblib.dump(pipeline, model_file)
print(f"✅ Model saved to {model_file}")
