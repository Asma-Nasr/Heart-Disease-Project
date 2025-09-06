import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Loading the trained model and scaler
@st.cache_resource
def load_model():
    return joblib.load('../results/model_pipeline.pkl')

model = load_model()

st.title("Heart Disease Prediction App")

# User input for health data
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=200, value=100)
ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
oldpeak = st.number_input("Oldpeak (depression induced by exercise relative to rest)", min_value=-2.0, max_value=6.0, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# Create input DataFrame for prediction
input_data = pd.DataFrame({
    'age': [age],
    'cp': [cp],
    'thalach': [thalach],
    'oldpeak': [oldpeak],
    'ca': [ca]
})

# real-time prediction output
if st.button("Predict"):
    with st.spinner("Calculating..."):
        prediction_proba = model.predict_proba(input_data)[0]
        probability = prediction_proba[1]  # Probability of the positive class (1)
        prediction = 1 if probability >= 0.5 else 0
        risk_level = "No Risk" if prediction == 0 else "High Risk"
        color = "green" if prediction == 0 else "red"
        
        # Display prediction with color
        st.markdown(f"<h3 style='color: {color};'>{risk_level}</h3>", unsafe_allow_html=True)
        st.write(f"Probability of High Risk: {probability:.2f}")
