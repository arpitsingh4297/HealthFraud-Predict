import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Healthcare Provider Fraud Detection", layout="wide")

# Title and description
st.title("Healthcare Provider Fraud Detection")
st.markdown("""
This app predicts whether a healthcare provider is potentially fraudulent based on input features.
Enter the values below, and the model will provide a prediction along with the probability of fraud.
""")

# Load model and selected features
try:
    model = joblib.load('best_fraud_detection_model.joblib')
    with open('selected_features.json', 'r') as f:
        selected_features = json.load(f)
except FileNotFoundError:
    st.error("Model or features file not found. Ensure 'best_fraud_detection_model.joblib' and 'selected_features.json' are in the same directory.")
    st.stop()

# Load or recreate scaler (assuming same training data statistics)
# Since we don't have the original scaler, we approximate it with reasonable ranges based on training data
scaler = StandardScaler()
# Approximate mean and std based on outlier analysis from training data
feature_stats = {
    'Age': {'mean': 72.61, 'std': 13.81},  # From outlier output
    'Claims_Per_Provider': {'mean': 573.93, 'std': 616.34},
    'Avg_Claim_Amt': {'mean': 8230.37, 'std': 5368.01},
    'Chronic_Score': {'mean': 17.37, 'std': 2.53},
    'InscClaimAmtReimbursed': {'mean': 607009.3, 'std': 559191.4}
}
scaler.mean_ = np.array([feature_stats[feat]['mean'] for feat in selected_features])
scaler.scale_ = np.array([feature_stats[feat]['std'] for feat in selected_features])

# Input form
st.header("Input Provider Details")
st.markdown("Enter values for the following features. Suggested ranges are provided based on training data.")
inputs = {}
with st.form(key='input_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Age'] = st.number_input(
            'Average Patient Age (years)', 
            min_value=0.0, max_value=120.0, value=73.0, 
            help="Range from training: ~33 to 99 (mean: 72.61)"
        )
        inputs['Claims_Per_Provider'] = st.number_input(
            'Number of Claims per Provider', 
            min_value=0.0, value=100.0, 
            help="Range from training: ~1 to 8240 (mean: 573.93)"
        )
        inputs['Avg_Claim_Amt'] = st.number_input(
            'Average Claim Amount ($)', 
            min_value=0.0, value=5000.0, 
            help="Range from training: ~3377 to 57000 (mean: 8230.37)"
        )
    
    with col2:
        inputs['Chronic_Score'] = st.number_input(
            'Chronic Condition Score', 
            min_value=0.0, max_value=30.0, value=5.0, 
            help="Range from training: ~12 to 22 (mean: 17.37)"
        )
        inputs['InscClaimAmtReimbursed'] = st.number_input(
            'Total Claim Amount Reimbursed ($)', 
            min_value=0.0, value=500000.0, 
            help="Range from training: ~207970 to 5996050 (mean: 607009.3)"
        )
    
    submit_button = st.form_submit_button(label="Predict")

# Process prediction
if submit_button:
    # Input validation
    valid = True
    for feature, value in inputs.items():
        if value < 0:
            st.error(f"{feature} cannot be negative.")
            valid = False
        elif feature == 'Age' and value > 120:
            st.error("Age seems unrealistic (>120 years). Please check.")
            valid = False
        elif feature == 'Chronic_Score' and value > 30:
            st.error("Chronic Score seems high (>30). Please check.")
            valid = False
    
    if valid:
        # Create DataFrame
        input_df = pd.DataFrame([inputs])
        
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=selected_features)
        
        # Make prediction
        try:
            prediction = model.predict(input_scaled_df)
            probability = model.predict_proba(input_scaled_df)[0][1] if hasattr(model, 'predict_proba') else None
            
            # Display prediction
            st.header("Prediction Result")
            st.write(f"**Prediction**: {'Fraudulent' if prediction[0] == 1 else 'Non-Fraudulent'}")
            if probability is not None:
                st.write(f"**Probability of Fraud**: {probability:.4f}")
            else:
                st.write("Probability not available for this model.")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Sidebar with additional info
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a Naive Bayes model trained on healthcare provider data to detect potential fraud.
The model was trained on features like average claim amount, number of claims, and patient demographics.
Key insights from training:
- Fraudulent providers often have higher claim amounts and more frequent claims.
- Older patients are more associated with fraudulent providers.
""")

st.sidebar.header("Model Details")
st.sidebar.markdown(f"""
- **Model**: NaiveBayes_Original_Tuned
- **Test F1-Score**: 0.605
- **Features Used**: {', '.join(selected_features)}
- **Data Source**: Healthcare Provider Fraud Detection Dataset
""")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter values for each feature in the form.
2. Use the suggested ranges to ensure realistic inputs.
3. Click 'Predict' to see the prediction result and fraud probability.
""")