import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Healthcare Provider Fraud Detection", layout="wide")

# Title and description
st.title("Healthcare Provider Fraud Detection")
st.markdown("""
This app predicts whether a healthcare provider is potentially fraudulent based on input features.
Enter the values below, and the model will provide a prediction along with the probability of fraud (if available).
**Note**: Claims_Per_Provider and InscClaimAmtReimbursed are log-transformed (use ln(1 + value)).
""")

# Load model, scaler, and selected features
try:
    model = joblib.load('best_fraud_detection_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('selected_features.json', 'r') as f:
        selected_features = json.load(f)
except FileNotFoundError:
    st.error("Required files not found. Ensure 'best_fraud_detection_model.joblib', 'scaler.joblib', and 'selected_features.json' are in the same directory.")
    st.stop()

# Verify feature consistency
expected_features = ['Age', 'Claims_Per_Provider', 'Avg_Claim_Amt', 'Chronic_Score', 'InscClaimAmtReimbursed']
if selected_features != expected_features:
    st.error(f"Feature mismatch! Expected features: {expected_features}, but found: {selected_features}. Please retrain the model with the correct features.")
    st.stop()

# Define feature ranges (adjusted for log-transformed features)
feature_ranges = {
    'Age': (33.0, 99.0),
    'Claims_Per_Provider': (0.0, 9.02),  # log1p(1) to log1p(8240)
    'Avg_Claim_Amt': (3377.0, 57000.0),
    'Chronic_Score': (12.0, 22.0),
    'InscClaimAmtReimbursed': (12.24, 15.61)  # log1p(207970) to log1p(5996050)
}

# Feature descriptions for user guidance
feature_descriptions = {
    'Age': "Average patient age in years",
    'Claims_Per_Provider': "Log-transformed number of claims submitted by the provider (ln(1 + claims))",
    'Avg_Claim_Amt': "Average claim amount in dollars",
    'Chronic_Score': "Sum of chronic conditions per patient",
    'InscClaimAmtReimbursed': "Log-transformed total claim amount reimbursed in dollars (ln(1 + amount))"
}

# Input form
st.header("Input Provider Details")
st.markdown(f"Enter values for the following features: {', '.join(selected_features)}. Suggested ranges are based on training data. Use log-transformed values for Claims_Per_Provider and InscClaimAmtReimbursed.")
inputs = {}
with st.form(key='input_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Age'] = st.number_input(
            'Age',
            min_value=0.0, max_value=120.0, value=73.0,
            help=f"{feature_descriptions['Age']}. Range: {feature_ranges['Age'][0]} to {feature_ranges['Age'][1]}"
        )
        inputs['Claims_Per_Provider'] = st.number_input(
            'Claims_Per_Provider (log-transformed)',
            min_value=0.0, value=4.61,  # log1p(100)
            help=f"{feature_descriptions['Claims_Per_Provider']}. Range: {feature_ranges['Claims_Per_Provider'][0]} to {feature_ranges['Claims_Per_Provider'][1]}"
        )
        inputs['Avg_Claim_Amt'] = st.number_input(
            'Avg_Claim_Amt',
            min_value=0.0, value=5000.0,
            help=f"{feature_descriptions['Avg_Claim_Amt']}. Range: {feature_ranges['Avg_Claim_Amt'][0]} to {feature_ranges['Avg_Claim_Amt'][1]}"
        )
    
    with col2:
        inputs['Chronic_Score'] = st.number_input(
            'Chronic_Score',
            min_value=0.0, max_value=30.0, value=15.0,
            help=f"{feature_descriptions['Chronic_Score']}. Range: {feature_ranges['Chronic_Score'][0]} to {feature_ranges['Chronic_Score'][1]}"
        )
        inputs['InscClaimAmtReimbursed'] = st.number_input(
            'InscClaimAmtReimbursed (log-transformed)',
            min_value=0.0, value=12.61,  # log1p(300000)
            help=f"{feature_descriptions['InscClaimAmtReimbursed']}. Range: {feature_ranges['InscClaimAmtReimbursed'][0]} to {feature_ranges['InscClaimAmtReimbursed'][1]}"
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
        elif feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            if value < min_val or value > max_val:
                st.warning(f"{feature} is outside training range ({min_val} to {max_val}). Capping to fit range.")
                inputs[feature] = np.clip(value, min_val, max_val)
        if feature == 'Age' and value > 120:
            st.error("Age seems unrealistic (>120 years). Please check.")
            valid = False
        elif feature == 'Chronic_Score' and value > 30:
            st.error("Chronic Score seems high (>30). Please check.")
            valid = False
    
    if valid:
        # Create DataFrame
        input_df = pd.DataFrame([inputs], columns=selected_features)
        
        # Scale inputs
        try:
            input_scaled = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled, columns=selected_features)
        except ValueError as e:
            st.error(f"Error scaling inputs: {str(e)}. Ensure the scaler was trained with the features: {selected_features}.")
            st.stop()
        
        # Check for extreme scaled values
        extreme_scaled = False
        for i, (feature, scaled_val) in enumerate(zip(selected_features, input_scaled[0])):
            if abs(scaled_val) > 5:
                st.warning(f"{feature} has an extreme scaled value ({scaled_val:.2f}). Predictions may be unreliable.")
                extreme_scaled = True
        
        # Make prediction
        try:
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else None
            
            # Debug output
            st.write("**Debug Info** (for verification):")
            st.write(f"Raw inputs: {input_df.values[0]}")
            st.write(f"Scaled inputs: {input_scaled[0]}")
            if probability is not None:
                st.write(f"Prediction probabilities: {model.predict_proba(input_scaled)[0]}")
            
            # Display prediction
            st.header("Prediction Result")
            st.write(f"**Prediction**: {'Fraudulent' if prediction[0] == 1 else 'Non-Fraudulent'}")
            if probability is not None:
                st.write(f"**Probability of Fraud**: {probability:.4f}")
            else:
                st.write("**Note**: Probability not available for this model type.")
            if extreme_scaled:
                st.warning("Warning: Prediction may be unreliable due to extreme input values.")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Sidebar with additional info
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a machine learning model trained on healthcare provider data to detect potential fraud.
The model was selected from candidates (e.g., Naive Bayes, SVM, AdaBoost) based on F1-score.
Key insights:
- Fraudulent providers have higher claim amounts and more frequent claims.
- Older patients are more associated with fraudulent providers.
- Clusters identify high-risk providers for audits.
""")

st.sidebar.header("Model Details")
st.sidebar.markdown(f"""
- **Model**: {best_model_name if 'best_model_name' in locals() else 'Tuned Model (type varies)'}
- **Test F1-Score**: ~0.6 or higher
- **Features Used**: {', '.join(selected_features)}
- **Data Source**: Healthcare Provider Fraud Detection Dataset
""")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter values for each feature in the form.
2. Use suggested ranges. Claims_Per_Provider and InscClaimAmtReimbursed are log-transformed (ln(1 + value)).
3. Click 'Predict' to see the prediction result and fraud probability.
4. Check debug info for verification.
""")

st.sidebar.header("Training Insights")
st.sidebar.markdown("""
- **EDA**: Imbalanced dataset (~90.6% non-fraudulent). Fraudulent providers have higher claim amounts and claims.
- **Clustering**: High-risk clusters show higher fraud probability.
- **Recommendations**: Audit high-risk providers and implement real-time detection.
""")