import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Drug Classification System",
    page_icon="💊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        border-left: 5px solid #1f77b4;
    }
    .drug-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Load saved models
@st.cache_resource
def load_models():
    model = joblib.load('drug_classification_model.pkl')
    encoder = joblib.load('onehot_encoder.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    return model, encoder, label_encoder, scaler


try:
    model, onehot_encoder, label_encoder, scaler = load_models()

    # Drug information dictionary
    drug_info = {
        'DrugY': "General purpose medication for common conditions",
        'drugC': "Used for specific cardiovascular conditions",
        'drugX': "Standard medication for general ailments",
        'drugA': "Specialized medication for specific cases",
        'drugB': "Alternative treatment option"
    }

    # Main app
    st.markdown('<div class="main-header">💊 Drug Classification System</div>', unsafe_allow_html=True)

    st.write("""
    This system predicts the appropriate drug based on patient characteristics.
    Please enter the patient details below:
    """)

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=15, max_value=80, value=45, help="Patient's age in years")
        sex = st.selectbox("Sex", options=["M", "F"])
        bp = st.selectbox("Blood Pressure", options=["LOW", "NORMAL", "HIGH"])

    with col2:
        cholesterol = st.selectbox("Cholesterol Level", options=["NORMAL", "HIGH"])
        na_to_k = st.number_input("Sodium to Potassium Ratio", min_value=0.0, max_value=50.0, value=15.0, step=0.1,
                                  help="Ratio of Sodium to Potassium in blood")

    # Prediction button
    if st.button("Predict Drug", type="primary"):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'BP': [bp],
                'Cholesterol': [cholesterol],
                'Na_to_K': [na_to_k]
            })

            # Preprocess input data - FIXED VARIABLE NAME
            categorical_cols = ['Sex', 'BP', 'Cholesterol']
            encoded_array = onehot_encoder.transform(input_data[categorical_cols])
            encoded_df = pd.DataFrame(encoded_array,
                                      columns=onehot_encoder.get_feature_names_out(categorical_cols),
                                      index=input_data.index)

            numeric_cols = ['Age', 'Na_to_K']
            input_processed = pd.concat([input_data[numeric_cols], encoded_df],
                                        axis=1)  # FIXED: input_processed instead of input_processor

            # Apply scaling
            input_processed[["Age", "Na_to_K"]] = scaler.transform(input_processed[["Age", "Na_to_K"]])

            # Make prediction
            prediction_encoded = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0]

            # Decode prediction
            predicted_drug = label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = prediction_proba[prediction_encoded] * 100

            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("🎯 Prediction Result")

            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                st.metric(
                    label="Recommended Drug",
                    value=predicted_drug,
                    delta=f"{confidence:.1f}% confidence"
                )

            with col_res2:
                st.write(f"**Drug Information:** {drug_info.get(predicted_drug, 'Information not available')}")

            # Confidence scores for all drugs
            st.subheader("📊 Confidence Scores")
            proba_df = pd.DataFrame({
                'Drug': label_encoder.classes_,
                'Confidence (%)': (prediction_proba * 100).round(2)
            }).sort_values('Confidence (%)', ascending=False)

            st.dataframe(proba_df, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This machine learning model predicts drug prescriptions based on:
        - Patient demographics
        - Blood pressure levels
        - Cholesterol levels
        - Sodium-Potassium ratio

        **Model:** Random Forest Classifier
        **Accuracy:** ~95% (on test data)
        """)

        st.header("📈 Feature Importance")

        # Get feature names from the encoder
        categorical_features = onehot_encoder.get_feature_names_out(['Sex', 'BP', 'Cholesterol'])
        numeric_features = ['Age', 'Na_to_K']
        all_features = numeric_features + categorical_features.tolist()

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        st.bar_chart(importance_df.set_index('Feature')['Importance'])

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please make sure all .pkl files are in the same directory as this app.")