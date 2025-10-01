# 💊 Drug Classification System

A machine learning-based system that predicts appropriate drug prescriptions based on patient medical characteristics and demographics.

**Live Demo:** [https://drug-classification-faiz-shaikh.streamlit.app/](https://drug-classification-faiz-shaikh.streamlit.app/)

---

## 🎯 Project Overview

This project implements a comprehensive drug classification system to help healthcare professionals and researchers predict the most suitable drug for patients based on their medical profile. The system analyzes various patient attributes to recommend personalized medication.

---

## 🧠 What We Did

### Data Analysis & Preprocessing
- **Dataset Analysis:** Comprehensive exploration of patient demographics and medical parameters
- **Data Cleaning:** Handled missing values, outliers, and data consistency checks
- **Feature Engineering:** Processed categorical variables (Sex, BP, Cholesterol) and numerical features (Age, Na_to_K ratio)
- **Data Visualization:** Statistical analysis and distribution studies of all features

### Machine Learning Implementation
- **Multiple Algorithms:** Tested and compared Logistic Regression, Decision Trees, and Random Forest classifiers
- **Model Selection:** Random Forest emerged as the best performer with ~95% accuracy
- **Feature Importance:** Identified key factors influencing drug prescription decisions
- **Performance Optimization:** Fine-tuned hyperparameters for optimal prediction accuracy

### Model Performance
- **Accuracy:** Achieved approximately 95% prediction accuracy on test data
- **Confidence Scoring:** Provides probability scores for all possible drug classifications
- **Robust Validation:** Implemented stratified train-test splits and cross-validation techniques

---

## 🔬 Technical Implementation

### Data Features Used
- **Age:** Patient's age in years
- **Sex:** Gender (M/F)
- **BP:** Blood Pressure levels (LOW/NORMAL/HIGH)
- **Cholesterol:** Cholesterol levels (NORMAL/HIGH)
- **Na_to_K:** Sodium to Potassium ratio in blood

### Preprocessing Pipeline
- One-Hot Encoding for categorical variables
- Standard Scaling for numerical features
- Label Encoding for target variable (Drug types)
- Comprehensive data validation and cleaning

### Machine Learning Models
1. **Logistic Regression** - Baseline model
2. **Decision Tree Classifier** - Interpretable model
3. **Random Forest Classifier** - Best performing ensemble method

---

## 📊 Key Findings

- **Random Forest** demonstrated superior performance in drug classification
- **Sodium-Potassium ratio** and **Age** were among the most important features
- The model successfully handles complex non-linear relationships in medical data
- High confidence scores indicate reliable predictions across different patient profiles

---

## 🚀 Business Impact

This system can:
- Assist healthcare providers in making informed drug prescription decisions
- Reduce medication errors through data-driven recommendations
- Provide educational insights into factors influencing drug selection
- Serve as a foundation for more advanced personalized medicine applications

---

## 🛠 Technical Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Model:** Random Forest Classifier

---

## 📈 Model Evaluation

The final Random Forest model provides:
- High accuracy drug predictions
- Confidence scores for transparency
- Feature importance analysis
- Robust performance across diverse patient profiles

---

*This project demonstrates the practical application of machine learning in healthcare decision support systems, showcasing how data-driven approaches can enhance medical prescription accuracy.*
