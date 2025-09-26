import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = 'final_model.pkl'
SCALER_PATH = 'scaler.pkl'

def train_and_save_final_model():
    """
    Loads all data, preprocesses it, trains the definitive model,
    and saves the model and scaler to disk.
    """
    with st.spinner('Training the model... This may take a moment.'):
        try:
            df_usa = pd.read_csv('data_usa.csv')
            df_japan = pd.read_csv('data_japan.csv')
        except FileNotFoundError as e:
            st.error(f"Data file not found: {e}. Please ensure 'data_usa.csv' and 'data_japan.csv' are present.")
            return

        # --- 特徴量エンジニアリング（client.pyと完全に一致させる） ---
        df_japan['Annual_Income(USD)'] = df_japan['Annual_Income(JPY)'] / 145.0
        df_japan['Savings'] = df_japan['Saving(JPY)'] / 145.0
        if 'FICO_Score' not in df_japan.columns:
            df_japan['FICO_Score'] = 0

        df_usa.rename(columns={'Saving(USD)': 'Savings'}, inplace=True)
        df_combined = pd.concat([df_usa, df_japan], ignore_index=True)

        if 'FICO_Score' not in df_combined.columns:
            df_combined['FICO_Score'] = 0
        df_combined['FICO_Score'] = df_combined['FICO_Score'] ** 3
        df_combined['Loan_Status'] = df_combined['Loan_Status'].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0)
        df_combined['income_per_service'] = df_combined['Annual_Income(USD)'] / (df_combined['Years_of_Service'] + 1)
        df_combined['estimated_asset_score'] = df_combined['Annual_Income(USD)'] * df_combined['Years_of_Service']
        df_combined['total_financial_power'] = (df_combined['Annual_Income(USD)'] + df_combined['Savings']) ** 2 # 【重要】新しい特徴量を計算

        # 【重要】学習に使う特徴量リストをclient.pyと完全に一致させる
        features_df = df_combined[[
            'Age', 'Annual_Income(USD)', 'Years_of_Service', 'Loan_Status', 'FICO_Score',
            'income_per_service', 'estimated_asset_score', 'Savings', 'total_financial_power'
        ]]
        target = df_combined['Payment_Delay']

        # スケーラーを新しい特徴量で fit する
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)

        final_model = LogisticRegression(
            solver='saga', max_iter=1000, class_weight='balanced', random_state=42
        )
        final_model.fit(X_scaled, target)

        joblib.dump(final_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    st.success("✅ Model and scaler have been trained and saved successfully!")
    st.info("Reloading the app...")

    
def run_app():
    st.set_page_config(page_title="Global Credit Score Predictor", layout="wide")
    st.title("🌍 Global Credit Score Predictor")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.warning("Welcome! The prediction model is not yet available.")
        st.info("Please train the model to begin.")
        if st.button("Train and Initialize Model"):
            try:
                train_and_save_final_model()
                st.rerun()
            except Exception as e:
                st.error("An error occurred during model training.")
                st.exception(e)
        return

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return

    st.header("Check Your Credit Score Health")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=19)
        currency = st.selectbox("Currency", ("USD", "JPY"))
        income_label = f"Annual Income ({currency})"
        annual_income_input = st.number_input(income_label, min_value=0, value=60000 if currency == "USD" else 12000000)
        savings_label = f"Total Savings ({currency})"
        savings_input = st.number_input(savings_label, min_value=0, value=20000 if currency == "USD" else 60000000)

    with col2:
        years_of_service = st.number_input("Years of Service", min_value=0, max_value=50, value=1)
        loan_status = st.selectbox("Has Existing Loan?", ("No", "Yes"), index=0)
        fico_score = st.number_input("FICO Score (if available)", min_value=300, max_value=850, value=750)

    if st.button("Predict Payment Delay Risk", type="primary"):
        JPY_TO_USD_RATE = 145.0
        if currency == "JPY":
            annual_income_usd = annual_income_input / JPY_TO_USD_RATE
            savings_usd = savings_input / JPY_TO_USD_RATE
        else:
            annual_income_usd = annual_income_input
            savings_usd = savings_input

        loan_status_numeric = 1 if loan_status == "Yes" else 0
        income_per_service = annual_income_usd / (years_of_service + 1)
        estimated_asset_score = annual_income_usd * years_of_service
        total_financial_power = (annual_income_usd + savings_usd) ** 2

        fico_score = fico_score ** 3

        input_data = pd.DataFrame([[
            age, annual_income_usd, years_of_service, loan_status_numeric, fico_score,
            income_per_service, estimated_asset_score, savings_usd, total_financial_power
        ]], columns=[
            'Age', 'Annual_Income(USD)', 'Years_of_Service', 'Loan_Status', 'FICO_Score',
            'income_per_service', 'estimated_asset_score', 'Savings', 'total_financial_power'
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("High Risk of Payment Delay")
        else:
            st.success("Low Risk of Payment Delay")
        
        st.metric(label="Risk Score (Probability of Delay)", value=f"{prediction_proba[0][1]:.2%}")
        st.bar_chart(pd.DataFrame(prediction_proba, columns=["Low Risk", "High Risk"]).T)

if __name__ == "__main__":
    run_app()
