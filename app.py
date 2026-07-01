import streamlit as st
import pandas as pd
import joblib


# Load Model, Scaler & Columns
model = joblib.load("random_forest_creditcard.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file containing transaction data to detect fraudulent transactions.")
# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Predict"):

        test_df = df.copy()

        # Keep only required columns
        test_df = test_df[columns]

        # Scale Amount column
        if "Amount" in test_df.columns:
            test_df["Amount"] = scaler.transform(test_df[["Amount"]])

        # Prediction
        predictions = model.predict(test_df)

        # Probability (optional)
        probabilities = model.predict_proba(test_df)[:, 1]

        # Add Results
        result_df = df.copy()
        result_df["Prediction"] = predictions
        result_df["Prediction"] = result_df["Prediction"].map(
            {0: "Normal", 1: "Fraud"}
        )

        result_df["Fraud Probability"] = probabilities.round(4)
        # Show Prediction Results
        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True)
        # Fraud Transactions
        fraud_df = result_df[result_df["Prediction"] == "Fraud"]

        st.subheader("Detected Fraud Transactions")

        if len(fraud_df) > 0:
            st.dataframe(fraud_df, use_container_width=True)
        else:
            st.success("No Fraud Transactions Detected.")
        # Summary
        normal_count = (result_df["Prediction"] == "Normal").sum()
        fraud_count = (result_df["Prediction"] == "Fraud").sum()

        st.markdown("## Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"✅ Normal Transactions: {normal_count}")

        with col2:
            st.error(f"🚨 Fraud Transactions: {fraud_count}")
        # Download Results
        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv",
        )