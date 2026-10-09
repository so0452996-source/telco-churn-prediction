# Telecom Customer Churn Prediction

A Streamlit app for predicting telecom customer churn with Logistic Regression, K-Nearest Neighbors, and Naive Bayes.

## Features
- **Single Customer:** enter customer details and estimate churn probability.
- **Batch Prediction:** upload a CSV and download predictions.
- **Model Insights:** compare model metrics and explore churn patterns.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

The dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` must remain in the same folder as `app.py`.

## Deploy to Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `WA_Fn-UseC_-Telco-Customer-Churn.csv` to your public GitHub repository.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app** and select repository `so0452996-source/telco-churn-prediction`.
4. Set branch to `main` and main file path to `app.py`.
5. Click **Deploy** and wait for the build to finish.

## Notes
The app trains its models on startup and caches the results for the current app session. Dataset columns and target are based on the public IBM Telco Customer Churn dataset. Metrics can vary with dataset changes.
