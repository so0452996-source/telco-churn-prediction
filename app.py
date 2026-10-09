import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Telecom Customer Churn Prediction", page_icon="📡", layout="wide")
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
DROP_COLS = ["customerID", TARGET]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    return df

@st.cache_resource
def train_models():
    df = load_data()
    X = df[FEATURES].copy()
    y = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, NUMERIC), ("cat", categorical_pipe, CATEGORICAL)])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }
    fitted, metrics = {}, []
    for name, clf in models.items():
        # GaussianNB requires dense one-hot output; use a small separate dense pipeline for compatibility.
        prep_model = ColumnTransformer([("num", numeric_pipe, NUMERIC), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL)]) if name == "Naive Bayes" else prep
        pipe = Pipeline([("prep", prep_model), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        metrics.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1-score": f1_score(y_test, pred, zero_division=0)})
    return fitted, pd.DataFrame(metrics), X_test, y_test

st.title("📡 Telecom Customer Churn Prediction")
st.caption("Predict which telecom customers may leave, explore model performance, and score a batch of customers.")
try:
    df = load_data()
    models, scores, X_test, y_test = train_models()
except Exception as e:
    st.error(f"Could not load the dataset or train the models: {e}")
    st.stop()

single_tab, batch_tab, insights_tab = st.tabs(["Single Customer", "Batch Prediction", "Model Insights"])

with single_tab:
    st.subheader("Predict churn for one customer")
    st.write("Enter customer details below, then select **Predict Churn**.")
    with st.form("single_customer_form"):
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with b:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        with c:
            protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=float(monthly * tenure), step=10.0)
        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{"SeniorCitizen": senior, "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total, "gender": gender, "Partner": partner, "Dependents": dependents, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment}])[FEATURES]
        model = models["Logistic Regression"]
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])
        if prediction:
            st.error(f"Higher churn risk — estimated probability: {probability:.1%}")
            st.write("Consider reviewing service experience, contract options, and customer support needs.")
        else:
            st.success(f"Lower churn risk — estimated probability: {probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0), text="Estimated churn probability")

with batch_tab:
    st.subheader("Batch Prediction")
    st.write("Upload a CSV containing customer columns from the Telco dataset. `customerID` and `Churn` are optional; missing feature columns will be reported.")
    st.download_button("Download sample CSV", df.drop(columns=[TARGET]).head(10).to_csv(index=False).encode("utf-8"), "sample_customers.csv", "text/csv")
    upload = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_upload")
    if upload is not None:
        try:
            incoming = pd.read_csv(upload)
            incoming["TotalCharges"] = pd.to_numeric(incoming.get("TotalCharges", pd.Series(index=incoming.index, dtype=float)), errors="coerce")
            missing = [c for c in FEATURES if c not in incoming.columns]
            if missing:
                st.error("These required columns are missing: " + ", ".join(missing))
            else:
                result = incoming.copy()
                pred = models["Logistic Regression"].predict(incoming[FEATURES])
                prob = models["Logistic Regression"].predict_proba(incoming[FEATURES])[:, 1]
                result["PredictedChurn"] = ["Yes" if x == 1 else "No" for x in pred]
                result["ChurnProbability"] = prob.round(4)
                st.dataframe(result.head(100), use_container_width=True)
                st.download_button("Download predictions CSV", result.to_csv(index=False).encode("utf-8"), "churn_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Unable to process this CSV: {e}")

with insights_tab:
    st.subheader("Model Insights")
    st.metric("Customers in dataset", f"{len(df):,}")
    st.metric("Overall churn rate", f"{(df[TARGET].astype(str).str.strip() == 'Yes').mean():.1%}")
    st.markdown("#### Model comparison")
    st.dataframe(scores.style.format({"Accuracy":"{:.1%}", "Precision":"{:.1%}", "Recall":"{:.1%}", "F1-score":"{:.1%}"}), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        churn_counts = df[TARGET].value_counts().reindex(["No", "Yes"], fill_value=0)
        ax.bar(churn_counts.index, churn_counts.values)
        ax.set_title("Churn distribution")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Number of customers")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots()
        df.assign(ChurnLabel=df[TARGET].astype(str)).groupby("Contract", observed=False)["ChurnLabel"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_title("Churn rate by contract")
        ax.set_ylabel("Churn rate")
        ax.set_xlabel("Contract")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("#### Logistic Regression confusion matrix")
    chosen = models["Logistic Regression"]
    cm = confusion_matrix(y_test, chosen.predict(X_test), labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted No", "Predicted Yes"])
    ax.set_yticks([0, 1], labels=["Actual No", "Actual Yes"])
    ax.set_title("Confusion matrix")
    for (i, j), value in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Metrics are measured on a held-out test split (20%, random_state=42). These predictions are estimates, not guarantees.")
