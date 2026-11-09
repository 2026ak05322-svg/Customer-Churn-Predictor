import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Subscription Service - Customer Churn Predictor")
st.write("This interactive machine learning application helps you predict customer churn using "
         "five pre-trained models. Upload a test CSV containing subscriber features to view "
         "evaluation performance.")

# Sidebar options
st.sidebar.header("Configuration")
model_options = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree": "Decision_Tree.joblib",
    "KNN": "KNN.joblib",
    "Naive Bayes": "Naive_Bayes.joblib",
    "SVM": "SVM.joblib",
}
selected_model_name = st.sidebar.selectbox("Select Classifier Model", list(model_options.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Churn Test CSV File", type=["csv"])

model_path = os.path.join("model", model_options[selected_model_name])
try:
    pipeline = joblib.load(model_path)
    st.sidebar.success(f"Loaded {selected_model_name} successfully!")
except Exception as e:
    st.sidebar.error(f"Could not load model pipeline: {e}")
    pipeline = None

if uploaded_file is not None and pipeline is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Subscriber Dataset Preview")
    st.dataframe(df.head(10))

    if "Churn" not in df.columns:
        st.error("The dataset must contain the target column named 'Churn'.")
    else:
        X_eval = df.drop(columns=["Churn"])
        y_eval = df["Churn"]
        y_pred = pipeline.predict(X_eval)
        y_proba = pipeline.predict_proba(X_eval)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

        # Compute the 6 metrics
        acc = accuracy_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, y_proba)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_eval, y_pred)

        st.subheader(f"Evaluation Metrics - {selected_model_name}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_eval, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                        xticklabels=["Active (0)", "Churned (1)"],
                        yticklabels=["Active (0)", "Churned (1)"], ax=ax)
            ax.set_ylabel("True Status")
            ax.set_xlabel("Predicted Status")
            st.pyplot(fig)
        with fig_col2:
            st.write("**Classification Report**")
            report_dict = classification_report(y_eval, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4))

        # Download metrics and predictions
        st.subheader("Download Results")
        metrics_df = pd.DataFrame([{
            "Model": selected_model_name, "Accuracy": acc, "AUC": auc,
            "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc}]).round(4)
        results_df = df.copy()
        results_df["Predicted"] = y_pred
        file_tag = selected_model_name.replace(" ", "_")
        dl1, dl2 = st.columns(2)
        dl1.download_button("Download metrics (CSV)", metrics_df.to_csv(index=False),
                            file_name=f"metrics_{file_tag}.csv", mime="text/csv")
        dl2.download_button("Download predictions (CSV)", results_df.to_csv(index=False),
                            file_name=f"predictions_{file_tag}.csv", mime="text/csv")
else:
    st.info("Please upload 'test_data.csv' from the sidebar to begin evaluation.")
