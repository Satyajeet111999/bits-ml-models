# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             matthews_corrcoef, roc_curve, auc)
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io

# --- UI styling (small) ---
st.set_page_config(page_title="Model Predictor", layout="wide")

# --- App Introduction ---
st.title("EEG Epilepsy Detection App")
st.markdown(
        """
        This interactive app allows you to upload EEG data and apply trained machine learning models to detect and classify epileptic seizures using the BEED dataset.
    
        **Purpose:**
        - Demonstrate automated seizure detection and classification from EEG signals.
        - Compare the performance of different classical machine learning models.
    
        **How to use:**
        1. **Upload** a CSV file containing EEG data (with columns X1-X16 and optional label `y`).
        2. **Select** a trained model from the sidebar.
        3. View predictions, performance metrics, confusion matrix, ROC curve, and feature importances.
        4. Download the results for further analysis.
    
        Sample data is available for download in the sidebar.
        """
)
st.markdown(
        """
        <style>
            .stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .metric-card { background: #f6f8fa; padding: 12px; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

def list_models():
    return sorted([
        p.name for p in MODEL_DIR.glob("*.pkl")
        if p.name != "numerical_scaler.pkl"
    ])

def load_model(path):
    return joblib.load(path)

def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")

def preprocess_df(df):
    df = df.copy()
    # fill nans
    df = df.fillna(df.median(numeric_only=True)).fillna("")  # simple fallback
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler_path = MODEL_DIR / "numerical_scaler.pkl"
    scaler = None
    if len(num_cols) > 0:
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            df[num_cols] = scaler.transform(df[num_cols])
        else:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            joblib.dump(scaler, scaler_path)
    return df, num_cols, scaler

def compute_metrics(y_true, y_pred, y_proba=None):
    out = {}
    out['accuracy'] = accuracy_score(y_true, y_pred)
    out['precision'] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    out['recall'] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    out['f1'] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    out['mcc'] = matthews_corrcoef(y_true, y_pred)
    if y_proba is not None:
        try:
            out['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            out['roc_auc'] = None
    else:
        out['roc_auc'] = None
    return out

def get_metric_color(value):
    """
    Returns a color based on metric value (0 to 1 scale).
    0 = Red, 1 = Olive Green, with gradient in between.
    """
    if value is None:
        return "gray"
    value = max(0, min(1, value))  # Clamp to [0, 1]
    # Interpolate from red (255, 0, 0) to olive green (128, 128, 0)
    r = int(255 * (1 - value))
    g = int(128 * value)
    b = 0
    return f"rgb({r}, {g}, {b})"

def plot_confusion(cm, labels):
    fig = px.imshow(cm, text_auto=True, labels={"x": "Predicted", "y": "Actual"}, x=labels, y=labels, color_continuous_scale="Blues")
    fig.update_layout(title="Confusion Matrix", margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return fig

def plot_roc(y_true, y_proba):
    # If binary, y_proba should be (n,) probabilities for positive class
    fig = go.Figure()
    if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba.ravel())
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line={"dash": "dash"}, showlegend=False))
        fig.update_layout(title="ROC Curve (Binary Classification)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin={"l": 10, "r": 10, "t": 40, "b": 10})
        return fig
    else:
        # multiclass: plot per-class ROC using one-vs-rest
        n_classes = y_proba.shape[1]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"class {i} (AUC={roc_auc:.2f})"))
        fig.update_layout(title="ROC Curve (Multiclass Classification)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin={"l": 10, "r": 10, "t": 40, "b": 10})
        return fig

# --- Sidebar ---

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"], help="If it contains column 'y' it will be used as true labels.")

# Button to download sample data
with open(BASE / "data" / "BEED_Test_Data.csv", "rb") as f:
    sample_csv = f.read()
st.sidebar.download_button(
    label="Download sample data (BEED)",
    data=sample_csv,
    file_name="BEED_Test_Data.csv",
    mime="text/csv",
    help="Download the sample CSV, then upload it using the uploader above."
)

models = list_models()
selected_model_name = st.sidebar.selectbox("Choose a machine learning model", options=["-- none --"] + [p.replace(".pkl", "") for p in models])

if uploaded is None:
    st.info("Upload a test CSV to get started. You can download the sample data using the button above.")
    st.stop()

df = safe_read_csv(uploaded)
st.subheader("Data preview")
st.dataframe(df.head())

has_target = 'y' in df.columns
if has_target:
    y_true = df['y']
    X = df.drop(columns=['y'])
else:
    y_true = None
    X = df.copy()

X_proc, num_cols, scaler = preprocess_df(X)

st.write(f"Detected numeric columns: **{', '.join(num_cols)}**")

if selected_model_name == "-- none --":
    st.warning("Please select a trained model (.pkl) from the sidebar.")
    st.stop()

selected_model_name += ".pkl"
model_path = MODEL_DIR / selected_model_name
if not model_path.exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

model = load_model(model_path)
st.markdown(f"""
<div style='text-align:center; font-size:2em; font-weight:bold; margin-top:1em;'>
<span style='color:black'>Model:</span> <span style='color:green'>{selected_model_name.replace('.pkl', '')}</span>
</div>
""", unsafe_allow_html=True)

# predict
try:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_proc)
    else:
        y_proba = None
except Exception as e:
    st.error(f"Error computing probabilities: {e}")
    y_proba = None

# handle prediction
if y_proba is not None:
    if y_proba.ndim == 2:
        y_pred = y_proba.argmax(axis=1)
    else:
        y_pred = (y_proba.ravel() >= 0.5).astype(int)
else:
    y_pred = model.predict(X_proc)

# metrics & display
if has_target:
    metrics = compute_metrics(y_true, y_pred, y_proba if y_proba is not None and y_true is not None else None)
    cols = st.columns(6)
    
    # Display each metric with color-coded background
    metric_data = [
        ("Accuracy", metrics['accuracy']),
        ("Precision (macro)", metrics['precision']),
        ("Recall (macro)", metrics['recall']),
        ("F1 (macro)", metrics['f1']),
        ("MCC", metrics['mcc']),
        ("ROC AUC", metrics['roc_auc']),
    ]
    
    for i, (label, value) in enumerate(metric_data):
        color = get_metric_color(value)
        formatted_value = f"{value:.4f}" if value is not None else "n/a"
        cols[i].markdown(
            f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="margin: 0; color: white; font-weight: bold; font-size: 14px;">{label}</p>
                <p style="margin: 0; color: white; font-weight: bold; font-size: 20px;">{formatted_value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("No ground truth `y` found in uploaded file — showing predictions only.")

st.subheader("Predictions")
out_df = X.copy()
out_df['prediction'] = y_pred
if y_proba is not None:
    # if multiclass, keep all probs as columns
    if y_proba.ndim == 2:
        for i in range(y_proba.shape[1]):
            out_df[f"proba_class_{i}"] = y_proba[:, i]
    else:
        out_df['proba'] = y_proba
if has_target:
    out_df['y_true'] = y_true.values

st.dataframe(out_df.head())

# download
csv = out_df.to_csv(index=False).encode()
st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")


# confusion matrix
if has_target:
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true.unique(), y_pred]))
    fig_cm = plot_confusion(cm, labels)
    st.plotly_chart(fig_cm, use_container_width=True)

# ROC
if has_target and y_proba is not None:
    try:
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            fig_roc = plot_roc(y_true, y_proba[:, 1])
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            fig_roc = plot_roc(y_true, y_proba)
            st.plotly_chart(fig_roc, use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute ROC curve: {e}")

# feature importance / coefficients
st.subheader("Feature importances / coefficients (if available)")
try:
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        features = X_proc.columns
        df_fi = pd.DataFrame({"feature": features, "importance": fi}).sort_values("importance", ascending=False).head(30)
        fig = px.bar(df_fi, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        # Handle multiclass case: take mean of absolute coefficients across classes
        if coefs.ndim == 2:
            coefs = np.mean(np.abs(coefs), axis=0)
        else:
            coefs = np.ravel(coefs)
        features = X_proc.columns
        df_coefs = pd.DataFrame({"feature": features, "coef": coefs}).sort_values("coef", key=abs, ascending=False).head(30)
        fig = px.bar(df_coefs, x="coef", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model does not expose importances or coefficients.")
except Exception as e:
    st.error(f"Could not extract importances: {e}")
