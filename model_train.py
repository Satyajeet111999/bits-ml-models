
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, matthews_corrcoef,
                             roc_auc_score)
from xgboost import XGBClassifier
# Z score standardization for numerical features
from sklearn.preprocessing import StandardScaler


def fit_and_evaluate(name: str, model, X_train, X_test, y_train, y_test):
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # Metrics (multiclass-aware)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    #if name in ["logistic_regression", "random_forest", "xgboost", "decision_tree", "knn"]:
    y_pred_proba = model.predict_proba(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
    #else:
     #   auc_roc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"ROC AUC (ovr, macro): {auc_roc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    return model


def main():
    df = pd.read_csv("data/BEED_Data.csv")
    print("Loaded data:", df.shape)

    # basic cleaning
    df = df.drop_duplicates()
    print("After dropping duplicates:", df.shape)

    X = df.drop(columns=["y"])
    y = df["y"]

    # split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28, stratify=y)
    print("Train/Test shapes:", X_train.shape, X_test.shape)

    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    # save the scaler for later use in the Streamlit app
    joblib.dump(scaler, "models/numerical_scaler.pkl")
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=28),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=28),
        "xgboost": XGBClassifier(random_state=28)
    }

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, estimator in models.items():
        # wrap in pipeline with preprocessor
        trained = fit_and_evaluate(name, estimator, X_train, X_test, y_train, y_test)
        # save
        fname = out_dir / f"{name}.pkl"
        joblib.dump(trained, fname)
        print(f"Saved model pipeline to: {fname}")


if __name__ == "__main__":
    main()
