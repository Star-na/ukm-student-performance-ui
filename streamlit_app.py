import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ===== UI NAME (change here) =====
APP_TAB_TITLE = "UKM Student Performance UI"
APP_MAIN_TITLE = "UKM Student Performance Prediction System (Interface Prototype)"
APP_SUBTITLE = "Upload → Preview/Select Target → Train/Evaluate → Predict → Export"

st.set_page_config(page_title=APP_TAB_TITLE, layout="wide")
st.title(APP_MAIN_TITLE)
st.caption(APP_SUBTITLE)

# ===== Sidebar =====
st.sidebar.header("A) Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("B) Model & Split Settings")
model_name = st.sidebar.selectbox(
    "Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "SVM (RBF)"]
)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
run_train = st.sidebar.button("Run Training")

def get_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=300)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True)
    raise ValueError("Unknown model")

# ===== Main =====
if file is None:
    st.info("⬅️ Please upload a CSV file from the sidebar to start.")
    st.stop()

df = pd.read_csv(file)

st.subheader("1) Dataset Preview (Input)")
st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("2) Select Target Column (Label)")
target_col = st.selectbox("Choose the target/label column", df.columns)

if not target_col:
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Numeric cols", len(numeric_cols))
with c2:
    st.metric("Categorical cols", len(categorical_cols))
with c3:
    st.metric("Missing cells (X)", int(X.isna().sum().sum()))
with c4:
    st.metric("Unique labels", int(y.nunique()))

st.write("Label distribution:")
st.dataframe(y.value_counts(dropna=False).rename("count").to_frame(), use_container_width=True)

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", get_model(model_name))
])

st.subheader("3) Training & Evaluation (Process + Output)")
st.write("Press **Run Training** in the sidebar to train and evaluate the model.")

if run_train:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=y if y.nunique() > 1 else None
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy", f"{acc:.4f}")
    with m2:
        st.metric("F1 (weighted)", f"{f1:.4f}")

    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm), use_container_width=True)

    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("4) Predict & Export (Output)")
    out = df.copy()
    out["predicted_label"] = pipe.predict(X)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(X)
        out["confidence"] = np.max(proba, axis=1)
    else:
        out["confidence"] = np.nan

    st.dataframe(out.head(30), use_container_width=True)

    st.download_button(
        "Download predictions as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.success("Done. You can now take screenshots for the D4 Interface section.")
