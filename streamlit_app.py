import streamlit as st
import pandas as pd
import numpy as np
import io
import csv

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


# =========================
# 0) UI NAME (edit here)
# =========================
APP_TAB_TITLE = "UKM Student Performance UI"
APP_MAIN_TITLE = "UKM Student Performance Prediction System (Interface Prototype)"
APP_SUBTITLE = "Upload → Preview/Select Target → Train/Evaluate → Predict → Export"

st.set_page_config(page_title=APP_TAB_TITLE, layout="wide")
st.title(APP_MAIN_TITLE)
st.caption(APP_SUBTITLE)


# =========================
# Helpers
# =========================
def _sniff_delimiter(sample_text: str) -> str:
    """Try to guess delimiter for CSV; fallback to ';' for UCI datasets."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # UCI student datasets are typically ';'
        return ";"


@st.cache_data(show_spinner=False)
def read_csv_auto(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    # Try to decode as utf-8 (most common)
    text = raw.decode("utf-8", errors="replace")
    sample = text[:5000]
    delim = _sniff_delimiter(sample)

    # Read using detected delimiter
    df = pd.read_csv(io.StringIO(text), sep=delim)
    return df, delim


def get_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=500)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True)
    raise ValueError("Unknown model")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


# =========================
# Sidebar
# =========================
st.sidebar.header("A) Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("B) Model & Split Settings")
model_name = st.sidebar.selectbox(
    "Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "SVM (RBF)"]
)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

run_train = st.sidebar.button("Run Training", type="primary")


# =========================
# Main workflow
# =========================
if file is None:
    st.info("⬅️ Please upload a CSV file from the sidebar to start.")
    st.stop()

df, delim_used = read_csv_auto(file)

st.subheader("1) Dataset Preview (Input)")
st.write(f"Detected delimiter: `{delim_used}`  |  Rows: {len(df)}  |  Columns: {len(df.columns)}")
st.dataframe(df.head(20), use_container_width=True)

# ---------- Target options ----------
st.subheader("2) Select Target Column (Label)")

# Option A: use existing label column
target_col = st.selectbox("Choose the target/label column", df.columns, index=list(df.columns).index("G3") if "G3" in df.columns else 0)

# Option B: create Risk Level from a numeric column (recommended for Early Warning)
st.markdown("**Optional (Recommended): Create a 3-level risk label from a numeric score (e.g., G3).**")
use_risk_label = st.checkbox("Create risk_level label (High/Medium/Low) from a numeric column", value=("G3" in df.columns))

risk_source_col = None
if use_risk_label:
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    default_idx = numeric_candidates.index("G3") if "G3" in numeric_candidates else 0
    risk_source_col = st.selectbox("Numeric column for risk label", numeric_candidates, index=default_idx)

    # Typical passing line is 10 (UCI dataset grades 0-20)
    low_cut = st.slider("High risk if score < (cutoff)", 0, 20, 10, 1)
    mid_cut = st.slider("Medium risk if score < (cutoff)", 0, 20, 15, 1)
    if mid_cut <= low_cut:
        st.warning("Make sure Medium cutoff > High cutoff (e.g., 10 and 15).")

# Build X, y
if use_risk_label and risk_source_col is not None:
    y = df[risk_source_col].copy()

    def to_risk(v):
        if pd.isna(v):
            return np.nan
        if v < low_cut:
            return "High"
        if v < mid_cut:
            return "Medium"
        return "Low"

    y = y.apply(to_risk).astype("object")
    X = df.drop(columns=[risk_source_col])
    label_name = f"risk_level(from {risk_source_col})"
else:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    label_name = target_col

# Basic summary
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows", len(df))
with c2:
    st.metric("Features", X.shape[1])
with c3:
    st.metric("Missing cells (X)", int(X.isna().sum().sum()))
with c4:
    st.metric("Unique labels", int(pd.Series(y).nunique(dropna=True)))

st.write(f"Label distribution: **{label_name}**")
st.dataframe(pd.Series(y).value_counts(dropna=False).rename("count").to_frame(), use_container_width=True)

# ---------- Pipeline ----------
preprocessor = make_preprocessor(X)
pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", get_model(model_name))
])

st.subheader("3) Training & Evaluation (Process + Output)")
st.write("Press **Run Training** in the sidebar to train and evaluate the model.")

# Session state (keep trained model)
if "trained" not in st.session_state:
    st.session_state["trained"] = False
if "pipe" not in st.session_state:
    st.session_state["pipe"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None

if run_train:
    # Drop rows with missing label
    valid_mask = pd.Series(y).notna()
    X2 = X.loc[valid_mask].copy()
    y2 = pd.Series(y).loc[valid_mask].copy()

    # Some labels may have too many classes; warn user
    nunique = y2.nunique(dropna=True)
    if nunique > 30:
        st.warning(
            f"Your label has {nunique} unique classes. "
            "For Early Warning, it’s better to create 3 risk levels (High/Medium/Low)."
        )

    # Split
    strat = y2 if y2.nunique() > 1 and y2.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2,
        test_size=test_size,
        random_state=int(random_state),
        stratify=strat
    )

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.session_state["trained"] = True
    st.session_state["pipe"] = pipe
    st.session_state["metrics"] = {
        "acc": acc, "f1": f1,
        "y_test": y_test, "y_pred": y_pred
    }

if st.session_state["trained"] and st.session_state["metrics"] is not None:
    acc = st.session_state["metrics"]["acc"]
    f1 = st.session_state["metrics"]["f1"]
    y_test = st.session_state["metrics"]["y_test"]
    y_pred = st.session_state["metrics"]["y_pred"]

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy", f"{acc:.4f}")
    with m2:
        st.metric("F1 (weighted)", f"{f1:.4f}")

    st.write("Confusion Matrix")
    labels = sorted(pd.Series(y_test).dropna().unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
    st.dataframe(cm_df, use_container_width=True)

    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

# ---------- Predict & Export ----------
st.subheader("4) Predict & Export (Output)")

if not st.session_state["trained"]:
    st.info("Train the model first (click **Run Training**).")
    st.stop()

pipe = st.session_state["pipe"]

# Predict on all rows (exclude rows with missing feature issues is okay; pipeline imputes)
pred_all = pipe.predict(X)
out = df.copy()
out["predicted_label"] = pred_all

# Confidence if supported
model_obj = pipe.named_steps["model"]
if hasattr(model_obj, "predict_proba"):
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

st.success
