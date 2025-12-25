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

# Optional: Gradient Boosting main models
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    LGBMClassifier = None
    _HAS_LGBM = False


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
        return ";"


@st.cache_data(show_spinner=False)
def read_csv_auto(uploaded_file):
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="replace")
    sample = text[:5000]
    delim = _sniff_delimiter(sample)
    df = pd.read_csv(io.StringIO(text), sep=delim)
    return df, delim


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


def build_model(name: str, num_classes: int):
    # Baselines
    if name == "Logistic Regression (LR)":
        return LogisticRegression(max_iter=500)
    if name == "Decision Tree (DT)":
        return DecisionTreeClassifier(random_state=42)
    if name == "Random Forest (RF)":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True)

    # Main models: Gradient Boosting
    if name == "XGBoost (Main)":
        if not _HAS_XGB:
            raise RuntimeError("xgboost is not installed. Add `xgboost` to requirements.txt.")
        if num_classes <= 2:
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=42,
        )

    if name == "LightGBM (Main)":
        if not _HAS_LGBM:
            raise RuntimeError("lightgbm is not installed. Add `lightgbm` to requirements.txt.")
        if num_classes <= 2:
            return LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="multiclass",
            num_class=num_classes,
        )

    raise ValueError("Unknown model")


def build_pipeline(model_name: str, X: pd.DataFrame, y_for_class_count: pd.Series) -> Pipeline:
    preprocessor = make_preprocessor(X)
    num_classes = int(pd.Series(y_for_class_count).nunique(dropna=True))
    model = build_model(model_name, num_classes)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def encode_labels_for_boosting(y: pd.Series):
    """
    For XGBoost/LightGBM: map string labels -> integers.
    Special-case risk_level: High/Medium/Low -> 0/1/2 (stable order).
    Returns: y_encoded, decoder_func(int_or_array)->label(s)
    """
    y_str = y.astype(str)

    uniq = set(y_str.dropna().unique().tolist())
    if uniq == {"High", "Medium", "Low"}:
        mapping = {"High": 0, "Medium": 1, "Low": 2}
        inv = {v: k for k, v in mapping.items()}
        y_enc = y_str.map(mapping).astype(int)

        def decode(arr):
            if isinstance(arr, (list, tuple, np.ndarray, pd.Series)):
                return pd.Series(arr).map(inv).astype(str).values
            return inv.get(int(arr), str(arr))

        return y_enc, decode

    # generic fallback: alphabetical mapping (still valid for training)
    classes = sorted(y_str.dropna().unique().tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    inv = {i: c for c, i in mapping.items()}
    y_enc = y_str.map(mapping).astype(int)

    def decode(arr):
        if isinstance(arr, (list, tuple, np.ndarray, pd.Series)):
            return pd.Series(arr).map(inv).astype(str).values
        return inv.get(int(arr), str(arr))

    return y_enc, decode


# =========================
# Sidebar
# =========================
st.sidebar.header("A) Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("B) Model & Split Settings")

model_options = [
    "Logistic Regression (LR)",
    "Decision Tree (DT)",
    "Random Forest (RF)",
    "SVM (RBF)",
    "XGBoost",
    "LightGBM",
]

model_name = st.sidebar.selectbox("Model", model_options, index=4)
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
target_col = st.selectbox(
    "Choose the target/label column",
    df.columns,
    index=list(df.columns).index("G3") if "G3" in df.columns else 0
)

st.markdown("**Optional (Recommended): Create a 3-level risk label from a numeric score (e.g., G3).**")
use_risk_label = st.checkbox(
    "Create risk_level label (High/Medium/Low) from a numeric column",
    value=("G3" in df.columns)
)

risk_source_col = None
low_cut, mid_cut = 10, 15

if use_risk_label:
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    default_idx = numeric_candidates.index("G3") if "G3" in numeric_candidates else 0
    risk_source_col = st.selectbox("Numeric column for risk label", numeric_candidates, index=default_idx)

    low_cut = st.slider("High risk if score < (cutoff)", 0, 20, 10, 1)
    mid_cut = st.slider("Medium risk if score < (cutoff)", 0, 20, 15, 1)
    if mid_cut <= low_cut:
        st.warning("Make sure Medium cutoff > High cutoff (e.g., 10 and 15).")

# Build X, y (display label is STRING)
if use_risk_label and risk_source_col is not None:
    score = df[risk_source_col].copy()

    def to_risk(v):
        if pd.isna(v):
            return np.nan
        if v < low_cut:
            return "High"
        if v < mid_cut:
            return "Medium"
        return "Low"

    y_display = score.apply(to_risk).astype("object")
    X = df.drop(columns=[risk_source_col])
    label_name = f"risk_level(from {risk_source_col})"
else:
    X = df.drop(columns=[target_col])
    y_display = df[target_col]
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
    st.metric("Unique labels", int(pd.Series(y_display).nunique(dropna=True)))

st.write(f"Label distribution: **{label_name}**")
st.dataframe(pd.Series(y_display).value_counts(dropna=False).rename("count").to_frame(), use_container_width=True)

st.subheader("3) Training & Evaluation (Process + Output)")
st.write("Press **Run Training** in the sidebar to train and evaluate the model.")

# Warn if missing main model packages
if model_name == "XGBoost (Main)" and not _HAS_XGB:
    st.warning("You selected XGBoost, but `xgboost` is not installed. Add it to requirements.txt and redeploy.")
if model_name == "LightGBM (Main)" and not _HAS_LGBM:
    st.warning("You selected LightGBM, but `lightgbm` is not installed. Add it to requirements.txt and redeploy.")

# Session state
if "trained" not in st.session_state:
    st.session_state["trained"] = False
if "pipe" not in st.session_state:
    st.session_state["pipe"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None
if "decoder" not in st.session_state:
    st.session_state["decoder"] = None  # for boosting label decode

if run_train:
    # drop NaN label rows
    valid_mask = pd.Series(y_display).notna()
    X2 = X.loc[valid_mask].copy()
    y2 = pd.Series(y_display).loc[valid_mask].copy()

    nunique = y2.nunique(dropna=True)
    if nunique > 30:
        st.warning(
            f"Your label has {nunique} unique classes. "
            "For Early Warning, it’s better to create 3 risk levels (High/Medium/Low)."
        )

    strat = y2 if y2.nunique() > 1 and y2.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2,
        test_size=test_size,
        random_state=int(random_state),
        stratify=strat
    )

    # For boosting: encode y to 0/1/2 etc, then decode for display
    decoder = None
    y_train_fit = y_train
    y_test_fit = y_test

    if model_name in ["XGBoost (Main)", "LightGBM (Main)"]:
        y_train_fit, decoder = encode_labels_for_boosting(y_train)
        # y_test_fit is only for metrics; model outputs encoded labels
        y_test_fit, _ = encode_labels_for_boosting(y_test)  # mapping consistent for risk labels

    # Build & train
    try:
        pipe = build_pipeline(model_name, X_train, y_train_fit)
        pipe.fit(X_train, y_train_fit)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # Predict
    y_pred_fit = pipe.predict(X_test)

    # Decode for reporting (so confusion matrix/report show High/Medium/Low)
    if decoder is not None:
        y_pred_show = decoder(y_pred_fit)
        y_test_show = y_test.astype(str).values
    else:
        y_pred_show = y_pred_fit
        y_test_show = y_test

    acc = accuracy_score(y_test_show, y_pred_show)
    f1 = f1_score(y_test_show, y_pred_show, average="weighted")

    st.session_state["trained"] = True
    st.session_state["pipe"] = pipe
    st.session_state["decoder"] = decoder
    st.session_state["metrics"] = {
        "acc": acc,
        "f1": f1,
        "y_test_show": y_test_show,
        "y_pred_show": y_pred_show
    }

if st.session_state["trained"] and st.session_state["metrics"] is not None:
    acc = st.session_state["metrics"]["acc"]
    f1 = st.session_state["metrics"]["f1"]
    y_test_show = st.session_state["metrics"]["y_test_show"]
    y_pred_show = st.session_state["metrics"]["y_pred_show"]

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy", f"{acc:.4f}")
    with m2:
        st.metric("F1 (weighted)", f"{f1:.4f}")

    st.write("Confusion Matrix")
    labels = sorted(pd.Series(y_test_show).dropna().unique().tolist())
    cm = confusion_matrix(y_test_show, y_pred_show, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
    st.dataframe(cm_df, use_container_width=True)

    st.write("Classification Report")
    st.text(classification_report(y_test_show, y_pred_show))


st.subheader("4) Predict & Export (Output)")

if not st.session_state["trained"]:
    st.info("Train the model first (click **Run Training**).")
    st.stop()

pipe = st.session_state["pipe"]
decoder = st.session_state.get("decoder", None)

pred_all_fit = pipe.predict(X)

# Decode predictions if boosting model used
if decoder is not None:
    pred_all = decoder(pred_all_fit)
else:
    pred_all = pred_all_fit

out = df.copy()
out["predicted_label"] = pred_all

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

st.success("Done. You can now take screenshots for the D4 Interface section.")
