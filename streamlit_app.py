import streamlit as st
import pandas as pd
import numpy as np
import io
import csv
import time
from contextlib import redirect_stdout, redirect_stderr

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional: XGBoost / LightGBM
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


# =========================================================
# 0) UI NAME (edit here)
# =========================================================
APP_TAB_TITLE = "UKM Student Performance UI"
APP_MAIN_TITLE = "UKM Student Performance Prediction System (Interface Prototype)"
APP_SUBTITLE = "Upload → Preview/Select Label → Train/Evaluate → Predict → Export"

st.set_page_config(page_title=APP_TAB_TITLE, layout="wide")
st.title(APP_MAIN_TITLE)
st.caption(APP_SUBTITLE)


# =========================================================
# Helpers
# =========================================================
def _sniff_delimiter(sample_text: str) -> str:
    """Guess delimiter; UCI student dataset often uses ';'."""
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


def make_risk_label(score_series: pd.Series, low_cut: int, mid_cut: int) -> pd.Series:
    """
    Convert numeric score -> 3 risk levels:
    score < low_cut  -> High
    score < mid_cut  -> Medium
    else             -> Low
    """
    def to_risk(v):
        if pd.isna(v):
            return np.nan
        if v < low_cut:
            return "High"
        if v < mid_cut:
            return "Medium"
        return "Low"

    return score_series.apply(to_risk).astype("object")


def build_model(name: str, num_classes: int, trees: int, speed_mode: str):
    # -------------------------
    # Baselines
    # -------------------------
    if name == "Logistic Regression (LR)":
        return LogisticRegression(max_iter=800)

    if name == "Decision Tree (DT)":
        return DecisionTreeClassifier(random_state=42)

    if name == "Random Forest (RF)":
        return RandomForestClassifier(
            n_estimators=min(250, max(80, trees)),
            random_state=42,
            n_jobs=-1
        )

    if name == "SVM (RBF)":
        # SVM can be slow on larger data; keep default
        return SVC(kernel="rbf", probability=True)

    # -------------------------
    # Main models (Boosting)
    # -------------------------
    if name == "XGBoost (Main)":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed. Add `xgboost` to requirements.txt and redeploy.")

        if speed_mode == "Fast":
            max_depth, lr, subs = 3, 0.10, 0.85
        elif speed_mode == "Accurate":
            max_depth, lr, subs = 4, 0.05, 0.90
        else:
            max_depth, lr, subs = 3, 0.07, 0.90

        common = dict(
            n_estimators=trees,
            learning_rate=lr,
            max_depth=max_depth,
            subsample=subs,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,   # no spam
        )

        if num_classes <= 2:
            return XGBClassifier(
                **common,
                objective="binary:logistic",
                eval_metric="logloss"
            )

        return XGBClassifier(
            **common,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss"
        )

    if name == "LightGBM (Main)":
        if not _HAS_LGBM:
            raise RuntimeError("lightgbm not installed. Add `lightgbm` to requirements.txt and redeploy.")

        # Make it MUCH faster by default (and still reasonable accuracy)
        if speed_mode == "Fast":
            lr, leaves, depth, max_bin, mcs = 0.12, 31, -1, 127, 40
        elif speed_mode == "Accurate":
            lr, leaves, depth, max_bin, mcs = 0.06, 63, -1, 255, 20
        else:  # Balanced
            lr, leaves, depth, max_bin, mcs = 0.08, 31, -1, 255, 30

        common = dict(
            n_estimators=trees,
            learning_rate=lr,
            num_leaves=leaves,
            max_depth=depth,
            max_bin=max_bin,              # speed up
            min_child_samples=mcs,        # reduce over-splitting & warnings
            min_split_gain=0.0,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,                 # no spam
            force_col_wise=True
        )

        if num_classes <= 2:
            return LGBMClassifier(**common)

        return LGBMClassifier(
            **common,
            objective="multiclass",
            num_class=num_classes
        )

    raise ValueError("Unknown model")


def build_pipeline(model_name: str, X: pd.DataFrame, num_classes: int, trees: int, speed_mode: str) -> Pipeline:
    preprocessor = make_preprocessor(X)
    model = build_model(model_name, num_classes, trees, speed_mode)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("A) Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("B) Model & Split Settings")
model_options = [
    "Logistic Regression (LR)",
    "Decision Tree (DT)",
    "Random Forest (RF)",
    "SVM (RBF)",
    "XGBoost (Main)",
    "LightGBM (Main)",
]

if not _HAS_XGB:
    st.sidebar.caption("⚠️ xgboost not installed (XGBoost will fail until added).")
if not _HAS_LGBM:
    st.sidebar.caption("⚠️ lightgbm not installed (LightGBM will fail until added).")

model_name = st.sidebar.selectbox("Model", model_options, index=4)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

is_boosting = model_name in ["XGBoost (Main)", "LightGBM (Main)"]
if is_boosting:
    speed_mode = st.sidebar.selectbox("Boosting mode", ["Fast", "Balanced", "Accurate"], index=1)
    # keep default smaller so it won't run 20 minutes
    trees = st.sidebar.slider("Number of trees (n_estimators)", 60, 260, 140, 10)
else:
    speed_mode = "Balanced"
    trees = 140

run_train = st.sidebar.button("Run Training", type="primary")


# =========================================================
# Main workflow
# =========================================================
if file is None:
    st.info("⬅️ Please upload a CSV file from the sidebar to start.")
    st.stop()

df, delim_used = read_csv_auto(file)

st.subheader("1) Dataset Preview (Input)")
st.write(f"Detected delimiter: `{delim_used}` | Rows: {len(df)} | Columns: {len(df.columns)}")
st.dataframe(df.head(20), width="stretch")

st.subheader("2) Select Target Column (Label)")
default_target = "G3" if "G3" in df.columns else df.columns[0]
target_col = st.selectbox(
    "Choose the target/label column",
    df.columns,
    index=list(df.columns).index(default_target)
)

st.markdown("**Optional (Recommended): Create a 3-level risk label (High/Medium/Low) from a numeric score (e.g., G3).**")
use_risk_label = st.checkbox(
    "Create risk_level label (High/Medium/Low) from a numeric column",
    value=("G3" in df.columns)
)

risk_source_col = None
low_cut, mid_cut = 10, 15

if use_risk_label:
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_candidates) == 0:
        st.warning("No numeric columns found. Cannot create risk_level.")
        use_risk_label = False
    else:
        default_idx = numeric_candidates.index("G3") if "G3" in numeric_candidates else 0
        risk_source_col = st.selectbox("Numeric column for risk label", numeric_candidates, index=default_idx)
        low_cut = st.slider("High risk if score < (cutoff)", 0, 20, 10, 1)
        mid_cut = st.slider("Medium risk if score < (cutoff)", 0, 20, 15, 1)
        if mid_cut <= low_cut:
            st.warning("Make sure Medium cutoff > High cutoff (e.g., 10 and 15).")

# Build X, y_display
if use_risk_label and risk_source_col is not None:
    y_display = make_risk_label(df[risk_source_col], low_cut, mid_cut)
    X = df.drop(columns=[risk_source_col])
    label_name = f"risk_level (from {risk_source_col})"
else:
    y_display = df[target_col]
    X = df.drop(columns=[target_col])
    label_name = target_col

# Summary
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
st.dataframe(
    pd.Series(y_display).value_counts(dropna=False).rename("count").to_frame(),
    width="stretch"
)

# =========================================================
# Session state
# =========================================================
if "trained" not in st.session_state:
    st.session_state["trained"] = False
if "pipe" not in st.session_state:
    st.session_state["pipe"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None
if "label_encoder" not in st.session_state:
    st.session_state["label_encoder"] = None
if "last_cfg" not in st.session_state:
    st.session_state["last_cfg"] = None

current_cfg = {
    "model": model_name,
    "test_size": float(test_size),
    "random_state": int(random_state),
    "label_name": str(label_name),
    "use_risk_label": bool(use_risk_label),
    "risk_source_col": str(risk_source_col),
    "low_cut": int(low_cut),
    "mid_cut": int(mid_cut),
    "target_col": str(target_col),
    "trees": int(trees),
    "speed_mode": str(speed_mode),
}

if st.session_state["last_cfg"] is None:
    st.session_state["last_cfg"] = current_cfg
elif st.session_state["last_cfg"] != current_cfg:
    st.session_state["trained"] = False
    st.session_state["pipe"] = None
    st.session_state["metrics"] = None
    st.session_state["label_encoder"] = None
    st.session_state["last_cfg"] = current_cfg

# =========================================================
# 3) Training & Evaluation
# =========================================================
st.subheader("3) Training & Evaluation (Process + Output)")
st.write("Press **Run Training** in the sidebar to train and evaluate the model.")

valid_mask = pd.Series(y_display).notna()
X2 = X.loc[valid_mask].copy()
y2 = pd.Series(y_display).loc[valid_mask].copy()

if run_train:
    nunique = y2.nunique(dropna=True)
    if nunique < 2:
        st.error("Your label has < 2 unique classes. Cannot train a classifier.")
        st.stop()

    strat = y2 if y2.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2,
        test_size=test_size,
        random_state=int(random_state),
        stratify=strat
    )

    # Always encode labels (fixes High/Medium/Low issue)
    le = LabelEncoder()
    le.fit(y2.astype(str))
    y_train_fit = le.transform(y_train.astype(str))
    y_test_fit = le.transform(y_test.astype(str))
    num_classes = int(pd.Series(y_train_fit).nunique())

    start = time.time()
    with st.spinner("Training model... (Boosting may take longer, try Fast mode if slow)"):
        try:
            pipe = build_pipeline(model_name, X_train, num_classes, trees, speed_mode)

            # Capture model logs (avoid console spam)
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                pipe.fit(X_train, y_train_fit)

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    elapsed = time.time() - start

    y_pred_fit = pipe.predict(X_test)
    y_test_show = le.inverse_transform(y_test_fit.astype(int))
    y_pred_show = le.inverse_transform(y_pred_fit.astype(int))

    acc = accuracy_score(y_test_show, y_pred_show)
    f1 = f1_score(y_test_show, y_pred_show, average="weighted")

    st.session_state["trained"] = True
    st.session_state["pipe"] = pipe
    st.session_state["label_encoder"] = le
    st.session_state["metrics"] = {
        "acc": acc,
        "f1": f1,
        "y_test_show": y_test_show,
        "y_pred_show": y_pred_show,
        "elapsed": elapsed,
    }

if st.session_state["trained"] and st.session_state["metrics"] is not None:
    acc = st.session_state["metrics"]["acc"]
    f1 = st.session_state["metrics"]["f1"]
    y_test_show = st.session_state["metrics"]["y_test_show"]
    y_pred_show = st.session_state["metrics"]["y_pred_show"]
    elapsed = st.session_state["metrics"]["elapsed"]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", f"{acc:.4f}")
    with m2:
        st.metric("F1 (weighted)", f"{f1:.4f}")
    with m3:
        st.metric("Train time (s)", f"{elapsed:.2f}")

    st.write("Confusion Matrix")
    labels = sorted(pd.Series(y_test_show).dropna().unique().tolist())
    cm = confusion_matrix(y_test_show, y_pred_show, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
    st.dataframe(cm_df, width="stretch")

    st.write("Classification Report")
    st.text(classification_report(y_test_show, y_pred_show))

# =========================================================
# 4) Predict & Export
# =========================================================
st.subheader("4) Predict & Export (Output)")

if not st.session_state["trained"]:
    st.info("Train the model first (click **Run Training**).")
    st.stop()

pipe = st.session_state["pipe"]
le = st.session_state["label_encoder"]

pred_all_fit = pipe.predict(X)
pred_all = le.inverse_transform(pred_all_fit.astype(int))

out = df.copy()
out["predicted_label"] = pred_all

model_obj = pipe.named_steps["model"]
try:
    if hasattr(model_obj, "predict_proba"):
        proba = pipe.predict_proba(X)
        out["confidence"] = np.max(proba, axis=1)
    else:
        out["confidence"] = np.nan
except Exception:
    out["confidence"] = np.nan

st.dataframe(out.head(30), width="stretch")

st.download_button(
    "Download predictions as CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)

