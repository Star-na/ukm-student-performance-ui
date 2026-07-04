import streamlit as st
import pandas as pd
import numpy as np
import io
import csv
import time
from contextlib import redirect_stdout, redirect_stderr

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# =========================================================
# Optional libraries: XGBoost / LightGBM
# The app still runs if these libraries are not installed.
# =========================================================
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
# 0) App Configuration
# =========================================================
APP_TAB_TITLE = "Student Risk Prediction"
APP_MAIN_TITLE = "Student Performance Prediction System"
APP_SUBTITLE = (
    "Educational Data Mining Prototype | Upload Dataset → Preprocess → Train Models "
    "→ Evaluate → Visualize → Predict → Export"
)

st.set_page_config(page_title=APP_TAB_TITLE, layout="wide")

# A small amount of CSS to make the prototype look more like a complete system.
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #f8fafc 0%, #eef4ff 100%);
        border: 1px solid #dbe5f5;
        margin-bottom: 1.3rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        line-height: 1.55;
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.95rem;
    }
    .section-card {
        padding: 1.1rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        min-height: 140px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(APP_MAIN_TITLE)
st.caption(APP_SUBTITLE)


# =========================================================
# 1) Helper Functions
# =========================================================
def _sniff_delimiter(sample_text: str) -> str:
    """Guess CSV delimiter. The UCI Student Performance dataset often uses ';'."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ";"


@st.cache_data(show_spinner=False)
def read_csv_auto(uploaded_file):
    """Read uploaded CSV file and automatically detect delimiter."""
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="replace")
    sample = text[:5000]
    delim = _sniff_delimiter(sample)
    df = pd.read_csv(io.StringIO(text), sep=delim)
    return df, delim


def clean_duplicate_rows(df: pd.DataFrame):
    """Remove duplicate rows and return cleaned dataframe with duplicate count."""
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, duplicate_rows


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing transformer:
    - Numeric: median imputation + standardization.
    - Categorical: most-frequent imputation + one-hot encoding.
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
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
    Convert numeric score into 3-level risk label:
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


def build_model(name: str, num_classes: int, trees: int, speed_mode: str, class_weight_mode: str):
    """Construct a classifier by model name."""
    cw = None if class_weight_mode == "None" else "balanced"

    if name == "Logistic Regression (LR)":
        return LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
            class_weight=cw
        )

    if name == "Decision Tree (DT)":
        return DecisionTreeClassifier(
            random_state=42,
            class_weight=cw
        )

    if name == "Random Forest (RF)":
        return RandomForestClassifier(
            n_estimators=min(250, max(80, trees)),
            random_state=42,
            n_jobs=-1,
            class_weight=cw
        )

    if name == "SVM (RBF)":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight=cw
        )

    if name == "XGBoost (Main)":
        if not _HAS_XGB:
            raise RuntimeError("xgboost is not installed. Add `xgboost` to requirements.txt if deployment needs this model.")

        if speed_mode == "Fast":
            max_depth, lr, subs = 3, 0.12, 0.85
        elif speed_mode == "Accurate":
            max_depth, lr, subs = 4, 0.05, 0.90
        else:
            max_depth, lr, subs = 3, 0.08, 0.90

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
            verbosity=0,
        )

        if num_classes <= 2:
            return XGBClassifier(
                **common,
                objective="binary:logistic",
                eval_metric=["logloss", "error"]
            )

        return XGBClassifier(
            **common,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric=["mlogloss", "merror"]
        )

    if name == "LightGBM (Main)":
        if not _HAS_LGBM:
            raise RuntimeError("lightgbm is not installed. Add `lightgbm` to requirements.txt if deployment needs this model.")

        if speed_mode == "Fast":
            lr, leaves, depth, max_bin, mcs = 0.12, 31, -1, 127, 40
        elif speed_mode == "Accurate":
            lr, leaves, depth, max_bin, mcs = 0.06, 63, -1, 255, 20
        else:
            lr, leaves, depth, max_bin, mcs = 0.08, 31, -1, 255, 30

        common = dict(
            n_estimators=trees,
            learning_rate=lr,
            num_leaves=leaves,
            max_depth=depth,
            max_bin=max_bin,
            min_child_samples=mcs,
            min_split_gain=0.0,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
            force_col_wise=True
        )

        if num_classes <= 2:
            return LGBMClassifier(**common)

        return LGBMClassifier(
            **common,
            objective="multiclass",
            num_class=num_classes
        )

    raise ValueError(f"Unknown model: {name}")


def build_pipeline(model_name: str, X: pd.DataFrame, num_classes: int, trees: int, speed_mode: str, class_weight_mode: str) -> Pipeline:
    """Build complete preprocessing + model pipeline."""
    preprocessor = make_preprocessor(X)
    model = build_model(model_name, num_classes, trees, speed_mode, class_weight_mode)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def plot_confusion_matrix_df(y_true, y_pred, labels_order):
    """Return confusion matrix as dataframe."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    return pd.DataFrame(
        cm,
        index=[f"true:{label}" for label in labels_order],
        columns=[f"pred:{label}" for label in labels_order]
    )


def plot_learning_curve(pipe, X, y, title="Learning Curve (Train vs Test/CV)", cv=3):
    """Generate learning curve for non-boosting models."""
    train_sizes, train_scores, test_scores = learning_curve(
        pipe,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    fig = plt.figure()
    plt.plot(train_sizes, train_mean, marker="o", label="Train Accuracy")
    plt.plot(train_sizes, test_mean, marker="o", label="Test/CV Accuracy")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return fig


def plot_boosting_history(history: dict, title="Training History (Train vs Test)"):
    """Plot XGBoost / LightGBM training history."""
    if not history:
        return None

    keys = list(history.keys())
    train_key = keys[0]
    valid_key = keys[1] if len(keys) > 1 else None

    if train_key not in history or len(history[train_key]) == 0:
        return None

    metrics = list(history[train_key].keys())

    preferred_metric = None
    for metric_name in ["merror", "error", "multi_error", "binary_error"]:
        if metric_name in metrics:
            preferred_metric = metric_name
            break

    metric = preferred_metric if preferred_metric else metrics[0]

    train_curve = history[train_key][metric]
    valid_curve = history[valid_key][metric] if valid_key else None

    fig = plt.figure()
    plt.plot(train_curve, label=f"Train {metric}")
    if valid_curve is not None:
        plt.plot(valid_curve, label=f"Test {metric}")

    plt.title(title)
    plt.xlabel("Iteration / Trees")
    plt.ylabel(metric)

    if metric in ["merror", "error", "multi_error", "binary_error"]:
        plt.figtext(0.01, 0.01, "Note: error = 1 - accuracy. Lower is better.", fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.legend()
    return fig


def plot_accuracy_bar(results_df: pd.DataFrame, title="Test Accuracy Comparison Across Models"):
    """Plot test accuracy comparison across models."""
    fig = plt.figure()
    plt.bar(results_df["model"], results_df["test_accuracy"])
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.ylabel("Test Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    return fig


def plot_train_test_accuracy(results_df: pd.DataFrame, title="Train vs Test Accuracy Across Models"):
    """Plot train vs test accuracy across all trained models."""
    fig = plt.figure()
    x = np.arange(len(results_df))
    w = 0.38
    plt.bar(x - w / 2, results_df["train_accuracy"], width=w, label="Train")
    plt.bar(x + w / 2, results_df["test_accuracy"], width=w, label="Test")
    plt.xticks(x, results_df["model"], rotation=25, ha="right")
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    return fig


def get_feature_importance_df(pipe: Pipeline, top_n: int = 20):
    """
    Extract feature importance or coefficient influence from a trained pipeline.
    Supports:
    - Tree-based models: feature_importances_
    - Logistic Regression: absolute coefficient values
    """
    try:
        preprocessor = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if len(coef.shape) == 1:
                values = np.abs(coef)
            else:
                values = np.mean(np.abs(coef), axis=0)
        else:
            return None

        if len(feature_names) != len(values):
            return None

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": values
        })

        importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)
        return importance_df

    except Exception:
        return None


def create_classification_report_df(y_true, y_pred):
    """Return classification report as dataframe instead of plain text."""
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    return pd.DataFrame(report_dict).transpose()


def render_landing_page():
    """Render professional opening page before dataset upload."""
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Student Performance Prediction System</div>
            <div class="hero-subtitle">
                A Streamlit-based educational data mining prototype designed to identify at-risk
                students using supervised machine learning, model evaluation metrics, and
                interpretable visual outputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Project Purpose")
        st.write(
            "Support early identification of students who may be at academic risk by analysing "
            "educational datasets and generating risk-level predictions."
        )

    with c2:
        st.markdown("### Main Workflow")
        st.write(
            "Upload CSV data, configure the prediction label, train multiple models, compare "
            "performance, visualize results, and export predictions."
        )

    with c3:
        st.markdown("### Supported Models")
        st.write(
            "Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, and LightGBM are "
            "supported for comparative model evaluation."
        )

    st.markdown("### System Workflow")
    st.markdown(
        """
        1. Upload an anonymized student performance CSV dataset.  
        2. Select the target label or generate High / Medium / Low risk labels.  
        3. Run multiple machine learning models under the same train-test split.  
        4. Review accuracy, weighted F1-score, confusion matrix, learning curves, and feature influence.  
        5. Generate final predictions and export them as a CSV file.
        """
    )

    with st.expander("Expected Dataset Format"):
        st.write(
            "The system expects a structured CSV dataset containing student academic, demographic, "
            "or behavioural attributes."
        )

    with st.expander("Privacy and Prototype Scope"):
        st.write(
            "This prototype is designed for public or anonymized educational datasets. It does not "
            "connect to real university databases and does not permanently store confidential student records."
        )

    st.info("Please upload a CSV file from the sidebar to start the prediction workflow.")


# =========================================================
# 2) Sidebar
# =========================================================
st.sidebar.header("A) Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("B) Training Settings")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.header("C) Boosting Settings")
speed_mode = st.sidebar.selectbox("Boosting mode", ["Fast", "Balanced", "Accurate"], index=1)
trees = st.sidebar.slider("Number of trees (n_estimators)", 60, 260, 140, 10)

st.sidebar.header("D) Class Imbalance Handling")
class_weight_mode = st.sidebar.selectbox("class_weight", ["None", "Balanced"], index=1)

st.sidebar.header("E) Run")
run_all = st.sidebar.button("Run ALL Models", type="primary")

with st.sidebar.expander("System Notes"):
    st.write("Models to run: LR, DT, RF, SVM, XGBoost, LightGBM.")
    st.write("XGBoost and LightGBM will be skipped automatically if they are not installed.")
    st.write("SVM may be slower on datasets with many rows or many one-hot encoded features.")

if not _HAS_XGB:
    st.sidebar.caption("XGBoost not installed: XGBoost will be skipped.")
if not _HAS_LGBM:
    st.sidebar.caption("LightGBM not installed: LightGBM will be skipped.")


# =========================================================
# 3) Main Workflow: Landing Page
# =========================================================
if file is None:
    render_landing_page()
    st.stop()


# =========================================================
# 4) Load and Prepare Dataset
# =========================================================
df_original, delim_used = read_csv_auto(file)
original_rows = len(df_original)
df, duplicate_rows_removed = clean_duplicate_rows(df_original)

tabs = st.tabs([
    "Dataset & Label",
    "Run & Results",
    "History / Curves",
    "Comparison",
    "Predict & Export"
])


# =========================================================
# Tab 1: Dataset & Label
# =========================================================
with tabs[0]:
    st.subheader("1) Dataset Preview and Data Quality Overview")
    st.write(f"Detected delimiter: `{delim_used}` | Current rows: `{len(df)}` | Columns: `{len(df.columns)}`")

    st.dataframe(df.head(30), use_container_width=True)

    st.markdown("### Data Quality Overview")
    numeric_count = len(df.select_dtypes(include=["number"]).columns)
    categorical_count = len(df.columns) - numeric_count
    missing_cells = int(df.isna().sum().sum())

    q1, q2, q3, q4, q5 = st.columns(5)
    with q1:
        st.metric("Original Rows", original_rows)
    with q2:
        st.metric("Duplicates Removed", duplicate_rows_removed)
    with q3:
        st.metric("Current Rows", len(df))
    with q4:
        st.metric("Missing Cells", missing_cells)
    with q5:
        st.metric("Numeric / Categorical", f"{numeric_count} / {categorical_count}")

    with st.expander("Missing Values by Column"):
        missing_df = df.isna().sum().rename("missing_count").to_frame()
        missing_df = missing_df[missing_df["missing_count"] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.write("No missing values detected.")

    with st.expander("Numeric Correlation Matrix"):
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] >= 2:
            corr_df = numeric_df.corr(numeric_only=True)
            st.dataframe(corr_df, use_container_width=True)
        else:
            st.write("Not enough numeric columns to calculate correlation matrix.")

    st.subheader("2) Select Target Column or Generate Risk Label")

    default_target = "G3" if "G3" in df.columns else df.columns[0]
    target_col = st.selectbox(
        "Choose the target/label column",
        df.columns,
        index=list(df.columns).index(default_target)
    )

    st.markdown("**Optional and recommended:** Create a 3-level risk label from a numeric score column, such as `G3`.")
    use_risk_label = st.checkbox(
        "Create risk_level label (High / Medium / Low) from a numeric column",
        value=("G3" in df.columns)
    )

    risk_source_col = None
    low_cut, mid_cut = 10, 15

    if use_risk_label:
        numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_candidates) == 0:
            st.warning("No numeric columns found. Risk-level generation is unavailable.")
            use_risk_label = False
        else:
            default_idx = numeric_candidates.index("G3") if "G3" in numeric_candidates else 0
            risk_source_col = st.selectbox("Numeric column for risk label", numeric_candidates, index=default_idx)
            low_cut = st.slider("High risk if score is lower than", 0, 20, 10, 1)
            mid_cut = st.slider("Medium risk if score is lower than", 0, 20, 15, 1)

            if mid_cut <= low_cut:
                st.error("Medium cutoff must be greater than High cutoff, for example 10 and 15.")
                st.stop()
                
    if use_risk_label and risk_source_col is not None and mid_cut > low_cut:
        y_display = make_risk_label(df[risk_source_col], low_cut, mid_cut)
        X = df.drop(columns=[risk_source_col])
        label_name = f"risk_level (from {risk_source_col})"
    else:
        y_display = df[target_col]
        X = df.drop(columns=[target_col])
        label_name = target_col

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows for Analysis", len(df))
    with c2:
        st.metric("Features", X.shape[1])
    with c3:
        st.metric("Missing Cells in X", int(X.isna().sum().sum()))
    with c4:
        st.metric("Unique Labels", int(pd.Series(y_display).nunique(dropna=True)))

    st.markdown(f"### Label Distribution: `{label_name}`")
    label_dist_df = pd.Series(y_display).value_counts(dropna=False).rename("count").to_frame()
    st.dataframe(label_dist_df, use_container_width=True)
    st.bar_chart(label_dist_df)


# Keep final X/y for other tabs
valid_mask = pd.Series(y_display).notna()
X2 = X.loc[valid_mask].copy()
y2 = pd.Series(y_display).loc[valid_mask].astype(str).copy()


# =========================================================
# 5) Session State
# =========================================================
for key, default_value in {
    "trained_all": False,
    "results_df": None,
    "models_pack": None,
    "last_cfg": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

current_cfg = {
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
    "class_weight_mode": str(class_weight_mode),
    "columns": tuple(df.columns),
    "n_rows": int(len(df)),
    "file_name": getattr(file, "name", "uploaded.csv")
}

if st.session_state["last_cfg"] is None:
    st.session_state["last_cfg"] = current_cfg
elif st.session_state["last_cfg"] != current_cfg:
    st.session_state["trained_all"] = False
    st.session_state["results_df"] = None
    st.session_state["models_pack"] = None
    st.session_state["last_cfg"] = current_cfg


# =========================================================
# 6) Model Training Function
# =========================================================
def train_all_models(X_data: pd.DataFrame, y_data: pd.Series):
    """Train all available models and return summary results plus model pack."""
    if y_data.nunique(dropna=True) < 2:
        st.error("The selected label has fewer than 2 classes. A classifier cannot be trained.")
        return None, None

    # Same split for all models to ensure fair comparison.
    strat = y_data if y_data.value_counts().min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            test_size=test_size,
            random_state=int(random_state),
            stratify=strat
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            test_size=test_size,
            random_state=int(random_state),
            stratify=None
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(y_data.astype(str))

    y_train_fit = label_encoder.transform(y_train.astype(str))
    y_test_fit = label_encoder.transform(y_test.astype(str))

    num_classes = int(pd.Series(y_train_fit).nunique())
    labels_order = list(label_encoder.classes_)

    model_list = [
        "Logistic Regression (LR)",
        "Decision Tree (DT)",
        "Random Forest (RF)",
        "SVM (RBF)",
        "XGBoost (Main)",
        "LightGBM (Main)",
    ]

    results_rows = []
    pack = {}

    progress = st.progress(0)
    status = st.status("Starting model training...", expanded=True)

    for i, model_name in enumerate(model_list, start=1):
        if model_name == "XGBoost (Main)" and not _HAS_XGB:
            status.write("Skipping XGBoost because it is not installed.")
            progress.progress(int(i / len(model_list) * 100))
            continue

        if model_name == "LightGBM (Main)" and not _HAS_LGBM:
            status.write("Skipping LightGBM because it is not installed.")
            progress.progress(int(i / len(model_list) * 100))
            continue

        status.write(f"Training: **{model_name}** ...")
        start_time = time.time()

        history = None
        learning_fig = None

        try:
            if model_name in ["XGBoost (Main)", "LightGBM (Main)"]:
                preprocessor = make_preprocessor(X_train)
                Xtr = preprocessor.fit_transform(X_train)
                Xte = preprocessor.transform(X_test)

                model = build_model(model_name, num_classes, trees, speed_mode, class_weight_mode)

                buffer = io.StringIO()
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    if model_name == "XGBoost (Main)":
                        model.fit(
                            Xtr,
                            y_train_fit,
                            eval_set=[(Xtr, y_train_fit), (Xte, y_test_fit)],
                            verbose=False
                        )
                        history = model.evals_result()
                    else:
                        model.fit(
                            Xtr,
                            y_train_fit,
                            eval_set=[(Xtr, y_train_fit), (Xte, y_test_fit)],
                            eval_metric=("multi_logloss" if num_classes > 2 else "binary_logloss")
                        )
                        history = model.evals_result_

                pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

            else:
                pipe = build_pipeline(model_name, X_train, num_classes, trees, speed_mode, class_weight_mode)

                buffer = io.StringIO()
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    pipe.fit(X_train, y_train_fit)

                try:
                    learning_fig = plot_learning_curve(
                        pipe,
                        X_train,
                        y_train_fit,
                        title=f"{model_name} - Learning Curve (Train vs Test/CV)",
                        cv=3
                    )
                except Exception:
                    learning_fig = None

            y_pred_test_fit = pipe.predict(X_test)
            y_pred_train_fit = pipe.predict(X_train)

            y_pred_test_show = label_encoder.inverse_transform(y_pred_test_fit.astype(int))
            y_test_show = label_encoder.inverse_transform(y_test_fit.astype(int))

            y_pred_train_show = label_encoder.inverse_transform(y_pred_train_fit.astype(int))
            y_train_show = label_encoder.inverse_transform(y_train_fit.astype(int))

            train_acc = accuracy_score(y_train_show, y_pred_train_show)
            test_acc = accuracy_score(y_test_show, y_pred_test_show)
            test_f1 = f1_score(y_test_show, y_pred_test_show, average="weighted", zero_division=0)
            elapsed = time.time() - start_time

            pack[model_name] = {
                "pipe": pipe,
                "label_encoder": label_encoder,
                "labels_order": labels_order,
                "y_test_show": y_test_show,
                "y_pred_show": y_pred_test_show,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "f1": test_f1,
                "elapsed": elapsed,
                "history": history,
                "learning_fig": learning_fig,
            }

            results_rows.append({
                "model": model_name,
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "f1_weighted": float(test_f1),
                "train_time_s": float(elapsed),
                "has_history": bool(history is not None),
                "has_learning_curve": bool(learning_fig is not None),
            })

            status.write(
                f"Done: {model_name} | test_acc={test_acc:.4f} | "
                f"f1={test_f1:.4f} | time={elapsed:.2f}s"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            status.write(f"Failed: {model_name} ({elapsed:.2f}s) → {e}")

        progress.progress(int(i / len(model_list) * 100))

    status.update(label="All available models completed.", state="complete", expanded=False)

    if len(results_rows) == 0:
        return None, None

    results_df = pd.DataFrame(results_rows).sort_values(
        ["test_accuracy", "f1_weighted"],
        ascending=False
    ).reset_index(drop=True)

    return results_df, pack


# =========================================================
# Tab 2: Run & Results
# =========================================================
with tabs[1]:
    st.subheader("3) Model Training and Evaluation")
    st.write(
        "Click **Run ALL Models** in the sidebar to train all available models and compare "
        "their train accuracy, test accuracy, weighted F1-score, and training time."
    )

    if run_all:
        if y2.nunique(dropna=True) < 2:
            st.error("The selected label has fewer than 2 unique classes. Please choose another label.")
        else:
            with st.spinner("Running all models. SVM and boosting models may take longer on large datasets."):
                results_df, model_pack = train_all_models(X2, y2)

            st.session_state["trained_all"] = bool(results_df is not None and len(results_df) > 0)
            st.session_state["results_df"] = results_df
            st.session_state["models_pack"] = model_pack

    if not st.session_state["trained_all"]:
        st.info("Run training first using the sidebar button: **Run ALL Models**.")
    else:
        results_df = st.session_state["results_df"]
        model_pack = st.session_state["models_pack"]

        st.markdown("### Summary Table")
        st.dataframe(results_df, use_container_width=True)

        best_row = results_df.iloc[0]
        st.success(
            f"Recommended Model: {best_row['model']} | "
            f"Test Accuracy: {best_row['test_accuracy']:.4f} | "
            f"Weighted F1-score: {best_row['f1_weighted']:.4f}"
        )

        with st.expander("Why this model is recommended"):
            st.write(
                "The recommended model is selected based on the highest test accuracy, with weighted "
                "F1-score used as an additional comparison indicator. Test accuracy reflects model "
                "generalization on unseen data, while weighted F1-score balances precision and recall "
                "across classes."
            )

        st.markdown("### Detailed Results by Model")
        chosen = st.selectbox("Select a model to view details", results_df["model"].tolist())
        info = model_pack[chosen]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Train Accuracy", f"{info['train_acc']:.4f}")
        with m2:
            st.metric("Test Accuracy", f"{info['test_acc']:.4f}")
        with m3:
            st.metric("Weighted F1-score", f"{info['f1']:.4f}")
        with m4:
            st.metric("Training Time (s)", f"{info['elapsed']:.2f}")

        st.markdown("#### Confusion Matrix")
        cm_df = plot_confusion_matrix_df(info["y_test_show"], info["y_pred_show"], info["labels_order"])
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("#### Classification Report")
        report_df = create_classification_report_df(info["y_test_show"], info["y_pred_show"])
        st.dataframe(report_df, use_container_width=True)

        st.markdown("#### Feature Importance / Feature Influence")
        fi_df = get_feature_importance_df(info["pipe"], top_n=20)

        if fi_df is not None and len(fi_df) > 0:
            st.dataframe(fi_df, use_container_width=True)
            st.bar_chart(fi_df.set_index("feature")["importance"])
        else:
            st.info(
                "Feature importance is not directly available for this model. Tree-based models "
                "and Logistic Regression usually provide clearer feature influence outputs."
            )

        st.download_button(
            "Download metrics table as results.csv",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="results.csv",
            mime="text/csv"
        )


# =========================================================
# Tab 3: History / Curves
# =========================================================
with tabs[2]:
    st.subheader("4) Training History and Learning Curves")
    st.write(
        "Boosting models show iteration-based training history. Other models show learning "
        "curves based on train size and cross-validation accuracy."
    )

    if not st.session_state["trained_all"]:
        st.info("Run all models first from the sidebar.")
    else:
        results_df = st.session_state["results_df"]
        model_pack = st.session_state["models_pack"]

        chosen_curve_model = st.selectbox(
            "Select a model for curve visualization",
            results_df["model"].tolist(),
            key="curve_select"
        )

        info = model_pack[chosen_curve_model]

        if info["history"] is not None:
            fig = plot_boosting_history(
                info["history"],
                title=f"{chosen_curve_model} - Training History (Train vs Test)"
            )
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("Training history is not available for plotting.")

        if info["learning_fig"] is not None:
            st.pyplot(info["learning_fig"])
        elif info["history"] is None:
            st.warning("Learning curve is not available for this model or dataset.")

        with st.expander("Interpretation Notes"):
            st.markdown(
                """
                - A large gap between train and test performance may indicate overfitting.  
                - Low train and test performance may indicate underfitting.  
                - Boosting history helps observe convergence across iterations or trees.  
                - Learning curves help explain whether more data may improve performance.
                """
            )


# =========================================================
# Tab 4: Comparison
# =========================================================
with tabs[3]:
    st.subheader("5) Model Comparison Dashboard")
    st.write(
        "This section provides overall model comparison using test accuracy and train-versus-test accuracy."
    )

    if not st.session_state["trained_all"]:
        st.info("Run all models first from the sidebar.")
    else:
        results_df = st.session_state["results_df"].copy()
        st.dataframe(results_df, use_container_width=True)

        st.markdown("### Test Accuracy Comparison")
        st.pyplot(plot_accuracy_bar(results_df))

        st.markdown("### Train vs Test Accuracy")
        st.pyplot(plot_train_test_accuracy(results_df))

        with st.expander("How to Explain This in the Report"):
            st.markdown(
                """
                - The test accuracy chart compares generalization performance across all trained models.  
                - The train-versus-test chart helps identify whether a model is overfitting or underfitting.  
                - The best model should be selected based on test performance, weighted F1-score, and interpretability.
                """
            )


# =========================================================
# Tab 5: Predict & Export
# =========================================================
with tabs[4]:
    st.subheader("6) Predict and Export Results")
    st.write(
        "Choose a trained model to generate predictions for the full uploaded dataset and export the results."
    )

    if not st.session_state["trained_all"]:
        st.info("Train models first using the sidebar button: **Run ALL Models**.")
    else:
        results_df = st.session_state["results_df"]
        model_pack = st.session_state["models_pack"]

        chosen_model_for_prediction = st.selectbox(
            "Choose a trained model for final prediction",
            results_df["model"].tolist(),
            index=0
        )

        pipe = model_pack[chosen_model_for_prediction]["pipe"]
        label_encoder = model_pack[chosen_model_for_prediction]["label_encoder"]

        pred_all_fit = pipe.predict(X)
        pred_all = label_encoder.inverse_transform(pred_all_fit.astype(int))

        output_df = df.copy()
        output_df["predicted_label"] = pred_all

        try:
            model_obj = pipe.named_steps["model"]
            if hasattr(model_obj, "predict_proba"):
                proba = pipe.predict_proba(X)
                output_df["confidence"] = np.max(proba, axis=1)
            else:
                output_df["confidence"] = np.nan
        except Exception:
            output_df["confidence"] = np.nan

        st.markdown(f"**Selected model:** `{chosen_model_for_prediction}`")

        st.markdown("### Prediction Preview")
        st.dataframe(output_df.head(40), use_container_width=True)

        st.markdown("### Predicted Risk / Label Distribution")
        pred_dist_df = output_df["predicted_label"].value_counts().rename("count").to_frame()
        st.dataframe(pred_dist_df, use_container_width=True)
        st.bar_chart(pred_dist_df)

        st.download_button(
            "Download predictions as predictions.csv",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
