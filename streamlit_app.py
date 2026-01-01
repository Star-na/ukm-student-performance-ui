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
APP_TAB_TITLE = "Student Performance UI"
APP_MAIN_TITLE = "Student Performance Prediction System (Interface Prototype)"
APP_SUBTITLE = "Upload → Preview/Select Label → Run ALL Models → History/Learning Curves → Accuracy Comparison → Predict → Export"

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


def build_model(name: str, num_classes: int, trees: int, speed_mode: str, class_weight_mode: str):
    # class_weight_mode: "None" or "Balanced"
    cw = None if class_weight_mode == "None" else "balanced"

    # -------------------------
    # Baselines
    # -------------------------
    if name == "Logistic Regression (LR)":
        # More stable defaults for high-dimensional one-hot
        return LogisticRegression(
            max_iter=1200,
            solver="lbfgs",
            class_weight=cw
        )

    if name == "Decision Tree (DT)":
        return DecisionTreeClassifier(random_state=42, class_weight=cw)

    if name == "Random Forest (RF)":
        return RandomForestClassifier(
            n_estimators=min(250, max(80, trees)),
            random_state=42,
            n_jobs=-1,
            class_weight=cw
        )

    if name == "SVM (RBF)":
        # SVM can be slow on large data; keep default; class_weight supported
        return SVC(kernel="rbf", probability=True, class_weight=cw)

    # -------------------------
    # Main models (Boosting)
    # -------------------------
    if name == "XGBoost (Main)":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed. Add `xgboost` to requirements.txt and redeploy.")

        # speed presets
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
            # error = 1-accuracy over iterations
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
            raise RuntimeError("lightgbm not installed. Add `lightgbm` to requirements.txt and redeploy.")

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

    raise ValueError("Unknown model")


def build_pipeline(model_name: str, X: pd.DataFrame, num_classes: int, trees: int, speed_mode: str, class_weight_mode: str) -> Pipeline:
    preprocessor = make_preprocessor(X)
    model = build_model(model_name, num_classes, trees, speed_mode, class_weight_mode)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def plot_confusion_matrix_df(y_true, y_pred, labels_order):
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{l}" for l in labels_order],
        columns=[f"pred:{l}" for l in labels_order]
    )
    return cm_df


def plot_learning_curve(pipe, X, y, title="Learning Curve (Train vs Test/CV)", cv=3):
    # Small CV for speed; still valid for visualization
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y,
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
    """
    XGBoost history: dict like {"validation_0": {"logloss":[...], "error":[...]}, "validation_1": {...}}
    LightGBM history: similar via evals_result_.
    We'll prefer an "error/merror" metric (for accuracy history) if available, else logloss.
    """
    if not history or len(history.keys()) == 0:
        return None

    keys = list(history.keys())
    train_key = keys[0]
    valid_key = keys[1] if len(keys) > 1 else None

    metrics = list(history[train_key].keys())

    # Prefer error-like metric (lower is better); show it and label as 1-accuracy
    prefer = None
    for m in ["merror", "error", "multi_error", "binary_error"]:
        if m in metrics:
            prefer = m
            break
    metric = prefer if prefer else metrics[0]

    train_curve = history[train_key][metric]
    valid_curve = history[valid_key][metric] if valid_key else None

    fig = plt.figure()
    plt.plot(train_curve, label=f"Train {metric}")
    if valid_curve is not None:
        plt.plot(valid_curve, label=f"Test {metric}")

    plt.title(title)
    plt.xlabel("Iteration / Trees")
    plt.ylabel(metric)

    # If metric is error-like, also hint accuracy
    if metric in ["merror", "error", "multi_error", "binary_error"]:
        plt.figtext(0.01, 0.01, "Note: error = 1 - accuracy (lower is better).", fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.legend()
    return fig


def plot_accuracy_bar(results_df: pd.DataFrame, title="Accuracy Comparison Across Models"):
    fig = plt.figure()
    plt.bar(results_df["model"], results_df["test_accuracy"])
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.ylabel("Test Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    return fig


def plot_train_test_accuracy(results_df: pd.DataFrame, title="Train vs Test Accuracy (All Models)"):
    fig = plt.figure()
    x = np.arange(len(results_df))
    w = 0.38
    plt.bar(x - w/2, results_df["train_accuracy"], width=w, label="Train")
    plt.bar(x + w/2, results_df["test_accuracy"], width=w, label="Test")
    plt.xticks(x, results_df["model"], rotation=25, ha="right")
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    return fig


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("A) Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("B) Settings")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.header("C) Boosting Settings (XGB/LGBM)")
speed_mode = st.sidebar.selectbox("Boosting mode", ["Fast", "Balanced", "Accurate"], index=1)
trees = st.sidebar.slider("Number of trees (n_estimators)", 60, 260, 140, 10)

st.sidebar.header("D) Imbalance Handling")
class_weight_mode = st.sidebar.selectbox("class_weight", ["None", "Balanced"], index=1)

st.sidebar.header("E) Run")
run_all = st.sidebar.button("Run ALL Models", type="primary")

st.sidebar.caption("Models to run: LR, DT, RF, SVM, XGBoost, LightGBM")
if not _HAS_XGB:
    st.sidebar.caption("⚠️ xgboost not installed → XGBoost will be skipped.")
if not _HAS_LGBM:
    st.sidebar.caption("⚠️ lightgbm not installed → LightGBM will be skipped.")
st.sidebar.caption("Tip: SVM can be slow on large datasets (many rows / many one-hot features).")


# =========================================================
# Main workflow
# =========================================================
if file is None:
    st.info("⬅️ Please upload a CSV file from the sidebar to start.")
    st.stop()

df, delim_used = read_csv_auto(file)

tabs = st.tabs(["Dataset & Label", "Run & Results", "History / Curves", "Comparison", "Predict & Export"])

# -------------------------
# Tab 1: Dataset & Label
# -------------------------
with tabs[0]:
    st.subheader("1) Dataset Preview (Input)")
    st.write(f"Detected delimiter: `{delim_used}` | Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df.head(30), width="stretch")

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

    # Summary cards
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

# Keep these for other tabs
valid_mask = pd.Series(y_display).notna()
X2 = X.loc[valid_mask].copy()
y2 = pd.Series(y_display).loc[valid_mask].copy()

# =========================================================
# Session state
# =========================================================
for k, v in {
    "trained_all": False,
    "results_df": None,
    "models_pack": None,   # dict model_name -> {"pipe":..., "le":..., "labels":..., "metrics":..., "history":..., "learning_fig":...}
    "last_cfg": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
}

if st.session_state["last_cfg"] is None:
    st.session_state["last_cfg"] = current_cfg
elif st.session_state["last_cfg"] != current_cfg:
    st.session_state["trained_all"] = False
    st.session_state["results_df"] = None
    st.session_state["models_pack"] = None
    st.session_state["last_cfg"] = current_cfg


# =========================================================
# Run ALL models
# =========================================================
def train_all_models(X2, y2):
    nunique = y2.nunique(dropna=True)
    if nunique < 2:
        st.error("Your label has < 2 unique classes. Cannot train a classifier.")
        return None, None

    # Single split for fair comparison
    strat = y2 if y2.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2,
        test_size=test_size,
        random_state=int(random_state),
        stratify=strat
    )

    # Encode labels once for all models
    le = LabelEncoder()
    le.fit(y2.astype(str))
    y_train_fit = le.transform(y_train.astype(str))
    y_test_fit = le.transform(y_test.astype(str))
    num_classes = int(pd.Series(y_train_fit).nunique())
    labels_order = list(le.classes_)  # stable order for confusion matrix

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
    status = st.status("Starting training...", expanded=True)

    # For history/curve figures, we may want to limit costs; use smaller CV for learning curve
    for i, mname in enumerate(model_list, start=1):
        # Skip missing optional libs
        if mname == "XGBoost (Main)" and not _HAS_XGB:
            status.write("Skipping XGBoost (not installed).")
            progress.progress(int(i / len(model_list) * 100))
            continue
        if mname == "LightGBM (Main)" and not _HAS_LGBM:
            status.write("Skipping LightGBM (not installed).")
            progress.progress(int(i / len(model_list) * 100))
            continue

        status.write(f"Training: **{mname}** ...")
        t0 = time.time()

        history = None
        learning_fig = None

        try:
            # Boosting: manual preprocess to capture eval history (train vs test)
            if mname in ["XGBoost (Main)", "LightGBM (Main)"]:
                preprocessor = make_preprocessor(X_train)
                Xtr = preprocessor.fit_transform(X_train)
                Xte = preprocessor.transform(X_test)

                model = build_model(mname, num_classes, trees, speed_mode, class_weight_mode)

                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    if mname == "XGBoost (Main)":
                        model.fit(
                            Xtr, y_train_fit,
                            eval_set=[(Xtr, y_train_fit), (Xte, y_test_fit)],
                            verbose=False
                        )
                        history = model.evals_result()
                    else:
                        # LightGBM
                        model.fit(
                            Xtr, y_train_fit,
                            eval_set=[(Xtr, y_train_fit), (Xte, y_test_fit)],
                            eval_metric=("multi_logloss" if num_classes > 2 else "binary_logloss"),
                        )
                        history = model.evals_result_

                pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

            else:
                # Non-boosting: standard pipeline
                pipe = build_pipeline(mname, X_train, num_classes, trees, speed_mode, class_weight_mode)
                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    pipe.fit(X_train, y_train_fit)

                # Learning curve (Train vs Test in same plot)
                # Note: learning_curve expects y labels; we already have encoded labels
                try:
                    learning_fig = plot_learning_curve(
                        pipe,
                        X_train,
                        y_train_fit,
                        title=f"{mname} - Learning Curve (Train vs Test/CV)",
                        cv=3
                    )
                except Exception:
                    learning_fig = None

            # Predict & metrics
            y_pred_fit = pipe.predict(X_test)
            y_pred_show = le.inverse_transform(y_pred_fit.astype(int))
            y_test_show = le.inverse_transform(y_test_fit.astype(int))

            # Also compute train accuracy for comparison plot
            y_pred_train_fit = pipe.predict(X_train)
            y_pred_train_show = le.inverse_transform(y_pred_train_fit.astype(int))
            y_train_show = le.inverse_transform(y_train_fit.astype(int))

            train_acc = accuracy_score(y_train_show, y_pred_train_show)
            test_acc = accuracy_score(y_test_show, y_pred_show)
            test_f1 = f1_score(y_test_show, y_pred_show, average="weighted")

            elapsed = time.time() - t0

            # Save per-model artifacts
            pack[mname] = {
                "pipe": pipe,
                "label_encoder": le,
                "labels_order": labels_order,
                "y_test_show": y_test_show,
                "y_pred_show": y_pred_show,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "f1": test_f1,
                "elapsed": elapsed,
                "history": history,
                "learning_fig": learning_fig,
            }

            results_rows.append({
                "model": mname,
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "f1_weighted": float(test_f1),
                "train_time_s": float(elapsed),
                "has_history": bool(history is not None),
                "has_learning_curve": bool(learning_fig is not None),
            })

            status.write(f"✅ Done: {mname} | test_acc={test_acc:.4f} | f1={test_f1:.4f} | time={elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - t0
            status.write(f"❌ Failed: {mname} ({elapsed:.2f}s) → {e}")

        progress.progress(int(i / len(model_list) * 100))

    status.update(label="All models completed.", state="complete", expanded=False)
    results_df = pd.DataFrame(results_rows).sort_values("test_accuracy", ascending=False)
    return results_df, pack


# -------------------------
# Tab 2: Run & Results
# -------------------------
with tabs[1]:
    st.subheader("3) Run ALL Models (Training + Evaluation Output)")
    st.write("Click **Run ALL Models** in the sidebar. This will train LR, DT, RF, SVM, XGBoost, LightGBM (if installed) and produce metrics + confusion matrices.")

    if run_all:
        if y2.nunique(dropna=True) < 2:
            st.error("Your label has < 2 unique classes. Cannot train.")
        else:
            with st.spinner("Running all models... (SVM may be slow; Boosting may take longer in Accurate mode)"):
                results_df, pack = train_all_models(X2, y2)
            st.session_state["trained_all"] = True if results_df is not None and len(results_df) > 0 else False
            st.session_state["results_df"] = results_df
            st.session_state["models_pack"] = pack

    if not st.session_state["trained_all"]:
        st.info("Run training first (sidebar → **Run ALL Models**).")
    else:
        results_df = st.session_state["results_df"]
        pack = st.session_state["models_pack"]

        st.markdown("### Summary Table (All Models)")
        st.dataframe(results_df, width="stretch")

        st.markdown("### Detailed Results (per model)")
        chosen = st.selectbox("Select a model to view details", results_df["model"].tolist())
        info = pack[chosen]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Train Acc", f"{info['train_acc']:.4f}")
        with m2:
            st.metric("Test Acc", f"{info['test_acc']:.4f}")
        with m3:
            st.metric("F1 (weighted)", f"{info['f1']:.4f}")
        with m4:
            st.metric("Train time (s)", f"{info['elapsed']:.2f}")

        st.write("Confusion Matrix")
        cm_df = plot_confusion_matrix_df(info["y_test_show"], info["y_pred_show"], info["labels_order"])
        st.dataframe(cm_df, width="stretch")

        st.write("Classification Report")
        st.text(classification_report(info["y_test_show"], info["y_pred_show"]))

        # Export results table
        st.download_button(
            "Download metrics table (results.csv)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="results.csv",
            mime="text/csv"
        )

# -------------------------
# Tab 3: History / Curves
# -------------------------
with tabs[2]:
    st.subheader("4) Training History / Learning Curves (Train vs Test in same plot)")
    st.write("Boosting models (XGBoost/LightGBM) show **iteration history**. Other models show **learning curves** (train size vs accuracy), which also plots Train and Test in the same figure.")

    if not st.session_state["trained_all"]:
        st.info("Run ALL models first (sidebar → **Run ALL Models**).")
    else:
        results_df = st.session_state["results_df"]
        pack = st.session_state["models_pack"]

        chosen2 = st.selectbox("Select a model for curves", results_df["model"].tolist(), key="curve_select")
        info = pack[chosen2]

        # Boosting history
        if info["history"] is not None:
            fig = plot_boosting_history(info["history"], title=f"{chosen2} - Training History (Train vs Test)")
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("History not available for plotting.")

        # Learning curve
        if info["learning_fig"] is not None:
            st.pyplot(info["learning_fig"])
        elif info["history"] is None:
            st.warning("Learning curve not available (may fail for certain datasets).")

        # Helpful notes
        with st.expander("Model Notes (for report write-up)"):
            st.markdown("""
- **Boosting history** (XGBoost/LightGBM): shows model performance across iterations/trees, allowing you to compare Train vs Test dynamics and detect overfitting.
- **Learning curve** (LR/DT/RF/SVM): shows how Train and Test/CV accuracy change as the training set grows; helpful to explain bias/variance and data sufficiency.
            """)

# -------------------------
# Tab 4: Comparison
# -------------------------
with tabs[3]:
    st.subheader("5) Accuracy Visualization (All Models)")
    st.write("This section provides the **accuracy visualization** requested: overall model comparison and Train vs Test accuracy in the same chart set.")

    if not st.session_state["trained_all"]:
        st.info("Run ALL models first (sidebar → **Run ALL Models**).")
    else:
        results_df = st.session_state["results_df"].copy()
        st.dataframe(results_df, width="stretch")

        st.pyplot(plot_accuracy_bar(results_df, title="Test Accuracy Comparison (All Models)"))
        st.pyplot(plot_train_test_accuracy(results_df, title="Train vs Test Accuracy (All Models)"))

        with st.expander("Interpretation Tips (copy to report)"):
            st.markdown("""
- The **Test Accuracy bar chart** ranks models based on generalization performance on the held-out test set.
- The **Train vs Test chart** helps identify overfitting (very high train accuracy but lower test accuracy) or underfitting (both low).
- Combine this with confusion matrices to explain which classes are hardest to predict.
            """)

# -------------------------
# Tab 5: Predict & Export
# -------------------------
with tabs[4]:
    st.subheader("6) Predict & Export")
    st.write("Choose the best-performing model (or any model) to generate predictions for the full dataset and export as CSV.")

    if not st.session_state["trained_all"]:
        st.info("Train models first (sidebar → **Run ALL Models**).")
        st.stop()

    results_df = st.session_state["results_df"]
    pack = st.session_state["models_pack"]

    default_best = results_df.iloc[0]["model"] if len(results_df) else None
    chosen_model_for_pred = st.selectbox(
        "Choose a trained model for final prediction",
        results_df["model"].tolist(),
        index=0
    )

    pipe = pack[chosen_model_for_pred]["pipe"]
    le = pack[chosen_model_for_pred]["label_encoder"]

    # Predict on full X (including rows with NaN label; that's fine)
    pred_all_fit = pipe.predict(X)
    pred_all = le.inverse_transform(pred_all_fit.astype(int))

    out = df.copy()
    out["predicted_label"] = pred_all

    # confidence if available
    model_obj = pipe.named_steps["model"]
    try:
        if hasattr(model_obj, "predict_proba"):
            proba = pipe.predict_proba(X)
            out["confidence"] = np.max(proba, axis=1)
        else:
            out["confidence"] = np.nan
    except Exception:
        out["confidence"] = np.nan

    st.markdown(f"**Selected model:** {chosen_model_for_pred}")
    st.dataframe(out.head(40), width="stretch")

    st.download_button(
        "Download predictions as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )
