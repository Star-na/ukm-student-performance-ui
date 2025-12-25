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

# =========================================================
# 0) UI NAME (改名字就改这里)
# =========================================================
APP_TAB_TITLE = "UKM Student Performance UI"
APP_MAIN_TITLE = "UKM Student Performance Prediction System (Interface Prototype)"
APP_SUBTITLE = "Upload → Preview/Select Target → Train/Evaluate → Predict → Export"

st.set_page_config(page_title=APP_TAB_TITLE, layout="wide")
st.title(APP_MAIN_TITLE)
st.caption(APP_SUBTITLE)

# Step indicator (for D4 screenshots)
st.divider()
st.write("### Workflow")
cA, cB, cC, cD = st.columns(4)
cA.success("1) Upload")
cB.info("2) Select Target")
cC.warning("3) Train & Evaluate")
cD.error("4) Predict & Export")
st.divider()

# =========================================================
# Helper: model factory
# =========================================================
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

# Keep trained pipeline in session
if "pipe" not in st.session_state:
    st.session_state["pipe"] = None
if "trained" not in st.session_state:
    st.session_state["trained"] = False

# =========================================================
# 1) Sidebar: inputs
# =========================================================
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

# =========================================================
# 2) Main: read data
# =========================================================
if file is None:
    st.info("⬅️ Please upload a CSV file from the sidebar to start.")
    st.stop()

try:
    df = pd.read_csv(file)
except Exception as e:
    st.error("Failed to read CSV. Please check your file encoding/format.")
    st.exception(e)
    st.stop()

# Basic validation
if df.shape[1] < 2:
    st.error("CSV must contain at least 2 columns (features + target).")
    st.stop()

st.subheader("1) Dataset Preview (Input)")
st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# 3) Select target + features
# =========================================================
st.subheader("2) Select Target Column (Label)")
target_col = st.selectbox("Choose the target/label column", df.columns)

X_all = df.drop(columns=[target_col])
y_all = df[target_col]

# Let user optionally select features (nice for interface)
st.write("Optional: Select feature columns (default = all features).")
feature_cols = st.multiselect(
    "Feature columns",
    options=X_all.columns.tolist(),
    default=X_all.columns.tolist()
)

if len(feature_cols) == 0:
    st.warning("Please select at least 1 feature column.")
    st.stop()

X = X_all[feature_cols]
y = y_all

# Data summary (D4 interface evidence)
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

# Check if label is usable for classification
if y.nunique() < 2:
    st.error("Target column must contain at least 2 different classes for classification.")
    st.stop()

# =========================================================
# 4) Preprocessing + Pipeline
# =========================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

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

# =========================================================
# 5) Train & Evaluate
# =========================================================
st.subheader("3) Training & Evaluation (Process + Output)")
st.write("Press **Run Training** in the sidebar to train and evaluate the model.")

if run_train:
    # Split (try stratify; if fails, fallback)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=int(random_state)
        )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.session_state["pipe"] = pipe
    st.session_state["trained"] = True

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Model", model_name)
    with m2:
        st.metric("Accuracy", f"{acc:.4f}")
    with m3:
        st.metric("F1 (weighted)", f"{f1:.4f}")

    st.write("Confusion Matrix")
    labels = sorted(list(pd.Series(y).dropna().unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True:{l}" for l in labels], columns=[f"Pred:{l}" for l in labels])
    st.dataframe(cm_df, use_container_width=True)

    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

# =========================================================
# 6) Predict & Export
# =========================================================
st.subheader("4) Predict & Export (Output)")

if not st.session_state["trained"]:
    st.info("Train the model first (click **Run Training**).")
    st.stop()

pipe = st.session_state["pipe"]

# Predict on all rows
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

# Optional: map to risk level if looks like 3-class numeric (0/1/2 etc.)
st.write("Optional: If your label represents risk levels, you can map predicted labels to High/Medium/Low.")
map_risk = st.checkbox("Map predicted_label to risk_level (auto for 3-class numeric labels)", value=False)

if map_risk:
    # Auto-map only when numeric and 3 unique labels
    unique_labels = pd.Series(pred_all).dropna().unique()
    try:
        unique_sorted = sorted([float(x) for x in unique_labels])
        if len(unique_sorted) == 3:
            # lowest -> Low, middle -> Medium, highest -> High
            mapping = {unique_sorted[0]: "Low", unique_sorted[1]: "Medium", unique_sorted[2]: "High"}
            # convert each prediction to float for mapping
            out["risk_level"] = pd.Series(pred_all).apply(lambda v: mapping.get(float(v), str(v)))
        else:
            out["risk_level"] = pd.Series(pred_all).astype(str)
            st.warning("Auto mapping works best for 3-class numeric labels. Your labels are not exactly 3 numeric classes.")
    except Exception:
        out["risk_level"] = pd.Series(pred_all).astype(str)
        st.warning("Labels are not numeric; used string labels as risk_level.")

st.dataframe(out.head(30), use_container_width=True)

st.download_button(
    "Download predictions as CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)

st.success("Done. You can now take screenshots for the D4 Interface section.")
