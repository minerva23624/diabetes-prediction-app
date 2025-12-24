# =========================================================
# DIABETES PREDICTION APP - FINAL VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction using Machine Learning")
st.markdown("**Comparaison de modèles ML & essai patient interactif**")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    return pd.read_csv(url, names=cols)

data = load_data()
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# =========================================================
# TRAIN / TEST + SMOTE
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🔧 Navigation")
mode = st.sidebar.radio(
    "Choisir le mode",
    ["Comparaison des modèles", "Essai Patient"]
)

# =========================================================
# MODE 1 : MODEL COMPARISON
# =========================================================
if mode == "Comparaison des modèles":

    st.subheader("🔍 Comparaison des modèles ML")

    model_name = st.sidebar.selectbox(
        "Modèle",
        ["Logistic Regression", "Random Forest", "SVM"]
    )

    # -------- Model selection --------
    if model_name == "Logistic Regression":
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        model = LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced"
        )

    elif model_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 300, 100)
        max_depth = st.sidebar.slider("max_depth", 2, 20, 6)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42
        )

    else:
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf"])
        model = SVC(
            C=C, kernel=kernel,
            probability=True,
            class_weight="balanced"
        )

    # -------- Train & Predict --------
    model.fit(X_train_scaled, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    col1, col2 = st.columns(2)

    # -------- Classification Report --------
    with col1:
        st.subheader("📄 Classification Report")
        st.text(classification_report(y_test, y_pred))

    # -------- Confusion Matrix --------
    with col2:
        st.subheader("🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # -------- ROC Curve --------
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0,1],[0,1],'--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

# =========================================================
# MODE 2 : PATIENT TEST
# =========================================================
else:
    st.subheader("🧪 Essai Patient")

    model_choice = st.selectbox(
        "Choisir le modèle",
        ["Logistic Regression", "Random Forest", "SVM"]
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(
            class_weight="balanced", random_state=42
        )
    else:
        model = SVC(
            probability=True, class_weight="balanced"
        )

    model.fit(X_train_scaled, y_train_res)

    with st.form("patient_form"):
        c1, c2 = st.columns(2)

        with c1:
            preg = st.number_input("Pregnancies", 0, 20, 1)
            glu = st.number_input("Glucose", 50, 250, 120)
            bp = st.number_input("Blood Pressure", 40, 150, 70)
            skin = st.number_input("Skin Thickness", 5, 100, 20)

        with c2:
            ins = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 10, 100, 30)

        submit = st.form_submit_button("🔮 Prédire")

    if submit:
        patient = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
        patient_scaled = scaler.transform(patient)

        pred = model.predict(patient_scaled)[0]
        prob = model.predict_proba(patient_scaled)[0][1]

        st.divider()
        if pred == 1:
            st.error("⚠️ Risque élevé de diabète")
        else:
            st.success("✅ Risque faible de diabète")

        st.metric("Probabilité estimée", f"{prob*100:.2f}%")

        st.warning(
            "Cette application est un outil d’aide à la décision "
            "et ne remplace pas un diagnostic médical."
        )
