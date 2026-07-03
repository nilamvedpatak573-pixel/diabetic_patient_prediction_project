import streamlit as st
import numpy as np
import pickle

# ---------------- Page Setup ---------------- #
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# ---------------- Professional Medical Theme & Layout ---------------- #
st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
}

/* Clinical Typography */
.main-title {
    text-align: center;
    color: #0369A1; /* Clinical Deep Blue-Teal */
    font-weight: 700;
    font-size: 38px;
    margin-bottom: 8px;
}

.sub-title {
    text-align: center;
    color: #0EA5E9; /* Medical Accent Blue */
    font-weight: 500;
    font-size: 18px;
    margin-bottom: 5px;
}

/* Styled Input Form Container & Border Design */
div[data-testid="stForm"] {
    background-color: #F0F7FF !important; /* Soft Clinical Blue Background */
    border: 2px solid #0369A1 !important; /* Emphasized Deep Blue Border Design */
    border-radius: 12px !important;
    padding: 25px !important;
    box-shadow: 0 4px 12px rgba(3, 105, 161, 0.08) !important;
}

/* Forces the Diagnostic Parameters header to be visible and clear */
div[data-testid="stForm"] h3 {
    color: #0369A1 !important;
    font-weight: 600 !important;
}

/* Professional Input Rows & Fields Styling */
div[data-testid="stForm"] div[data-baseweb="input"] {
    background-color: #0369A1 !important; /* Professional Deep Blue */
    border-radius: 6px !important;
    border: 1px solid #0284C7 !important;
}

/* Input Field Text & Icons */
div[data-testid="stForm"] input {
    color: #FFFFFF !important; /* Crisp White Text for readability */
    font-weight: 500 !important;
}

/* --- Dark Blue Color Fix for + and - Step Buttons --- */
div[data-testid="stForm"] button[data-testid="stNumberInputStepDown"], 
div[data-testid="stForm"] button[data-testid="stNumberInputStepUp"] {
    color: #0369A1 !important; /* Custom Dark Blue sign color */
    background-color: #FFFFFF !important; /* High contrast clean button background */
    border: 1px solid #BAE6FD !important;
    font-weight: bold !important;
    opacity: 1 !important;
}

div[data-testid="stForm"] button[data-testid="stNumberInputStepDown"]:hover, 
div[data-testid="stForm"] button[data-testid="stNumberInputStepUp"]:hover {
    background-color: #E0F2FE !important; /* Soft tint hover effect */
    color: #01476D !important;
}

/* Labels above the input fields */
div[data-testid="stForm"] label {
    color: #0F172A !important; /* Dark slate for high-contrast legible labels */
    font-weight: 600 !important;
}

/* Medical Action Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #0EA5E9, #0369A1);
    color: white;
    font-size: 18px;
    border-radius: 8px;
    height: 50px;
    border: none;
    font-weight: bold;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0369A1, #0EA5E9);
    transform: scale(1.01);
    transition: all 0.2s ease-in-out;
}

/* Custom Footer Styling */
.footer {
    text-align: center;
    color: #64748B;
    padding: 20px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title & Headers ---------------- #
st.markdown('<div class="main-title">🩺 Clinical Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Assisted Patient Risk Assessment Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title" style="font-size:15px; color:#64748B;">Input diagnostic metrics below for immediate laboratory screening results</div>', unsafe_allow_html=True)

# ---------------- Model Loading (Safe Load) ---------------- #
@st.cache_resource
def load_models():
    try:
        pipe = pickle.load(open("model.pkl", "rb"))
        sc = pickle.load(open("sc.pkl", "rb"))
        return pipe, sc
    except FileNotFoundError:
        return None, None

pipe, sc = load_models()

# Show error info if files are missing, but let the rest of the page render
if pipe is None or sc is None:
    st.error("⚠️ System Diagnostic Error: `model.pkl` or `sc.pkl` configuration files missing from application directory.")

# ---------------- Input Form ---------------- #
with st.form("form"):
    st.markdown("### 🧾 Diagnostic Parameters")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        input_insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    with col2:
        glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        bmi = st.number_input("BMI (Weight in kg/(height in m)²)", 0.0, 70.0, 25.0)
        age = st.number_input("Age (Years)", 1, 120, 30)

    submit = st.form_submit_button("🔍 Run Diagnostic Screening")

# ---------------- Prediction Logic ---------------- #
if submit:
    if pipe is not None and sc is not None:
        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            input_insulin, bmi, dpf, age
        ]])

        input_scaled = sc.transform(input_data)
        prediction = pipe.predict(input_scaled)

        status = "High Risk ⚠️" if prediction[0] == 1 else "Low Risk ✅"
        st.metric("Screening Metric Status", status)

        if prediction[0] == 1:
            st.error("⚠️ High Risk Indicators Found: Patient metrics align with diabetic classifications.")
        else:
            st.success("✅ Low Risk Confirmed: Patient metrics register within normal clinical limits.")
    else:
        st.error("Screening could not be executed due to unlinked classification models.")

# ---------------- Footer ---------------- #
st.markdown("---")
st.markdown('<div class="footer">🔒 Secure Medical Decision Support Tool | Powered by Machine Learning</div>', unsafe_allow_html=True)