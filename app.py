import streamlit as st
import joblib
import numpy as np

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("kt_eligibility_model.pkl")

model = load_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎓 KT Eligibility Checker")
st.write("Enter the number of **KTs** per subject to check student eligibility.")

# Define input fields for each feature
feature_names = [
    "FY Sem1 External", "FY Sem1 Internal", "FY Sem2 External", "FY Sem2 Internal",
    "SY Sem3 External", "SY Sem3 Internal", "SY Sem4 External", "SY Sem4 Internal",
    "TY Sem5 External", "TY Sem5 Internal", "TY Sem6 External", "TY Sem6 Internal",
    "Final Year Sem7 External", "Final Year Sem7 Internal",
    "Final Year Sem8 External", "Final Year Sem8 Internal",
    "Current Year (1, 2, 3, 4)"
]

student_data = []
for name in feature_names:
    val = st.number_input(f"{name}:", min_value=0, max_value=10, value=0)
    student_data.append(val)

if st.button("Check Eligibility"):
    student_array = np.array(student_data).reshape(1, -1)
    prediction = model.predict(student_array)[0]

    if prediction == 1:
        st.success("✅ Student is **ELIGIBLE** for promotion.")
    else:
        st.error("❌ Student is **NOT ELIGIBLE** for promotion.")