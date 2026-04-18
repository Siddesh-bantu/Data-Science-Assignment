import streamlit as st
import pickle
import numpy as np
import os

# Load model from the same directory as this script so Streamlit's working
# directory doesn't cause a FileNotFoundError.
model_path = os.path.join(os.path.dirname(__file__), "titanic_model.pkl")
model = None
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # We'll show a friendly error in the Streamlit UI below instead of crashing
    model = None

st.title("Titanic Survival Prediction App")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100)
sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=10)
parch = st.number_input("Number of Parents/Children", min_value=0, max_value=10)
fare = st.number_input("Fare Paid", min_value=0.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs
sex = 1 if sex == "Male" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_dict[embarked]

if st.button("Predict Survival"):
    if model is None:
        st.error("Model file 'titanic_model.pkl' not found. Put the file next to this script.")
    else:
        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.success("The passenger is likely to SURVIVE.")
        else:
            st.error("The passenger is likely to NOT SURVIVE.")
