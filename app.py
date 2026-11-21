import streamlit as st
imort joblib
import pandas as pd
import numpy as np

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title("Churn Prediction App")

st.divider()

st.write("please enter the values and hit the predict button to get the result")

st.divider()

age = st.number_input("Enter the age of the customer", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter the tenure of the customer", min_value=0, max_value=130, value=10)

monthly_charges = st.number_input("Enter the monthly charges of the customer", min_value=30, max_value=150)

gender = st.selectbox("Select the gender of the customer", options=["Male", "Female"])

st.divider()

predict_button = st.button("Predict")

if predict_button:
    gender_selected = 1 if gender == "Male" else 0
    
    x =[age, tenure, monthly_charges, gender_selected]
    
    x1 = np.array(x)
    
    X_array = scaler.transform([x1])

    prediction = model.predict(X_array)[0]
    
    predicted = "Yes" if prediction == 1 else "No"
    
    st.write(f"The predicted result is: {predicted}")
    
else:
    st.write("Please enter the values and hit the predict button to get the result")