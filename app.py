import streamlit as st
import pandas as pd
import joblib

model=joblib.load("KNN_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")
st.title("Heart Disease Prediction By Sachin using KNN")
st.markdown("provide the following details")
age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX",['M','F'])
chest_pain=st.selectbox("chest pain type",["ATA","NAP","TA","ASY"])

resting_blood_pressure=st.number_input("Resting Blood Pressure",80,200,120)
cholesterol=st.number_input("Cholesterol",100,600,200)
fasting_blood_sugar=st.selectbox("Fasting Blood Sugar>120 mg/dl",["0","1"])
resting_ecg=st.selectbox("Resting ECG",["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])

max_heart_rate=st.slider("Max Heart Rate",60,220,150)
exercise_induced_angina=st.selectbox("Exercise Induced Angina",["Y","N"])
oldpeak=st.slider("Oldpeak(ST depression )",0.0,6.0,1.0)
slope=st.selectbox("Slope of the peak exercise ST segment",["Upsloping","Flat","Downsloping"])


if st.button("Predict"):
    raw_input={
        "age": age,
        "sex_" + sex:1,
        "chest_pain_"+ chest_pain:1,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_ecg_"+ resting_ecg:+1,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_angina_" +exercise_induced_angina:+1,
        "oldpeak": oldpeak,
        "slope_"+ slope:1
    }
    input_df=pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
    input_df=input_df[expected_columns] 
    input_scaled=scaler.transform(input_df)
    prediction=model.predict(input_scaled)[0]

    if prediction==1:
        st.error("The model predicts that you have heart disease. Please consult a doctor for further evaluation.")
    else:
        st.success("Low risk of heart disease")
