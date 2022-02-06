import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file :
        data=pickle.load(file)
    return data

data = load_model()

classifier = data["model"]

def show_predict_page():
    st.title("Diabetes Prediction Model")
    st.write("### We need some information to predict your Diabetic Condition")

    Age = st.text_input("Age", 0)
    Glucose = st.text_input("Glucose", 0)
    BloodPressure = st.text_input("BloodPressure", 0)
    SkinThickness = st.text_input("SkinThickness", 0)
    Insulin = st.text_input("Insulin", 0)
    BMI = st.text_input("BMI", 0)
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction", 0)
    Pregnancies = st.text_input("Pregnancies", 0)


    ok = st.button("CHECK PREDICTED RESULTS")
    if ok:
        X = np.array([Pregnancies , Glucose , BloodPressure , SkinThickness , Insulin , BMI , DiabetesPedigreeFunction , Age ])
        input_data_reshaped = X.reshape(1,-1)
        prediction = classifier.predict(input_data_reshaped)
        if (prediction[0] == 0):
            st.subheader("Prediction : The person is non diabetic")
        else:
            st.subheader("Prediction : The person is diabetic")
