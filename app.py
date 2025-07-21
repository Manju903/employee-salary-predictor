import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")

st.title("Employee Income Predictor")
st.markdown("Fill out the details below to predict if the income is >50K or <=50K.")


def user_input():
    return {
        "age": st.number_input("Age", 18, 100, 30),
        "workclass": st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', '?']),
        "fnlwgt": st.number_input("Fnlwgt", 10000, 1000000, 200000),
        "education": st.selectbox("Education", ['Bachelors', 'Some-college', 'HS-grad', '11th', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th']),
        "educational-num": st.slider("Educational Number", 1, 16, 10),
        "marital-status": st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed']),
        "occupation": st.selectbox("Occupation", ['Exec-managerial', 'Craft-repair', 'Other-service', 'Sales', 'Tech-support', 'Adm-clerical', 'Prof-specialty']),
        "relationship": st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife']),
        "race": st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']),
        "gender": st.selectbox("Gender", ['Male', 'Female']),
        "capital-gain": st.number_input("Capital Gain", 0, 100000, 0),
        "capital-loss": st.number_input("Capital Loss", 0, 5000, 0),
        "hours-per-week": st.slider("Hours per Week", 1, 99, 40),
        "native-country": st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India'])
    }


inputs = user_input()
input_df = pd.DataFrame([inputs])  # Make into a DataFrame


if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Income: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
