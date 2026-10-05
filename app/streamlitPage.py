import streamlit as st
import pandas as pd
import numpy as np
import joblib
from helperFuncitons import get_scaler_stream
st.set_page_config(layout="wide")

"""
### Select model
"""
st.selectbox("Select Model", ["Forest", "KNN", "XGB"], key="Model")

"""
### Input the RFM values of the client
"""
st.number_input("Monetary Value", key="Monetary")
st.number_input("Frequency Value", key="Frequency", step=1)
st.number_input("Recency Value", key="Recency", step=1)
d = {"Monetary": st.session_state.Monetary, "Frequency": st.session_state.Frequency,"Recency": st.session_state.Recency}
input_data = pd.DataFrame(d, index=[0])
st.header("Input Data")
st.write(input_data)


def ready_data(data):
    cleaned_data = data.copy()
    cleaned_data.dropna(inplace=True)
    cleaned_data = np.log1p(cleaned_data)
    scaler = get_scaler_stream()

    cleaned_data = scaler.transform(cleaned_data)
    return cleaned_data

selected_model = st.session_state.Model
loaded_model = joblib.load(f"./artifacts/{selected_model}.joblib")

def predict_class():
    cleaned_data = ready_data(input_data)
    y_pred = loaded_model.predict(cleaned_data)
    classes = {0: "Loyal", 1: "Lost", 2: "Casual"}

    st.subheader("Prediction:")
    st.write(classes[y_pred[0]])



st.button("Predict", key="Predict", on_click=predict_class)
