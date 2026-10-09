import gradio as gr
import pandas as pd
import numpy as np
import joblib
from helperFuncitons import get_scaler_stream

def ready_data(data):
    cleaned_data = data.copy()
    cleaned_data.dropna(inplace=True)
    cleaned_data = np.log1p(cleaned_data)
    scaler = get_scaler_stream()

    cleaned_data = scaler.transform(cleaned_data)
    return cleaned_data

def predict_class(input_data, model):
    cleaned_data = ready_data(input_data)
    y_pred = model.predict(cleaned_data)
    classes = {0: "Loyal", 1: "Lost", 2: "Casual"}

    return classes[y_pred[0]]

def predict(Monetary, Frequency, Recency, selected_model):
    d = {"Monetary": Monetary, "Frequency": Frequency, "Recency": Recency}
    model = joblib.load(f"./artifacts/{selected_model}.joblib")
    input_data = pd.DataFrame(d, index=[0])
    cl = predict_class(input_data, model)

    return cl


page = gr.Interface(
    fn=predict,
    inputs=["number", "number", "number", gr.Radio(["Forest", "KNN", "XGB"])],
    outputs=["text"],
)

page.launch()