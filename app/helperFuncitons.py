from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
def clean_data_model(df):
    df = df.dropna(subset=["CustomerID"]).copy()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"].astype(str).str.replace(",", ".", regex=False),
        errors="raise",
    )

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.drop_duplicates()

    return df


def clean_data_user(df):
    temp = df.dropna().copy()
    temp["UnitPrice"] = pd.to_numeric()


    return df


def get_scaler():
    CRFM = pd.read_csv("../data/CRFM.csv")
    temp = CRFM[["Monetary", "Frequency", "Recency"]].copy()

    temp["Monetary"] = np.log1p(temp["Monetary"])
    temp["Frequency"] = np.log1p(temp["Frequency"])
    temp["Recency"] = np.log1p(temp["Recency"])


    scaler = StandardScaler().fit(temp)

    return scaler

def get_scaler_stream():
    CRFM = pd.read_csv("./data/CRFM.csv")
    temp = CRFM[["Monetary", "Frequency", "Recency"]].copy()

    temp["Monetary"] = np.log1p(temp["Monetary"])
    temp["Frequency"] = np.log1p(temp["Frequency"])
    temp["Recency"] = np.log1p(temp["Recency"])


    scaler = StandardScaler().fit(temp)

    return scaler