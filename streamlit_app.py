import json
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")

st.title("Concrete Compressive Strength Predictor")
st.write("Enter concrete mix features below and click Predict.")

with open("model_features.json", "r") as f:
    feature_columns = json.load(f)

model = xgb.Booster()
model.load_model("xgb_concrete_model.json")

inputs = {}
defaults = {
    "cement": 300.0,
    "blast_furnace_slag": 0.0,
    "fly_ash": 0.0,
    "water": 160.0,
    "superplasticizer": 0.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 700.0,
    "age": 28.0,
}

for feature in feature_columns:
    inputs[feature] = st.number_input(
        feature.replace("_", " ").capitalize(),
        value=float(defaults.get(feature, 0.0)),
        format="%.3f",
    )

if st.button("Predict"):
    data = pd.DataFrame([inputs])
    data = data[feature_columns]
    dmatrix = xgb.DMatrix(data)
    prediction = model.predict(dmatrix)
    st.success(f"Predicted concrete compressive strength: {float(prediction[0]):.2f} MPa")
c