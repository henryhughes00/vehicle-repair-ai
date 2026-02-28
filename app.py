import joblib
import json
import pandas as pd
import streamlit as st

MODEL_PATH = "model/model_bundle.joblib"
META_PATH = "model/metadata.json"
DATA_PATH = "data/data.xlsx"
SHEET_NAME = "Vehicle_Maintenance_Data"

@st.cache_resource
def load_models():
    bundle = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return bundle["class_pipe"], bundle["cost_pipe"], meta

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

def main():
    st.set_page_config(page_title="Vehicle Repair Predictor")
    st.title("AI Vehicle Repair and Cost Predictor")

    class_pipe, cost_pipe, meta = load_models()
    df = load_data()

    st.subheader("Model Performance")
    st.json(meta["metrics"])

    with st.form("prediction_form"):
        inputs = {}

        for col in meta["features"]:
            if col in meta["categorical_features"]:
                options = sorted(df[col].dropna().unique())
                inputs[col] = st.selectbox(col, options)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                median_val = float(df[col].median())
                inputs[col] = st.slider(col, min_val, max_val, median_val)

        submitted = st.form_submit_button("Predict")

    if submitted:
        X = pd.DataFrame([inputs])
        pred_category = class_pipe.predict(X)[0]
        pred_cost = float(cost_pipe.predict(X)[0])

        st.metric("Likely Component Category", pred_category)
        st.metric("Estimated Repair Cost", f"${pred_cost:,.2f}")

if __name__ == "__main__":
    main()
