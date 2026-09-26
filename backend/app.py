from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_api = Flask("SuperKart Sales Forecast API")
model = joblib.load(Path(__file__).with_name("superkart_model.joblib"))

MODEL_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
    "Product_Sugar_Content",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
]


def validate_columns(frame):
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    return frame[MODEL_FEATURES]


@superkart_api.get("/")
def home():
    return jsonify({"service": "SuperKart Sales Forecast API", "version": "v1"})


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})


@superkart_api.post("/v1/predict")
def predict_sales():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        input_data = validate_columns(pd.DataFrame([payload]))
        prediction = round(float(model.predict(input_data)[0]), 2)
        return jsonify({"Predicted Sales": prediction})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file using form field 'file'."}), 400

    try:
        input_data = validate_columns(pd.read_csv(request.files["file"]))
        predictions = [round(float(value), 2) for value in model.predict(input_data)]
        return jsonify({str(index): value for index, value in enumerate(predictions)})
    except (KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
