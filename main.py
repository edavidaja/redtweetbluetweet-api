import os
import joblib
from flask import Flask, request, jsonify, abort
from google.cloud import storage

bucket_name = os.getenv("BUCKET_NAME")
PORT = os.getenv("PORT", 8080)
model_blob = "model.joblib"
dest_file = "/tmp/model.joblib"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)


model = joblib.load("model/model.joblib")

app = Flask(__name__)


@app.route("/")
def index():
    return {"status": "it's alive"}


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or not "text" in request.json:
        abort(400)
    text = [request.json["text"]]

    prediction = model.predict_proba(text)
    out = prediction.tolist()

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
