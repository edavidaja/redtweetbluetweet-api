import os
import pickle
from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin

PORT = os.getenv("PORT", 8080)


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model("model/model.pkl")

app = Flask(__name__)


@app.route("/")
def index():
    return {"status": "it's alive"}


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if not request.json or not "text" in request.json:
        abort(400)
    text = [request.json["text"]]
    prediction = model.predict_proba(text)
    out = prediction.tolist()

    return jsonify(out)


if __name__ == "__main__":
    app.run(port=PORT)
