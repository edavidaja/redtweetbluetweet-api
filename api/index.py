from flask import Flask, request, jsonify, abort
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("index.py")
app = Flask(__name__)
logger.debug("before model")
model = joblib.load("model/model.joblib")
logger.debug("after model")


@app.route("/")
def index():
    return "Index Page"


@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or not "text" in request.json:
        abort(400)
    text = [request.json["text"]]

    prediction = model.predict_proba(text)
    out = prediction.tolist()

    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=False)
