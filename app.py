"""
app.py
------
Flask web app for the Emotion Detector. Serves a single-page UI where the
user types text and gets a live emotion + emoji prediction via an AJAX
call to /predict (no page reload).

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import joblib
from flask import Flask, request, jsonify, render_template

from emotion_mapper import map_emotion

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "emotion_model.joblib")

app = Flask(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run `python train_model.py` first "
                "(after generating data with data/generate_dataset.py)."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    model = get_model()
    emotion = model.predict([text])[0]

    # Confidence scores per class, if the model supports it
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        proba = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}

    result = map_emotion(emotion)
    return jsonify({
        "emotion": emotion,
        "emoji": result["emoji"],
        "label": result["label"],
        "confidence": proba,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
