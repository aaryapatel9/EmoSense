"""
api.py — Flask REST API backend for Emotion Detection.
Replaces Streamlit entirely.

Usage:
    python api.py --model_dir ./outputs/best_model
    python api.py --model_dir ./outputs/best_model --port 5000
"""

import argparse
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import BertTokenizerFast, BertForSequenceClassification
import os

app = Flask(__name__, static_folder=".")
CORS(app)

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

EMOTION_EMOJI = {
    "admiration": "🤩", "amusement": "😄", "anger": "😠", "annoyance": "😤",
    "approval": "👍", "caring": "🤗", "confusion": "😕", "curiosity": "🤔",
    "desire": "😍", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
    "embarrassment": "😳", "excitement": "🎉", "fear": "😨", "gratitude": "🙏",
    "grief": "😢", "joy": "😊", "love": "❤️", "nervousness": "😰",
    "optimism": "🌟", "pride": "🦁", "realization": "💡", "relief": "😌",
    "remorse": "😔", "sadness": "😭", "surprise": "😲", "neutral": "😐",
}

EMOTION_COLOR = {
    "admiration": "#F4C430", "amusement": "#32CD32", "anger": "#DC143C",
    "annoyance": "#FF8C00", "approval": "#3CB371", "caring": "#FF85A1",
    "confusion": "#9370DB", "curiosity": "#00CED1", "desire": "#FF69B4",
    "disappointment": "#708090", "disapproval": "#CD5C5C", "disgust": "#6B8E23",
    "embarrassment": "#DDA0DD", "excitement": "#FF4500", "fear": "#8B008B",
    "gratitude": "#DAA520", "grief": "#4682B4", "joy": "#FFD700",
    "love": "#FF1493", "nervousness": "#D2691E", "optimism": "#00BFFF",
    "pride": "#B8860B", "realization": "#7B68EE", "relief": "#66CDAA",
    "remorse": "#696969", "sadness": "#1E90FF", "surprise": "#FF6347",
    "neutral": "#A9A9A9",
}

# Global model state
model_state = {"tokenizer": None, "model": None, "device": None}


def load_model(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from '{model_dir}' on {device}…")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    model_state["device"] = device
    print("✅ Model loaded and ready.")


@torch.no_grad()
def run_prediction(text, threshold=0.5):
    tokenizer = model_state["tokenizer"]
    model     = model_state["model"]
    device    = model_state["device"]

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    logits = model(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
    ).logits
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    scores   = {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(len(EMOTION_LABELS))}
    emotions = [e for e, p in scores.items() if p >= threshold]
    if not emotions:
        emotions = [max(scores, key=scores.get)]

    return {
        "text":     text,
        "emotions": emotions,
        "scores":   scores,
        "emoji":    {e: EMOTION_EMOJI.get(e, "") for e in emotions},
        "colors":   EMOTION_COLOR,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text      = data["text"].strip()
    threshold = float(data.get("threshold", 0.5))

    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 characters)"}), 400

    result = run_prediction(text, threshold)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_state["model"] is not None})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./outputs/best_model")
    parser.add_argument("--port",      type=int, default=5000)
    parser.add_argument("--host",      type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_model(args.model_dir)
    print(f"\n🚀 Server running at http://localhost:{args.port}")
    print(f"   Open index.html in your browser or visit http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)
