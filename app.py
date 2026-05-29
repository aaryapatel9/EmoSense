"""
app.py — Streamlit web demo for emotion detection.

Usage:
    streamlit run app.py -- --model_dir ./outputs/best_model
"""

import argparse
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import streamlit as st

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
    "admiration":"#F4C430","amusement":"#32CD32","anger":"#DC143C","annoyance":"#FF8C00",
    "approval":"#3CB371","caring":"#FFB6C1","confusion":"#9370DB","curiosity":"#00CED1",
    "desire":"#FF69B4","disappointment":"#708090","disapproval":"#CD5C5C","disgust":"#556B2F",
    "embarrassment":"#DDA0DD","excitement":"#FF4500","fear":"#8B008B","gratitude":"#DAA520",
    "grief":"#4682B4","joy":"#FFD700","love":"#FF1493","nervousness":"#D2691E",
    "optimism":"#00BFFF","pride":"#B8860B","realization":"#7B68EE","relief":"#66CDAA",
    "remorse":"#696969","sadness":"#1E90FF","surprise":"#FF6347","neutral":"#A9A9A9",
}
EXAMPLES = [
    "I can't believe how amazing this concert was!",
    "I'm so frustrated with myself.",
    "Thanks so much — I really appreciate it.",
    "I have no idea what to do next.",
    "My dog passed away today. I miss him.",
    "Wow, I didn't expect that at all!",
]


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_dir):
    tok   = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tok, model


@torch.no_grad()
def predict(text, tok, model, threshold):
    enc    = tok(text, return_tensors="pt", truncation=True, max_length=128)
    logits = model(**enc).logits
    probs  = torch.sigmoid(logits).squeeze().numpy()
    scores   = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
    emotions = [e for e, p in scores.items() if p >= threshold] or [max(scores, key=scores.get)]
    return emotions, scores


def main(model_dir):
    st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")
    st.title("🎭 Emotion Detection in Text")
    st.markdown("Fine-tuned **BERT** on GoEmotions — 28 emotion labels.")
    st.divider()

    tok, model = load_model(model_dir)

    col1, col2 = st.columns([3, 1])
    with col2:
        threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
    with col1:
        example = st.selectbox("Try an example…", ["(enter your own below)"] + EXAMPLES)

    text = st.text_area(
        "Enter text:",
        value="" if example.startswith("(") else example,
        height=100,
        placeholder="Type something with emotion…",
    )

    if st.button("Analyse 🔍", use_container_width=True, type="primary") and text.strip():
        with st.spinner("Analysing…"):
            emotions, scores = predict(text.strip(), tok, model, threshold)

        st.success("**Detected:** " + "  ".join(f"{EMOTION_EMOJI.get(e,'')} {e}" for e in emotions))

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        st.markdown("#### Top-10 Probabilities")
        for emo, val in sorted_scores:
            active = emo in emotions
            col = EMOTION_COLOR.get(emo, "#888")
            st.markdown(
                f"""<div style='display:flex;align-items:center;margin-bottom:6px;'>
                    <span style='width:190px;font-weight:{"700" if active else "400"}'>
                        {EMOTION_EMOJI.get(emo,'')} {emo}</span>
                    <div style='flex:1;background:#eee;border-radius:8px;height:18px;overflow:hidden;'>
                        <div style='height:100%;width:{val*100:.1f}%;background:{col};border-radius:8px;'></div>
                    </div>
                    <span style='width:52px;text-align:right;font-size:.85rem'>{val*100:.1f}%</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption("Model: bert-base-uncased | Dataset: GoEmotions (211k rows, 28 labels)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./outputs/best_model")
    args, _ = parser.parse_known_args()
    main(args.model_dir)
