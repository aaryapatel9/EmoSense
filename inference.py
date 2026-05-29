"""
inference.py — Load fine-tuned BERT and predict emotions for new text.

Usage:
    python inference.py --text "I can't believe how amazing this is!"
    python inference.py --text "Thank you!||I hate this." --threshold 0.4
"""

import argparse
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

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


class EmotionPredictor:
    def __init__(self, model_dir: str = "./outputs/best_model", threshold: float = 0.5):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        print(f"Loading model from '{model_dir}'…")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model     = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("Model ready.\n")

    @torch.no_grad()
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        logits = self.model(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
        ).logits
        probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for text, prob in zip(texts, probs):
            scores   = {EMOTION_LABELS[i]: round(float(prob[i]), 4) for i in range(len(EMOTION_LABELS))}
            emotions = [e for e, p in scores.items() if p >= self.threshold]
            if not emotions:
                emotions = [max(scores, key=scores.get)]   # fallback: top-1
            results.append({"text": text, "emotions": emotions, "scores": scores})

        return results

    def pretty_print(self, result: dict, top_n: int = 5):
        print(f'\n📝 "{result["text"]}"')
        print("🎯 Detected:", "  ".join(f"{EMOTION_EMOJI.get(e,'')} {e}" for e in result["emotions"]))

        sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
        print(f"\n📊 Top-{top_n} scores:")
        for label, score in sorted_scores[:top_n]:
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            print(f"   {EMOTION_EMOJI.get(label,'')} {label:<18} {bar} {score*100:5.1f}%")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",       type=str,   default=None)
    parser.add_argument("--model_dir",  type=str,   default="./outputs/best_model")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--top_n",      type=int,   default=5)
    args = parser.parse_args()

    predictor = EmotionPredictor(args.model_dir, args.threshold)

    demo_texts = args.text.split("||") if args.text else [
        "I can't believe how amazing this concert was! Best night ever!",
        "I'm so frustrated with myself. I keep making the same mistakes.",
        "Thanks so much for your help — I really appreciate it.",
        "I have no idea what to do next. Everything feels uncertain.",
        "My dog passed away today. I miss him so much.",
    ]

    for r in predictor.predict(demo_texts):
        predictor.pretty_print(r, top_n=args.top_n)
