"""
explore_data.py — EDA on your GoEmotions CSV file.
Saves all plots to ./eda_outputs/

Usage:
    python explore_data.py
    python explore_data.py --csv data/go_emotions_dataset.csv
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

os.makedirs("eda_outputs", exist_ok=True)

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["example_very_unclear"] != True]
    df = df[df[EMOTION_LABELS].sum(axis=1) > 0]
    df["text"] = df["text"].astype(str).str.strip()
    print(f"Loaded {len(df):,} usable rows from {csv_path}\n")
    return df


def plot_label_distribution(df):
    counts = df[EMOTION_LABELS].sum().sort_values(ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(counts)))

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors, edgecolor="white")
    for bar, v in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height()/2,
                f"{int(v):,}", va="center", fontsize=9, color="#555")
    ax.set_xlabel("Number of samples", fontsize=12)
    ax.set_title("Label Distribution", fontsize=15, fontweight="bold", pad=15)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda_outputs/01_label_distribution.png", dpi=150)
    plt.close()
    print("✔ saved 01_label_distribution.png")


def plot_text_lengths(df):
    lengths = df["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=40, color="#4A90D9", edgecolor="white")
    ax.axvline(lengths.median(), color="#E74C3C", linestyle="--", lw=2,
               label=f"Median: {lengths.median():.0f} words")
    ax.axvline(lengths.mean(),   color="#F39C12", linestyle="--", lw=2,
               label=f"Mean: {lengths.mean():.1f} words")
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Text Length Distribution", fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_outputs/02_text_lengths.png", dpi=150)
    plt.close()
    print("✔ saved 02_text_lengths.png")


def plot_labels_per_example(df):
    n_labels = df[EMOTION_LABELS].sum(axis=1)
    count    = Counter(n_labels.astype(int))
    x   = sorted(count.keys())
    pct = [count[k] / len(df) * 100 for k in x]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, pct, color="#8E44AD", edgecolor="white")
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{p:.1f}%", ha="center", fontsize=10)
    ax.set_xlabel("Labels per example", fontsize=12)
    ax.set_ylabel("% of examples",      fontsize=12)
    ax.set_title("Labels per Example", fontsize=15, fontweight="bold", pad=15)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("eda_outputs/03_labels_per_example.png", dpi=150)
    plt.close()
    print("✔ saved 03_labels_per_example.png")


def plot_cooccurrence(df):
    mat = df[EMOTION_LABELS].values
    co  = mat.T @ mat                           # (28, 28) co-occurrence counts
    diag = np.maximum(np.diag(co), 1)
    co_norm = co / diag[:, None]
    np.fill_diagonal(co_norm, 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(co_norm, xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="#eee",
                cbar_kws={"label": "P(col | row)"})
    ax.set_title("Emotion Co-occurrence (normalised by row)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    plt.savefig("eda_outputs/04_cooccurrence_heatmap.png", dpi=150)
    plt.close()
    print("✔ saved 04_cooccurrence_heatmap.png")


def print_summary(df):
    total  = len(df)
    multi  = (df[EMOTION_LABELS].sum(axis=1) > 1).sum()
    counts = df[EMOTION_LABELS].sum().sort_values()

    stats = {
        "total_rows":         total,
        "multi_label_rows":   int(multi),
        "multi_label_pct":    round(multi / total * 100, 2),
        "most_common_emotion": counts.idxmax(),
        "rarest_emotion":      counts.idxmin(),
        "neutral_pct":         round(df["neutral"].sum() / total * 100, 2),
    }
    print("\n📊 Dataset Summary:")
    for k, v in stats.items():
        print(f"   {k:<28}: {v}")
    with open("eda_outputs/summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("✔ saved summary_stats.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/go_emotions_dataset.csv")
    args = parser.parse_args()

    df = load(args.csv)
    plot_label_distribution(df)
    plot_text_lengths(df)
    plot_labels_per_example(df)
    plot_cooccurrence(df)
    print_summary(df)
    print("\n✅ EDA complete. Figures saved to ./eda_outputs/")
