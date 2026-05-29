"""
train.py — Fine-tune BERT on your GoEmotions CSV (one-hot format).

Usage:
    python train.py [--epochs 3] [--batch_size 32] [--lr 2e-5] [--output_dir ./outputs]
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
NUM_LABELS = len(EMOTION_LABELS)
ID2LABEL   = {i: l for i, l in enumerate(EMOTION_LABELS)}
LABEL2ID   = {l: i for i, l in enumerate(EMOTION_LABELS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────
class GoEmotionsCSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=128):
        self.texts      = df["text"].tolist()
        self.labels     = df[EMOTION_LABELS].values.astype(np.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Load & split CSV ──────────────────────────────────────────────────────────
def load_csv(csv_path: str):
    print(f"▶ Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop unclear rows and rows with no label
    df = df[df["example_very_unclear"] != True].copy()
    df = df[df[EMOTION_LABELS].sum(axis=1) > 0].copy()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()

    print(f"  Total usable rows : {len(df):,}")

    train_df, temp_df = train_test_split(df,       test_size=0.2,  random_state=42)
    val_df,   test_df = train_test_split(temp_df,  test_size=0.5,  random_state=42)

    print(f"  Train : {len(train_df):,}")
    print(f"  Val   : {len(val_df):,}")
    print(f"  Test  : {len(test_df):,}\n")
    return train_df, val_df, test_df


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0.0
    criterion  = nn.BCEWithLogitsLoss()

    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion  = nn.BCEWithLogitsLoss()

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > threshold).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    micro_f1   = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    return total_loss / len(loader), macro_f1, micro_f1, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    print(f"\n{'='*60}")
    print(f"  Emotion Detection with BERT — Custom CSV")
    print(f"  Device : {DEVICE}")
    print(f"  CSV    : {args.csv}")
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    train_df, val_df, test_df = load_csv(args.csv)

    tokenizer    = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_ds     = GoEmotionsCSVDataset(train_df, tokenizer)
    val_ds       = GoEmotionsCSVDataset(val_df,   tokenizer)
    test_ds      = GoEmotionsCSVDataset(test_df,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model
    print("▶ Loading bert-base-uncased…")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)

    # 3. Optimizer & Scheduler
    no_decay  = ["bias", "LayerNorm.weight"]
    params    = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer    = torch.optim.AdamW(params, lr=args.lr)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 4. Training
    best_val_f1 = 0.0
    history     = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──────────────────────────")
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        vl_loss, macro_f1, micro_f1, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_macro_f1"].append(macro_f1)
        history["val_micro_f1"].append(micro_f1)

        print(f"  Train loss   : {tr_loss:.4f}")
        print(f"  Val   loss   : {vl_loss:.4f}")
        print(f"  Val macro F1 : {macro_f1:.4f}  |  micro F1 : {micro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"  ✔ Best model saved (macro F1 = {best_val_f1:.4f})")

    # 5. Test
    print("\n▶ Evaluating on test set…")
    test_loss, test_macro, test_micro, preds, labels_arr = evaluate(model, test_loader)
    print(f"  Test loss    : {test_loss:.4f}")
    print(f"  Test macro F1: {test_macro:.4f}  |  micro F1 : {test_micro:.4f}\n")

    report = classification_report(labels_arr, preds, target_names=EMOTION_LABELS, zero_division=0)
    print(report)

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\n✅ Done. Artefacts saved to '{args.output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str,   default="data/go_emotions_dataset.csv")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str,   default="./outputs")
    main(parser.parse_args())
