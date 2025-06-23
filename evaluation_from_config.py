import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import f1_score, classification_report
import os

# --- VERY IMPORTANT: CHANGE THIS LINE ---
# Put the path to the folder containing your saved model (e.g., './results/distilbert-base-uncased-5epochs/checkpoint-1234')
PATH_TO_SAVED_MODEL = "./results/bert-base-uncased-5epochs-final"
# -----------------------------------------

# --- 1. Load Dataset and Prepare Splits (Same as your train.py) ---
print("🔄 Loading and preparing dataset...")
dataset = load_dataset("go_emotions", "raw")["train"]
non_label_cols = {'text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear'}
label_names = [col for col in dataset.column_names if col not in non_label_cols]
num_labels = len(label_names)

# Re-create the same splits to get the validation and test sets
dataset = dataset.shuffle(seed=42)
train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
val_ds = val_test_split["train"]
test_ds = val_test_split["test"]
print("✅ Datasets loaded.")

# --- 2. Load Saved Model and Tokenizer ---
print(f"🔄 Loading saved model from: {PATH_TO_SAVED_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_SAVED_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(PATH_TO_SAVED_MODEL)
print("✅ Model and tokenizer loaded.")

# --- 3. Preprocess Data (Same as your train.py) ---
def preprocess(examples):
    encodings = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    labels = np.array([examples[label] for label in label_names]).T.astype(np.float32)
    encodings["labels"] = labels.tolist()
    return encodings

print("⚙️  Preprocessing data...")
val_enc = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
test_enc = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)
print("✅ Data preprocessed.")

# --- 4. Find Optimal Threshold on Validation Set ---
# We need a dummy Trainer to run predictions
trainer = Trainer(model=model)

def find_best_threshold(trainer, eval_dataset):
    print("🔍 Finding the best prediction threshold on the validation set...")
    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    sigmoid = 1 / (1 + np.exp(-logits))

    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (sigmoid > threshold).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print(f"✅ Best threshold found: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
    return best_threshold

best_threshold = find_best_threshold(trainer, val_enc)

# --- 5. Generate Final Report on Test Set ---
print("\n\n📊 Generating final report on the TEST set...")
predictions = trainer.predict(test_enc)
logits = predictions.predictions
labels = predictions.label_ids

sigmoid = 1 / (1 + np.exp(-logits))
final_preds = (sigmoid > best_threshold).astype(int)

# This will print the report to your console
report_str = classification_report(
    labels,
    final_preds,
    target_names=label_names,
    zero_division=0,
    digits=4
)

print("\n\n--- FINAL CLASSIFICATION REPORT ---\n")
print(report_str)
print("\n--- END OF REPORT ---")
print("\n✅ Process complete. Please copy the report above.")