import torch
import numpy as np
import random
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, classification_report
import os

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# --- Configuration ---
#STEP1: Fill in the constants
MODEL_NAME = 'bert-base-uncased' # 'distilbert-base-uncased' or 'bert-base-uncased' 'roberta-base'
EPOCHS = 15 # A smaller number of epochs is often better for larger models
BATCH_SIZE = 8 # Reduced batch size for larger models to avoid memory issues,8 is ok.
MAX_LENGTH = 128
LEARNING_RATE = 2e-5

# --- 1. Load and Prepare Dataset ---
print("🔄 Loading and preparing dataset...")
dataset = load_dataset("go_emotions", "raw")["train"]
non_label_cols = {'text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear'}
label_names = [col for col in dataset.column_names if col not in non_label_cols]
num_labels = len(label_names)

dataset = dataset.shuffle(seed=42)
# Create a three-way split: train, validation (for thresholding), and test (for final report)
train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = train_val_split["train"]
val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
val_ds = val_test_split["train"]
test_ds = val_test_split["test"]

print(f"✅ Dataset split: {len(train_ds)} train, {len(val_ds)} validation, {len(test_ds)} test samples.")

# --- 2. Calculate Class Weights for Imbalance ---
print("⚖️ Calculating class weights...")
pos_counts = [train_ds[label].count(1) for label in label_names]
total_samples = len(train_ds)
pos_weights = torch.tensor(
    [(total_samples - count) / count for count in pos_counts],
    dtype=torch.float
)
print("✅ Class weights calculated.")

# --- 3. Preprocessing ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def preprocess(examples):
    encodings = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    labels = np.array([examples[label] for label in label_names]).T.astype(np.float32)
    encodings["labels"] = labels.tolist()
    return encodings

print("⚙️  Preprocessing data...")
train_enc = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_enc = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
test_enc = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

# --- 4. Custom Trainer & Metrics ---
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs:dict, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        #loss funtion with my specially calcuated weights
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    sigmoid = 1 / (1 + np.exp(-logits))
    # For quick evaluation during training, we use a fixed threshold
    preds = (sigmoid > 0.5).astype(int)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"f1_weighted": f1}

# --- 5. Training ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification")
output_dir = f"./results/{MODEL_NAME}-optimized"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    weight_decay=0.01,
    report_to="none",
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=val_enc, # Use the validation set for early stopping
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Stop if no improvement for 2 epochs
)

print("🚀 Training starting...")
trainer.train()

# --- 6. STEP 2: FIND THE OPTIMAL PREDICTION THRESHOLD ---
def find_best_threshold(trainer, eval_dataset):
    """Find the best F1 threshold on the validation set."""
    print("🔍 Finding the best prediction threshold...")
    # Get model predictions (logits) on the validation dataset
    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    sigmoid = 1 / (1 + np.exp(-logits))

    # Test a range of thresholds
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (sigmoid > threshold).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        print(f"Threshold: {threshold:.2f}, F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"✅ Best threshold found: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
    return best_threshold

best_threshold = find_best_threshold(trainer, val_enc)

# --- 7. Final Evaluation on Test Set ---
print("📊 Generating final report on the TEST set with the best threshold...")
predictions = trainer.predict(test_enc)
logits = predictions.predictions
labels = predictions.label_ids

sigmoid = 1 / (1 + np.exp(-logits))
# Use the best threshold we found for the final predictions
final_preds = (sigmoid > best_threshold).astype(int)

report = classification_report(
    labels,
    final_preds,
    target_names=label_names,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(output_dir, f"classification_report-{MODEL_NAME}_{EPOCHS}epochs_final.csv")
os.makedirs(output_dir, exist_ok=True)
report_df.to_csv(report_path)

print(f"✅ Final report saved to {report_path}")
trainer.save_model(os.path.join(output_dir, f"{MODEL_NAME}-{EPOCHS}epochs-final_model"))
print("✅ All steps complete.")