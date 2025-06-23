import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURE THIS SECTION ---
# Set the path to the results folder of the completed training run
RESULTS_DIR = './results/bert-base-uncased-optimized/checkpoint-63369'
# --- End of Configuration ---


# --- 2. Load the Trainer State ---
state_path = os.path.join(RESULTS_DIR, 'trainer_state.json')
print(f"🔄 Loading trainer state from: {state_path}")

try:
    with open(state_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find 'trainer_state.json' in the specified directory.")
    print("Please make sure the RESULTS_DIR path is correct and the training has completed.")
    exit()

# Extract the log history
log_history = data.get('log_history')
if not log_history:
    print("Error: Could not find 'log_history' in the trainer state file.")
    exit()

# Try to get model name from path for the plot title
model_name_from_path = os.path.basename(RESULTS_DIR)

# --- 3. Process the Logs ---
print("⚙️  Processing log data...")
train_logs = [log for log in log_history if 'loss' in log]
eval_logs = [log for log in log_history if 'eval_loss' in log]

train_df = pd.DataFrame(train_logs)
eval_df = pd.DataFrame(eval_logs)

# --- 4. Generate and Save the Plot ---
print("📈 Generating and saving training plot...")
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot training loss and evaluation loss on the primary y-axis (ax1)
color = 'tab:blue'
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Loss', color=color)
ax1.plot(train_df["step"], train_df["loss"], label="Training Loss", color=color, alpha=0.6)
ax1.plot(eval_df["step"], eval_df["eval_loss"], label="Validation Loss", color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for the F1 score that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('F1 Score (Weighted)', color=color)
ax2.plot(eval_df["step"], eval_df["eval_f1_weighted"], label="Validation F1", color=color, marker='s')
ax2.tick_params(axis='y', labelcolor=color)

# Add titles and legends
plt.title(f"Training & Validation Metrics for\n{model_name_from_path}", fontsize=16)
fig.tight_layout() # Adjust layout to make room for everything
# To combine legends from two axes, we collect them first
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')


# Save the plot
plot_path = os.path.join(RESULTS_DIR, "training_plot.png")
plt.savefig(plot_path, dpi=300) # dpi=300 for higher resolution

print(f"✅ Plot saved to {plot_path}")