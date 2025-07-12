import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the files provided by the user
files = {
    "BERT (5 epochs)": "./Models-Analysis/bert/bert-5-epochs-trainer-state-fixed.json",
    "BERT (15 epochs)": "./Models-Analysis/bert/bert-15epochs-trainer-state.txt",
    "DistilBERT (5 epochs)": "distilbert-5epochs-trainer-state.txt",
    "DistilBERT (15 epochs)": "distilbert-15epochs-trainer-state.txt",
    "RoBERTa (5 epochs)": "roberta-5epochs-trainer-state.txt",
    "RoBERTa (15 epochs)": "roberta-15epochs-trainer-state.txt"
}

# Dictionary to store the extracted data
plot_data = {}

# Process each file
for model_run, filename in files.items():
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # This handles the original format, where the file is a dictionary
            log_history = data.get("log_history", [])
        elif isinstance(data, list):
            # This handles your fixed file, where the file is just the list of logs
            log_history = data
        else:
            log_history = []
            print("Warning: Could not recognize the JSON structure.")

        epochs = []
        f1_scores = []

        for entry in log_history:
            if "eval_f1_weighted" in entry:
                epochs.append(entry["epoch"])
                f1_scores.append(entry["eval_f1_weighted"])

        if epochs:
            plot_data[model_run] = pd.DataFrame({'epoch': epochs, 'f1_score': f1_scores})
            print(f"Successfully processed {filename} for {model_run}. Found {len(epochs)} evaluation points.")
        else:
            print(f"Warning: No evaluation data found in {filename} for {model_run}.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing {filename}: {e}")

# Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

colors = {
    'BERT': 'blue',
    'DistilBERT': 'green',
    'RoBERTa': 'red'
}
linestyles = {
    '5 epochs': ':',  # Dotted line for 5-epoch runs
    '15 epochs': '-'  # Solid line for 15-epoch runs
}

for model_run, df in plot_data.items():
    model_name = model_run.split(' (')[0]
    run_type = model_run.split('(')[1].replace(')', '')

    # Ensure epochs are plotted correctly, especially if training stopped early
    # The 5-epoch runs are subsets of the 15-epoch runs, so we only plot the 15-epoch runs
    # for a clearer graph, as they contain the full history.
    if "15 epochs" in model_run:
        ax.plot(df['epoch'], df['f1_score'],
                label=model_name,
                color=colors[model_name],
                linestyle=linestyles[run_type],
                marker='o', markersize=4)

# Formatting the plot
ax.set_title('Model Performance on Validation Set vs. Training Epoch', fontsize=16)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation F1-Score (Weighted)', fontsize=12)
ax.legend(title='Model', fontsize=10)
ax.grid(True)

# Set integer ticks for epochs if possible
max_epoch = 0
for model_run, df in plot_data.items():
    if "15 epochs" in model_run:
        if df['epoch'].max() > max_epoch:
            max_epoch = df['epoch'].max()
ax.set_xticks(np.arange(0, max_epoch + 1, 1.0))
plt.tight_layout()

# Save the plot to a file
plot_filename = 'learning_curves.png'
plt.savefig(plot_filename)

print(f"\nPlot saved as {plot_filename}")

# Display the data tables for the text draft
print("\n--- Data for Thesis Text ---")
for model_run, df in plot_data.items():
    # We only need the 15-epoch data for the plot and analysis of overfitting
    if "15 epochs" in model_run:
        print(f"\n--- {model_run} ---")
        print(df)
print("\n--- End of Data ---")