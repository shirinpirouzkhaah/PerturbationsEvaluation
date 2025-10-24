import os
import re
import pandas as pd
import numpy as np
import difflib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from itertools import product
import ast
import matplotlib.pyplot as plt



# ==============================================================
# === Setup ===
# ==============================================================

models = ["gpt-3.5-turbo", "Llama3.3_70B", "DeepSeek-V3", "T5", "LLaMaLoRa"]
items = [
    "IfElseSwap", "DeadException", "DeadVariable", "TryNcatchWrapper",
    "ReturnViaVariable", "ShuffleNames", "RandomNames",
    "IndependentLineSwap", "defUseBreak",
]
scenarios = ["Org", "CodeRepetition", "InlineComment", "CoT"]

input_path = "Reports/Final_Evaluation_With_CodeBlue.csv"
output_dir = "RQ3_Delta_EXM"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================
# === Load Data and Filter ===
# ==============================================================

df = pd.read_csv(input_path)
print(f"✅ Loaded {len(df)} rows from {input_path}")

# Keep only rows where all models have EXM == True for Source_Org
exm_cols = [f"{m}_Source_Org_EXM" for m in models]
mask_all_true = df[exm_cols].apply(lambda row: all(row == True), axis=1)
df_filtered = df[mask_all_true].copy()
print(f"✅ Filtered to {len(df_filtered)} rows (all models Source_Org_EXM=True)")

# ==============================================================
# === Compute Per-Model EXM Ratios and Deltas + Summary Stats ===
# ==============================================================

for model in models:
    rows = []
    for item in items:
        row_data = {"Item": item}

        # Compute EXM ratio for each scenario
        exm_vals = {}
        for scenario in scenarios:
            col_name = f"{model}_{item}_{scenario}_EXM"
            if col_name not in df_filtered.columns:
                exm_vals[scenario] = np.nan
            else:
                exm_vals[scenario] = df_filtered[col_name].mean(skipna=True)

        # Map scenario names
        row_data["Org"] = round(exm_vals.get("Org", np.nan), 3)
        row_data["CR"] = round(exm_vals.get("CodeRepetition", np.nan), 3)
        row_data["IC"] = round(exm_vals.get("InlineComment", np.nan), 3)
        row_data["COT"] = round(exm_vals.get("CoT", np.nan), 3)

        # Compute deltas relative to Org
        org_val = exm_vals.get("Org", np.nan)
        for sc, short in zip(["CodeRepetition", "InlineComment", "CoT"], ["CR", "IC", "COT"]):
            delta = (
                round(exm_vals[sc] - org_val, 3)
                if not np.isnan(exm_vals[sc]) and not np.isnan(org_val)
                else np.nan
            )
            row_data[f"Δ{short}"] = delta

        rows.append(row_data)

    # Convert to DataFrame
    df_model = pd.DataFrame(rows)

    # === Compute summary statistics ===
    numeric_cols = ["Org", "CR", "IC", "COT", "ΔCR", "ΔIC", "ΔCOT"]
    summary = {
        "Item": ["MAX", "MIN", "AVG"],
    }
    for col in numeric_cols:
        summary[col] = [
            round(df_model[col].max(skipna=True), 3),
            round(df_model[col].min(skipna=True), 3),
            round(df_model[col].mean(skipna=True), 3),
        ]
    df_summary = pd.DataFrame(summary)

    # Append summary to CSV (with empty line separator)
    output_path = os.path.join(output_dir, f"{model}_EXM_Deltas.csv")
    with open(output_path, "w", encoding="utf-8") as f:
        df_model.to_csv(f, index=False)
        f.write("\n")  # blank line for readability
        df_summary.to_csv(f, index=False)

    print(f"💾 Saved {model} results with summary → {output_path}")

print("\n🎉 All model EXM delta CSVs generated successfully with summary stats!")









