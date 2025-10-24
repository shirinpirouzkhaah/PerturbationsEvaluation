# ==============================================================
# ==============================================================
# ==============================================================
# IntersectDataset — Dual-Axis Plots (Improved Final Version)
# ==============================================================
# ==============================================================
# ==============================================================

import os
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# ==============================================================
# === Configurations ===
# ==============================================================

models = ["gpt-3.5-turbo", "Llama3.3_70B", "DeepSeek-V3", "T5", "LLaMaLoRa"]

items = [
    "IfElseSwap",
    "DeadException",
    "DeadVariable",
    "TryNcatchWrapper",
    "ReturnViaVariable",
    "ShuffleNames",
    "RandomNames",
    "IndependentLineSwap",
    "defUseBreak",
]

scenarios = ["Org"]

# ==============================================================
# === Load Data ===
# ==============================================================

output_dir = "RQ1ResultsIntersect"
os.makedirs(output_dir, exist_ok=True)

main_df = pd.read_csv("Reports/Final_Evaluation_With_CodeBlue.csv")

# ==============================================================
# Filter rows where ALL models have EXM=True for Source_Org
# ==============================================================

exm_cols = [f"{m}_Source_Org_EXM" for m in models]
mask_all_exm_true = main_df[exm_cols].apply(lambda row: all(row == True), axis=1)
df_filtered = main_df[mask_all_exm_true].copy()
print(f"✅ Filtered subset size: {len(df_filtered)} rows (all Source_Org_EXM=True)")

# ==============================================================
# Compute per-model metrics for each item (Org scenario)
# ==============================================================

for model in models:
    rows = []
    for item in items:
        scenario = "Org"

        exact_col = f"{model}_{item}_{scenario}_EXM"
        bleu_col = f"{model}_{item}_{scenario}_CodeBleu"
        coverage_col = f"{model}_{item}_{scenario}_Edit_coverage"
        total_ratio_col = f"{model}_{item}_{scenario}_Edit_total_ratio"
        removed_ratio_col = f"{model}_{item}_{scenario}_Edit_removed_ratio"
        added_ratio_col = f"{model}_{item}_{scenario}_Edit_added_ratio"

        if exact_col not in df_filtered.columns:
            print(f"⚠️ Missing {exact_col} — skipping {model} {item}")
            continue

        exm_true_ratio = df_filtered[exact_col].mean(skipna=True)
        codebleu_avg = df_filtered[bleu_col].mean(skipna=True) if bleu_col in df_filtered else np.nan
        coverage_true_ratio = df_filtered[coverage_col].mean(skipna=True) if coverage_col in df_filtered else np.nan

        if coverage_col in df_filtered:
            covered_df = df_filtered[df_filtered[coverage_col] == True]
            if not covered_df.empty:
                ratios = covered_df[
                    [total_ratio_col, removed_ratio_col, added_ratio_col]
                ].apply(pd.to_numeric, errors="coerce")
                ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna(how="all")
                total_avg = ratios[total_ratio_col].mean(skipna=True)
                removed_avg = ratios[removed_ratio_col].mean(skipna=True)
                added_avg = ratios[added_ratio_col].mean(skipna=True)
            else:
                total_avg = removed_avg = added_avg = np.nan
        else:
            total_avg = removed_avg = added_avg = np.nan

        rows.append({
            "Item": item,
            "EXM": round(exm_true_ratio, 3),
            "CodeBLEU": round(codebleu_avg, 3),
            "Coverage": round(coverage_true_ratio, 3),
            "Relative_Edit_Error": round(total_avg, 3),
            "Edit_Removed_Ratio": round(removed_avg, 3),
            "Edit_Added_Ratio": round(added_avg, 3),
        })

    df_model_summary = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, f"{model}_IntersectData_Results_{len(df_filtered)}.csv")
    df_model_summary.to_csv(output_path, index=False, encoding="utf-8")
    print(f"💾 Saved summary for {model}: {output_path}")

print("✅ All model summaries generated successfully.")


# ==============================================================
# === Dual-Axis Plotting ===
# ==============================================================

input_dir = "RQ1ResultsIntersect"
output_dir = "RQ1ResultsIntersect/plots_dual_axis_intersect"
os.makedirs(output_dir, exist_ok=True)

# Custom item order
items = [
    "RandomNames",
    "ShuffleNames",
    "TryNcatchWrapper",
    "DeadException",
    "DeadVariable",
    "ReturnViaVariable",
    "IndependentLineSwap",
    "IfElseSwap",
    "defUseBreak",
]

# ==============================================================
# === Loop through all CSV files and generate two-y-axis plots ===
# ==============================================================

for file in os.listdir(input_dir):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(input_dir, file)
    print(f"📄 Processing: {file_path}")

    df = pd.read_csv(file_path)
    df = df[df["Item"].isin(items)]
    df["Item"] = pd.Categorical(df["Item"], categories=items, ordered=True)
    df = df.sort_values("Item")
    df["Item"] = df["Item"].replace("defUseBreak", "DefUseBreak")

    if df.empty:
        print(f"⚠️ Skipping empty file: {file}")
        continue

    # --- Two-axis plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis — bounded metrics
    ax1.plot(df["Item"], df["EXM"], marker="o", label="ExactMatch", linewidth=2)
    ax1.plot(df["Item"], df["CodeBLEU"], marker="s", label="CodeBLEU", linewidth=2)
    ax1.plot(df["Item"], df["Coverage"], marker="^", label="EditMatch", linewidth=2)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Ratio (0–1)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right axis — Relative Edit Error
    color_edit = "red"
    ax2 = ax1.twinx()
    ax2.plot(df["Item"], df["Relative_Edit_Error"], marker="d", color=color_edit,
             label="RelativeEditError", linewidth=2)
    ax2.set_ylabel("REE", color=color_edit)
    ax2.tick_params(axis="y", colors=color_edit)
    ax2.spines["right"].set_color(color_edit)
    ax2.set_ylim(0, max(2, df["Relative_Edit_Error"].max()))

    # X-axis
    ax1.set_xticks(range(len(df["Item"])))
    ax1.set_xticklabels(df["Item"], rotation=45, ha="right")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
            loc="center left", bbox_to_anchor=(0.02, 0.5),
            frameon=True, fancybox=True, framealpha=0.8)



    plt.tight_layout()

    # Save as PDF
    model_name = os.path.splitext(file)[0]
    save_path = os.path.join(output_dir, f"{model_name}_metrics_dual_axis_plot.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"✅ Saved dual-axis plot: {save_path}")

print("\n🎉 All dual-axis model plots generated successfully!")