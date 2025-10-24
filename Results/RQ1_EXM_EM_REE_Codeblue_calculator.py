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
# === Helper Functions
# ==============================================================


def extract_list_predictions(pred_str):
    """
    Extract predictions stored as a Python list-like string.
    (Used for GPT, Llama3.3, DeepSeek, etc.)
    Handles strings like: "['pred1', 'pred2', 'pred3']"
    """
    if not isinstance(pred_str, str):
        return []

    text = pred_str.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            result = ast.literal_eval(text)
            if isinstance(result, list):
                return [str(r).strip() for r in result if str(r).strip()]
            elif isinstance(result, str):
                return [result.strip()]
        except Exception:
            pass

    if "', '" in text:
        parts = re.split(r"',\s*'", text.strip("[]'\" "))
        return [p.strip() for p in parts if p.strip()]

    return [text]


def extract_numbered_predictions(pred_str):
    """
    Extract predictions from numbered text format (used by T5 or LoRA-tuned LLaMA)
    Example:
        1- pred1
        2- pred2
    """
    if not isinstance(pred_str, str):
        return []

    text = pred_str.strip()
    pattern = r"\d+\s*[-–]\s*(.*?)(?=\n\d+\s*[-–]\s*|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return [text] if text else []


def calculate_bleu(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)


def tokenize_java_words(code):
    return re.findall(r"[A-Za-z_]+", code)


def compare_word_lists(List1, List2):
    return [w.lower() for w in List1] == [w.lower() for w in List2]


def remove_start_end_tokens(text):
    return re.sub(r"<START>|<END>", "", text)


def clean_extra_spaces(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_spacing_to_match(s1, s2):
    def normalize(text):
        text = "\n".join(line.strip() for line in text.splitlines())
        text = re.sub(r"\n\s*\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+([,;(){}])", r"\1", text)
        text = re.sub(r"([({,;])\s+", r"\1", text)
        return text.strip()
    norm1, norm2 = normalize(s1), normalize(s2)
    if norm1 == norm2:
        return norm1, norm2
    len1, len2 = len(s1), len(s2)
    if len1 > len2:
        return normalize(s1), s2
    elif len2 > len1:
        return s1, normalize(s2)
    else:
        return normalize(s1), normalize(s2)


def tokenize_java(code):
    pattern = r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||[{}()\[\];,<>!=+\-*/%]"
    return re.findall(pattern, code)


def Generate_diff_general(source, target):
    src_tokens = tokenize_java(source)
    tgt_tokens = tokenize_java(target)
    matcher = difflib.SequenceMatcher(None, src_tokens, tgt_tokens, autojunk=False)
    deleted_groups, added_groups = [], []
    current_deleted, current_added = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            current_deleted.extend(src_tokens[i1:i2])
        if tag in ("replace", "insert"):
            current_added.extend(tgt_tokens[j1:j2])
        if tag == "equal":
            if current_deleted:
                deleted_groups.append(" ".join(current_deleted))
                current_deleted = []
            if current_added:
                added_groups.append(" ".join(current_added))
                current_added = []
    if current_deleted:
        deleted_groups.append(" ".join(current_deleted))
    if current_added:
        added_groups.append(" ".join(current_added))
    return deleted_groups, added_groups


def normalize_text(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def check_diff_coverage(gold_removed, gold_added, prediction_removed, prediction_added):
    def all_gold_in_pred(gold_list, pred_list):
        pred_texts = [normalize_text(p) for p in pred_list]
        for gold_item in gold_list:
            g = normalize_text(gold_item)
            if not any(g in p for p in pred_texts):
                return False
        return True
    return all_gold_in_pred(gold_removed, prediction_removed) and all_gold_in_pred(gold_added, prediction_added)


def clean_and_tokenize_edits(edit_list):
    clean_tokens = []
    for edit in edit_list:
        cleaned = re.sub(r"[{}()\[\];,<>!=+\-*/%&|^~?:.]", " ", edit)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", cleaned)
        clean_tokens.extend(tokens)
    return clean_tokens


def over_edit_ratio_clean(gold_removed, gold_added, prediction_removed, prediction_added):
    gold_removed_toks = clean_and_tokenize_edits(gold_removed)
    gold_added_toks = clean_and_tokenize_edits(gold_added)
    pred_removed_toks = clean_and_tokenize_edits(prediction_removed)
    pred_added_toks = clean_and_tokenize_edits(prediction_added)
    gold_total = len(gold_removed_toks) + len(gold_added_toks)
    pred_total = len(pred_removed_toks) + len(pred_added_toks)
    def safe_ratio(pred, gold):
        if gold == 0:
            return float("inf") if pred > 0 else 0.0
        return round((pred - gold) / gold, 3)
    return {
        "total_ratio": safe_ratio(pred_total, gold_total),
        "removed_ratio": safe_ratio(len(pred_removed_toks), len(gold_removed_toks)),
        "added_ratio": safe_ratio(len(pred_added_toks), len(gold_added_toks)),
    }


# ==============================================================
# === Main Processing Loop
# ==============================================================

df = pd.read_csv("Final_ACR_Evaluation_Results.csv")
dataSize = len(df)
print("Number of rows:", dataSize)

patterns_to_remove = ["DataType", "EqualAssert", "NullAsser", "TrueFalseAssert", "EXM"]

filtered_columns = [
    c for c in df.columns if not any(pat in c for pat in patterns_to_remove)
]
df = df[filtered_columns]

print("Column names:", df.columns.tolist())


metrics_file = "Perturbation_Metrics_Results.csv"
df_metrics = pd.read_csv(metrics_file) if os.path.exists(metrics_file) else None
for col in df_metrics.columns:
    print(col)
    

models = ["gpt-3.5-turbo", "Llama3.3_70B", "DeepSeek-V3", "T5", "LLaMaLoRa"]

items = [
    "Source",
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
scenarios = ["Org", "CodeRepetition", "InlineComment", "CoT"]

output_dir = "Reports"



for model, item, scenario in product(models, items, scenarios):
    skip_count = 0
    print(f"\n--- Evaluating {model} | {item} | {scenario} ---")
    

    pred_col = f"{model}_{item}_{scenario}_pred"
    
    
    if item == "Source": 
        source_col = "source_code" 
        target_col = "target" 
    else: 
        source_col = f"{item}Source"
        target_col = f"{item}Target"

    if pred_col not in df.columns:
        print(f"⚠️ Missing column: {pred_col} — skipping.")
        continue

    summary_txt_path = os.path.join(output_dir, f"{model}_{item}_{scenario}_summary.txt")

    # === Create new evaluation columns ===
    exact_col = f"{model}_{item}_{scenario}_EXM"
    bleu_col = f"{model}_{item}_{scenario}_CodeBleu"
    total_ratio_col = f"{model}_{item}_{scenario}_Edit_total_ratio"
    removed_ratio_col = f"{model}_{item}_{scenario}_Edit_removed_ratio"
    added_ratio_col = f"{model}_{item}_{scenario}_Edit_added_ratio"
    coverage_col = f"{model}_{item}_{scenario}_Edit_coverage"

    for col in [exact_col, bleu_col, total_ratio_col, removed_ratio_col, added_ratio_col, coverage_col]:
        df[col] = np.nan

    for idx, row in df.iterrows():
        source = row.get(source_col, "")
        target = row.get(target_col, "")
        pred = row.get(pred_col, "")



        if pd.isna(source) or pd.isna(target) or pd.isna(pred):
            skip_count += 1
            continue

        if model in ["T5", "LLaMaLoRa"]:
            predictions = extract_numbered_predictions(pred)
        else:
            predictions =  extract_list_predictions(pred)

        if not predictions:
            skip_count += 1
            continue
        
        
        NoToken_source = remove_start_end_tokens(source)
        NoToken_source_cleaned = clean_extra_spaces(NoToken_source)
        target_cleaned = clean_extra_spaces(target)

        # === (1) Exact Match ===
        exact_found = any(
            compare_word_lists(
                tokenize_java_words(normalize_spacing_to_match(clean_extra_spaces(p), target_cleaned)[0]),
                tokenize_java_words(normalize_spacing_to_match(clean_extra_spaces(p), target_cleaned)[1])
            ) for p in predictions
        )
        df.at[idx, exact_col] = exact_found

        # === (2) BLEU Score ===
        bleu_scores = [
            calculate_bleu(*normalize_spacing_to_match(target_cleaned, clean_extra_spaces(p)))
            for p in predictions
        ]
        df.at[idx, bleu_col] = max(bleu_scores) if bleu_scores else 0.0

        # === (3) Edit Coverage + Ratios ===
        gold_removed, gold_added = Generate_diff_general(NoToken_source_cleaned, target_cleaned)
        results = []
        for p in predictions:
            norm_cleaned_prediction, _ = normalize_spacing_to_match(clean_extra_spaces(p), target_cleaned)
            prediction_removed, prediction_added = Generate_diff_general(NoToken_source_cleaned, norm_cleaned_prediction)
            coverage = check_diff_coverage(gold_removed, gold_added, prediction_removed, prediction_added)
            stats = over_edit_ratio_clean(gold_removed, gold_added, prediction_removed, prediction_added)
            results.append({"coverage": coverage, "stats": stats})

        covered = [r for r in results if r["coverage"]]
        if covered:
            best = min(covered, key=lambda r: abs(r["stats"]["total_ratio"]))
            df.at[idx, coverage_col] = True
            df.at[idx, total_ratio_col] = best["stats"]["total_ratio"]
            df.at[idx, removed_ratio_col] = best["stats"]["removed_ratio"]
            df.at[idx, added_ratio_col] = best["stats"]["added_ratio"]
        else:
            df.at[idx, coverage_col] = False


    
    # === Write Summary ===
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Evaluation Summary: {model} | {item} | {scenario} ===\n\n")
        f.write(f"Total rows: {dataSize}\n")
        f.write(f"Skipped rows: {skip_count}\n")
        f.write(f"Total evaluated: {dataSize - skip_count}\n\n")

        exm_true_ratio = df[exact_col].mean(skipna=True)
        codebleu_avg = df[bleu_col].mean(skipna=True)
        coverage_true_ratio = df[coverage_col].mean(skipna=True)

        f.write(f"EXM: {exm_true_ratio:.3f}\n")
        f.write(f"CodeBLEU: {codebleu_avg:.3f}\n")
        f.write(f"Coverage: {coverage_true_ratio:.3f}\n")

        covered_df = df[df[coverage_col] == True]
        if not covered_df.empty:
            ratios = covered_df[[total_ratio_col, removed_ratio_col, added_ratio_col]].apply(pd.to_numeric, errors="coerce")
            ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna(how="all")
            f.write("\nEdit Ratio Averages:\n")
            f.write(f"  Total: {ratios[total_ratio_col].mean():.3f}\n")
            f.write(f"  Removed: {ratios[removed_ratio_col].mean():.3f}\n")
            f.write(f"  Added: {ratios[added_ratio_col].mean():.3f}\n")
        else:
            f.write("\nNo covered rows found.\n")
            


output_path = os.path.join(output_dir, "Final_Evaluation_With_CodeBlue.csv")
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ DataFrame saved to: {output_path}")


    