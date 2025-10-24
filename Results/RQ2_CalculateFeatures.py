import pandas as pd
import re
import numpy as np
import difflib
from itertools import combinations




csv_path = 'Final_ACR_Evaluation_Results.csv'
df = pd.read_csv(csv_path)

print("Column names:", df.columns.tolist())


def clean_code(code: str) -> str:
    """
    Remove <START> and <END> tags from the code.
    """
    return code.replace("<START>", "").replace("<END>", "").strip()



def tokenize_java_clean(code: str) -> list:
    """
    Tokenize Java code into meaningful tokens including:
    - Java keywords, identifiers, and operators
    - Special tags <START> and <END> as individual tokens
    Excludes structural punctuation like (), {}, ;, ,.
    """

    code = re.sub(r'(<START>)', r' \1 ', code)
    code = re.sub(r'(<END>)', r' \1 ', code)
    code = re.sub(r'[(){};,]', ' ', code)

    return re.findall(r'<START>|<END>|\w+|[+\-*/=<>!&|%^~]', code)




def count_java_tokens(code: str) -> int:
    """
    Return the number of meaningful tokens in Java code,
    excluding structural punctuation.
    """
    return len(tokenize_java_clean(code))



def token_edit_distance(source: str, mutated: str) -> int:
    """
    Compute token-level edit distance between two Java code strings
    that may contain <START> and <END> tags.
    """
    clean_source = clean_code(source)
    clean_mutated = clean_code(mutated)

    tokens_source = tokenize_java_clean(clean_source)
    tokens_mutated = tokenize_java_clean(clean_mutated)

    dp = np.zeros((len(tokens_source) + 1, len(tokens_mutated) + 1), dtype=int)
    for i in range(len(tokens_source) + 1):
        dp[i][0] = i
    for j in range(len(tokens_mutated) + 1):
        dp[0][j] = j

    # Compute DP for edit distance
    for i in range(1, len(tokens_source) + 1):
        for j in range(1, len(tokens_mutated) + 1):
            cost = 0 if tokens_source[i - 1] == tokens_mutated[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # delete
                dp[i][j - 1] + 1,        # insert
                dp[i - 1][j - 1] + cost  # substitute
            )

    return dp[-1][-1]


def find_token_differences(original_tokens, cf_tokens):
    s = difflib.SequenceMatcher(None, original_tokens, cf_tokens)
    differences = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != 'equal':
            differences.append((tag, i1, i2, j1, j2))
    return differences


def get_tagged_segment_indices(code: str) -> tuple:
    """
    Return the (start_idx, end_idx) of the tokens between <START> and <END>,
    in the tokenized list (with tags preserved).
    If no tags are found, the function adds them around the full string.
    """
    # Add tags if missing
    if "<START>" not in code or "<END>" not in code:
        code = f"<START>{code}<END>"

    tokens = tokenize_java_clean(code)
    start_idx = tokens.index("<START>") + 1
    end_idx = tokens.index("<END>") - 1
    return start_idx, end_idx



def get_tag_boundaries(src: str, mut: str) -> dict:
    """
    Uses get_tagged_segment_indices on both source and mutated code
    and returns a dictionary with start and end indices of tagged segments.
    """
    src_start, src_end = get_tagged_segment_indices(src)
    mut_start, mut_end = get_tagged_segment_indices(mut)

    return {
        'src_start': src_start,
        'src_end': src_end,
        'mut_start': mut_start,
        'mut_end': mut_end
    }


    
def categorize_and_measure_distances(differences, src: str, mut: str):
    tag_boundaries = get_tag_boundaries(src, mut)
    tag_start = tag_boundaries['mut_start']
    tag_end = tag_boundaries['mut_end']

    tokens_src = tokenize_java_clean(src)
    tokens_mut = tokenize_java_clean(mut)

    weighted_distances = []
    distances = []
    total_weights = 0

    categories = set()

    for _, _, _, j1, j2 in differences:
        segment_length = j2 - j1
        if segment_length == 0:
            continue

        if j2 <= tag_start:
            distance = abs(tag_start - j2)
            categories.add("Before")

        elif j1 >= tag_end + 1:
            distance = abs(j1 - tag_end - 1)
            categories.add("After")

        elif j1 >= tag_start and j2 <= tag_end + 1:
            distance = 0
            categories.add("Inside")

        elif j1 < tag_start and j2 > tag_end:
            distance = 0
            categories.add("Overlap-Both")

        elif j1 < tag_start and j2 > tag_start and j2 <= tag_end:
            distance = 0
            categories.add("Overlap-Before")

        elif j1 >= tag_start and j1 <= tag_end and j2 > tag_end:
            distance = 0
            categories.add("Overlap-After")

        distances.append(distance)
        weighted_distances.append(distance * segment_length)
        total_weights += segment_length

    weighted_average_distance = round(sum(weighted_distances) / total_weights, 2) if total_weights else 0
    average_distance = round(sum(distances) / len(distances), 2) if distances else 0


    # Rule 1: If 'Overlap-Both' is present, it's the primary category
    if "Overlap-Both" in categories:
        primary_category = "Overlap-Both"
    else:
        # Rule 2: Lookup table for other category combinations
        category_combinations = {
            frozenset(["Before", "After"]): "Surrounding",
            frozenset(["Before", "Inside"]): "Overlap-Before",
            frozenset(["Overlap-Before", "Before"]): "Overlap-Before",
            frozenset(["Before", "Overlap-After"]): "Overlap-Both",
            frozenset(["Inside", "After"]): "Overlap-After",
            frozenset(["Overlap-Before", "After"]): "Overlap-Both",
            frozenset(["Overlap-After", "After"]): "Overlap-After",
            frozenset(["Overlap-Before", "Inside"]): "Overlap-Before",
            frozenset(["Inside", "Overlap-After"]): "Overlap-After",
            frozenset(["Overlap-Before", "Overlap-After"]): "Overlap-Both",
            frozenset(["Before", "Inside", "After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "Before", "After"]): "Overlap-Both",
            frozenset(["Overlap-After", "Before", "After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "Before", "Inside"]): "Overlap-Before",
            frozenset(["Before", "Inside", "Overlap-After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "Before", "Overlap-After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "Inside", "After"]): "Overlap-Both",
            frozenset(["Overlap-After", "Inside", "After"]): "Overlap-After",
            frozenset(["Overlap-After", "Overlap-Before", "After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "Inside", "Overlap-After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "Before", "Inside", "After"]): "Overlap-Both",
            frozenset(["Overlap-After", "Before", "Inside", "After"]): "Overlap-Both",
            frozenset(["Overlap-After", "Overlap-Before", "Before", "After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "Before", "Inside", "Overlap-After"]): "Overlap-Both",
            frozenset(["Overlap-After", "Overlap-Before", "Inside", "After"]): "Overlap-Both",
            frozenset(["Overlap-Before", "After", "Inside", "Before", "Overlap-After"]): "Overlap-Both",
        }

        primary_category = category_combinations.get(frozenset(categories), next(iter(categories), "Overlap-Both"))

    return primary_category, weighted_average_distance, average_distance


#-------------------------------------------------------------------------------------------
# Constants
BaseTypes = ["IfElseSwap", "EqualAssert", "NullAssert", "TrueFalseAssert", "DeadException", "DeadVariable",
             "TryNcatchWrapper", "IndependentLineSwap", "ReturnViaVariable", "defUseBreak", "DataType",
             "RandomNames", "ShuffleNames"]

Models = ['T5', 'LLaMaLoRa', 'Llama3.3_70B', 'gpt-3.5-turbo', 'DeepSeek-V3']
Scenarios = ['Org']

result_df = pd.DataFrame(index=df.index)

for Type in BaseTypes:
    src_col = 'source_code'
    org_tgt_col = 'target'
    mut_col = f'{Type}Source'
    tgt_col = f'{Type}Target'

    org_pert_ed = []
    pert_fix_ed = []
    org_tgt_ed = []
    pert_len = []
    pos_list = []
    wdist = []
    dist = []


    for idx, row in df.iterrows():
        src, mut, tgt, org_tgt = row.get(src_col), row.get(mut_col), row.get(tgt_col), row.get(org_tgt_col)
        if pd.notna(mut) and pd.notna(tgt):
            for model in Models:
                col_name = f"{model}_{Type}_Org_EXM"
                if col_name in df.columns:
                    result_df[col_name] = df[col_name]
            try:
                ted_org_mut = token_edit_distance(src, mut)
                ted_mut_tgt = token_edit_distance(clean_code(mut), tgt)
                ted_org_tgt = token_edit_distance(src, org_tgt) if pd.notna(org_tgt) else np.nan 

            
                org_pert_ed.append(ted_org_mut)
                pert_fix_ed.append(ted_mut_tgt)
                org_tgt_ed.append(ted_org_tgt) 

        
            
                # Continue with the rest
                pert_len.append(count_java_tokens(mut))
                diffs = find_token_differences(tokenize_java_clean(src), tokenize_java_clean(mut))
                pos, w, d = categorize_and_measure_distances(diffs, src, mut)
                pos_list.append(pos)
                wdist.append(w)
                dist.append(d)

            except Exception as e:
                print(f"[{Type}] Error at row {idx}:\n{e}")

        else:
            org_pert_ed.append(np.nan)
            pert_fix_ed.append(np.nan)
            pert_len.append(np.nan)
            pos_list.append(np.nan)
            wdist.append(np.nan)
            dist.append(np.nan)
            org_tgt_ed.append(np.nan)

    result_df[f"{Type}_org_pert_tokenEdit"] = org_pert_ed
    result_df[f"{Type}_pert_fix_tokenEdit"] = pert_fix_ed
    result_df[f"{Type}_Pert_length"] = pert_len
    result_df[f"{Type}_pos"] = pos_list
    result_df[f"{Type}_Wpert_dist"] = wdist
    result_df[f"{Type}_pert_dist"] = dist
    result_df[f"{Type}_org_tgt_tokenEdit"] = org_tgt_ed 

        

result_df.to_csv("Perturbation_Metrics_Results.csv", index=False)
print("Column names:", result_df.columns.tolist())



# ---------------------- Overall Summary Statistics Across All BaseTypes ----------------------

# List of metric suffixes to analyze
metric_suffixes = [
    "_org_pert_tokenEdit",
    "_pert_fix_tokenEdit",
    "_org_tgt_tokenEdit",  # <-- include new metric in statistics
    "_Pert_length",
    "_Wpert_dist",
    "_pert_dist"
]

# Prepare a dictionary to collect all valid values per metric suffix across BaseTypes
metric_values = {suffix: [] for suffix in metric_suffixes}

# Gather all values from result_df
for base in BaseTypes:
    for suffix in metric_suffixes:
        col_name = f"{base}{suffix}"
        if col_name in result_df.columns:
            valid_data = result_df[col_name].dropna().tolist()
            metric_values[suffix].extend(valid_data)

# Calculate summary stats per metric (across all BaseTypes)
overall_summary = []

for suffix, values in metric_values.items():
    if values:  # Only process if we have data
        series = pd.Series(values)

        # Exclude zeros for the Min calculation
        nonzero_values = [v for v in values if v != 0]
        min_nonzero = min(nonzero_values) if nonzero_values else 0

        stats = {
            "Metric": suffix,
            "Min": min_nonzero,
            "Max": series.max(),
            "Median": series.median(),
            "Mean": series.mean(),
            "StdDev": series.std()
        }
        overall_summary.append(stats)

# Save to CSV
overall_summary_df = pd.DataFrame(overall_summary)
overall_summary_df.to_csv("Perturbation_Metrics_Statistics.csv", index=False)

print("Overall statistics saved to 'Perturbation_Metrics_Statistics.csv'")



# ---------------------- Count Cases Per Position Category Across All BaseTypes ----------------------

from collections import Counter

# Collect all position labels from all BaseTypes
all_pos_labels = []

for base in BaseTypes:
    col_name = f"{base}_pos"
    if col_name in result_df.columns:
        all_pos_labels.extend(result_df[col_name].dropna().tolist())

# Count frequency of each position category
pos_counts = Counter(all_pos_labels)

# Print results
print("\nPosition Category Distribution (across all BaseTypes):")
for category, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{category}: {count} cases")


