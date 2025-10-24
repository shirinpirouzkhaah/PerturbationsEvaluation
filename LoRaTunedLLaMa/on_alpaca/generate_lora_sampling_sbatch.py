import re
import logging
import time
import torch
import pandas as pd
from xturing.models.base import BaseModel
from tqdm import tqdm
import random
import numpy as np


import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(ROOT_DIR)

from utils.smooth_bleu import bleu_fromstr
 

def is_invalid(row):
    input_text = str(row["input"])
    output_text = str(row["output"]).strip().lower()
    match = re.search(r"The comment is: '(.*?)'\s*(?:\\n|\n)The code is: '(.*?)'", input_text, re.IGNORECASE)
    if match:
        comment = match.group(1).strip().lower()
        code = match.group(2).strip().lower()
        return comment == "nan" or code == "nan" or output_text == "nan"
    else:
        return True  


def clean_java_code(java_code):
    java_code = java_code.strip()
    java_code = re.sub(r'<START>|<END>', '', java_code)  # Remove <START> and <END>
    java_code = re.sub(r'\s+', ' ', java_code)
    java_code = re.sub(r'\s*([\{\};()])\s*', r'\1', java_code)
    return java_code



def compare_by_words(str1, str2):
    
    str1 = str1.lower()
    str2 = str2.lower()
   
    words1 = re.findall(r'[a-zA-Z]+', str1)
    words2 = re.findall(r'[a-zA-Z]+', str2)
    
    return words1 == words2



# Configs
Types = [
    "Source", "IfElseCF", "ExceptionCF", "DeadCodeCF", "TryNcatchCF",
    "DataFlowCF", "EqualAssertCF", "NullAssertCF", "TrueFalseAssertCF",
    "ShuffleNamesCF", "RandomNamesCF", "IndependentSwapCF", "defUseBreakCF"
]
Scenarios = ["Org", "Mitigation", "Inline", "CoT"]

MODEL_PATH = "./tufano_cr_alpaca_weights_16_16"
TEST_DIR = "../"  
OUTPUT_DIR = "./predictions_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model once
model = BaseModel.load(MODEL_PATH)
generation_config = model.generation_config()


generation_config = model.generation_config()
generation_config.do_sample = True
generation_config.penalty_alpha = 0.6
generation_config.top_k = 50
generation_config.max_new_tokens = 600
model.engine.model.config.eos_token_id = model.engine.tokenizer.eos_token_id
model.engine.model.config.num_beams = 1


# Set up general logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_all")
logger.setLevel(logging.INFO)

# Utility: reproducible seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# Results collection
results = []

def evaluate_file(type_name, scenario_name):
    test_file_path = os.path.join(TEST_DIR, f"tufano_cr_test_{type_name}_{scenario_name}.jsonl")
    prediction_file_path = os.path.join(OUTPUT_DIR, f"preds_{type_name}_{scenario_name}.jsonl")
    gold_file_path = os.path.join(OUTPUT_DIR, f"golds_{type_name}_{scenario_name}.jsonl")
    log_file_path = os.path.join(OUTPUT_DIR, f"log_{type_name}_{scenario_name}.txt")

    # Redirect log for each run
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    t0 = time.time()
    dfF = pd.read_json(test_file_path, lines=True, orient='records')
    print(f"Loaded {len(dfF)} records from {test_file_path}")
    
    df = dfF[~dfF.apply(is_invalid, axis=1)]
    
    print(f"Remaining clean records: {len(df)}")
    print(f"Records removed: {len(dfF) - len(df)}")

    df = df.head(10)
    inputs = df["input"].tolist()
    golds = df["output"].tolist()


    with open(gold_file_path, "w") as f:
        for g in golds:
            print(g)
            f.write(g + "\n")

    
    with open(prediction_file_path, "w") as pred_file:
        for input_text in tqdm(inputs, desc=f"{type_name}-{scenario_name}"):
            num_preds = 0
            seed_counter = 0
            while num_preds < 10:
                set_seed(seed_counter)
                seed_counter += 1
                with torch.no_grad():
                    try:
                        prediction = model.generate(texts=[input_text])[0]
                    except Exception as e:
                        print(f"Generation failed at seed {seed_counter - 1}: {e}")
                        continue
            
                if prediction == "" or prediction == "nan":
                    print(f"Discarded invalid prediction at seed {seed_counter - 1}: '{prediction}'")
                    continue
            
                print(prediction)
                pred_file.write(prediction + "\n")
                torch.cuda.empty_cache()
                num_preds += 1




    # Compute EM using 10 predictions per input
    em_count = 0
    with open(prediction_file_path, "r") as f:
        all_preds = f.readlines()

    assert len(all_preds) == len(golds) * 10, "Mismatch: Predictions count is not 10x golds"

    for i, gold in enumerate(golds):
        gold_cleaned = clean_java_code(gold)
        pred_chunk = all_preds[i*10:(i+1)*10]
        matched = any(compare_by_words(clean_java_code(pred.strip()), gold_cleaned) for pred in pred_chunk)
        if matched:
            em_count += 1

    em_ratio = em_count / len(golds)


    logger.info(f"Total records: {len(golds)}")
    logger.info(f"Exact matches: {em_count}")
    logger.info(f"EM Ratio: {em_ratio:.4f}")

    results.append({
        "Type": type_name,
        "Scenario": scenario_name,
        "Total": len(golds),
        "ExactMatchCount": em_count,
        "EMRatio": round(em_ratio, 4)
    })

    # Remove handler to avoid duplicate logs
    logger.removeHandler(fh)

# === Run Single Evaluation from CLI Args ===
if len(sys.argv) != 3:
    print("Usage: python script.py <TYPE> <SCENARIO>")
    sys.exit(1)

type_arg = sys.argv[1]
scenario_arg = sys.argv[2]

if type_arg not in Types:
    raise ValueError(f"Invalid TYPE: {type_arg}")
if scenario_arg not in Scenarios:
    raise ValueError(f"Invalid SCENARIO: {scenario_arg}")

evaluate_file(type_arg, scenario_arg)

# Save single result
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, f"summary_results_{type_arg}_{scenario_arg}.csv"), index=False)
