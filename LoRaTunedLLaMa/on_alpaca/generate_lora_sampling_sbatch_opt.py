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

def load_completed_indices(csv_path):
    df_done = pd.read_csv(csv_path)
    completed = set(df_done[df_done["exact_match"] == 1].index)
    return completed

def is_invalid(row, completed_indices):
    if row.name not in completed_indices:
        return False

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
    java_code = re.sub(r'<START>|<END>', '', java_code)
    java_code = re.sub(r'\s+', ' ', java_code)
    java_code = re.sub(r'\s*([\{\};()])\s*', r'\1', java_code)
    return java_code

def compare_by_words(str1, str2):
    str1L = str1.lower()
    str2L = str2.lower()
    words1 = re.findall(r'[a-zA-Z]+', str1L)
    words2 = re.findall(r'[a-zA-Z]+', str2L)
    return words1 == words2

def set_seed(seed):
    seed = seed % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

MODEL_PATH = "./tufano_cr_alpaca_weights_16_16"
TEST_DIR = "../"
OUTPUT_DIR = "./predictions_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = BaseModel.load(MODEL_PATH)
generation_config = model.generation_config()
generation_config.do_sample = True
generation_config.penalty_alpha = 0.6
generation_config.top_k = 50
generation_config.max_new_tokens = 600
model.engine.model.config.eos_token_id = model.engine.tokenizer.eos_token_id
model.engine.model.config.num_beams = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_all")
logger.setLevel(logging.INFO)

def evaluate_file(type_name, scenario_name):
    test_file_path = os.path.join(TEST_DIR, f"tufano_cr_test_{type_name}_{scenario_name}.jsonl")
    csv_file_path = os.path.join(OUTPUT_DIR, f"updated_df_{type_name}_{scenario_name}.csv")
    log_file_path = os.path.join(OUTPUT_DIR, f"log_{type_name}_{scenario_name}.txt")
    Source_file_path = os.path.join(OUTPUT_DIR, "updated_df_Source_Org.csv")

    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if fh not in logger.handlers:
        logger.addHandler(fh)
    log_stream = fh.stream  # For manual flushing

    df = pd.read_json(test_file_path, lines=True, orient='records')
    logger.info(f"Loaded {len(df)} rows from test file")
    log_stream.flush()

    completed_indices = load_completed_indices(Source_file_path)
    logger.info(f"{len(completed_indices)} completed indices loaded from {Source_file_path}")
    log_stream.flush()

    df["predictions"] = ""
    df["exact_match"] = 0

    inputs = df["input"].tolist()
    golds = df["output"].tolist()

    first_write = True
    valid_processed = 0

    for idx, input_text in enumerate(tqdm(inputs, desc=f"{type_name}-{scenario_name}")):
        row = df.iloc[idx]
        if is_invalid(row, completed_indices):
            logger.info(f"Skipping invalid or completed row at index {idx}")
            log_stream.flush()
            continue

        logger.info(f"Processing row {idx}")
        log_stream.flush()

        valid_processed += 1
        num_preds = 0
        seed_counter = 0
        max_attempts = 50
        preds_list = []

        while num_preds < 10 and seed_counter < max_attempts:
            seed = (hash(input_text) + seed_counter) % (2**32 - 1)
            set_seed(seed)
            seed_counter += 1

            with torch.no_grad():
                try:
                    prediction = model.generate(texts=[input_text])[0]
                    prediction = prediction.replace(model.engine.tokenizer.eos_token, "").strip()
                except Exception as e:
                    logger.info(f"Generation failed at seed {seed_counter - 1}: {e}")
                    log_stream.flush()
                    continue

            if prediction.lower() in ["", "nan"]:
                logger.info(f"Discarded invalid prediction at seed {seed_counter - 1}: '{prediction}'")
                log_stream.flush()
                continue

            preds_list.append(f"{num_preds + 1}- {prediction}")
            num_preds += 1

        if num_preds < 10:
            logger.info(f"Only {num_preds}/10 predictions generated for index {idx}")
            log_stream.flush()

        preds_str = "\n".join(preds_list)
        gold_cleaned = clean_java_code(golds[idx])
        pred_cleaned = [clean_java_code(p.split("-", 1)[-1]) for p in preds_list]
        matched = any(compare_by_words(pred, gold_cleaned) for pred in pred_cleaned)

        df.at[df.index[idx], "predictions"] = preds_str
        df.at[df.index[idx], "exact_match"] = 1 if matched else 0

        torch.cuda.empty_cache()

        df.iloc[[idx]].to_csv(csv_file_path, mode='a', header=first_write, index=False)
        first_write = False

    exact_matches = df["exact_match"].sum()
    logger.info(f"Processed {valid_processed} valid rows.")
    logger.info(f"Exact matches among them: {exact_matches}")
    logger.info(f"Final CSV saved at {csv_file_path}")
    log_stream.flush()

    logger.removeHandler(fh)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <TYPE> <SCENARIO>")
        sys.exit(1)

    type_arg = sys.argv[1]
    scenario_arg = sys.argv[2]

    evaluate_file(type_arg, scenario_arg)

