import re
import logging
import time
import torch
import pandas as pd
from xturing.models.base import BaseModel
from tqdm import tqdm


import sys
import os

# Adjust this path to point to your repo root where `utils/smooth_bleu.py` lives
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(ROOT_DIR)

from utils.smooth_bleu import bleu_fromstr


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
generation_config.penalty_alpha = 0.6
generation_config.top_k = 1
generation_config.max_new_tokens = 600
model.engine.model.config.eos_token_id = model.engine.tokenizer.eos_token_id
model.engine.model.config.num_beams = 10

# Set up general logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_all")
logger.setLevel(logging.INFO)

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
    df = pd.read_json(test_file_path, lines=True, orient='records')
    inputs = df["input"].tolist()
    golds = df["output"].tolist()

    predictions = []
    for input_text in tqdm(inputs, desc=f"{type_name}-{scenario_name}"):
        with torch.no_grad():
            prediction = model.generate(texts=[input_text])[0]
        predictions.append(prediction)
        torch.cuda.empty_cache()

    # Save predictions and golds
    with open(prediction_file_path, "w") as f:
        for p in predictions:
            f.write(repr(p)[1:-1] + "\n")

    with open(gold_file_path, "w") as f:
        for g in golds:
            f.write(repr(g)[1:-1] + "\n")

    # Compute EM
    em_count = sum(
        re.sub(r'\s+', ' ', pred).lower() == re.sub(r'\s+', ' ', gold).lower()
        for pred, gold in zip(predictions, golds)
    )
    em_ratio = em_count / len(golds)

    logger.info(f"Total records: {len(golds)}")
    logger.info(f"Exact matches: {em_count}")
    logger.info(f"EM Ratio: {em_ratio:.4f}")
    logger.info(f"BLEU: {bleu_fromstr(predictions, golds)}")
    logger.info(f"Time: {time.time() - t0:.2f}s | Throughput: {len(predictions) / (time.time() - t0):.2f} samples/s")
    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    results.append({
        "Type": type_name,
        "Scenario": scenario_name,
        "Total": len(golds),
        "ExactMatchCount": em_count,
        "EMRatio": round(em_ratio, 4)
    })

    # Remove handler to avoid duplicate logs
    logger.removeHandler(fh)

# === Run All Evaluations ===
for t in Types:
    for s in Scenarios:
        evaluate_file(t, s)

# Save summary
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "summary_em_results.csv"), index=False)
