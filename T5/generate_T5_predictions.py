import os, sys, re, logging, torch, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

# === CLI Arguments ===
if len(sys.argv) != 3:
    print("Usage: python script.py <TYPE> <SCENARIO>")
    sys.exit(1)

CF_Type = sys.argv[1]
scenario = sys.argv[2]
pred_col = f"predictions_{CF_Type}_{scenario}"
em_col = f"exact_match_{CF_Type}_{scenario}"

# === Setup Logging ===
OUTPUT_DIR = Path("./predictions_T5")
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_file_path = os.path.join(OUTPUT_DIR, f"{CF_Type}_{scenario}_predictions.csv")
log_file_path = os.path.join(OUTPUT_DIR, f"log_{CF_Type}_{scenario}.txt")

fh = logging.FileHandler(log_file_path)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(fh)
log_stream = fh.stream

# === Reproducibility ===
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# === Model Setup ===
data_dir = "./Perturbations.csv"
tokenizer_name = "./tokenizer/TokenizerModel.model"
model_name_or_path = "./PytorchFinetunedT5/model.bin"
config_name = "./config.json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
config = T5Config.from_pretrained(config_name, output_attentions=True)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config).to(DEVICE)
model.eval()

# === EvalDataset Using Only Valid Rows ===
class EvalDataset(Dataset):
    def __init__(self, data_path, scenario):
        self.df_full = pd.read_csv(data_path)
        self.df_full[pred_col] = ""
        self.df_full[em_col] = 0
        
    
        self.source_col = CF_Type + 'Source'
        self.comment_col = CF_Type + 'Comment'
        self.target_col = CF_Type + 'Target'
        

        valid_mask = (
            self.df_full[self.source_col].notna() &
            self.df_full[self.comment_col].notna() &
            self.df_full[self.target_col].notna()
        )
        valid_mask &= ~self.df_full[self.source_col].astype(str).str.upper().eq("NA")
        valid_mask &= ~self.df_full[self.comment_col].astype(str).str.upper().eq("NA")
        valid_mask &= ~self.df_full[self.target_col].astype(str).str.upper().eq("NA")
        
        valid_count = valid_mask.sum()
        total_count = len(self.df_full)
        logger.info(f"Valid rows after filtering: {valid_count} / {total_count}")


        self.df = self.df_full[valid_mask].copy()
        self.df.reset_index(inplace=True)  # preserve original index in 'index' column

        self.df[self.source_col] = self.df[self.source_col].astype(str).str.strip()
        self.df[self.comment_col] = self.df[self.comment_col].astype(str).str.strip()
        self.df[self.target_col] = self.df[self.target_col].astype(str).str.strip()

        self.samples = []
        for _, row in self.df.iterrows():
            source = row[self.source_col]
            comment = row[self.comment_col]
            if scenario == "Org":
                prompt = f"<code>{source}</code><technical_language>{comment}</technical_language>"
            elif scenario == "Mitigation":
                match = re.search(r"<START>(.*?)<END>", source, re.DOTALL)
                extracted = match.group(1).strip() if match else ""
                mitigation_comment = f"For this part of the Java Code : {extracted}, this review comment is provided: {comment}."
                prompt = f"<code>{source}</code><technical_language>{mitigation_comment}</technical_language>"
            elif scenario == "Inline":
                insertion = "<END> // " + comment
                if "<END>" in source:
                    source = source.replace("<END>", insertion, 1)
                else:
                    source += f"// {comment}"
                prompt = f"<code>{source}</code><technical_language></technical_language>"
            elif scenario == "CoT":
                cot_comment = comment + " Provide step-by-step reasoning about how the review comment relates to the code."
                prompt = f"<code>{source}</code><technical_language>{cot_comment}</technical_language>"
            else:
                continue

            self.samples.append((prompt, row[self.target_col], row["index"]))

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

# === Prepare Dataset & Dataloader ===
dataset = EvalDataset(data_dir, scenario)
dloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)

# === Inference Loop ===
beam_size = 10
first_write = True
em_count = 0
valid_processed = 0
df = dataset.df_full  # Includes all rows

for batch in tqdm(dloader, total=len(dloader), desc=f"{CF_Type}-{scenario}"):
    input_text, gold, index = batch[0][0], batch[1][0], batch[2].item()

    valid_processed += 1

    encoded = tokenizer([input_text], add_special_tokens=False, return_tensors='pt', padding=True)
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=beam_size,
            early_stopping=True,
            num_return_sequences=beam_size,
            output_attentions=True,
            return_legacy_cache=True
        )

    decoded_predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    preds_list = [f"{i+1}- {pred.strip()}" for i, pred in enumerate(decoded_predictions)]
    match = any(str(pred).strip() == str(gold).strip() for pred in decoded_predictions)

    if match:
        em_count += 1

    df.at[index, pred_col] = "\n".join(preds_list)
    df.at[index, em_col] = 1 if match else 0

    pd.DataFrame([df.iloc[index]]).to_csv(csv_file_path, mode='a', header=first_write, index=False)
    first_write = False

    torch.cuda.empty_cache()

# === Final Summary ===
logger.info(f"Processed {valid_processed} valid rows.")
logger.info(f"Exact matches: {em_count}")
logger.info(f"Final CSV saved at {csv_file_path}")
logger.info(f"Final columns in output CSV: {list(df.columns)}")

log_stream.flush()

