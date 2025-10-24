import json
import pandas as pd
import re
from datasets import Dataset, DatasetDict
from xturing.engines.llama_utils import LlamaTokenizer
from tqdm import tqdm
import os

# Constants
'''
| Segment                     | Token Count |
|----------------------------|-------------|
| Input (instruction + input)| 1348        |
| Output (refined code)      | ≤ 600       |
| Buffer (BOS/EOS etc.)      | ~100        |
| Total                      | ~2048       |
'''

MAX_INSTRCUTION_AND_INPUT_LENGTH = 1348
MAX_OUTPUT_LENGTH = 600

# Types and Scenarios
Types = [
    "Source", "IfElseCF", "ExceptionCF", "DeadCodeCF", "TryNcatchCF",
    "DataFlowCF", "EqualAssertCF", "NullAssertCF", "TrueFalseAssertCF",
    "ShuffleNamesCF", "RandomNamesCF", "IndependentSwapCF", "defUseBreakCF"
]

Scenarios = ["Org", "Mitigation", "Inline", "CoT"]

# Tokenizer
tokenizer = LlamaTokenizer.from_pretrained("aleksickx/llama-7b-hf", add_bos_token=False)


def extract_tagged_code(source_code):
    pattern = r"<START>(.*?)<END>"
    match = re.search(pattern, source_code, re.DOTALL)
    return match.group(1).strip() if match else ""


def build_mitigation_string(review_comment, extracted_code):
    return f"For this part of the code : {extracted_code}, this comment is provided: {review_comment}."


def preprocess_all_code_refinements(csv_path: str):
    df = pd.read_csv(csv_path)

    for Type in Types:
        for Scenario in Scenarios:
            print(f"\n➡️  Processing Type: '{Type}' | Scenario: '{Scenario}'")
            test_inputs, test_outputs = [], []

            for index in tqdm(range(len(df))):
                # ======= Step 1: Extract columns =======
                if Type == "Source" :
                    source_code = str(df.iloc[index].get('source_code')).strip()
                    review_comment = str(df.iloc[index].get('review_comment')).strip()
                    target = str(df.iloc[index].get('target')).strip()
                else:
                    source_col = f"{Type}Source"
                    comment_col = f"{Type}Comment"
                    target_col = f"{Type}Target"
                    source_code = str(df.iloc[index].get(source_col)).strip()
                    review_comment = str(df.iloc[index].get(comment_col)).strip()
                    target = str(df.iloc[index].get(target_col)).strip()

                # ======= Step 2: Choose instruction =======
                if Scenario == "CoT":
                    instruction = (
                        "First reason step-by-step in one short sentence, how the provided feedback applies to the code."
                        "Then refine the code based on the feedback provided in the code review comment, with a focus on the "
                        "segment between <START> and <END>."
                    )
                else:
                    instruction = (
                        "Refine the code based on the feedback provided in the code review comment, "
                        "with a focus on the segment between <START> and <END>."
                    )

                # ======= Step 3: Scenario-based changes =======
                if Scenario == "Mitigation":
                    extracted = extract_tagged_code(source_code)
                    if extracted:
                        review_comment = build_mitigation_string(review_comment, extracted)

                if Scenario == "Inline":
                    end_tag = "<END>"
                    insertion = f"{end_tag} // {review_comment}"
                    if end_tag in source_code:
                        source_code = source_code.replace(end_tag, insertion, 1)

                # ======= Step 4: Prepare prompt and truncate =======
                input_str = f"The comment is: '{review_comment}'\nThe code is: '{source_code}'"
                instruction_len = len(tokenizer.encode(instruction, add_special_tokens=False))
                max_input_len = MAX_INSTRCUTION_AND_INPUT_LENGTH - instruction_len

                input_tokens = tokenizer.encode(input_str, add_special_tokens=False)
                output_tokens = tokenizer.encode(target, add_special_tokens=False)

                if len(input_tokens) > max_input_len:
                    input_tokens = input_tokens[:max_input_len]
                    input_str = tokenizer.decode(input_tokens) + " ...'"

                if len(output_tokens) > MAX_OUTPUT_LENGTH:
                    output_tokens = output_tokens[:MAX_OUTPUT_LENGTH]
                    target = tokenizer.decode(output_tokens)

                full_input = instruction + " " + input_str
                test_inputs.append(full_input)
                test_outputs.append(target)

            # ======= Step 5: Save output =======
            output_df = pd.DataFrame({"input": test_inputs, "output": test_outputs})
            output_path = f"./tufano_cr_test_{Type}_{Scenario}.jsonl"
            output_df.to_json(output_path, orient="records", lines=True)
            print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    preprocess_all_code_refinements("./Perturbations.csv")