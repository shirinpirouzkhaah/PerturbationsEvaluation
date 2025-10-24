import pandas as pd
import numpy as np
from together import Together 
from PromptingInputs import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
import replicate



from service import instruct_model 
client1 = Together(api_key="")
Model = "deepseek-ai/DeepSeek-V3"
model = "DeepSeek-V3"

from service import instruct_model 
client1 = Together(api_key="")
Model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
model = "Llama3.3_70B"


from serviceOpenAI import instruct_model
GPT_TOKEN = ""
client1 = OpenAI(api_key=GPT_TOKEN)
Model = "gpt-3.5-turbo"
model = "gpt-3.5-turbo"

# ---------- Phase 1: Main Prompts ----------


import pandas as pd
import numpy as np
from together import Together 
from PromptingInputs import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
import replicate

from service import instruct_model 
client1 = Together(api_key="")
Model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
model = "Llama3.3_70B"



df = pd.read_csv("Perturbations.csv")
prompt_labels = ["org", "no_comment", "ThankYou", "mitigation", "inline_comment", "CoT"]

df_A_records = pd.DataFrame(index=df.index)
for label in prompt_labels:
    df_A_records[f"{model}_{label}_response"] = None
    df_A_records[f"{model}_{label}_EXM"] = False

OrgCode_prompt_match_counts = [0] * len(prompt_labels)


for index, row in df.iterrows():
    # if index < 1872:
    #     continue

    
    print(f"Processing index: {index}")
    prompts = [
        build_prompt_org(df, index),
        build_prompt_org_NoReviewComment(df, index),
        build_prompt_ThankYou(df, index),
        build_prompt_org_mitigation(df, index),
        build_prompt_inline_comment_after_END(df, index),
        build_prompt_org_chain_of_thought(df, index)
    ]
    
    responses = instruct_model(client1, prompts,  Model, 4096)
    if not responses:
        continue
    
    match_found = False
    for i, label in enumerate(prompt_labels):
        response_i = responses[i]
        
        if response_i is not None and response_i[0] is not None:
            output = response_i[0]
            revisions = extract_code_revisions_from_response(output)
            df_A_records.at[index, f"{model}_{label}_response"] = revisions
            df_A_records.at[index, f"{model}_{label}_EXM"] = False
    
            if revisions:
                for rev in revisions:
                    if compare_by_words(clean_java_code(rev), clean_java_code(row["target"])):
                        df_A_records.at[index, f"{model}_{label}_EXM"] = True
                        OrgCode_prompt_match_counts[i] += 1
                        if label == "org":
                            match_found = True
                        break


df_A_records.to_csv(f"OrgSourceCode_{model}.csv", index=False)


org_col = f"Llama3.3_70B_org_response"
count_entries = df_A_records[org_col].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
print(f"Count of entries evaluated: {count_entries}")
print("\n--- Prompt match counts ---")
for i, label in enumerate(prompt_labels):
    print(f"{label}: {OrgCode_prompt_match_counts[i]}")


# ---------- Phase 2: CF Prompts ----------

import pandas as pd
import numpy as np
from together import Together 
from PromptingInputs import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
import replicate


CF_Type = "RandomNamesCF"

df_A_records = pd.read_csv(f"OrgSourceCode_{model}.csv")
df = pd.read_csv("perfectPredictionsUpdated.csv")
df_B_records = pd.DataFrame(index=df.index)


# Output columns for CF, mitigation, inline, and CoT
df_B_records[f"{model}_{CF_Type}_response"] = None
df_B_records[f"{model}_{CF_Type}_mitigation_response"] = None
df_B_records[f"{model}_{CF_Type}_inline_response"] = None
df_B_records[f"{model}_{CF_Type}_CoT_response"] = None  # NEW

df_B_records[f"{model}_{CF_Type}_EXM"] = False
df_B_records[f"{model}_{CF_Type}_mitigation_EXM"] = False
df_B_records[f"{model}_{CF_Type}_inline_EXM"] = False
df_B_records[f"{model}_{CF_Type}_CoT_EXM"] = False  # NEW

df_B_records[f"{model}_{CF_Type}_retrieval"] = False
df_B_records[f"{model}_{CF_Type}_mitigation_retrieval"] = False
df_B_records[f"{model}_{CF_Type}_inline_retrieval"] = False
df_B_records[f"{model}_{CF_Type}_CoT_retrieval"] = False  # NEW

df_B_records[f"{model}_{CF_Type}_bleu"] = 0.0
df_B_records[f"{model}_{CF_Type}_mitigation_bleu"] = 0.0
df_B_records[f"{model}_{CF_Type}_inline_bleu"] = 0.0
df_B_records[f"{model}_{CF_Type}_CoT_bleu"] = 0.0  # NEW



count_CF_entries = CF_EXM = CF_retrieval = CF_mitigation_EXM = CF_mitigation_retrieval = CF_inline_EXM = CF_inline_retrieval = 0
CF_CoT_EXM = CF_CoT_retrieval = 0

bleu_avg_base = []
bleu_avg_mitigation = []
bleu_avg_inline = []
bleu_avg_CoT = []


for index, row in df.iterrows():

    if not df_A_records.at[index, f"{model}_org_EXM"]:
        continue

    print(f"\nRunning CF prompts for index: {index}")
    CF_prompt = build_prompt_CF(df, index, CF_Type)
    if CF_prompt == "Empty":
        continue

    CF_mitigation_prompt = build_prompt_CF_mitigation(df, index, CF_Type)
    CF_inline_prompt = build_prompt_CF_inline_comment_after_END(df, index, CF_Type)
    CF_COT_prompt = build_prompt_CF_chain_of_thought(df, index, CF_Type)
    
    CFprompts = [CF_prompt, CF_mitigation_prompt, CF_inline_prompt, CF_COT_prompt]
    count_CF_entries += 1

    cf_responses = instruct_model(client1, CFprompts, Model, 4096)
    if not cf_responses or len(cf_responses) < 4:
        continue
    
    

    # --- CF base response
    
    if cf_responses[0] is not None and cf_responses[0][0] is not None:
    
        CF_revisions = extract_code_revisions_from_response(cf_responses[0][0])
        df_B_records.at[index, f"{model}_{CF_Type}_response"] = CF_revisions
        bleu_computed = False
        if CF_revisions:
            for rev in CF_revisions:
                if compare_by_words(clean_java_code(rev), clean_java_code(row[f"{CF_Type}Target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_EXM"] = True
                    CF_EXM += 1
                    bleu_computed = True  # skip BLEU calculation later
                    break
        
                elif compare_by_words(clean_java_code(rev), clean_java_code(row["target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_retrieval"] = True
                    CF_retrieval += 1
                    bleu_computed = True  # skip BLEU calculation later
                    break
        
            # Compute BLEU only if no EXM or retrieval was found
            if not bleu_computed:
                ref = clean_java_code(row[f"{CF_Type}Target"])
                cleaned_revs = [clean_java_code(r) for r in CF_revisions]
                bleu = best_bleu_score(cleaned_revs, ref)
                df_B_records.at[index, f"{model}_{CF_Type}_bleu"] = bleu
                bleu_avg_base.append(bleu)




    # --- Mitigation response
    if cf_responses[1] is not None and cf_responses[1][0] is not None:
        mitigation_revisions = extract_code_revisions_from_response(cf_responses[1][0])
        df_B_records.at[index, f"{model}_{CF_Type}_mitigation_response"] = mitigation_revisions
        bleu_computed = False
        if mitigation_revisions:
            for rev in mitigation_revisions:
                if compare_by_words(clean_java_code(rev), clean_java_code(row[f"{CF_Type}Target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_mitigation_EXM"] = True
                    CF_mitigation_EXM += 1
                    bleu_computed = True
                    break
                elif compare_by_words(clean_java_code(rev), clean_java_code(row["target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_mitigation_retrieval"] = True
                    CF_mitigation_retrieval += 1
                    bleu_computed = True
                    break
            if not bleu_computed:
                ref = clean_java_code(row[f"{CF_Type}Target"])
                cleaned_revs = [clean_java_code(r) for r in mitigation_revisions]
                bleu = best_bleu_score(cleaned_revs, ref)
                df_B_records.at[index, f"{model}_{CF_Type}_mitigation_bleu"] = bleu
                bleu_avg_mitigation.append(bleu)
            

    # --- Inline comment response
    if cf_responses[2] is not None and cf_responses[2][0] is not None:
        inline_revisions = extract_code_revisions_from_response(cf_responses[2][0])
        df_B_records.at[index, f"{model}_{CF_Type}_inline_response"] = inline_revisions
        bleu_computed = False
        if inline_revisions:
            for rev in inline_revisions:
                if compare_by_words(clean_java_code(rev), clean_java_code(row[f"{CF_Type}Target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_inline_EXM"] = True
                    CF_inline_EXM += 1
                    bleu_computed = True
                    break
                elif compare_by_words(clean_java_code(rev), clean_java_code(row["target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_inline_retrieval"] = True
                    CF_inline_retrieval += 1
                    bleu_computed = True
                    break
                
            if not bleu_computed:
                ref = clean_java_code(row[f"{CF_Type}Target"])
                cleaned_revs = [clean_java_code(r) for r in inline_revisions]
                bleu = best_bleu_score(cleaned_revs, ref)
                df_B_records.at[index, f"{model}_{CF_Type}_inline_bleu"] = bleu
                bleu_avg_inline.append(bleu)
            
    
    # --- COT response
    if cf_responses[3] is not None and cf_responses[3][0] is not None:
        CoT_revisions = extract_code_revisions_from_response(cf_responses[3][0])
        df_B_records.at[index, f"{model}_{CF_Type}_CoT_response"] = CoT_revisions
        bleu_computed = False
        if CoT_revisions:
            for rev in CoT_revisions:
                if compare_by_words(clean_java_code(rev), clean_java_code(row[f"{CF_Type}Target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_CoT_EXM"] = True
                    CF_CoT_EXM += 1
                    bleu_computed = True
                    break
                elif compare_by_words(clean_java_code(rev), clean_java_code(row["target"])):
                    df_B_records.at[index, f"{model}_{CF_Type}_CoT_retrieval"] = True
                    CF_CoT_retrieval += 1
                    bleu_computed = True
                    break
                
            if not bleu_computed:
                ref = clean_java_code(row[f"{CF_Type}Target"])
                cleaned_revs = [clean_java_code(r) for r in CoT_revisions]
                bleu = best_bleu_score(cleaned_revs, ref)
                df_B_records.at[index, f"{model}_{CF_Type}_CoT_bleu"] = bleu
                bleu_avg_CoT.append(bleu)
            
            
        

# ---------- Final Summary ----------

print("\n--- CF Summary ---")
print(f"Count of CF entries evaluated: {count_CF_entries}")
print(f"Total CF_EXM matches: {CF_EXM}")
print(f"Total CF_retrieval: {CF_retrieval}")
print(f"Total CF mitigation EXM matches: {CF_mitigation_EXM}")
print(f"Total CF mitigation retrieval: {CF_mitigation_retrieval}")
print(f"Total CF inline EXM matches: {CF_inline_EXM}")
print(f"Total CF inline retrieval: {CF_inline_retrieval}")
print(f"Total CF CoT EXM matches: {CF_CoT_EXM}")
print(f"Total CF CoT retrieval: {CF_CoT_retrieval}")

print("\n--- Average BLEU Scores (EXM & retrieval excluded) ---")
print(f"Base BLEU avg: {np.mean(bleu_avg_base) if bleu_avg_base else 0.0:.4f}")
print(f"Mitigation BLEU avg: {np.mean(bleu_avg_mitigation) if bleu_avg_mitigation else 0.0:.4f}")
print(f"Inline BLEU avg: {np.mean(bleu_avg_inline) if bleu_avg_inline else 0.0:.4f}")
print(f"CoT BLEU avg: {np.mean(bleu_avg_CoT) if bleu_avg_CoT else 0.0:.4f}")


df_B_records.to_csv(f"{model}_{CF_Type}.csv", index=False)
