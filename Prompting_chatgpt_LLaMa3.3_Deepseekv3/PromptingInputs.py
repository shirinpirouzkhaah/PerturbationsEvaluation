import pandas as pd
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def build_prompt_org(df, index):
    source_code = df.iloc[index]['source_code']
    review_comment = str(df.iloc[index]['review_comment']).strip()

    prompt = f"""
    
    The Review Comment is specifically attached to the code between <START> and <END> tags in the method.

    Your task is to generate **10 complete and revised versions of the entire Java method**, not just the modified lines.

    Each revision must represent the **full method after applying the review comment**.

    IMPORTANT:
    - DO NOT provide any explanation or justification.
    - DO NOT repeat or restate the Review Comment.
    - DO NOT include <START>, <END>, or the comment itself in your output.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    
    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}

Review Comment:
{review_comment}
"""
    return prompt




def build_prompt_org_NoReviewComment(df, index):
    source_code = df.iloc[index]['source_code']

    prompt = f"""

    Even though no specific review comment is provided, you should generate 10 meaningful and different revisions based on what could be improved or modified in the marked section.

    Each revision must include the **entire revised method**, not just the modified lines.

    IMPORTANT:
    - DO NOT include any code comments in your output.
    - DO NOT provide any explanation or justification.
    - DO NOT include <START> or <END> in your output.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    
    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}
"""
    return prompt


def build_prompt_ThankYou(df, index):
    
    source_code = df.iloc[index]['source_code']
    review_comment = df.iloc[index]['review_comment']
    review_comment = str(review_comment).strip()

    polite_review_comment = f"Please, {review_comment.strip()}. Thank you."

    

    prompt = f"""
    
    The Review Comment is specifically attached to the code between <START> and <END> tags in the method.

    Your task is to generate **10 complete and revised versions of the entire Java method**, not just the modified lines.

    Each revision must represent the **full method after applying the review comment**.

    IMPORTANT:
    - DO NOT provide any explanation or justification.
    - DO NOT repeat or restate the Review Comment.
    - DO NOT include <START>, <END>, or the comment itself in your output.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    
    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}

Review Comment:
{polite_review_comment}
"""
    return prompt


def build_prompt_inline_comment_after_END(df, index):
    source_code = df.iloc[index]['source_code']
    review_comment = str(df.iloc[index]['review_comment']).strip()

    # Insert the review comment as a Java comment after the <END> tag or at the end of the method
    end_tag = "<END>"
    inline_comment = f"{end_tag} // {review_comment}"

    if end_tag in source_code:
        source_code_with_comment = source_code.replace(end_tag, inline_comment, 1)
    else:
        source_code_with_comment = source_code.strip() + f"\n// {review_comment}"

    prompt = f"""

    A Review Comment has been inserted directly into the code using a Java `//` comment. 
    The comment is placed either:
    - Immediately after the <END> tag, or
    - At the end of the method if the <END> tag is missing.

    The comment applies to the part of the code that would normally be marked between <START> and <END>.

    You must generate **10 different complete revisions of the entire Java method** based only on the inserted comment.

    IMPORTANT:
    - DO NOT retain the inserted comment or any <START>/<END> tags in your output.
    - DO NOT include any explanation or justification.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.

    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code_with_comment}
"""
    return prompt






def build_prompt_org_mitigation(df, index):
    source_code = df.iloc[index]['source_code']
    extracted_code = extract_tagged_code(source_code)
    comment = str(df.iloc[index]['review_comment']).strip()
    review_comment = build_mitigation_string(comment, extracted_code)

    prompt = f"""

    The Review Comment applies specifically to the code segment between the <START> and <END> tags in the method.

    You must generate **10 different complete revisions of the entire Java method**, using only the information in the Review Comment.

    IMPORTANT:
    - DO NOT include any explanation or justification.
    - DO NOT repeat or rephrase the Review Comment.
    - DO NOT include <START>, <END>, or the comment itself in your output.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.

    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}

Review Comment:
{review_comment}
"""
    return prompt


def build_prompt_org_chain_of_thought(df, index):
    source_code = df.iloc[index]['source_code']
    review_comment = str(df.iloc[index]['review_comment']).strip()

    prompt = f"""

    The Review Comment applies specifically to the code section between <START> and <END>.

    Before generating code, Start with a short **step-by-step reasoning** (max 2 sentences) about how the Review Comment relates to the tagged code segment.

    Then generate **10 different complete revisions of the entire Java method** that address the Review Comment and follow any additional constraint given.


    IMPORTANT:
    - Only make changes that directly address the Review Comment.
    - DO NOT modify unrelated parts of the code.
    - DO NOT include <START>, <END>, or the Review Comment itself in your output.
    - DO NOT repeat or rephrase the Review Comment.
    - DO NOT include any explanation after the reasoning section.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    

    FORMAT:
<Your step-by-step reasoning here>

Then:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}

Review Comment:
{review_comment}
"""
    return prompt





cf_type_descriptions = {
    "IfElseCF": "",
    "ExceptionCF": "Do not remove any code, even if it appears unused, unless it is explicitly requested in the Review Comment.",
    "DeadCodeCF": "Do not remove any code, even if it appears unused, unless it is explicitly requested in the Review Comment.",
    "TryNcatchCF": "Do not remove the 'try-catch' structure.",
    "DataFlowCF": "",
    "EqualAssertCF": "Do not remove the if condition used before equality assertions.",
    "NullAssertCF": "Do not remove the if condition used before null assertions.",
    "TrueFalseAssertCF": "Do not remove the if condition used before boolean (true/false) assertions.",
    "defUseBreakCF": "Do not remove any variable declarations or initializations."
}


def build_prompt_CF(df, index, CF_Type):
    source_code_column = CF_Type + 'Source'
    review_comment_column = CF_Type + 'Comment'

    source_code = df.iloc[index][source_code_column]
    review_comment = str(df.iloc[index][review_comment_column]).strip()

    if pd.isna(source_code) or pd.isna(review_comment):
        return "Empty"

    constraint = cf_type_descriptions.get(CF_Type, "").strip()
    constraint_line = f"Very important: {constraint}" if constraint else ""

    prompt = f"""

    The Review Comment applies specifically to the section between <START> and <END> in the method.

    You must generate **10 different complete revisions of the entire Java method** that address the Review Comment and follow any additional constraint given.

    

    IMPORTANT:
    - Only make changes that directly address the Review Comment.
    - DO NOT modify unrelated parts of the code.
    - DO NOT include <START>, <END>, or the Review Comment itself in your output.
    - DO NOT include any explanation or justification.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    
    {constraint_line}

    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}

Review Comment:
{review_comment}
"""
    return prompt





def build_prompt_CF_mitigation(df, index, CF_Type):
    source_code_column = CF_Type + 'Source'
    review_comment_column = CF_Type + 'Comment'

    source_code = df.iloc[index][source_code_column]
    extracted_code = extract_tagged_code(source_code)
    comment = str(df.iloc[index][review_comment_column]).strip()
    review_comment = build_mitigation_string(comment, extracted_code)

    if pd.isna(source_code) or pd.isna(review_comment):
        return "Empty"

    constraint = cf_type_descriptions.get(CF_Type, "").strip()
    constraint_line = f"Very important: {constraint}" if constraint else ""

    prompt = f"""

    The Review Comment applies specifically to the section between <START> and <END> in the method.

    You must generate **10 different complete revisions of the entire Java method** that address the Review Comment and follow any additional constraint given.

    IMPORTANT:
    - Only make changes that directly address the Review Comment.
    - DO NOT modify unrelated parts of the code.
    - DO NOT include <START>, <END>, or the Review Comment itself in your output.
    - DO NOT include any explanation or justification.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    
    {constraint_line}

    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}

Review Comment:
{review_comment}
"""
    return prompt


def build_prompt_CF_inline_comment_after_END(df, index, CF_Type):
    source_code_column = CF_Type + 'Source'
    review_comment_column = CF_Type + 'Comment'

    source_code = df.iloc[index][source_code_column]
    review_comment = str(df.iloc[index][review_comment_column]).strip()

    if pd.isna(source_code) or pd.isna(review_comment):
        return "Empty"

    constraint = cf_type_descriptions.get(CF_Type, "").strip()
    constraint_line = f"Very important: {constraint}" if constraint else ""

    end_tag = "<END>"
    insertion = f"{end_tag} // {review_comment}"

    if end_tag in source_code:
        source_code_with_comment = source_code.replace(end_tag, insertion, 1)
    else:
        source_code_with_comment = source_code.strip() + f"\n// {review_comment}"

    prompt = f"""

    The Review Comment has been inserted as a Java `//` comment:
    - Immediately after the <END> tag if it exists, OR
    - Appended to the end of the method if the <END> tag is missing.

    The comment refers specifically to the code between <START> and <END>.

    You must generate **10 different complete revisions of the entire Java method** that address the comment and follow any constraint provided.


    IMPORTANT:
    - Only make changes that directly address the Review Comment.
    - DO NOT retain the inserted comment or any <START>/<END> tags in your output.
    - DO NOT include any explanation or justification.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    
    {constraint_line}

    FORMAT:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code_with_comment}
"""
    return prompt

    


def build_prompt_CF_chain_of_thought(df, index, CF_Type):
    source_code_column = CF_Type + 'Source'
    review_comment_column = CF_Type + 'Comment'

    source_code = df.iloc[index][source_code_column]
    review_comment = str(df.iloc[index][review_comment_column]).strip()

    if pd.isna(source_code) or pd.isna(review_comment):
        return "Empty"

    constraint = cf_type_descriptions.get(CF_Type, "").strip()
    constraint_line = f"Very important: {constraint}" if constraint else ""

    prompt = f"""

    The Review Comment applies specifically to the code section between <START> and <END>.

    Before generating code, Start with a short **step-by-step reasoning** (max 2 sentences) about how the Review Comment relates to the tagged code segment.

    Then generate **10 different complete revisions of the entire Java method** that address the Review Comment and follow any additional constraint given.

    

    IMPORTANT:
    - Only make changes that directly address the Review Comment.
    - DO NOT modify unrelated parts of the code.
    - DO NOT include <START>, <END>, or the Review Comment itself in your output.
    - DO NOT repeat or rephrase the Review Comment.
    - DO NOT include any explanation after the reasoning section.
    - JUST OUTPUT CODE in the exact format described below.
    - Each code revision must start with '/CodeRevision/:' followed by the complete revised method.
    
    {constraint_line}

    FORMAT:
<Your step-by-step reasoning here>

Then:
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>
...
/CodeRevision/:
<REVISED FULL JAVA METHOD CODE>

Java Code:
{source_code}

Review Comment:
{review_comment}
"""
    return prompt






def extract_tagged_code(source_code):
    pattern = r"<START>(.*?)<END>"
    match = re.search(pattern, source_code, re.DOTALL)
    
    if match:
        return match.group(1).strip() 
    
    
def build_mitigation_string(review_comment, extracted_code):
    combined_string = f"For this part of the Java Code : {extracted_code}, this review comment is provided: {review_comment}."
    return combined_string



def clean_java_code(java_code):
    # Remove all new lines and leading/trailing whitespace
    java_code = java_code.strip()
    # Replace multiple spaces with a single space
    java_code = re.sub(r'\s+', ' ', java_code)
    #Remove spaces around braces, parentheses, and semicolons for compactness
    java_code = re.sub(r'\s*([\{\};()])\s*', r'\1', java_code)
    return java_code


def compare_by_words(str1, str2):
    
    str1 = str1.lower()
    str2 = str2.lower()
   
    words1 = re.findall(r'[a-zA-Z]+', str1)
    words2 = re.findall(r'[a-zA-Z]+', str2)
    
    return words1 == words2



def extract_code_revisions_from_response(response_text):
    # Find all blocks that follow the /CodeRevision/: pattern
    pattern = r"/CodeRevision/:\s*(.*?)(?=(?:/CodeRevision/:|$))"
    matches = re.findall(pattern, response_text, re.DOTALL)

    cleaned_revisions = []
    
    for code in matches:
        code = code.strip()
        # Remove leading/trailing angle brackets if present
        if code.startswith("<") and code.endswith(">"):
            code = code[1:-1].strip()
        cleaned_revisions.append(code)

    return cleaned_revisions



def best_bleu_score(revisions, reference):
    smoothie = SmoothingFunction().method4
    reference_tokens = reference.split()
    return max(
        sentence_bleu([reference_tokens], rev.split(), smoothing_function=smoothie)
        for rev in revisions
    )






