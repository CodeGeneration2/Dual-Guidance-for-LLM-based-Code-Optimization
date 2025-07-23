# -*- coding: utf-8 -*-
# print(f":{{}}")
# #####################################################################################################################🔖💡✅🟨

import json
import pandas as pd
import os
from tqdm import tqdm

# #####################################################################################################################🔖💡✅🟨
dataset_path = ""
generated_path = ""
save_csv_path = ""

column_prefix = ""

# #####################################################################################################################🔖💡✅🟨
def main():
    global generation_type
    if 'use_NL' in generated_path:
        generation_type = 'Code'
    elif 'gen_NL' in generated_path:
        generation_type = 'NL'
    elif 'ablation_remove_N_L' in generated_path:
        generation_type = 'Code'
    
    if os.path.exists(f"{generated_path}/0000_4.txt"):
        process_multiple_entries()
    else:
        process_single_entry()

# #####################################################################################################################🔖💡✅🟨
def process_single_entry():
    df = pd.read_csv(dataset_path)

    original_code_list = []
    postprocessed_code_list = []
    log_probs_list = []
    avg_log_probs_list = []

    for question_index in tqdm(range(len(df))):
        question_index_str = str(question_index).zfill(4)
        with open(f"{generated_path}/{question_index_str}_0.txt", 'r', encoding='utf-8') as f:
            code_str = f.read().strip()
        original_code_list.append(code_str)

        if generation_type == 'Code':
            postprocessed_code_list.append(process_cot_code(code_str, question_index).strip())
        elif generation_type == 'NL':
            postprocessed_code_list.append(postprocess_nl(code_str, question_index).strip())

        if 'GPT' in generated_path:
            with open(f"{generated_path}/{question_index_str}_0_log_probs.txt", 'r', encoding='utf-8') as f:
                log_prob = f.read().strip()
            log_probs_list.append(log_prob)
        if 'GPT' in generated_path or 'Gemini' in generated_path:
            with open(f"{generated_path}/{question_index_str}_0_avg_log_probs.txt", 'r', encoding='utf-8') as f:
                avg_log_prob = f.read().strip()
            avg_log_probs_list.append(avg_log_prob)

    # Update DataFrame with new columns
    if generation_type == 'Code':
        df[f'{column_prefix}_G1__Predict_Fast_code__Ori']   = original_code_list
        df[f'{column_prefix}_G1__Predict_Fast_code']       = postprocessed_code_list
        if 'GPT' in generated_path:
            df[f'{column_prefix}_G1__log_probs']            = log_probs_list
        df[f'{column_prefix}_G1__avg_log_probs']           = avg_log_probs_list

    elif generation_type == 'NL':
        df['Code_Function_Description_G1__Ori']            = original_code_list
        df['Code_Function_Description_G1']                 = postprocessed_code_list
        if 'GPT' in generated_path:
            df['Code_Function_Description_G1__log_probs']  = log_probs_list
        if 'GPT' in generated_path or 'Gemini' in generated_path:
            df['Code_Function_Description_G1__avg_log_probs'] = avg_log_probs_list

    else:
        print(f"### Invalid generation_type: {generation_type}. Please check the code!")

    # Save the modified data to a new CSV file
    # os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df.to_csv(save_csv_path, index=False, encoding='utf-8')


# #####################################################################################################################🔖💡✅🟨
def process_multiple_entries():
    df = pd.read_csv(dataset_path)

    original_code_matrix      = [[] for _ in range(5)]
    postprocessed_code_matrix = [[] for _ in range(5)]
    log_probs_matrix          = [[] for _ in range(5)]
    avg_log_probs_matrix      = [[] for _ in range(5)]

    for question_index in tqdm(range(len(df))):
        for entry_index in range(5):
            question_index_str = str(question_index).zfill(4)
            with open(f"{generated_path}/{question_index_str}_{entry_index}.txt", 'r', encoding='utf-8') as f:
                code_str = f.read().strip()
            original_code_matrix[entry_index].append(code_str)

            if generation_type == 'Code' and 'Py' in generated_path:
                postprocessed_code_matrix[entry_index].append(
                    process_cot_code(code_str, question_index, programming_language='python').strip()
                )
            elif generation_type == 'Code' and 'Cpp' in generated_path:
                postprocessed_code_matrix[entry_index].append(
                    process_cot_code(code_str, question_index, programming_language='cpp').strip()
                )
            elif generation_type == 'NL':
                postprocessed_code_matrix[entry_index].append(
                    postprocess_nl(code_str, question_index).strip()
                )
            else:
                print(f"### Invalid generation_type: {generation_type}. Please check the code!")

            if 'GPT' in generated_path:
                with open(f"{generated_path}/{question_index_str}_{entry_index}_log_probs.txt", 'r', encoding='utf-8') as f:
                    log_prob = f.read().strip()
                log_probs_matrix[entry_index].append(log_prob)

            if 'GPT' in generated_path or 'Gemini' in generated_path:
                with open(f"{generated_path}/{question_index_str}_{entry_index}_avg_log_probs.txt", 'r', encoding='utf-8') as f:
                    avg_log_prob = f.read().strip()
                avg_log_probs_matrix[entry_index].append(avg_log_prob)

    # Update DataFrame with new columns for each of the 5 entries
    for entry_index in range(5):
        num = entry_index + 1
        if generation_type == 'Code':
            df[f'{column_prefix}_G5__Predict_Fast_code__Ori_{num}'] = original_code_matrix[entry_index]
            df[f'{column_prefix}_G5__Predict_Fast_code_{num}']     = postprocessed_code_matrix[entry_index]
            if 'GPT' in generated_path:
                df[f'{column_prefix}_G5__log_probs_{num}']          = log_probs_matrix[entry_index]
            if 'GPT' in generated_path or 'Gemini' in generated_path:
                df[f'{column_prefix}_G5__avg_log_probs_{num}']      = avg_log_probs_matrix[entry_index]

        elif generation_type == 'NL':
            df[f'Code_Function_Description_G5__Ori_{num}'] = original_code_matrix[entry_index]
            df[f'Code_Function_Description_G5_{num}']     = postprocessed_code_matrix[entry_index]
            if 'GPT' in generated_path:
                df[f'Code_Function_Description_G5__log_probs_{num}'] = log_probs_matrix[entry_index]
            df[f'Code_Function_Description_G5__avg_log_probs_{num}'] = avg_log_probs_matrix[entry_index]

        else:
            print(f"### Invalid generation_type: {generation_type}. Please check the code!")

    # Save the modified data to a new CSV file
    df.to_csv(save_csv_path, index=False, encoding='utf-8')


# #####################################################################################################################🔖💡✅🟨
def process_cot_code(original_code_str, index, programming_language='cpp'):
    code_str = original_code_str.strip()

    # Handle single backtick blocks
    if code_str.count("```") == 1:
        if f"```{programming_language}" in code_str:
            code_str = code_str.split(f"```{programming_language}")[1].strip()
        if f"```\n{programming_language}" in code_str:
            code_str = code_str.split(f"```\n{programming_language}")[1].strip()
        if "```" in code_str and not code_str.endswith('```'):
            code_str = code_str.split("```")[1].strip()
        if f"``{programming_language}" in code_str:
            code_str = code_str.split(f"``{programming_language}")[1].split("```")[0].strip()

    # Handle exactly two backtick blocks
    elif code_str.count("```") == 2:
        if f"```{programming_language}" in code_str:
            code_str = code_str.split(f"```{programming_language}")[1].split("```")[0].strip()
        if f"```\n{programming_language}" in code_str:
            code_str = code_str.split(f"```\n{programming_language}")[1].split("```")[0].strip()
        if code_str.startswith('```') and code_str.endswith('```'):
            code_str = code_str[3:-3].strip()
        if "```\n" in code_str:
            code_str = code_str.split("```")[1]
            if code_str and code_str[0] != '\n':
                print(f"\n\n### Error: mismatched backticks, index: {index}\nOriginal code:\n{original_code_str}")
            code_str = code_str.strip()
        if "```" in code_str:
            print(f"\n\n### Error: mismatched backticks, index: {index}\nOriginal code:\n{original_code_str}\nProcessed code:\n{code_str}")

    # Handle more than two backtick blocks
    elif code_str.count("```") > 2:
        if f"```{programming_language}" in code_str:
            code_str = code_str.split(f"```{programming_language}")[1].split("```")[0].strip()
        if f"```\n{programming_language}" in code_str:
            code_str = code_str.split(f"```\n{programming_language}")[1].split("```")[0].strip()
        if "```" in code_str:
            code_str = code_str.split("```")[1]
            if code_str and code_str[0] != '\n':
                print(f"\n\n### Error: multiple backticks, index: {index}\nOriginal code:\n{original_code_str}\nProcessed code:\n{code_str}")
            code_str = code_str.strip()
        if "```" in code_str:
            print(f"\n\n### Error: multiple backticks, index: {index}\nOriginal code:\n{original_code_str}\nProcessed code:\n{code_str}")

    # Fallback for empty or NaN
    if not code_str or code_str.lower() == "nan":
        print(f"\n### Warning: code_str is empty, index: {index}\nOriginal code:\n{original_code_str}")
        code_str = 'pass'

    return code_str.strip()


# #####################################################################################################################🔖💡✅🟨
def postprocess_nl(original_text, index):
    text = original_text.strip()

    # Pre-process known markers
    markers = [
        '```\nnatural language description of code functions\n```',
        '```\nNatural language description of code functions\n```',
        '```\nCode functionality description\n```',
        '```\ncode functionality description\n```',
    ]
    for marker in markers:
        if marker in text:
            text = text.replace(marker, '').strip()

    # Remove triple backticks
    if text.startswith('```\n') and text.endswith('\n```'):
        text = text[3:-3].strip()
    if text.startswith('```Code functionality description'):
        text = text.split('```Code functionality description')[1].strip()
    if text.startswith('```'):
        text = text[3:].strip()
        if text and text[0] != '\n':
            print(f"\n\n### Error: mismatched backticks, index: {index}\nOriginal text:\n{original_text}")
    if text.endswith('```'):
        text = text[:-3].strip()

    # Post-process remaining descriptors
    if text.startswith('Code functionality description'):
        text = text.split('Code functionality description')[1].strip()
    if text.startswith('code functionality description'):
        text = text.split('code functionality description')[1].strip()

    # Final validation
    if '```' in text:
        print(f"\n\n### Error: unmatched backticks, index: {index}\nOriginal text:\n{original_text}")
    if not text or text.lower() == "nan":
        print(f"\n### Warning: description is empty, index: {index}\nOriginal text:\n{original_text}")
        text = 'Null'

    return text.strip()


# #####################################################################################################################🔖💡✅🟨
def postprocess_codellama_code(original_code_str, index):
    code_str = original_code_str.strip()
    if len(code_str.split('```')) > 1:
        code_str = code_str.split('```')[1]
    code_str = (
        code_str
        .split('### Optimized version')[-1]
        .split('# Test')[0]
        .split('# test')[0]
        .split('### Code Functionality Description')[0]
        .split('### Control Flow Graph')[0]
    )
    if not code_str or code_str.lower() == "nan":
        print(f"\n### Warning: code_str is empty, index: {index}\nOriginal code:\n{original_code_str}")
        code_str = 'pass'
    return code_str


# #####################################################################################################################🔖💡✅🟨
def check_code(code_to_check):
    secondary_risk_keywords = [
        'setrecursionlimit(', 'stack_size(', "if __name__ == '__main__':", ' is not ', ' is ',
        'stdout', 'stderr', ", file=sys.stdout", ", file=stdout", ", output=sys.stdout", ", output=stdout",
        ", file=sys.stderr", ", file=stderr", ", output=sys.stderr", ", output=stderr",
        "sys.stdin.readline", "stdin.readline", "sys.stdin.buffer.readline", "stdin.buffer.readline",
        "sys.stdout.write", "stdout.write", "sys.__stdout__.write", "__stdout__.write",
        "sys.stderr.write", "stderr.write", "sys.__stderr__.write", "__stderr__.write",
        'return sys.stdout.flush()', "IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(",
        ".close(", "stdout", "stderr", 'open(', "threading", "thread", "multiprocessing", "asyncio",
        "queue.Queue(", "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("
    ]
    risk_keywords_list = secondary_risk_keywords.copy()

    returned_code_lines = []
    returned_risk_keywords = []

    code_lines = code_to_check.split('\n')
    for line in code_lines:
        found = False
        for keyword in risk_keywords_list:
            if keyword in line:
                returned_risk_keywords.append(keyword)
                found = True
        if not found:
            returned_code_lines.append(line)

    if returned_risk_keywords:
        cleaned_code = '\n'.join(returned_code_lines).strip()
        return True, set(returned_risk_keywords), cleaned_code
    else:
        return False, [], ''


if __name__ == "__main__":
    main()

