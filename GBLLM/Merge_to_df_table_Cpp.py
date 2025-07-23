# -*- coding: utf-8 -*-
# print(f":{}")
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

    # ------------------------------------------------------------------------------------------
    if generation_type == 'Code':
        df[f'{column_prefix}_G1__Predict_Fast_code__Ori'] = original_code_list
        df[f'{column_prefix}_G1__Predict_Fast_code']     = postprocessed_code_list
        if 'GPT' in generated_path:
            df[f'{column_prefix}_G1__log_probs']        = log_probs_list
        df[f'{column_prefix}_G1__avg_log_probs']      = avg_log_probs_list

    elif generation_type == 'NL':
        df['Code_Function_Description_G1__Ori']        = original_code_list
        df['Code_Function_Description_G1']             = postprocessed_code_list
        if 'GPT' in generated_path:
            df['Code_Function_Description_G1__log_probs'] = log_probs_list
        if 'GPT' in generated_path or 'Gemini' in generated_path:
            df['Code_Function_Description_G1__avg_log_probs'] = avg_log_probs_list

    else:
        print(f"### Invalid generation_type: {generation_type}. Please check the code!")

    # Save the modified data to a new CSV file
    # os.makedirs(f'{save_csv_path}', exist_ok=True)
    df.to_csv(save_csv_path, index=False, encoding='utf-8')
    print(f"### DataFrame length: {len(df)}")

# #####################################################################################################################🔖💡✅🟨
def process_multiple_entries():
    df = pd.read_csv(dataset_path)

    original_code_matrix     = [[] for _ in range(5)]
    postprocessed_code_matrix = [[] for _ in range(5)]
    log_probs_matrix         = [[] for _ in range(5)]
    avg_log_probs_matrix     = [[] for _ in range(5)]

    for question_index in tqdm(range(len(df))):
        for entry_index in range(5):
            question_index_str = str(question_index).zfill(4)
            with open(f"{generated_path}/{question_index_str}_{entry_index}.txt", 'r', encoding='utf-8') as f:
                code_str = f.read().strip()
            original_code_matrix[entry_index].append(code_str)

            if generation_type == 'Code':
                postprocessed_code_matrix[entry_index].append(process_cot_code(code_str, question_index).strip())
            elif generation_type == 'NL':
                postprocessed_code_matrix[entry_index].append(postprocess_nl(code_str, question_index).strip())
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

    # --------------------------------------------------------------------------------------------
    for entry_index in range(5):
        entry_num = entry_index + 1
        if generation_type == 'Code':
            df[f'{column_prefix}_G5__Predict_Fast_code__Ori_{entry_num}'] = original_code_matrix[entry_index]
            df[f'{column_prefix}_G5__Predict_Fast_code_{entry_num}']     = postprocessed_code_matrix[entry_index]
            if 'GPT' in generated_path:
                df[f'{column_prefix}_G5__log_probs_{entry_num}']     = log_probs_matrix[entry_index]
            if 'GPT' in generated_path or 'Gemini' in generated_path:
                df[f'{column_prefix}_G5__avg_log_probs_{entry_num}'] = avg_log_probs_matrix[entry_index]

        elif generation_type == 'NL':
            df[f'Code_Function_Description_G5__Ori_{entry_num}'] = original_code_matrix[entry_index]
            df[f'Code_Function_Description_G5_{entry_num}']     = postprocessed_code_matrix[entry_index]
            if 'GPT' in generated_path:
                df[f'Code_Function_Description_G5__log_probs_{entry_num}'] = log_probs_matrix[entry_index]
            df[f'Code_Function_Description_G5__avg_log_probs_{entry_num}'] = avg_log_probs_matrix[entry_index]

        else:
            print(f"### Invalid generation_type: {generation_type}. Please check the code!")

    # Save the modified data to a new CSV file
    df.to_csv(save_csv_path, index=False, encoding='utf-8')
    print(f"### DataFrame length: {len(df)}")

# #####################################################################################################################🔖💡✅🟨
def process_cot_code(original_code_str, index):
    code_str = original_code_str.strip()

    # Handle single backtick cases
    if code_str.count("```") == 1:
        if "```cpp" in code_str:
            code_str = code_str.split("```cpp")[1].strip()
        if "```\ncpp" in code_str:
            code_str = code_str.split("```\ncpp")[1].strip()
        if "```" in code_str and not code_str.endswith('```'):
            code_str = code_str.split("```")[1].strip()
        if "``cpp" in code_str:
            code_str = code_str.split("``cpp")[1].split("```")[0].strip()

    # Handle exactly two backticks
    elif code_str.count("```") == 2:
        if "```cpp" in code_str:
            code_str = code_str.split("```cpp")[1].split("```")[0].strip()
        if "```\ncpp" in code_str:
            code_str = code_str.split("```\ncpp")[1].split("```")[0].strip()
        if code_str.startswith('```') and code_str.endswith('```'):
            code_str = code_str[3:-3].strip()
        if "```\n" in code_str:
            code_str = code_str.split("```")[1]
            if code_str and code_str[0] != '\n':
                print(f"\n\n### Error: mismatched backticks, index: {index}\nOriginal code:\n{original_code_str}")
            code_str = code_str.strip()
        if "```" in code_str:
            print(f"\n\n### Error: mismatched backticks, index: {index}\nOriginal code:\n{original_code_str}")
            print(f"\n\n### Processed code:\n{code_str}")

    # Handle more than two backticks
    elif code_str.count("```") > 2:
        if "```cpp" in code_str:
            code_str = code_str.split("```cpp")[1].split("```")[0].strip()
        if "```\ncpp" in code_str:
            code_str = code_str.split("```\ncpp")[1].split("```")[0].strip()
        if "```" in code_str:
            code_str = code_str.split("```")[1]
            if code_str and code_str[0] != '\n':
                print(f"\n\n### Error: multiple backticks, index: {index}\nOriginal code:\n{original_code_str}")
                print(f"\n\n### Processed code:\n{code_str}")
            code_str = code_str.strip()
        if "```" in code_str:
            print(f"\n\n### Error: multiple backticks, index: {index}\nOriginal code:\n{original_code_str}")
            print(f"\n\n### Processed code:\n{code_str}")

    # Fallback for empty or NaN
    if not code_str or code_str.lower() == "nan":
        print(f"\n### Warning: code_str is empty, index: {index}\nOriginal code:\n{original_code_str}")
        code_str = 'pass'

    return code_str.strip()

# #####################################################################################################################🔖💡✅🟨
def postprocess_nl(original_code_str, index):
    code_str = original_code_str.strip()

    # Pre-processing: remove known markers
    markers = [
        '```\nnatural language description of code functions\n```',
        '```\nNatural language description of code functions\n```',
        '```\nCode functionality description\n```',
        '```\ncode functionality description\n```',
    ]
    for marker in markers:
        if marker in code_str:
            code_str = code_str.replace(marker, '').strip()

    # Remove triple backticks
    if code_str.startswith('```\n') and code_str.endswith('\n```'):
        code_str = code_str[3:-3].strip()
    if code_str.startswith('```Code functionality description'):
        code_str = code_str.split('```Code functionality description')[1].strip()
    if code_str.startswith('```'):
        code_str = code_str[3:].strip()
        if code_str and code_str[0] != '\n':
            print(f"\n\n### Error: mismatched backticks, index: {index}\nOriginal code:\n{original_code_str}")
    if code_str.endswith('```'):
        code_str = code_str[:-3].strip()

    # Post-processing: handle remaining descriptors
    if code_str.startswith('Code functionality description'):
        code_str = code_str.split('Code functionality description')[1].strip()
    if code_str.startswith('code functionality description'):
        code_str = code_str.split('code functionality description')[1].strip()

    # Final checks
    if '```' in code_str:
        print(f"\n\n### Error: unmatched backticks, index: {index}\nOriginal code:\n{original_code_str}")
    if not code_str or code_str.lower() == "nan":
        print(f"\n### Warning: description is empty, index: {index}\nOriginal description:\n{original_code_str}")
        code_str = 'Null'

    return code_str.strip()

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
        if any(keyword in line for keyword in risk_keywords_list):
            for keyword in risk_keywords_list:
                if keyword in line:
                    returned_risk_keywords.append(keyword)
        else:
            returned_code_lines.append(line)

    if returned_risk_keywords:
        cleaned_code = '\n'.join(returned_code_lines).strip()
        return True, set(returned_risk_keywords), cleaned_code
    else:
        return False, [], ''

if __name__ == "__main__":
    main()
