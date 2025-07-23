# -*- coding: utf-8 -*-
# Utilities for batch-evaluating code predictions with IO unit tests
import sys
import re
import os
from tqdm import tqdm
import io
import json
import keyword
import pandas as pd
from API__sandbox import create_subprocess_and_run_python_io_test

import gc
import statistics
import pprint

# -----------------------------------------------------------------------------
# Toggle debugging mode
DEBUG = False

# IO test scopes: 'public' or 'private'
TEST_IO_SCOPES = ['public', 'private']

# Number of initial runs to ignore for warm-up
IGNORE_RUNS = 1
# Number of runs to measure after ignoring warm-up runs
MEASURE_RUNS = 30

# Directory containing dataset CSV files
DATASET_DIR = r"code_datasets"
# Directory to save evaluated results
OUTPUT_DIR = r"io_evaluated_datasets"

# Programming language of submitted code ('python' or 'cpp')
LANGUAGE = 'python'

# List of DataFrame columns containing code predictions to test
TEST_COLUMNS = [
    # add column names here, e.g. 'predicted_code'
]


def main():
    """Entry point: iterate over all datasets."""
    iterate_multiple_datasets()


def iterate_multiple_datasets():
    """Process each CSV file in the dataset directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_files = os.listdir(DATASET_DIR)
    for filename in dataset_files:
        df = pd.read_csv(os.path.join(DATASET_DIR, filename))

        # PIE datasets
        if 'PIE_' in filename:
            if 'public' in TEST_IO_SCOPES:
                io_type = 'Public_IO_unit_tests'
                process_dataframe(filename, df, io_type)
            if 'private' in TEST_IO_SCOPES:
                io_type = 'Private_IO_unit_tests'
                process_dataframe(filename, df, io_type)

        # DB datasets (deduplicated)
        elif 'DB_' in filename:
            if 'public' in TEST_IO_SCOPES:
                io_type = 'Public_IO_unit_tests_Dedup'
                process_dataframe(filename, df, io_type)
            if 'private' in TEST_IO_SCOPES:
                io_type = 'Private_IO_unit_tests_Dedup'
                process_dataframe(filename, df, io_type)


def process_dataframe(file_name: str, df: pd.DataFrame, io_type: str) -> pd.DataFrame:
    """Evaluate each specified column and save the updated DataFrame."""
    for col in TEST_COLUMNS:
        df = process_column(df, prediction_column=col, save_column=col, io_type=io_type)

    # Generate new file name with incremented PIE ID
    if 'PIE_Cpp_' in file_name:
        pie_id = file_name.split('PIE_Cpp_')[-1].split('_')[0]
    elif 'DB_Py_' in file_name:
        pie_id = file_name.split('DB_Py_')[-1].split('_')[0]
    elif 'PIE_' in file_name:
        pie_id = file_name.split('PIE_')[-1].split('_')[0]
    else:
        pie_id = '000'
    new_pie_id = str(int(pie_id) + 1).zfill(3)
    base_name = file_name.split('__')[0].replace(pie_id, new_pie_id)

    # Determine output file suffix based on IO type
    if 'Public_IO_unit_tests' in io_type:
        new_name = f"{base_name}__intermediate_public_io_complete.csv"
    elif 'Private_IO_unit_tests' in io_type:
        new_name = f"{base_name}__hidden_io_complete.csv"
    elif 'Generate_IO' in io_type or 'Gen_IO' in io_type:
        new_name = f"{base_name}__intermediate_generated_io_complete.csv"
    else:
        new_name = f"{base_name}__io_complete.csv"

    df.to_csv(os.path.join(OUTPUT_DIR, new_name), index=False)
    return df


def process_column(df: pd.DataFrame, prediction_column: str, save_column: str, io_type: str) -> pd.DataFrame:
    """Run IO tests on a single column of code predictions."""
    overall_pass_rate, pass_rate_list, time_list, result_list = \
        iterate_over_column(df, prediction_column=prediction_column, io_type=io_type)
    print(f"### Column: {prediction_column}  IO type: {io_type}  Overall pass rate: {overall_pass_rate}%")

    # Add results as new DataFrame columns
    if 'Public_IO_unit_tests' in io_type:
        df[f'{save_column}__Public_IO_pass_rate_(%)'] = pass_rate_list
        df[f'{save_column}__Public_time(ms)'] = time_list
        df[f'{save_column}__Public_results'] = result_list
    elif 'Private_IO_unit_tests' in io_type:
        df[f'{save_column}__IO_pass_rate_(%)'] = pass_rate_list
        df[f'{save_column}__time(ms)'] = time_list
        df[f'{save_column}__results'] = result_list

    # Save intermediate results per column
    temp_name = f'IO_temp_{prediction_column}.csv'
    df.to_csv(os.path.join(OUTPUT_DIR, temp_name), index=False)
    return df


def iterate_over_column(df: pd.DataFrame, prediction_column: str, io_type: str):
    """Evaluate IO tests for each code entry in the specified column."""
    code_list = df[prediction_column].astype(str).tolist()
    io_tests_list = df[io_type].tolist()

    pass_rate_list = []
    time_list = []
    result_list = []

    for idx in tqdm(range(len(code_list)), desc=f"Evaluating {prediction_column}"):
        code_str = code_list[idx]
        
        # Load IO test dictionary
        io_dict = eval(io_tests_list[idx])
        if not io_dict.get('inputs'):
            # No IO tests provided
            pass_rate_list.append(0.0)
            time_list.append(1234567890)
            result_list.append({
                'pass_rates': [],
                'time_lists': [],
                'time_std': 1234567890,
                'time_unit': 'milliseconds_ms = e-3 seconds',
                'error_types': ['no_io'],
                'return_code': 0,
                'stderr': ''
            })
            continue

        # Execute IO tests via sandbox runner
        io_dict_copy = eval(io_tests_list[idx])
        pass_rate, exec_time, result_dict = evaluate_io(
            code_str, io_dict_copy, IGNORE_RUNS, MEASURE_RUNS, pie_index=idx)

        pass_rate_list.append(pass_rate)
        time_list.append(exec_time)
        result_list.append(result_dict)

    # Compute overall pass rate
    if 'Generate_IO' in io_type or 'Gen_IO' in io_type:
        print(f"### Pass rates set: {set(pass_rate_list)}")
        overall_pass_rate = round(statistics.mean(pass_rate_list) * 100, 2)
    else:
        overall_pass_rate = round(
            len([v for v in pass_rate_list if v > 1]) / len(pass_rate_list) * 100, 2)

    return overall_pass_rate, pass_rate_list, time_list, result_list


def check_code_compliance(code_str: str):
    """Check code for disallowed operations; return offending keyword or True if compliant."""
    risk_keywords = [
        'setrecursionlimit(', 'stack_size(', "if __name__ == '__main__':",
        'stdout', 'stderr', 'open(', 'threading', 'subprocess.run('
        # add more patterns as needed
    ]
    for kw in risk_keywords:
        if kw in code_str and 'open(0' not in code_str:
            return kw
    return True


def evaluate_io(code_str: str, io_dict: dict, ignore_runs: int,
                measure_runs: int, pie_index: int = 0):
    """Run IO tests by writing a temp JSON, invoking the sandbox, and processing results."""
    temp_json = 'temp_io_test.json'
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(io_dict, f, ensure_ascii=False, indent=4)

    result_dict = create_subprocess_and_run_python_io_test(
        code_str=code_str.strip(),
        io_file=temp_json,
        ignore_runs=ignore_runs,
        runs_after_ignore=measure_runs,
        language=LANGUAGE,
        pie_index=pie_index,
    )
    os.remove(temp_json)

    if DEBUG:
        print(f"### Result dict: {result_dict}")
        pprint.pprint(result_dict)
        return 0.0, 1234567890, result_dict

    # Map pass rates and time lists from result dict, accommodating both English and original keys
    raw_pass_rates = result_dict.get('pass_rates', result_dict.get('IO', []))
    raw_time_lists = result_dict.get('time_list', result_dict.get('time', []))

    pass_rate = statistics.mean(raw_pass_rates) if raw_pass_rates else 0.0
    if pass_rate == 0:
        return pass_rate, 1234567890, result_dict
    if pass_rate == 1:
        medians = [statistics.median(lst) for lst in raw_time_lists]
        avg_time = round(statistics.mean(medians), 2)
        return pass_rate, avg_time, result_dict
    # Partial success: consider only fully passing runs
    successful_rates = [r for r in raw_pass_rates if r == 1]
    if not successful_rates:
        return 0.0, 1234567890, result_dict
    new_rate = statistics.mean(successful_rates)
    filtered_times = [raw_time_lists[i] for i, r in enumerate(raw_pass_rates) if r == 1]
    medians = [statistics.median(lst) for lst in filtered_times]
    avg_time = round(statistics.mean(medians), 2)
    return new_rate, avg_time, result_dict


if __name__ == '__main__':
    main()
