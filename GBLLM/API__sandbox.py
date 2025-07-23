# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨

global_timeout = 1

# utilities and code that handles actual execution of programs
import pathlib
import shlex
import subprocess
from typing import Dict, List, Tuple, Union
import time
import numpy as np
import resource
import logging
import psutil
import os
import sys
import traceback
import pdb
import glob

import argparse
import json
import statistics
import io

# disable logging from psutil
logging.getLogger("psutil").setLevel(logging.WARNING)

# disable logging from resource
logging.getLogger("resource").setLevel(logging.WARNING)

# disable logging from subprocess
logging.getLogger("subprocess").setLevel(logging.WARNING)

logging.basicConfig(level=logging.CRITICAL)

DEBUG = True

# #####################################################################################################################🔖💡✅🟨
def create_subprocess_and_run_python_io_test(code_str: str, io_file: str, ignore_runs: int = 0, runs_after_ignore: int = 1, language: str = 'python', pie_index: str = '0') -> Dict:
    """Run a new subprocess to execute Python code for IO unit tests.
    io_file: path to dictionary file with inputs and outputs.
    """
    cmd_args = ["python", "API__PIE_sandbox_latest4.py", code_str, io_file, str(ignore_runs), str(runs_after_ignore), language, pie_index]
    process_result = subprocess.run(
        cmd_args,
        start_new_session=True,
        capture_output=True,
        text=True
    )
    return_code = process_result.returncode
    stderr = process_result.stderr

    # parse stdout to dictionary
    io_test_result = json.loads(process_result.stdout.strip())
    io_test_result["return_code"] = return_code
    io_test_result["stderr"] = stderr

    return io_test_result

# #####################################################################################################################
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tests from command line.")
    parser.add_argument("code_str")
    parser.add_argument("io_file")
    parser.add_argument("ignore_runs", type=int)
    parser.add_argument("runs_after_ignore", type=int)
    parser.add_argument("language")
    parser.add_argument("pie_index")
    return parser.parse_args()

# #####################################################################################################################
def main():
    args = get_args()

    # determine temporary code filename
    if args.language == "python":
        temp_code_filename = "temp_PY_Code.py"
    elif args.language == "cpp":
        temp_code_filename = "temp_Cpp_Code.cpp"
    else:
        temp_code_filename = "temp_code"

    # write code to temporary file
    with open(temp_code_filename, "w") as f:
        f.write(args.code_str.strip())

    # read IO file
    if isinstance(args.io_file, str):
        with open(args.io_file, 'r', encoding='UTF-8') as f:
            io_test_dict = eval(f.read())
    else:
        io_test_dict = args.io_file

    expected_outputs = io_test_dict['outputs']
    num_tests = len(expected_outputs)
    assert num_tests > 0, f"No IO test cases found in {args.io_file}"

    # run code IO tests
    io_test_result = run_code_io_tests(
        code_path=temp_code_filename,
        io_dict=io_test_dict,
        ignore_runs=args.ignore_runs,
        runs_after_ignore=args.runs_after_ignore,
        timeout=global_timeout,
        io_expected_outputs=expected_outputs,
        language=args.language,
        pie_index=args.pie_index,
    )

    print(json.dumps(io_test_result))
    os.remove(temp_code_filename)

# #####################################################################################################################🔖💡✅🟨
def run_code_io_tests(
    code_path: str,
    io_dict: Dict,
    ignore_runs: int,
    runs_after_ignore: int,
    timeout: int = global_timeout,
    io_expected_outputs: List[str] = None,
    num_tests: int = None,
    cpu_core: int = 1,
    language: str = "python",
    pie_index: str = "0",
) -> Dict:
    """
    Run the given code on specified IO test inputs and return a dictionary with:
    - pass_rates: List of float
    - time_list: List of List of float (milliseconds)
    - time_std: float
    - time_unit: str
    - error_types: List of str
    - code_outputs: List of str
    """
    if num_tests is None and io_expected_outputs is not None:
        num_tests = len(io_expected_outputs)

    executable_path = None
    if language == "cpp":
        try:
            executable_path = compile_cpp_code(code_path, cflags="--std=c++17 -O3", pie_index=pie_index)
        except Exception as e:
            return {
                "pass_rates": [0],
                "time_list": [[1234567890]],
                "time_std": 0,
                "time_unit": "milliseconds_ms = e-3 seconds",
                "error_types": ["compilation_error", str(e)],
                "code_outputs": [],
            }

    overall_pass_list = []
    overall_time_list = []
    overall_error_list = []
    overall_outputs = []

    for test_index in range(num_tests):
        # build command
        if is_linux() and language == "python":
            cmd_str = f"taskset --cpu-list {cpu_core} {language} {code_path}"
        elif is_linux() and language == "cpp":
            cmd_str = f"taskset --cpu-list {cpu_core} {executable_path}"
        else:
            cmd_str = f"{language} {code_path}"
        cmd_args = shlex.split(cmd_str)

        input_text = io_dict['inputs'][test_index]
        test_pass_list = []
        test_time_list = []

        for run_idx in range(ignore_runs + runs_after_ignore):
            try:
                output, error, duration = run_cmd_and_get_time(
                    cmd_args,
                    input_text=input_text,
                    timeout=timeout,
                )

                if output is None:
                    test_pass_list.append(0)
                    test_time_list.append(1234567890)
                    overall_error_list.append("timeout")
                    overall_outputs.append("timeout_output")
                    break

                if run_idx >= ignore_runs:
                    test_time_list.append(duration * 1000)
                    if io_expected_outputs is not None and run_idx == ignore_runs:
                        pass_rate = compare_output_with_truth(output, io_expected_outputs[test_index])
                        test_pass_list.append(pass_rate)
                        overall_outputs.append(output)

            except Exception as e:
                test_pass_list.append(0)
                test_time_list.append(1234567890)
                overall_error_list.append(str(e))
                overall_outputs.append("error_output")
                break

        overall_pass_list.append(statistics.mean(test_pass_list))
        overall_time_list.append(test_time_list)

    assert len(overall_pass_list) == len(overall_time_list), f"Pass list length {len(overall_pass_list)} does not match time list length {len(overall_time_list)}"

    # clean up executable if created
    if is_linux() and executable_path and os.path.exists(executable_path):
        os.remove(executable_path)

    if (statistics.mean(overall_pass_list) < 0.9999) and not overall_error_list:
        overall_error_list.append("incomplete_IO")

    rounded_time_list = [[round(time_ms, 2) for time_ms in times] for times in overall_time_list]

    return {
        "pass_rates": overall_pass_list,
        "time_list": rounded_time_list,
        "time_std": round(statistics.pstdev([t for times in overall_time_list for t in times]), 4),
        "time_unit": "milliseconds_ms = e-3 seconds",
        "error_types": overall_error_list,
        "code_outputs": overall_outputs,
    }

# #####################################################################################################################🔖💡✅🟨
def is_linux() -> bool:
    """Check if the current platform is Linux."""
    from sys import platform
    return platform.startswith("linux")

# #####################################################################################################################🔖💡✅🟨
# Maximum virtual memory for subprocesses (in bytes).
MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50  # 500 MB

def limit_virtual_memory():
    """Limit virtual memory for subprocesses."""
    if is_linux():
        resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY * 10))

# #####################################################################################################################🔖💡✅🟨
def run_cmd_and_get_time(cmd_args: List[str], input_text: str, timeout: int = global_timeout) -> Union[Tuple[str, str, float], Tuple[None, str, None]]:
    """Run a command with given input and capture its output, error, and execution time."""
    def _kill(proc_pid):
        process = psutil.Process(proc_pid)
        for child in process.children(recursive=True):
            child.kill()
        process.kill()

    try:
        proc = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=limit_virtual_memory,
        )
        start_time = time.time()
        output, error = proc.communicate(input=input_text.encode('utf-8'), timeout=timeout)
        duration = time.time() - start_time
        return output.decode("utf-8").strip(), error.decode("utf-8").strip(), duration
    except subprocess.TimeoutExpired:
        _kill(proc.pid)  # type: ignore
        return None, "subprocess.TimeoutExpired", None

# ####################################################################################################################################################
def compare_output_with_truth(test_output: str, truth_output: str) -> float:
    """
    Compare the code output with the ground truth and return accuracy as a float between 0 and 1.
    """
    correct_count = 0
    test_lines = test_output.strip().splitlines()
    truth_lines = truth_output.strip().splitlines()
    for line_test, line_truth in zip(test_lines, truth_lines):
        match = line_test == line_truth
        if not match:
            match = line_test.strip() == line_truth.strip()
        if not match:
            try:
                num_test = float(line_test)
                num_truth = float(line_truth)
                match = abs(num_test - num_truth) < 1e-3
            except:
                pass
        if not match:
            try:
                match = line_test.lower() == line_truth.lower()
            except:
                pass
        correct_count += int(match)
    if correct_count == 0:
        return 0.0
    return correct_count / len(truth_lines)

# #####################################################################################################################🔖💡✅🟨
def write_io_test_inputs(inputs: List[str] = ["10000", "1000000"]) -> Tuple[List[str], str]:
    """Write test inputs to temporary files and return their expected outputs and directory name."""
    temp_dir = create_temp_dir()
    for i, input_val in enumerate(inputs):
        file_path = f"{temp_dir}/input.{i}.txt"
        with open(file_path, "w") as f:
            print(f"Wrote input #{i} to {file_path}")
            f.write(input_val)
    outputs = [str(sum(range(int(i) + 1))) for i in inputs]
    return outputs, temp_dir

# #####################################################################################################################🔖💡✅🟨
def create_temp_dir() -> str:
    """Create a unique temporary directory and return its path."""
    import uuid
    temp_dir = f"/tmp/{uuid.uuid4()}"
    pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)
    return temp_dir

# #####################################################################################################################🔖💡✅🟨
def compile_cpp_code(code_path: str, executable_path: str = None, cflags: str = "", pie_index: str = "0") -> str:
    """Compile C++ code and return the path to the executable."""
    if executable_path is None:
        executable_path = os.path.join(os.path.dirname(code_path), f"./Cpp_{pie_index}.out")
    cmd = ["/usr/bin/g++-9", code_path, "-o", executable_path] + shlex.split(cflags.replace('"', "").replace("'", ""))
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise Exception(f"Error compiling code: {code_path} with command: {' '.join(cmd)}, return code: {p.returncode}, stderr: {p.stderr.decode('utf-8')}")
    return executable_path

# #####################################################################################################################🔖💡✅🟨
def run_c_code_io_tests(
    code_path: str,
    unit_test_data_basepath: str,
    ignore_runs: int,
    runs_after_ignore: int,
    timeout: int,
    io_expected_outputs: List[str] = None,
    num_tests: int = None,
    cpu_core: int = 1,
    return_per_trial_times: bool = False,
    python_bin: str = "python",
    return_dict: bool = False,
    remove_code_after_run: bool = True,
    debug_stream = sys.stderr,
    cflags: str = "--std=c++17 -O1",
    return_if_acc_below: float = 0.0,
) -> Union[Tuple[float, float, float], Dict]:
    """
    Run C++ code on IO unit tests defined by input/output files in a directory.
    Returns either (avg_time, std_time, avg_acc) or a dictionary with those metrics.
    """
    try:
        executable_path = compile_cpp_code(code_path, cflags=cflags)
    except Exception as e:
        logging.warning(f"Error: {e}")
        return (float('nan'), float('nan'), 0.0)

    if num_tests is None and io_expected_outputs is not None:
        num_tests = len(io_expected_outputs)

    all_times: List[float] = []
    all_pass_rates: List[float] = []

    for test_index in range(num_tests):
        if is_linux():
            cmd = f"taskset --cpu-list {cpu_core} {executable_path}"
        else:
            cmd = executable_path
        subprocess_args = shlex.split(cmd)
        input_file_path = f"{unit_test_data_basepath}/input.{test_index}.txt"

        for run_idx in range(ignore_runs + runs_after_ignore):
            try:
                start = time.time()
                output, error, _ = run_cmd_and_get_time(subprocess_args, input_text=input_file_path, timeout=timeout)
                elapsed = time.time() - start
                if output is None:
                    if remove_code_after_run:
                        os.remove(executable_path)
                    return (float('nan'), float('nan'), 0.0)

                if run_idx >= ignore_runs:
                    all_times.append(elapsed * 1000)
                    if io_expected_outputs is not None:
                        pass_rate = compare_output_with_truth(output, io_expected_outputs[test_index])
                        if pass_rate < return_if_acc_below:
                            if remove_code_after_run:
                                os.remove(executable_path)
                            logging.info(f"Accuracy {pass_rate} below threshold {return_if_acc_below}. Exiting.")
                            return (float('nan'), float('nan'), 0.0)
                        all_pass_rates.append(pass_rate)

            except Exception as e:
                logging.warning("Error", e)
                return (float('nan'), float('nan'), 0.0)

    # clean up
    if remove_code_after_run and os.path.exists(executable_path):
        os.remove(executable_path)

    avg_time = float(np.mean(all_times))
    std_time = float(np.std(all_times))
    avg_acc = float(np.mean(all_pass_rates))

    if return_dict:
        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "avg_acc": avg_acc,
        }
    else:
        return avg_time, std_time, avg_acc

# #####################################################################################################################🔖💡✅🟨
def test_cpp_cases():
    import shutil
    from pprint import pprint
    slow_code = "src/codenet_eval/cpp_examples/slow_num.cpp"
    fast_code = "src/codenet_eval/cpp_examples/fast_num.cpp"
    wrong_code = "src/codenet_eval/cpp_examples/fast_but_wrong.cpp"
    test_cases = {
        "slow": slow_code,
        "fast": fast_code,
        "fast_but_wrong": wrong_code
    }
    ground_truths, temp_dir = write_io_test_inputs()
    results: Dict[str, Dict] = {key: {} for key in test_cases}

    for case_name, code_path in test_cases.items():
        result = run_c_code_io_tests(
            code_path=code_path,
            unit_test_data_basepath=temp_dir,
            runs_after_ignore=10,
            ignore_runs=2,
            timeout=10,
            io_expected_outputs=ground_truths,
            cpu_core=2,
            return_dict=True
        )
        results[case_name].update(result)  # type: ignore

    assert results["slow"]["avg_time"] > results["fast"]["avg_time"]
    assert results["fast"]["avg_acc"] == 1.0
    assert results["slow"]["avg_acc"] == 1.0
    assert results["fast_but_wrong"]["avg_acc"] == 0.0

    shutil.rmtree(temp_dir)
    print("Test passed! Results:")
    pprint(results)

# #####################################################################################################################🔖💡✅🟨
def run_all_tests():
    test_cpp_cases()

# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()
