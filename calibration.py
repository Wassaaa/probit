import subprocess
import time
import numpy as np
from scipy.stats import norm
from typing import Tuple


NUM_TESTS = 1_000_000

CPP_BATCH_PROGRAM = "./build/probit_batch"
CPP_BASE_PROGRAM = "./build/probit_base"
# ---------------------


def generate_test_data(n_tests: int) -> np.ndarray:
    """Generates an array of evenly-spaced test probabilities."""
    print(f"Generating {n_tests} test points...")
    return np.linspace(0.00000001, 0.99999999, n_tests)


def get_scipy_answers(test_data: np.ndarray) -> np.ndarray:
    """Gets the "true" answers from SciPy."""
    print("Generating SciPy answers...")
    start_time = time.time()
    scipy_answers = norm.ppf(test_data)
    print(f"SciPy generation took: {time.time() - start_time:.4f}s")
    return scipy_answers


def _parse_time_from_stderr(stderr_output: str) -> int:
    """
    Finds the 'time_ns:...' line in stderr and returns the time in nanoseconds.
    Returns 0 if the line isn't found.
    """
    for line in stderr_output.strip().split("\n"):
        if line.startswith("time_ns:"):
            try:
                time_str = line.split(":", 1)[1]
                return int(time_str)
            except (IndexError, ValueError):
                print(f"Warning: Found malformed time line in stderr: {line}")
                return 0
    if stderr_output:
        print(
            f"Warning: No 'time_ns:' line found in stderr. Got this instead:\n{stderr_output}"
        )
    return 0


def run_cpp_test(program_path: str, test_data: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Runs a C++ program, feeds it data, and captures its
    stdout (results) and stderr (timing).
    """
    print(f"\n--- Testing C++ Program: {program_path} ---")
    print(f"Testing {len(test_data)} numbers in a single batch...")

    input_string = "\n".join(map(str, test_data))

    result = subprocess.run(
        [program_path],
        input=input_string,
        capture_output=True,
        text=True,
        check=True,
    )

    output_lines = result.stdout.strip().split("\n")
    cpp_answers = np.array([float(line) for line in output_lines])
    cpp_time_ns = _parse_time_from_stderr(result.stderr)
    return cpp_answers, cpp_time_ns


def compare_results(scipy_answers: np.ndarray, cpp_answers: np.ndarray):
    """Calculates and prints the final error between the two results."""
    # Filter out both NaN and infinity values
    valid_indices = np.isfinite(cpp_answers)

    num_invalid = len(cpp_answers) - np.sum(valid_indices)

    if len(scipy_answers) != len(cpp_answers):
        print(
            f"Error: Mismatched results! SciPy: {len(scipy_answers)}, C++: {len(cpp_answers)}"
        )
        return

    if np.sum(valid_indices) == 0:
        print(f"\nNo valid (finite) values to compare!")
        return

    abs_error = np.abs(scipy_answers[valid_indices] - cpp_answers[valid_indices])
    max_error = np.max(abs_error)

    print(f"\nProcessed {len(cpp_answers)} numbers ({num_invalid} invalid/infinite).")
    print(f"Maximum Error (vs SciPy): {max_error: .2e}")


if __name__ == "__main__":

    test_data = generate_test_data(NUM_TESTS)
    scipy_answers = get_scipy_answers(test_data)

    cpp_results, cpp_time_ns = run_cpp_test(CPP_BATCH_PROGRAM, test_data)
    print("\n--- Performance Report ---")
    if cpp_time_ns > 0:
        print(f"C++ ${CPP_BATCH_PROGRAM} core math took: {cpp_time_ns / 1.0e9:.4f}s")
    else:
        print("C++ ${CPP_BATCH_PROGRAM} did not report a valid time.")
    compare_results(scipy_answers, cpp_results)

    # cpp_results, cpp_time_ns = run_cpp_test(CPP_BASE_PROGRAM, test_data)
    # print("\n--- Performance Report ---")
    # if cpp_time_ns > 0:
    #     print(f"C++ ${CPP_BASE_PROGRAM} core math took: {cpp_time_ns / 1.0e9:.4f}s")
    # else:
    #     print("C++ ${CPP_BASE_PROGRAM} did not report a valid time.")

    # compare_results(scipy_answers, cpp_results)
