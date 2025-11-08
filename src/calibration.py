import subprocess
import time
import numpy as np
from scipy.stats import norm
import multiprocessing
from typing import List


NUM_TESTS = 1_000_000
N_CORES = multiprocessing.cpu_count()

CPP_STREAM_PROGRAM = "../build/probit_stream"
CPP_BATCH_PROGRAM = "../build/probit_batch"
CPP_PARALLEL_PROGRAM = "../build/probit_parallel"
# ---------------------


def generate_test_data(n_tests: int) -> np.ndarray:
    """Generates an array of evenly-spaced test probabilities."""
    print(f"Generating {n_tests} test points...")
    return np.linspace(0.00001, 0.99999, n_tests)


def get_scipy_answers(test_data: np.ndarray) -> np.ndarray:
    """Gets the "true" answers from SciPy."""
    print("Generating SciPy answers...")
    start_time = time.time()
    scipy_answers = norm.ppf(test_data)
    print(f"SciPy generation took: {time.time() - start_time:.4f}s")
    return scipy_answers


def run_stream_test(test_data: np.ndarray):
    """
    Tests the "slow" C++ streamer by calling it
    once for every single number.
    """
    print("\n--- Starting one-by-one process call Test ---")
    print(f"Testing {len(test_data)} numbers, one process at a time...")

    cpp_answers: List[float] = []
    start_cpp = time.time()

    for p in test_data:
        result = subprocess.run(
            [CPP_STREAM_PROGRAM],
            input=str(p),
            capture_output=True,
            text=True,
            check=True,
        )
        cpp_answers.append(float(result.stdout.strip()))

    cpp_time = time.time() - start_cpp
    print(f"C++ (slow stream) took: {cpp_time:.4f}s")
    print(f"Time per calculation: {cpp_time / len(test_data) * 1000:.4f} milliseconds")
    return np.array(cpp_answers)


def run_batch_test(test_data: np.ndarray) -> np.ndarray:
    """
    Tests the "fast" C++ serial batch program.
    It calls it ONCE for all numbers.
    """
    print("\n--- Starting Serial C++ Test ---")
    print(f"Testing {len(test_data)} numbers in a single batch...")

    input_string = "\n".join(map(str, test_data))

    start_cpp = time.time()
    result = subprocess.run(
        [CPP_BATCH_PROGRAM],
        input=input_string,
        capture_output=True,
        text=True,
        check=True,
    )
    cpp_time = time.time() - start_cpp
    print(f"C++ (fast batch) took: {cpp_time:.4f}s")

    output_lines = result.stdout.strip().split("\n")
    return np.array([float(line) for line in output_lines])


def _worker_run_batch(chunk_of_data: np.ndarray) -> List[float]:
    """
    This is the "worker" function that each process will run.
    It runs the SERIAL C++ batch program on its assigned chunk.
    """
    input_string = "\n".join(map(str, chunk_of_data))

    result = subprocess.run(
        [CPP_BATCH_PROGRAM],
        input=input_string,
        capture_output=True,
        text=True,
        check=True,
    )
    output_lines = result.stdout.strip().split("\n")
    return [float(line) for line in output_lines]


def run_python_parallel_orchestrator(test_data: np.ndarray) -> np.ndarray:
    """
    Tests the "Python-as-Orchestrator" model.
    It splits the work into chunks and runs them on
    parallel C++ processes using multiprocessing.Pool.
    """
    print(f"\n--- Starting 'Python Parallel Orchestrator' Test ---")
    print(f"Detected {N_CORES} CPU cores.")
    print(f"Splitting {len(test_data)} numbers into {N_CORES} chunks...")

    chunks = np.array_split(test_data, N_CORES)
    print(f"Each chunk has ~{len(chunks[0])} numbers.")

    start_cpp = time.time()

    with multiprocessing.Pool(processes=N_CORES) as pool:
        results_from_all_chunks = pool.map(_worker_run_batch, chunks)

    cpp_time = time.time() - start_cpp
    print(f"Python Orchestrator took: {cpp_time:.4f}s")

    all_results = [item for sublist in results_from_all_chunks for item in sublist]
    return np.array(all_results)


def run_parallel_test(test_data: np.ndarray) -> np.ndarray:
    """
    Tests the "fastest" C++ parallel program.
    It calls it ONCE for all numbers.
    """
    print("\n--- Starting Parallel C++ Test ---")
    print(f"Testing {len(test_data)} numbers in a single batch (C++ parallel)...")

    input_string = "\n".join(map(str, test_data))

    start_cpp = time.time()
    result = subprocess.run(
        [CPP_PARALLEL_PROGRAM],
        input=input_string,
        capture_output=True,
        text=True,
        check=True,
    )
    cpp_time = time.time() - start_cpp
    print(f"C++ (fastest batch) took: {cpp_time:.4f}s")

    output_lines = result.stdout.strip().split("\n")
    return np.array([float(line) for line in output_lines])


def compare_results(scipy_answers: np.ndarray, cpp_answers: np.ndarray):
    """Calculates and prints the final error between the two results."""
    # Filter out any 'nan' values from our C++ error handling
    valid_indices = ~np.isnan(cpp_answers)

    if len(scipy_answers) != len(cpp_answers):
        print(
            f"Error: Mismatched results! SciPy: {len(scipy_answers)}, C++: {len(cpp_answers)}"
        )
        return

    abs_error = np.abs(scipy_answers[valid_indices] - cpp_answers[valid_indices])
    max_error = np.max(abs_error)

    print("\n--- Test Results ---")
    print(f"Processed {len(cpp_answers)} numbers.")
    print(f"Maximum Error (vs SciPy): {max_error: .2e}")


if __name__ == "__main__":

    test_data = generate_test_data(NUM_TESTS)
    scipy_answers = get_scipy_answers(test_data)

    # cpp_results = run_stream_test(test_data)

    # --- Test B: The "Fast" Serial Batch ---
    cpp_results = run_batch_test(test_data)

    # --- Test C: The "Fastest" Parallel Batch (C++ handles threads) ---
    # cpp_results = run_parallel_test(test_data)

    # --- Test D: The "Python Orchestrator" (Python handles processes) ---
    # cpp_results = run_python_parallel_orchestrator(test_data)

    # 4. Compare the results
    compare_results(scipy_answers, cpp_results)
