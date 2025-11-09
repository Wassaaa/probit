import subprocess
import numpy as np
from scipy.stats import norm

NUM_TESTS = 1_000_000

PROGRAMS = [
    ("./build/probit_base", "Bisection baseline"),
    ("./build/probit_scalar_single", "Acklam + refinement"),
    ("./build/probit_simd_single", "Acklam + SIMD + refinement"),
    ("./build/probit_scalar_omp", "Acklam + OpenMP + refinement"),
    ("./build/probit", "Acklam + SIMD + OpenMP + refinement"),
    ("./build/probit_simd_omp", "Acklam + SIMD + OpenMP (no refinement)"),
]


def generate_test_data(n: int) -> np.ndarray:
    print(f"Generating {n:,} test points...")
    return np.linspace(1e-12, 1 - 1e-12, n)


def get_scipy_reference(data: np.ndarray) -> np.ndarray:
    print("Computing SciPy reference values...\n")
    return norm.ppf(data)


def run_program(path: str, data: np.ndarray) -> tuple[np.ndarray, float]:
    input_str = "\n".join(map(str, data))
    result = subprocess.run(
        [path], input=input_str, capture_output=True, text=True, check=True
    )

    outputs = np.array([float(line) for line in result.stdout.strip().split("\n")])

    time_ns = 0
    for line in result.stderr.strip().split("\n"):
        if line.startswith("time_ns:"):
            time_ns = int(line.split(":", 1)[1])
            break

    return outputs, time_ns / 1e9


def compute_error(reference: np.ndarray, results: np.ndarray) -> dict:
    valid = np.isfinite(results)
    if not valid.any():
        return {"max": float("inf"), "mean": float("inf"), "valid_count": 0}

    errors = np.abs(reference[valid] - results[valid])
    return {
        "max": np.max(errors),
        "mean": np.mean(errors),
        "valid_count": np.sum(valid),
    }


if __name__ == "__main__":
    test_data = generate_test_data(NUM_TESTS)
    reference = get_scipy_reference(test_data)

    print("=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(
        f"{'Program':<45} {'Time (s)':>10} {'Throughput':>12} {'Speedup':>12} {'Max Error':>12}"
    )
    print("-" * 100)

    baseline_time = None
    for path, description in PROGRAMS:
        try:
            results, elapsed = run_program(path, test_data)
            error = compute_error(reference, results)

            throughput = NUM_TESTS / elapsed / 1e6
            if baseline_time is None:
                baseline_time = elapsed
                speedup_str = "(baseline)"
            else:
                speedup = baseline_time / elapsed
                speedup_str = f"({speedup:.1f}x)"

            print(
                f"{description:<45} {elapsed:>10.4f} "
                f"{throughput:>8.1f} M/s {speedup_str:>12} {error['max']:>12.2e}"
            )

        except Exception as e:
            print(f"{description:<45} FAILED: {e}")

    print("=" * 100)
