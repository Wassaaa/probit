#ifdef USE_BASE_HEADER
# include "InverseCumulativeNormalBase.h"
#else
# include "InverseCumulativeNormal.h"
#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <atomic>

#ifdef _OPENMP
# include <omp.h>
#endif

int main(int argc, char *argv[]) {
    quant::InverseCumulativeNormal icn; // mean=0, sigma=1

    // Generate test data - focus on central region for SIMD fast path
    size_t N = 10'000'000;
    if (argc > 1) {
        N = std::stoull(argv[1]);
    }
    std::cout << "Generating " << N << " test values...\n";

    std::vector<double> probabilities(N);
    std::vector<double> results(N);

    // Use fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.1, 0.9); // Central region

    // Thread-local counter for unique seeds
    static std::atomic<int> thread_counter{0};
#pragma omp parallel
    {
        int tid = thread_counter.fetch_add(1);
        std::mt19937_64 rng(12345 + tid);

#pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            probabilities[i] = dist(rng);
        }
    }

    std::cout << "Running inverse CDF computation...\n";

    // Warmup
    icn(probabilities.data(), results.data(), 1000);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    icn(probabilities.data(), results.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nResults:\n";
    std::cout << "  Elements:     " << N << "\n";
    std::cout << "  Time:         " << elapsed << " seconds\n";
    std::cout << "  Throughput:   " << (N / elapsed / 1e6) << " M/sec\n";
    std::cout << "  Per element:  " << (elapsed / N * 1e9) << " ns\n";

    // Show a few sample results
    std::cout << "\nSample results:\n";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "  x=" << probabilities[i] << "  z=" << results[i] << "\n";
    }

    return 0;
}
