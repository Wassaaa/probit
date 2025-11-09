#include "InverseCumulativeNormal.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>

int main() {
    quant::InverseCumulativeNormal icn; // mean=0, sigma=1

    // Generate test data - focus on central region for SIMD fast path
    constexpr size_t N = 10'000'000;
    std::cout << "Generating " << N << " test values...\n";

    std::vector<double> probabilities(N);
    std::vector<double> results(N);

    // Use fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.1, 0.9); // Central region

#pragma omp parallel
    {
        // Each thread gets its own RNG with different seed
        int tid = omp_get_thread_num();
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
