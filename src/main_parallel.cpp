#include "ProbitSolver.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <future>

/**
 * @brief A simple worker function that processes a "chunk" of the vector.
 * * @param data A pointer to the start of the data chunk.
 * @param count The number of items in this chunk.
 * @return A vector of doubles containing the results.
 */
std::vector<double> process_chunk(const double *data, size_t count)
{
    std::vector<double> results;
    results.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        double p = data[i];
        if (p <= 0.0 || p >= 1.0) {
            results.push_back(std::nan("")); // Use nan for invalid input
        }
        else {
            results.push_back(ProbitSolver::probit(p));
        }
    }
    return results;
}

int main()
{
    // --- 1. THE "MAGIC" SPEEDUP LINES ---
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // --- 2. Read all numbers into a vector ---
    std::vector<double> probabilities;
    double p;
    while (std::cin >> p) {
        probabilities.push_back(p);
    }

    if (probabilities.empty()) {
        return 0;
    }

    // --- 3. Parallel Processing Setup ---
    // Get the number of available CPU cores
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) {
        n_threads = 2; // Default to 2 if detection fails
    }

    size_t total_size = probabilities.size();
    size_t chunk_size = (total_size + n_threads - 1) / n_threads; // Ceiling division

    // A vector to hold our "futures" (the results from each thread)
    std::vector<std::future<std::vector<double>>> futures;

    // --- 4. Launch all threads ---
    // We loop 'n_threads' times, launching one async task for each.
    for (size_t i = 0; i < n_threads; ++i) {
        size_t start = i * chunk_size;
        if (start >= total_size) {
            break; // No more data left to process
        }
        size_t count = std::min(chunk_size, total_size - start);

        // std::async runs our 'process_chunk' function on a new thread
        futures.push_back(
            std::async(std::launch::async, process_chunk, &probabilities[start], count));
    }

    // --- 5. Collect results and print ---
    std::cout << std::fixed << std::setprecision(16);

    // This loop waits for each thread to finish
    // and then prints its results in order.
    for (auto &fut : futures) {
        std::vector<double> chunk_results = fut.get(); // Wait for thread
        for (double res : chunk_results) {
            if (std::isnan(res)) {
                std::cout << "nan\n";
            }
            else {
                std::cout << res << "\n";
            }
        }
    }

    return 0;
}
