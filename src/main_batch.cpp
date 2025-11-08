#include "InverseCumulativeNormal.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // Read all numbers into a vector
    std::vector<double> probabilities;
    double p;
    while (std::cin >> p) {
        probabilities.push_back(p);
    }

    quant::InverseCumulativeNormal icn; // mean=0, sigma=1

    // prep the results array for the timed loop
    std::vector<double> results;
    results.reserve(probabilities.size());

    // time and math
    auto start = std::chrono::high_resolution_clock::now();
    for (double x : probabilities) {
        results.push_back(icn(x));
    }
    auto end = std::chrono::high_resolution_clock::now();

    // print out the result
    std::cout << std::fixed << std::setprecision(16);
    for (double z : results) {
        std::cout << z << "\n";
    }
    // print time to stderr for performance checks
    auto total_duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cerr << "time_ns:" << total_duration_ns << std::endl;

    return 0;
}
