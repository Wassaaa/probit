#ifdef USE_BASE_HEADER
# include "InverseCumulativeNormalBase.h"
#else
# include "InverseCumulativeNormal.h"
#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::vector<double> probabilities;
    double p;
    while (std::cin >> p) {
        probabilities.push_back(p);
    }

    quant::InverseCumulativeNormal icn;
    std::vector<double> results(probabilities.size());

    auto start = std::chrono::high_resolution_clock::now();
    icn(probabilities.data(), results.data(), probabilities.size());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(16);
    for (double z : results) {
        std::cout << z << "\n";
    }

    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cerr << "time_ns:" << duration_ns << std::endl;

    return 0;
}
