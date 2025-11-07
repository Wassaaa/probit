#include "ProbitSolver.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::vector<double> probabilities;
    double p;

    while (std::cin >> p) {
        probabilities.push_back(p);
    }

    std::cout << std::fixed << std::setprecision(16);

    for (double prob : probabilities) {
        if (prob <= 0.0 || prob >= 1.0) {
            std::cerr << "Input " << prob << " is not a valid probability. Skipping.\n";
            std::cout << "nan\n";
            continue;
        }
        std::cout << ProbitSolver::probit(prob) << "\n";
    }

    return 0;
}
