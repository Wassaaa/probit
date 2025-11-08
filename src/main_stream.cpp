#include "ProbitSolver.hpp"
#include <iostream>
#include <iomanip>

int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    double probability;
    std::cout << std::fixed << std::setprecision(16);

    std::string line;
    while (std::cin >> probability) {
        if (probability <= 0.0 || probability >= 1.0) {
            std::cerr << "Input " << probability << " is not a valid probability. Skipping.\n";
            std::cout << "nan\n";
            continue;
        }
        std::cout << ProbitSolver::probit(probability) << std::endl;
    }
    return 0;
}
