#include "ProbitSolver.hpp"
#include <iostream>

int main()
{
    std::string line;
    while (std::getline(std::cin, line, ' ')) {
        double probability = std::stod(line);
        if (probability <= 0.0 || probability >= 1.0) {
            std::cout << "not a valid probability" << std::endl;
            continue;
        }
        std::cout << ProbitSolver::probit(probability) << std::endl;
    }
    return 0;
}
