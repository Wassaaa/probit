#include "InverseCumulativeNormal.h"
#include <iostream>
#include <array>

int main() {
    // --- Scalar usage ---
    quant::InverseCumulativeNormal icn; // mean=0, sigma=1
    double xs[] = {1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1-1e-6, 1-1e-12};
    for (double x : xs) {
        double z = icn(x); // z = Φ^{-1}(x)
        std::cout << "scalar  x=" << x << "  z=" << z << "\n";
    }

    // --- Vector/array usage (multiple values at once) ---
    const double xin[] = {0.0001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999};
    double zout[std::size(xin)];
    icn(xin, zout, std::size(xin)); // out[i] = Φ^{-1}(xin[i])

    for (std::size_t i = 0; i < std::size(xin); ++i) {
        std::cout << "vector  x=" << xin[i] << "  z=" << zout[i] << "\n";
    }

    return 0;
}

