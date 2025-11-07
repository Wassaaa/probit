#include "ProbitSolver.hpp"

double ProbitSolver::probit(double p)
{
    // Limits
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    // Rational approximation for lower region.
    if (p < P_LOW) {
        double q = std::sqrt(-2.0 * std::log(p));
        return polynomial(q, C) / polynomial(q, D);
    }
    // Rational approximation for central region.
    if (p < P_HIGH) {
        double q = p - 0.5;
        double r = q * q;
        return q * polynomial(r, A) / polynomial(r, B);
    }
    // Rational approximation for upper region.
    double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -polynomial(q, C) / polynomial(q, D);
}
