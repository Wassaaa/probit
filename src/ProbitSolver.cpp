#include "ProbitSolver.hpp"

double ProbitSolver::probit(double p)
{
    // Limits
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    double guess;
    // Rational approximation for lower region.
    if (p < P_LOW) {
        double q = std::sqrt(-2.0 * std::log(p));
        guess = polynomial(q, C) / polynomial(q, D);
    }
    // Rational approximation for central region.
    else if (p <= P_HIGH) {
        double q = p - 0.5;
        double r = q * q;
        guess = q * polynomial(r, A) / polynomial(r, B);
    }
    // Rational approximation for upper region.
    else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        guess = -polynomial(q, C) / polynomial(q, D);
    }

    // Refinement
    // get the actual cdf from our guess
    double cdf = 0.5 * std::erfc(-guess / sqrt(2.0));
    // check how much the error was compared to input
    double error = cdf - p;
    // calculate the final correction (Newton's Method)
    guess -= error / norm_pdf(guess);
    return guess;
}
