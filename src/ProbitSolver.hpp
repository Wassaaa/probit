#pragma once

#include <array>
#include <cmath>
#include <limits>

class ProbitSolver
{
public:
    /**
     * @brief Calculates the Inverse Standard Normal CDF (Probit Function).
     * * Uses Peter J. Acklam's high-precision rational approximation.
     * @param p The probability (must be in the range 0.0 < p < 1.0).
     * @return The corresponding Z-score (double).
     */
    static double probit(double p);

private:
    /**
     * @brief This function implements "Horner's method" for fast
     * and stable polynomial evaluation.
     * @tparam N The number of coefficients in the array
     * @param x The variable
     * @param coeffs The array of coefficients
     * @returns The evaluated polynomial result
     */
    template <size_t N>
    static inline double polynomial(double x, const std::array<double, N> &coeffs)
    {
        double result = 0.0;
        for (double c : coeffs) {
            result = result * x + c;
        }
        return result;
    }

    /**
     * @brief Calculates the Standard Normal PDF (the "bell curve" formula).
     * This is needed for the refinement step.
     */
    static inline double norm_pdf(double x)
    {
        return (1.0 / (std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * x * x);
    }

    static constexpr double P_LOW = 0.02425;
    static constexpr double P_HIGH = 1.0 - P_LOW;

    static constexpr std::array<double, 6> A = {
        -3.969683028665376e+01, //
        2.209460984245205e+02,  //
        -2.759285104469687e+02, //
        1.383577518672690e+02,  //
        -3.066479806614716e+01, //
        2.506628277459239e+00   //
    };
    static constexpr std::array<double, 6> B = {
        -5.447609879822406e+01, //
        1.615858368580409e+02,  //
        -1.556989798598866e+02, //
        6.680131188771972e+01,  //
        -1.328068155288572e+01, //
        1.0                     //
    };
    static constexpr std::array<double, 6> C = {
        -7.784894002430293e-03, //
        -3.223964580411365e-01, //
        -2.400758277161838e+00, //
        -2.549732539343734e+00, //
        4.374664141464968e+00,  //
        2.938163982698783e+00   //
    };
    static constexpr std::array<double, 5> D = {
        7.784695709041462e-03, //
        3.224671290700398e-01, //
        2.445134137142996e+00, //
        3.754408661907416e+00, //
        1.0                    //
    };
};
