#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <omp.h>

#if defined(__AVX__) && !defined(ICN_USE_SIMD) && !defined(ICN_DISABLE_SIMD)
# define ICN_USE_SIMD
#endif

#ifdef ICN_USE_SIMD
# include <immintrin.h>
#endif

namespace quant {

class InverseCumulativeNormal {
  public:
    explicit InverseCumulativeNormal(double average = 0.0, double sigma = 1.0)
        : average_(average), sigma_(sigma) {}

    // Scalar call: return average + sigma * Φ^{-1}(x)
    inline double operator()(double x) const {
        return average_ + sigma_ * standard_value(x);
    }

    // Vector overload: out[i] = average + sigma * Φ^{-1}(in[i]) for i in [0, n)
    inline void operator()(const double *in, double *out, std::size_t n) const {
#ifdef ICN_USE_SIMD
        process_vector_simd(in, out, n);
#else
        process_vector_scalar(in, out, n);
#endif
    }

    // Standardized value: inverse CDF with average=0, sigma=1.
    static inline double standard_value(double x) {
        // Handle edge and invalid cases defensively.
        if (x <= 0.0) return -std::numeric_limits<double>::infinity();
        if (x >= 1.0) return std::numeric_limits<double>::infinity();

        // Piecewise structure left in place so you can drop in rational approximations.
        if (x < x_low_ || x > x_high_) {
            double z = tail_value_rational(x);
#ifdef ICN_ENABLE_HALLEY_REFINEMENT
            z = halley_refine(z, x);
#endif
            return z;
        } else {
            double z = central_value_rational(x);
#ifdef ICN_ENABLE_HALLEY_REFINEMENT
            z = halley_refine(z, x);
#endif
            return z;
        }
    }

  private:
#ifdef ICN_USE_SIMD
    // SIMD vector processing (AVX + FMA)
    inline void process_vector_simd(const double *in, double *out, std::size_t n) const {

        __m256d vec_avg = _mm256_set1_pd(average_);
        __m256d vec_sigma = _mm256_set1_pd(sigma_);
        __m256d vec_low = _mm256_set1_pd(x_low_);
        __m256d vec_high = _mm256_set1_pd(x_high_);

        std::size_t n_chunks = n / 4;
# pragma omp parallel for
        for (std::size_t chunk = 0; chunk < n_chunks; ++chunk) {
            std::size_t i = chunk * 4;
            __m256d vec_x = _mm256_loadu_pd(&in[i]);

            // Check if all 4 values are in central region
            __m256d is_central_mask = _mm256_and_pd(_mm256_cmp_pd(vec_x, vec_low, _CMP_GE_OQ),
                                                    _mm256_cmp_pd(vec_x, vec_high, _CMP_LE_OQ));

            int mask_bits = _mm256_movemask_pd(is_central_mask);

            // Calculate standardized z values (SIMD for all, then fix non-central)
            alignas(32) double z_values[4];
            __m256d vec_z = standard_value_simd_central(vec_x);
            _mm256_store_pd(z_values, vec_z);

            // Fix non-central values and apply Halley refinement to all
            for (int j = 0; j < 4; j++) {
                if (!(mask_bits & (1 << j))) {
                    // Non-central: use scalar path with refinement
                    z_values[j] = standard_value(in[i + j]);
                } else {
# ifdef ICN_ENABLE_HALLEY_REFINEMENT
                    // Central: apply Halley refinement to SIMD result
                    z_values[j] = halley_refine(z_values[j], in[i + j]);
# endif
                }
            }

            // Load refined z values and apply transform
            __m256d vec_z_refined = _mm256_load_pd(z_values);
            __m256d vec_result = _mm256_fmadd_pd(vec_sigma, vec_z_refined, vec_avg);
            _mm256_storeu_pd(&out[i], vec_result);
        }

        // Handle remaining elements
        for (size_t i = n_chunks * 4; i < n; ++i) {
            out[i] = average_ + sigma_ * standard_value(in[i]);
        }
    }
#endif

    // Scalar vector processing fallback
    inline void process_vector_scalar(const double *in, double *out, std::size_t n) const {
#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = average_ + sigma_ * standard_value(in[i]);
        }
    }

#ifdef ICN_USE_SIMD
    // SIMD version of central_value_rational for 4 doubles at once
    static inline __m256d standard_value_simd_central(__m256d vec_x) {
        // q = x - 0.5
        __m256d vec_half = _mm256_set1_pd(0.5);
        __m256d vec_q = _mm256_sub_pd(vec_x, vec_half);

        // r = q * q
        __m256d vec_r = _mm256_mul_pd(vec_q, vec_q);

        // Evaluate numerator polynomial A(r) using Horner's method with FMA
        __m256d vec_num = _mm256_set1_pd(A_[0]);
        for (size_t i = 1; i < A_.size(); ++i) {
            vec_num = _mm256_fmadd_pd(vec_num, vec_r, _mm256_set1_pd(A_[i]));
        }

        // Evaluate denominator polynomial B(r) using Horner's method with FMA
        __m256d vec_den = _mm256_set1_pd(B_[0]);
        for (size_t i = 1; i < B_.size(); ++i) {
            vec_den = _mm256_fmadd_pd(vec_den, vec_r, _mm256_set1_pd(B_[i]));
        }

        // result = q * (num / den)
        __m256d vec_ratio = _mm256_div_pd(vec_num, vec_den);
        __m256d vec_result = _mm256_mul_pd(vec_q, vec_ratio);

        return vec_result;
    }
#endif

    // ---- Baseline numerics (intentionally slow but stable) ------------------        //
    // Standard normal pdf
    static inline double phi(double z) {
        // 1/sqrt(2π) * exp(-z^2 / 2)
        constexpr double INV_SQRT_2PI =
            0.398942280401432677939946059934381868475858631164934657; // 1/sqrt(2π)
        return INV_SQRT_2PI * std::exp(-0.5 * z * z);
    }

    // Standard normal cdf using erfc: Φ(z) = 0.5 * erfc(-z/√2)
    static inline double Phi(double z) {
        constexpr double INV_SQRT_2 =
            0.707106781186547524400844362104849039284835937688474036588; // 1/√2
        return 0.5 * std::erfc(-z * INV_SQRT_2);
    }

    // Polynomial evaluation, Horner's method
    template <size_t N>
    static inline double polynomial(double x, const std::array<double, N> &coeffs) {
        double result = 0.0;
        for (double c : coeffs) {
            result = result * x + c;
        }
        return result;
    }

    static inline double central_value_rational(double x) {
        double q = x - 0.5;
        double r = q * q;
        return q * polynomial(r, A_) / polynomial(r, B_);
    }

    static inline double tail_value_rational(double x) {
        double q;
        if (x <= x_low_) {
            q = std::sqrt(-2.0 * std::log(x));
        } else {
            q = std::sqrt(-2.0 * std::log(1.0 - x));
        }

        double z = polynomial(q, C_) / polynomial(q, D_);

        return (x < x_low_) ? z : -z;
    }

#ifdef ICN_ENABLE_HALLEY_REFINEMENT
    // Use logarithmic form with expm1 to compute the difference on the extreme tails.
    static inline double halley_refine(double z, double x) {
        constexpr double TAIL_THRESHOLD = 1e-5;
        const double p = phi(z); // φ(z) = PDF value
        double r;                // The residual: (Φ(z) - x) / φ(z)

        if (x > 1.0 - TAIL_THRESHOLD) {
            // Right tail (x > 0.5): y = 1 - x ≪ 1 and q = Q(z) = 1 - Φ(z)
            // Formula: r = -y × expm1(log(q) - log(y)) / φ(z)
            const double y = 1.0 - x;
            const double q = Phi(-z); // Q(z) = 1 - Φ(z) = Φ(-z) by symmetry
            r = -y * std::expm1(std::log(q) - std::log(y)) /
                std::max(p, std::numeric_limits<double>::min());

        } else if (x < TAIL_THRESHOLD) {
            // Left tail (x < 0.5): y = x ≪ 1 and a = Φ(z)
            // Formula: r = y × expm1(log(a) - log(y)) / φ(z)
            const double y = x;
            const double a = Phi(z);
            r = y * std::expm1(std::log(a) - std::log(y)) /
                std::max(p, std::numeric_limits<double>::min());

        } else {
            // Central region: x not extreme, direct computation is stable
            // Simply compute r = (Φ(z) - x) / φ(z) directly
            const double f = Phi(z);
            r = (f - x) / std::max(p, std::numeric_limits<double>::min());
        }

        // Apply Halley's method correction: z_new = z - r / (1 - 0.5*z*r)
        const double denom = 1.0 - 0.5 * z * r;
        const double correction =
            r /
            (denom != 0.0 ? denom : std::copysign(std::numeric_limits<double>::infinity(), denom));
        return z - correction;
    }
#endif

    // ---- State & constants ---------------------------------------------------

    double average_, sigma_;

    // Region split (you may adjust in your improved version).
    static constexpr double x_low_ = 0.02425; // ~ Φ(-2.0)
    static constexpr double x_high_ = 1.0 - x_low_;

    // Acklam's coefficients
    static constexpr std::array<double, 6> A_ = {
        -3.969683028665376e+01, //
        2.209460984245205e+02,  //
        -2.759285104469687e+02, //
        1.383577518672690e+02,  //
        -3.066479806614716e+01, //
        2.506628277459239e+00   //
    };
    static constexpr std::array<double, 6> B_ = {
        -5.447609879822406e+01, //
        1.615858368580409e+02,  //
        -1.556989798598866e+02, //
        6.680131188771972e+01,  //
        -1.328068155288572e+01, //
        1.0                     //
    };
    static constexpr std::array<double, 6> C_ = {
        -7.784894002430293e-03, //
        -3.223964580411365e-01, //
        -2.400758277161838e+00, //
        -2.549732539343734e+00, //
        4.374664141464968e+00,  //
        2.938163982698783e+00   //
    };
    static constexpr std::array<double, 5> D_ = {
        7.784695709041462e-03, //
        3.224671290700398e-01, //
        2.445134137142996e+00, //
        3.754408661907416e+00, //
        1.0                    //
    };
};

} // namespace quant

/*
Minimal usage example (not part of API, kept here for convenience):

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
*/
