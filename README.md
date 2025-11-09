# Probit - Inverse Normal CDF Implementation

High-performance inverse cumulative distribution function for the standard normal distribution using Acklam's rational approximation with SIMD and OpenMP optimizations.

## Build

Requirements: C++17 (gcc, clang) compiler with OpenMP support, CMake 3.10+

```bash
cmake -B build -G Ninja
ninja -C build probit
```

Or without Ninja:

```bash
cmake -B build
make -C build probit
```

This builds the main `probit` target (SIMD + OpenMP + Halley refinement). Other variants: `probit_base`, `probit_scalar_single`, `probit_simd_single`, `probit_scalar_omp`, `probit_simd_omp`.

**Nix users:** A `flake.nix` is provided for reproducible environment setup.

## Testing

Run the calibration script to validate accuracy and performance:

```bash
cmake -B build -G Ninja
ninja -C build # for all targets
python calibration.py  # Requires: numpy, scipy
```

| Configuration                   | Time (s) | Throughput | Speedup | Max Error  | Mean Error |
| ------------------------------- | -------- | ---------- | ------- | ---------- | ---------- |
| Bisection baseline              | 1.1371   | 0.9 M/s    | 1.0×    | 7.74×10⁻⁶  | 7.74×10⁻¹² |
| Acklam scalar                   | 0.0259   | 38.5 M/s   | 44×     | 8.58×10⁻¹³ | 2.02×10⁻¹⁶ |
| Acklam + SIMD                   | 0.0215   | 46.6 M/s   | 53×     | 8.57×10⁻¹³ | 2.01×10⁻¹⁶ |
| Acklam + OpenMP                 | 0.0052   | 193.3 M/s  | 220×    | 8.58×10⁻¹³ | 2.01×10⁻¹⁶ |
| Acklam + SIMD + OpenMP          | 0.0035   | 285.5 M/s  | 325×    | 8.57×10⁻¹³ | 2.01×10⁻¹⁶ |
| Acklam + SIMD + OpenMP (no ref) | 0.0015   | 649.8 M/s  | 739×    | 7.36×10⁻⁹  | 5.71×10⁻¹⁰ |

Compares against SciPy's `norm.ppf` on 1M test points and reports throughput, speedup, and error metrics.

## API Usage

```cpp
#include "InverseCumulativeNormal.h"

quant::InverseCumulativeNormal icn(0.0, 1.0);  // mean, stddev
double z = icn(0.975);  // Scalar: z ≈ 1.96

// Vector (optimized with SIMD + OpenMP)
std::vector<double> x = {0.1, 0.5, 0.9};
std::vector<double> z(x.size());
icn(x.data(), z.data(), x.size());
```

## Implementation

See `DESIGN.md` for full documentation including rational approximation details, Halley refinement, and performance analysis.
