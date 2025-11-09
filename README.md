# Probit - Inverse Normal CDF Implementation

High-performance inverse cumulative distribution function for the standard normal distribution using Acklam's rational approximation with SIMD and OpenMP optimizations.

## Build

Requirements: C++17 compiler with OpenMP support, CMake 3.10+, AVX-capable CPU

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
python calibration.py  # Requires: numpy, scipy
```

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
