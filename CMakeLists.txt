cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(probit LANGUAGES CXX)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Compiler Options
# ============================================================================
add_compile_options(-O3 -fopt-info-vec-optimized)

# ============================================================================
# Find Dependencies
# ============================================================================
find_package(OpenMP REQUIRED)

# ============================================================================
# Interface Libraries (header-only)
# ============================================================================
add_library(icn INTERFACE)
target_include_directories(icn INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(icn_base INTERFACE)
target_include_directories(icn_base INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ============================================================================
# Test Executables - Different configurations for benchmarking
# ============================================================================

# SIMD + OpenMP (full optimization, no refinement for max speed)
add_executable(probit_simd_omp src/main.cpp)
target_link_libraries(probit_simd_omp PRIVATE
    icn
    OpenMP::OpenMP_CXX
)
target_compile_options(probit_simd_omp PRIVATE
    -mavx
    -mfma
    -fno-math-errno
)

# SIMD only (no OpenMP, with refinement)
add_executable(probit_simd_single src/main.cpp)
target_link_libraries(probit_simd_single PRIVATE
    icn
)
target_compile_options(probit_simd_single PRIVATE
    -mavx
    -mfma
    -fno-math-errno
)
target_compile_definitions(probit_simd_single PRIVATE
    ICN_ENABLE_HALLEY_REFINEMENT
)

# No SIMD + OpenMP (threading only)
add_executable(probit_scalar_omp src/main.cpp)
target_link_libraries(probit_scalar_omp PRIVATE
    icn
    OpenMP::OpenMP_CXX
)
target_compile_definitions(probit_scalar_omp PRIVATE
    ICN_ENABLE_HALLEY_REFINEMENT
    ICN_DISABLE_SIMD
)

# No SIMD + No OpenMP (pure scalar, single-threaded)
add_executable(probit_scalar_single src/main.cpp)
target_link_libraries(probit_scalar_single PRIVATE
    icn
)
target_compile_definitions(probit_scalar_single PRIVATE
    ICN_ENABLE_HALLEY_REFINEMENT
)

# Default target (alias to full optimization)
add_executable(probit src/main.cpp)
target_link_libraries(probit PRIVATE
    icn
    OpenMP::OpenMP_CXX
)
target_compile_options(probit PRIVATE
    -mavx
    -mfma
    -fno-math-errno
)
target_compile_definitions(probit PRIVATE
    ICN_ENABLE_HALLEY_REFINEMENT
)

# Baseline (using InverseCumulativeNormalBase.h with bisection)
add_executable(probit_base src/main.cpp)
target_link_libraries(probit_base PRIVATE
    icn_base
)
target_compile_definitions(probit_base PRIVATE
    ICN_ENABLE_HALLEY_REFINEMENT
    USE_BASE_HEADER
)

# ============================================================================
# Batch Processing Executable (for comparison)
# ============================================================================
add_executable(probit_nopy src/main_nopy.cpp)
target_link_libraries(probit_nopy PRIVATE
    icn
    OpenMP::OpenMP_CXX
)
target_compile_options(probit_nopy PRIVATE
  -mavx
  -mfma
  -fno-math-errno
)
target_compile_definitions(probit_nopy PRIVATE
    # ICN_ENABLE_HALLEY_REFINEMENT
)


