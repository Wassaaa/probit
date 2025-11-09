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
# Main Test Executable (SIMD enabled)
# ============================================================================
add_executable(probit src/main.cpp)
target_link_libraries(probit PRIVATE icn OpenMP::OpenMP_CXX mvec m)
target_compile_options(probit PRIVATE -mavx2 -mfma)
target_compile_definitions(probit PRIVATE ICN_ENABLE_HALLEY_REFINEMENT)

# ============================================================================
# Batch Processing Executable (for comparison)
# ============================================================================
add_executable(probit_batch src/main_batch.cpp)
target_link_libraries(probit_batch PRIVATE
    icn
    OpenMP::OpenMP_CXX
)
target_compile_options(probit_batch PRIVATE -mavx2 -mfma)
target_compile_definitions(probit_batch PRIVATE
    ICN_ENABLE_HALLEY_REFINEMENT
)

# ============================================================================
# Baseline Executable
# ============================================================================
add_executable(probit_base src/main_base.cpp)
target_link_libraries(probit_base PRIVATE
    icn_base
    OpenMP::OpenMP_CXX
)
target_compile_definitions(probit_base PRIVATE
    ICN_ENABLE_HALLEY_REFINEMENT
)
