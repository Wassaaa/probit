cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(probit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3)

add_library(icn INTERFACE)
add_library(icn_base INTERFACE)

target_include_directories(icn INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(icn_base INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(probit
    src/main_batch.cpp
)
# Link our batch program against the engine
target_link_libraries(probit PRIVATE icn)
# Add refinement flag
target_compile_definitions(probit PRIVATE ICN_ENABLE_HALLEY_REFINEMENT)

add_executable(probit_base
    src/main_base.cpp
)
# Link our batch program against the engine
target_link_libraries(probit_base PRIVATE icn_base)
# Add refinement flag
target_compile_definitions(probit_base PRIVATE ICN_ENABLE_HALLEY_REFINEMENT)
