cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(probit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_compile_options(-O3)

add_library(icn INTERFACE)

target_include_directories(icn INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(probit
    src/main_batch.cpp
)
# Link our batch program against the engine
target_link_libraries(probit PRIVATE icn)
# Add refinement flag
target_compile_definitions(probit PRIVATE ICN_ENABLE_HALLEY_REFINEMENT)
