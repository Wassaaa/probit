cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(probit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3)

add_library(probit_engine STATIC
    src/ProbitSolver.cpp
    src/ProbitSolver.hpp
)
target_include_directories(probit_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# target_compile_options(probit_engine PRIVATE -O3)

# --- Program A: The "Slow Streamer" ---
add_executable(probit_stream
    src/main_stream.cpp
)
# Link our streamer against the engine
target_link_libraries(probit_stream PRIVATE probit_engine)


# --- Program B: The "Fast Serial Batch" ---
add_executable(probit_batch
    src/main_batch.cpp
)
# Link our batch program against the engine
target_link_libraries(probit_batch PRIVATE probit_engine)


# --- Program C: The "Fast Parallel Batch" ---
# This one needs std::thread, so we have to find the 'Threads' package
find_package(Threads REQUIRED)

add_executable(probit_parallel
    src/main_parallel.cpp
)
# Link against the engine AND the Threads library
target_link_libraries(probit_parallel PRIVATE
    probit_engine
    Threads::Threads
)
