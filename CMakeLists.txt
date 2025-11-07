cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(probit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
add_compile_options(-Wall -Wextra -Werror)

file(GLOB_RECURSE SOURCES
    src/*.cpp
)

add_executable(probit ${SOURCES})
target_include_directories(probit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
