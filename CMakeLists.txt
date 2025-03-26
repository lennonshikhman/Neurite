cmake_minimum_required(VERSION 3.10)
project(Neurite VERSION 0.1 LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (assumes Eigen3 is installed in your system)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Specify the include directory for Neurite (header-only library)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create an INTERFACE library target for Neurite
add_library(neurite INTERFACE)
target_include_directories(neurite INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optionally, create a test executable
add_executable(test_neurite tests/test_neurite.cpp)
target_link_libraries(test_neurite PRIVATE neurite)
