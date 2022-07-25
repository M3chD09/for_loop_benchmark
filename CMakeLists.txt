cmake_minimum_required(VERSION 3.10)

project(for_loop_benchmark VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenMP_CXX_FLAGS "-openmp:llvm")

find_package(TBB CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
