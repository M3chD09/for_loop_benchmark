cmake_minimum_required(VERSION 3.10)

project(for_loop_benchmark VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CTEST_OUTPUT_ON_FAILURE ON)

find_package(TBB CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

enable_testing()

foreach(test seq par par_unseq unseq tbb omp)
    foreach(i 1 2 3)
        add_test(NAME ${PROJECT_NAME}_${test}_${i} COMMAND ${PROJECT_NAME} ${test})
    endforeach()
endforeach()
