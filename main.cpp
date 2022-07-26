#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <functional>
#include <iostream>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

const size_t N = 2 << 28;

template<class ExecutionPolicy>
std::function<double(void)> stl_for(ExecutionPolicy&& policy)
{
    return [&]() -> double {
        auto values = std::vector<double>(N);
        std::for_each(policy, values.begin(), values.end(), [](double& value) {
            value = 1.0 / (1 + std::exp(-std::sin(value * 0.001)));
        });

        double total = std::reduce(policy, values.begin(), values.end());
        return total;
    };
}

double tbb_for()
{
    auto values = std::vector<double>(N);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, values.size()), [&](tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            values[i] = 1.0 / (1 + std::exp(-std::sin(values[i] * 0.001)));
        }
    });

    double total = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, values.size()), 0.0,
        [&](tbb::blocked_range<size_t> r, double init) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            init += values[i];
        }
        return init; },
        [](double a, double b) {
            return a + b;
        });
    return total;
}

double omp_for()
{
    auto values = std::vector<double>(N);
#pragma omp parallel for
    for (long long i = 0; i < (long long)values.size(); ++i) {
        values[i] = 1.0 / (1 + std::exp(-std::sin(values[i] * 0.001)));
    }

    double total = 0;
#pragma omp parallel for reduction(+ \
                                   : total)
    for (long long i = 0; i < (long long)values.size(); ++i) {
        total += values[i];
    }
    return total;
}

void time_it(std::function<double(void)> fn_ptr, const std::string& fn_name)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    auto result = fn_ptr();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << fn_name << ", result = " << result << ", duration = " << duration << "us" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <seq|par|par_unseq|unseq|tbb|omp>" << std::endl;
        return 1;
    }

    std::string op(argv[1]);
    if (op == "seq") {
        time_it(stl_for(std::execution::seq), op);
    } else if (op == "par") {
        time_it(stl_for(std::execution::par), op);
    } else if (op == "par_unseq") {
        time_it(stl_for(std::execution::par_unseq), op);
    } else if (op == "unseq") {
        time_it(stl_for(std::execution::unseq), op);
    } else if (op == "tbb") {
        time_it(&tbb_for, op);
    } else if (op == "omp") {
        time_it(&omp_for, op);
    } else {
        std::cout << "Usage: " << argv[0] << " <seq|par|par_unseq|unseq|tbb|omp>" << std::endl;
    }
    return 0;
}
