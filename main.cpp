#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>

#include <tbb/parallel_for.h>

const size_t N = 2 << 28;

double seq_for()
{
    auto values = std::vector<double>(N);
    std::for_each(std::execution::seq, values.begin(), values.end(), [](double& value) {
        value = 1.0 / (1 + std::exp(-std::sin(value * 0.001)));
    });

    double total = 0;
    for (double value : values) {
        total += value;
    }
    return total;
}

double par_for()
{
    auto values = std::vector<double>(N);
    std::for_each(std::execution::par, values.begin(), values.end(), [](double& value) {
        value = 1.0 / (1 + std::exp(-std::sin(value * 0.001)));
    });

    double total = 0;
    for (double value : values) {
        total += value;
    }
    return total;
}

double tbb_for()
{
    auto values = std::vector<double>(N);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, values.size()), [&](tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            values[i] = 1.0 / (1 + std::exp(-std::sin(values[i] * 0.001)));
        }
    });

    double total = 0;
    for (double value : values) {
        total += value;
    }
    return total;
}

double omp_for()
{
    auto values = std::vector<double>(N);
#pragma omp parallel for
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 1.0 / (1 + std::exp(-std::sin(values[i] * 0.001)));
    }

    double total = 0;
    for (double value : values) {
        total += value;
    }
    return total;
}

void time_it(double (*fn_ptr)(), const std::string& fn_name)
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
        std::cout << "Usage: " << argv[0] << " <par|tbb|seq|omp>" << std::endl;
        return 1;
    }

    std::string op(argv[1]);
    if (op == "par") {
        time_it(&par_for, op);
    } else if (op == "tbb") {
        time_it(&tbb_for, op);
    } else if (op == "seq") {
        time_it(&seq_for, op);
    } else if (op == "omp") {
        time_it(&omp_for, op);
    } else {
        std::cout << "Usage: " << argv[0] << " <par|tbb|seq|omp>" << std::endl;
    }
    return 0;
}