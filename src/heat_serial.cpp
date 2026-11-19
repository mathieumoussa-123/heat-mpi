#include "heat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace heat {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

void initialize(const Config& c, std::vector<double>& u) {
    const int N = c.N;
    const double h = 1.0 / static_cast<double>(N);
    // Set fixed Dirichlet boundaries explicitly. Both time levels receive them.
    for (int j = 0; j <= N; ++j) {
        for (int i = 0; i <= N; ++i) {
            const bool boundary = (i == 0 || i == N || j == 0 || j == N);
            if (boundary) {
                if (c.test_case == '1') {
                    u[index(i, j, N)] = 1.0;
                } else if (c.test_case == '2') {
                    // Top and bottom win at corners (project PDF Section 3).
                    u[index(i, j, N)] = (j == 0 || j == N) ? 1.0 : 0.0;
                } else {
                    u[index(i, j, N)] = 0.0;
                }
            } else if (c.test_case == 'V') {
                const double x = i * h;
                const double y = j * h;
                double value = std::sin(kPi * x) * std::sin(kPi * y);
                if (c.high_mode_amplitude != 0.0) {
                    value += c.high_mode_amplitude *
                             std::sin((N - 1) * kPi * x) *
                             std::sin((N - 1) * kPi * y);
                }
                u[index(i, j, N)] = value;
            } // Otherwise interior starts cold (the vector is zero-initialized).
        }
    }
}
} // namespace

Result run_serial(const Config& c) {
    const std::size_t side = static_cast<std::size_t>(c.N) + 1U;
    const std::size_t nodes = side * side;
    std::vector<double> old_values(nodes, 0.0);
    std::vector<double> new_values(nodes, 0.0);
    initialize(c, old_values);
    initialize(c, new_values); // Fixed boundaries, ready for swap without copying.

    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t step = 0; step < c.steps; ++step) {
        for (int j = 1; j < c.N; ++j) {
            for (int i = 1; i < c.N; ++i) {
                const auto k = index(i, j, c.N);
                // Task 5.2(b): five-point explicit (forward Euler) stencil.
                // Preserve this floating-point expression/order for future MPI cmp tests.
                const double sum = old_values[index(i - 1, j, c.N)]
                                 + old_values[index(i + 1, j, c.N)]
                                 + old_values[index(i, j - 1, c.N)]
                                 + old_values[index(i, j + 1, c.N)];
                new_values[k] = old_values[k] + c.r * (sum - 4.0 * old_values[k]);
            }
        }
        // Task 8.2: swap vectors, never overwrite an old time level in place.
        old_values.swap(new_values);
    }
    const auto stop = std::chrono::steady_clock::now();
    Result result;
    result.compute_seconds = std::chrono::duration<double>(stop - start).count();
    result.field = std::move(old_values);
    return result;
}

Errors verification_errors(const Config& c, const std::vector<double>& u) {
    if (c.test_case != 'V') {
        throw std::invalid_argument("verification error norms require case V");
    }
    const double h = 1.0 / static_cast<double>(c.N);
    const double decay = std::exp(-2.0 * kPi * kPi * c.final_time);
    const double fast_decay = (c.high_mode_amplitude == 0.0)
                                  ? 0.0
                                  : std::exp(-2.0 * (c.N - 1.0) * (c.N - 1.0) *
                                             kPi * kPi * c.final_time);
    double max_error = 0.0;
    double squared_error = 0.0;
    for (int j = 0; j <= c.N; ++j) {
        for (int i = 0; i <= c.N; ++i) {
            // Mathematically the exact solution is exactly zero on Dirichlet edges.
            const bool edge = (i == 0 || j == 0 || i == c.N || j == c.N);
            double exact = 0.0;
            if (!edge) {
                const double x = static_cast<double>(i) * h;
                const double y = static_cast<double>(j) * h;
                exact = std::sin(kPi * x) * std::sin(kPi * y) * decay;
                if (c.high_mode_amplitude != 0.0) {
                    exact += c.high_mode_amplitude * fast_decay *
                             std::sin((c.N - 1) * kPi * x) *
                             std::sin((c.N - 1) * kPi * y);
                }
            }
            const double error = u[index(i, j, c.N)] - exact;
            max_error = std::max(max_error, std::abs(error));
            squared_error += error * error;
        }
    }
    return Errors{max_error, h * std::sqrt(squared_error)};
}

} // namespace heat
