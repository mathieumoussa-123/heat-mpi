#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace heat {

struct Config {
    char test_case = '1'; // '1', '2', or 'V'
    int N = 64;           // intervals in each direction: (N+1)^2 nodes
    double requested_r = 0.2;
    double r = 0.2;       // actual alpha*dt/h^2 (<= requested_r when --T)
    double dt = 0.0;
    double final_time = 0.0;
    std::uint64_t steps = 100;
    std::string decomp = "serial";
    std::string output;
    double high_mode_amplitude = 0.0; // for Task 6.7; zero for ordinary verification
    bool allow_unstable = false;       // explicitly opt-in for Task 6.7 only
};

struct Errors {
    double infinity = 0.0;
    double l2h = 0.0;
};

struct Result {
    std::vector<double> field;
    double compute_seconds = 0.0;
};

inline std::size_t index(int i, int j, int N) noexcept {
    return static_cast<std::size_t>(j) * (static_cast<std::size_t>(N) + 1U)
           + static_cast<std::size_t>(i);
}

Config parse_cli(int argc, char** argv);
void print_usage(const char* executable);
Result run_serial(const Config& config);
Errors verification_errors(const Config& config, const std::vector<double>& field);
void write_binary_field(const std::string& file, const std::vector<double>& field);

} // namespace heat
