#include "heat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace heat {
namespace {

std::string argument(int& i, int argc, char** argv, const std::string& flag) {
    if (++i >= argc) {
        throw std::invalid_argument("missing value after " + flag);
    }
    return argv[i];
}

double floating(const std::string& text, const std::string& name) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid floating-point value for " + name + ": " + text);
    }
    if (used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid finite floating-point value for " + name + ": " + text);
    }
    return value;
}

std::uint64_t positive_integer(const std::string& text, const std::string& name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(name + " must be a positive integer");
    }
    std::size_t used = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(text, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid integer for " + name + ": " + text);
    }
    if (used != text.size() || value == 0) {
        throw std::invalid_argument(name + " must be a positive integer");
    }
    return value;
}

} // namespace

void print_usage(const char* executable) {
    std::cout << "Usage: " << executable << " [options]\n"
              << "  --case 1|2|V          Four hot edges, hot top/bottom, or exact verification\n"
              << "  --N INTEGER           Grid intervals per side (at least 2; default 64)\n"
              << "  --r NUMBER            Requested alpha*dt/h^2 (default 0.2; alpha=1)\n"
              << "  --steps INTEGER       Number of steps (default 100, unless --T is set)\n"
              << "  --T NUMBER            Requested physical final time; adjust dt so r<=requested\n"
              << "  --out PATH            Optional raw little-endian (N+1)x(N+1) double field\n"
              << "  --decomp serial       Serial reference solver (strips/blocks come later)\n"
              << "  --seed-high-mode A    Verification test: add A*sin((N-1)*pi*x)*sin((N-1)*pi*y)\n"
              << "  --allow-unstable      TEST ONLY: allow r>0.25 in case V with seeded mode\n"
              << "  --help                Show this help\n";
}

Config parse_cli(int argc, char** argv) {
    Config c;
    bool given_T = false;
    bool given_steps = false;
    double wanted_T = 0.0;

    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (flag == "--case") {
            const auto value = argument(i, argc, argv, flag);
            if (value != "1" && value != "2" && value != "V" && value != "v") {
                throw std::invalid_argument("--case must be 1, 2, or V");
            }
            c.test_case = (value == "v") ? 'V' : value.front();
        } else if (flag == "--N") {
            const auto number = positive_integer(argument(i, argc, argv, flag), flag);
            if (number < 2 || number > static_cast<std::uint64_t>(std::numeric_limits<int>::max() - 1)) {
                throw std::invalid_argument("--N must be in [2, INT_MAX-1]");
            }
            c.N = static_cast<int>(number);
        } else if (flag == "--r") {
            c.requested_r = floating(argument(i, argc, argv, flag), flag);
        } else if (flag == "--steps") {
            c.steps = positive_integer(argument(i, argc, argv, flag), flag);
            given_steps = true;
        } else if (flag == "--T") {
            wanted_T = floating(argument(i, argc, argv, flag), flag);
            given_T = true;
        } else if (flag == "--out") {
            c.output = argument(i, argc, argv, flag);
            if (c.output.empty()) {
                throw std::invalid_argument("--out requires a nonempty path");
            }
        } else if (flag == "--decomp") {
            c.decomp = argument(i, argc, argv, flag);
        } else if (flag == "--seed-high-mode") {
            c.high_mode_amplitude = floating(argument(i, argc, argv, flag), flag);
        } else if (flag == "--allow-unstable") {
            c.allow_unstable = true;
        } else {
            throw std::invalid_argument("unknown option: " + flag);
        }
    }

    if (given_T && given_steps) {
        throw std::invalid_argument("pass either --T or --steps, not both");
    }
    if (c.decomp != "serial") {
        throw std::invalid_argument("Part 1 supports only --decomp serial (strips and blocks are not implemented yet)");
    }
    if (!(c.requested_r > 0.0)) {
        throw std::invalid_argument("--r must be positive");
    }
    if (c.high_mode_amplitude != 0.0 && c.test_case != 'V') {
        throw std::invalid_argument("--seed-high-mode is only valid for --case V");
    }
    if (c.allow_unstable && (c.test_case != 'V' || c.high_mode_amplitude == 0.0 || given_T || !given_steps)) {
        throw std::invalid_argument("--allow-unstable requires --case V, --seed-high-mode A, and --steps");
    }
    if (c.requested_r > 0.25 && !c.allow_unstable) {
        throw std::invalid_argument("unstable r: explicit 2D heat scheme requires 0 < r <= 0.25; use --allow-unstable only for Task 6.7");
    }
    if (c.allow_unstable && c.requested_r <= 0.25) {
        throw std::invalid_argument("--allow-unstable is only needed for r > 0.25");
    }

    if (given_T) {
        if (!(wanted_T > 0.0)) {
            throw std::invalid_argument("--T must be positive");
        }
        // Long-double arithmetic limits roundoff near exact integer ratios.
        const long double n = static_cast<long double>(c.N);
        const long double needed = (static_cast<long double>(wanted_T) * n * n) /
                                   static_cast<long double>(c.requested_r);
        if (!std::isfinite(needed) || needed > static_cast<long double>(std::numeric_limits<std::uint64_t>::max() - 1)) {
            throw std::invalid_argument("--T requires too many time steps");
        }
        const long double rounded = std::round(needed);
        const long double tolerance = 8 * std::numeric_limits<double>::epsilon() *
                                      std::max(1.0L, std::fabs(needed));
        const long double chosen = (std::fabs(needed - rounded) < tolerance) ? rounded : std::ceil(needed);
        c.steps = static_cast<std::uint64_t>(std::max(1.0L, chosen));
        // Guard against any insufficient step count after conversion to double.
        c.dt = wanted_T / static_cast<double>(c.steps);
        c.r = c.dt * static_cast<double>(c.N) * static_cast<double>(c.N);
        while (c.r > c.requested_r * (1 + 1e-14)) {
            ++c.steps;
            c.dt = wanted_T / static_cast<double>(c.steps);
            c.r = c.dt * static_cast<double>(c.N) * static_cast<double>(c.N);
        }
        c.final_time = wanted_T;
    } else {
        c.r = c.requested_r;
        const double n = static_cast<double>(c.N);
        c.dt = c.r / (n * n);
        c.final_time = static_cast<double>(c.steps) * c.dt;
    }

    if (!std::isfinite(c.dt) || !(c.dt > 0.0) || !std::isfinite(c.final_time) ||
        !std::isfinite(c.r) || c.final_time <= 0.0) {
        throw std::invalid_argument("invalid time-step configuration (overflow or underflow)");
    }
    const std::size_t side = static_cast<std::size_t>(c.N) + 1U;
    if (side > std::numeric_limits<std::size_t>::max() / side ||
        side * side > std::numeric_limits<std::size_t>::max() / sizeof(double) / 2U) {
        throw std::invalid_argument("grid size overflows allocated memory");
    }
    return c;
}

} // namespace heat

int main(int argc, char** argv) {
    try {
        const heat::Config config = heat::parse_cli(argc, argv);
        if (config.allow_unstable) {
            std::cerr << "WARNING: TEST ONLY, r=" << config.r
                      << " violates the stability limit 0.25; intentional Task 6.7 experiment\n";
        }
        auto result = heat::run_serial(config);
        // Output and error checks are explicitly OUTSIDE the timed time-stepping loop.
        if (!config.output.empty()) {
            heat::write_binary_field(config.output, result.field);
        }
        const heat::Errors errors = (config.test_case == 'V')
                                       ? heat::verification_errors(config, result.field)
                                       : heat::Errors{};

        std::cout << std::setprecision(17)
                  << "decomp=serial P=1 case=" << config.test_case
                  << " N=" << config.N
                  << " r=" << config.r
                  << " requested_r=" << config.requested_r
                  << " dt=" << config.dt
                  << " T=" << config.final_time
                  << " steps=" << config.steps
                  << " t_total=" << result.compute_seconds
                  << " t_comp=" << result.compute_seconds
                  << " t_comm=0";
        if (config.test_case == 'V') {
            std::cout << " err_inf=" << errors.infinity
                      << " err_l2=" << errors.l2h;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "heat: " << e.what() << '\n';
        return 1;
    }
}
