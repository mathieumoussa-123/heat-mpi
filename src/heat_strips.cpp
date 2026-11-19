#include "heat.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace heat {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kToUpperTag = 71;
constexpr int kToLowerTag = 72;

struct Partition {
    int first_row = 0; // Global index of first owned interior row (j = 1..N-1).
    int owned_rows = 0;
};

Partition partition_for(int N, int P, int rank) {
    const int interior = N - 1;
    const int base = interior / P;
    const int extras = interior % P;
    return Partition{1 + rank * base + std::min(rank, extras),
                     base + (rank < extras ? 1 : 0)};
}

// Same boundary values and floating-point evaluation as Part 1's initialize().
// j is GLOBAL; local j=0 and j=n_local+1 are ghosts.
void initialize_row(const Config& c, int global_j, int local_j, std::vector<double>& u) {
    const int N = c.N;
    const double h = 1.0 / static_cast<double>(N);
    for (int i = 0; i <= N; ++i) {
        const bool boundary = (i == 0 || i == N || global_j == 0 || global_j == N);
        const auto k = index(i, local_j, N);
        if (boundary) {
            if (c.test_case == '1') {
                u[k] = 1.0;
            } else if (c.test_case == '2') {
                // Match serial: hot top/bottom, including all four corners.
                u[k] = (global_j == 0 || global_j == N) ? 1.0 : 0.0;
            } else {
                u[k] = 0.0;
            }
        } else if (c.test_case == 'V') {
            const double x = i * h;
            const double y = global_j * h;
            double value = std::sin(kPi * x) * std::sin(kPi * y);
            if (c.high_mode_amplitude != 0.0) {
                value += c.high_mode_amplitude *
                         std::sin((N - 1) * kPi * x) *
                         std::sin((N - 1) * kPi * y);
            }
            u[k] = value;
        }
        // Otherwise an interior node remains the zero-initialized value.
    }
}

void exchange_halos(std::vector<double>& old_values, int N, int owned_rows,
                    int rank, int P) {
    const int lower = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int upper = (rank == P - 1) ? MPI_PROC_NULL : rank + 1;
    const int count = N + 1;
    // First: top owned row -> rank above; bottom ghost <- rank below.
    // Both partners execute the same calls, so there is no blocking-send deadlock.
    MPI_Sendrecv(old_values.data() + index(0, owned_rows, N), count, MPI_DOUBLE,
                 upper, kToUpperTag,
                 old_values.data() + index(0, 0, N), count, MPI_DOUBLE,
                 lower, kToUpperTag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Second: bottom owned row -> rank below; top ghost <- rank above.
    MPI_Sendrecv(old_values.data() + index(0, 1, N), count, MPI_DOUBLE,
                 lower, kToLowerTag,
                 old_values.data() + index(0, owned_rows + 1, N), count, MPI_DOUBLE,
                 upper, kToLowerTag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Diagnostic for the professor's N=8, one-step ghost-row debugging exercise.
// Called BEFORE the timed loop (and therefore doesn't contaminate timings).
void print_halos(const Config& c, const Partition& part, int rank, int P,
                 const char* stage, const std::vector<double>& old_values) {
    for (int turn = 0; turn < P; ++turn) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (turn == rank) {
            std::cerr << "[halo-debug stage=" << stage << " rank=" << rank << " P=" << P
                      << " global_owned=" << part.first_row << ".."
                      << (part.first_row + part.owned_rows - 1) << "]\n";
            std::cerr << std::setprecision(9);
            for (int local_j = 0; local_j < part.owned_rows + 2; ++local_j) {
                const char* kind = (local_j == 0) ? "lower-ghost"
                                   : (local_j == part.owned_rows + 1) ? "upper-ghost"
                                   : "owned";
                const int global_j = part.first_row + local_j - 1;
                std::cerr << "  local_j=" << local_j << " global_j=" << global_j
                          << " " << kind << " :";
                for (int i = 0; i <= c.N; ++i) {
                    std::cerr << ' ' << old_values[index(i, local_j, c.N)];
                }
                std::cerr << '\n';
            }
            std::cerr.flush();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// We assemble a FULL serial-layout field only when output or verification requires it.
std::vector<double> gather_field(const Config& c, const Partition& part,
                                 const std::vector<double>& local_values, int rank, int P) {
    const auto side = static_cast<std::size_t>(c.N) + 1U;
    const auto total_nodes = side * side;
    // MPI_Gatherv uses 32-bit int counts/displacements. Fail rather than silently overflow.
    if (total_nodes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("MPI_Gatherv output exceeds INT_MAX doubles; use smaller N");
    }
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> field;
    if (rank == 0) {
        counts.resize(static_cast<std::size_t>(P));
        displacements.resize(static_cast<std::size_t>(P));
        for (int p = 0; p < P; ++p) {
            const auto owned = partition_for(c.N, P, p);
            counts[static_cast<std::size_t>(p)] = owned.owned_rows * (c.N + 1);
            displacements[static_cast<std::size_t>(p)] = owned.first_row * (c.N + 1);
        }
        field.assign(total_nodes, 0.0);
        // Only first and last global boundary rows are missing from Gatherv.
        initialize_row(c, 0, 0, field);
        initialize_row(c, c.N, c.N, field);
    }
    MPI_Gatherv(local_values.data() + index(0, 1, c.N), part.owned_rows * (c.N + 1),
                MPI_DOUBLE,
                rank == 0 ? field.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return field;
}

} // namespace

StripResult run_strips(const Config& c, int rank, int P) {
    if (P < 1 || P > c.N - 1) {
        throw std::invalid_argument("strip decomposition requires 1 <= P <= N-1 (at least one interior row per MPI rank)");
    }
    const Partition part = partition_for(c.N, P, rank);
    const auto side = static_cast<std::size_t>(c.N) + 1U;
    const auto local_nodes = (static_cast<std::size_t>(part.owned_rows) + 2U) * side;
    std::vector<double> old_values(local_nodes, 0.0);
    std::vector<double> new_values(local_nodes, 0.0);

    for (int local_j = 1; local_j <= part.owned_rows; ++local_j) {
        const int global_j = part.first_row + local_j - 1;
        initialize_row(c, global_j, local_j, old_values);
        initialize_row(c, global_j, local_j, new_values);
    }
    // Exterior ghost rows are actual physical boundaries, not MPI-owned interior rows.
    if (rank == 0) {
        initialize_row(c, 0, 0, old_values);
        initialize_row(c, 0, 0, new_values);
    }
    if (rank == P - 1) {
        initialize_row(c, c.N, part.owned_rows + 1, old_values);
        initialize_row(c, c.N, part.owned_rows + 1, new_values);
    }

    if (c.debug_halos) {
        // A single untimed exchange makes the initial ghost layers inspectable.
        exchange_halos(old_values, c.N, part.owned_rows, rank, P);
        print_halos(c, part, rank, P, "initial", old_values);
    }

    MPI_Barrier(MPI_COMM_WORLD); // All ranks start measuring the time-stepping loop together.
    double comm_seconds = 0.0;
    double compute_seconds = 0.0;
    const double start = MPI_Wtime();
    for (std::uint64_t step = 0; step < c.steps; ++step) {
        const double comm_start = MPI_Wtime();
        exchange_halos(old_values, c.N, part.owned_rows, rank, P);
        comm_seconds += MPI_Wtime() - comm_start;

        const double comp_start = MPI_Wtime();
        for (int local_j = 1; local_j <= part.owned_rows; ++local_j) {
            for (int i = 1; i < c.N; ++i) {
                const auto k = index(i, local_j, c.N);
                // EXACT same arithmetic/order as run_serial (Part 1).
                const double sum = old_values[index(i - 1, local_j, c.N)]
                                 + old_values[index(i + 1, local_j, c.N)]
                                 + old_values[index(i, local_j - 1, c.N)]
                                 + old_values[index(i, local_j + 1, c.N)];
                new_values[k] = old_values[k] + c.r * (sum - 4.0 * old_values[k]);
            }
        }
        old_values.swap(new_values);
        compute_seconds += MPI_Wtime() - comp_start;
    }
    const double total_seconds = MPI_Wtime() - start;

    if (c.debug_halos) {
        // For the required N=8 one-step debug case: show evolved owned rows
        // and refreshed ghost copies AFTER the loop, without affecting timings.
        exchange_halos(old_values, c.N, part.owned_rows, rank, P);
        print_halos(c, part, rank, P, "final", old_values);
    }

    const double local_times[3] = {total_seconds, compute_seconds, comm_seconds};
    double max_times[3] = {0.0, 0.0, 0.0};
    MPI_Reduce(local_times, max_times, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    StripResult result;
    if (rank == 0) {
        result.total_seconds = max_times[0];
        result.compute_seconds = max_times[1];
        result.comm_seconds = max_times[2];
    }
    if (!c.output.empty() || c.test_case == 'V') {
        result.field = gather_field(c, part, old_values, rank, P);
    }
    return result;
}

} // namespace heat
