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
#include <utility>
#include <vector>

namespace heat {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kUpTag = 171;
constexpr int kDownTag = 172;
constexpr int kRightTag = 173;
constexpr int kLeftTag = 174;

struct AxisPart {
    int first = 0;  // First owned GLOBAL INTERIOR index; node 0 and node N are boundary.
    int count = 0;
};

AxisPart split_axis(int N, int partitions, int coordinate) {
    const int interior = N - 1;
    const int common = interior / partitions;
    const int extras = interior % partitions;
    return AxisPart{1 + coordinate * common + std::min(coordinate, extras),
                    common + (coordinate < extras ? 1 : 0)};
}

struct Block {
    AxisPart x;
    AxisPart y;
    int pitch = 0;
};

std::size_t local_index(int x, int y, int pitch) noexcept {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch)
         + static_cast<std::size_t>(x);
}

double boundary_value(const Config& config, int j) {
    if (config.test_case == '1') return 1.0;
    if (config.test_case == '2') return (j == 0 || j == config.N) ? 1.0 : 0.0;
    return 0.0;
}

// Match heat_serial.cpp initialization, including operation order in sin evaluations.
void initialize_block(const Config& config, const Block& block, std::vector<double>& data) {
    const double h = 1.0 / static_cast<double>(config.N);
    for (int local_j = 0; local_j <= block.y.count + 1; ++local_j) {
        const int global_j = block.y.first + local_j - 1;
        for (int local_i = 0; local_i <= block.x.count + 1; ++local_i) {
            const int global_i = block.x.first + local_i - 1;
            const bool physical = global_i == 0 || global_i == config.N ||
                                  global_j == 0 || global_j == config.N;
            auto& value = data[local_index(local_i, local_j, block.pitch)];
            if (physical) {
                value = boundary_value(config, global_j);
            } else if (local_i >= 1 && local_i <= block.x.count &&
                       local_j >= 1 && local_j <= block.y.count && config.test_case == 'V') {
                const double x = global_i * h;
                const double y = global_j * h;
                value = std::sin(kPi * x) * std::sin(kPi * y);
                if (config.high_mode_amplitude != 0.0) {
                    value += config.high_mode_amplitude *
                             std::sin((config.N - 1) * kPi * x) *
                             std::sin((config.N - 1) * kPi * y);
                }
            }
            // Interior ghost cells have no initial owner value; the exchange fills them.
        }
    }
}

// One MPI derived datatype describes each LOCAL column, strided by the row pitch.
// All rank groups on the same process-grid row own the same number of grid rows.
void exchange_halos(std::vector<double>& data, const Block& block,
                    MPI_Comm cart, int south, int north, int west, int east,
                    MPI_Datatype column_type) {
    const int nx = block.x.count;
    const int ny = block.y.count;
    const int pitch = block.pitch;

    // Only owned interior edge cells are sent. Corner ghost cells are NEVER sent.
    // Send top to north / receive bottom from south; then send bottom / receive top.
    MPI_Sendrecv(data.data() + local_index(1, ny, pitch), nx, MPI_DOUBLE,
                 north, kUpTag,
                 data.data() + local_index(1, 0, pitch), nx, MPI_DOUBLE,
                 south, kUpTag, cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(data.data() + local_index(1, 1, pitch), nx, MPI_DOUBLE,
                 south, kDownTag,
                 data.data() + local_index(1, ny + 1, pitch), nx, MPI_DOUBLE,
                 north, kDownTag, cart, MPI_STATUS_IGNORE);

    // Vertical columns are noncontiguous; use MPI_Type_vector, not manual packing.
    // First send right / receive left; then send left / receive right.
    MPI_Sendrecv(data.data() + local_index(nx, 1, pitch), 1, column_type,
                 east, kRightTag,
                 data.data() + local_index(0, 1, pitch), 1, column_type,
                 west, kRightTag, cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(data.data() + local_index(1, 1, pitch), 1, column_type,
                 west, kLeftTag,
                 data.data() + local_index(nx + 1, 1, pitch), 1, column_type,
                 east, kLeftTag, cart, MPI_STATUS_IGNORE);
}

void print_halo_debug(const Block& block, MPI_Comm cart,
                      int cart_rank, int P, const int coords[2], const char* stage,
                      const std::vector<double>& data) {
    for (int turn = 0; turn < P; ++turn) {
        MPI_Barrier(cart);
        if (cart_rank == turn) {
            std::cerr << "[block-halo-debug stage=" << stage << " rank=" << cart_rank
                      << " P=" << P << " cart_y=" << coords[0] << " cart_x=" << coords[1]
                      << " owned_x=" << block.x.first << ".." << (block.x.first + block.x.count - 1)
                      << " owned_y=" << block.y.first << ".." << (block.y.first + block.y.count - 1)
                      << "]\n";
            std::cerr << std::setprecision(9);
            for (int lj = 0; lj < block.y.count + 2; ++lj) {
                const char* row_type = lj == 0 ? "south-ghost" :
                                       lj == block.y.count + 1 ? "north-ghost" : "owned-row";
                std::cerr << "  local_j=" << lj << " global_j=" << (block.y.first + lj - 1)
                          << " " << row_type << " :";
                for (int li = 0; li < block.x.count + 2; ++li) {
                    std::cerr << ' ' << data[local_index(li, lj, block.pitch)];
                }
                std::cerr << '\n';
            }
            std::cerr << "  (first/last values in each row are west/east ghost cells;"
                      << " corner ghosts are unused)\n";
            std::cerr.flush();
        }
    }
    MPI_Barrier(cart);
}

std::vector<double> gather_field(const Config& config, const Block& block,
                                 const std::vector<double>& data, MPI_Comm cart,
                                 int cart_rank, int P, const int dims[2]) {
    // MPI_Gatherv uses int counts/displacements. All ranks independently validate.
    const std::size_t interior = static_cast<std::size_t>(config.N - 1);
    if (interior * interior > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("block MPI_Gatherv exceeds INT_MAX interior doubles");
    }
    const int local_count = block.x.count * block.y.count;
    std::vector<double> packed(static_cast<std::size_t>(local_count));
    for (int j = 0; j < block.y.count; ++j) {
        std::copy_n(data.data() + local_index(1, j + 1, block.pitch),
                    block.x.count, packed.data() + static_cast<std::size_t>(j) * block.x.count);
    }

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> gathered;
    if (cart_rank == 0) {
        counts.resize(static_cast<std::size_t>(P));
        displs.resize(static_cast<std::size_t>(P));
        int offset = 0;
        for (int rank = 0; rank < P; ++rank) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(cart, rank, 2, coords);
            const AxisPart part_y = split_axis(config.N, dims[0], coords[0]);
            const AxisPart part_x = split_axis(config.N, dims[1], coords[1]);
            const int count = part_x.count * part_y.count;
            counts[static_cast<std::size_t>(rank)] = count;
            displs[static_cast<std::size_t>(rank)] = offset;
            offset += count;
        }
        gathered.resize(static_cast<std::size_t>(offset));
    }

    MPI_Gatherv(packed.data(), local_count, MPI_DOUBLE,
                cart_rank == 0 ? gathered.data() : nullptr,
                cart_rank == 0 ? counts.data() : nullptr,
                cart_rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, cart);

    std::vector<double> field;
    if (cart_rank == 0) {
        const auto side = static_cast<std::size_t>(config.N) + 1U;
        field.assign(side * side, 0.0);
        // Restore physical boundary (including Case 2's hot corner precedence).
        for (int j = 0; j <= config.N; ++j) {
            for (int i = 0; i <= config.N; ++i) {
                if (i == 0 || i == config.N || j == 0 || j == config.N) {
                    field[index(i, j, config.N)] = boundary_value(config, j);
                }
            }
        }
        // A gathered 2D tile cannot be appended as if it were a full-width strip.
        // Unpack each tile to its original global coordinates (no arithmetic).
        for (int rank = 0; rank < P; ++rank) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(cart, rank, 2, coords);
            const AxisPart part_y = split_axis(config.N, dims[0], coords[0]);
            const AxisPart part_x = split_axis(config.N, dims[1], coords[1]);
            const std::size_t start = static_cast<std::size_t>(displs[static_cast<std::size_t>(rank)]);
            for (int local_j = 0; local_j < part_y.count; ++local_j) {
                const std::size_t source = start + static_cast<std::size_t>(local_j) * part_x.count;
                const auto destination = index(part_x.first, part_y.first + local_j, config.N);
                std::copy_n(gathered.data() + source, part_x.count, field.data() + destination);
            }
        }
    }
    return field;
}

} // namespace

StripResult run_blocks(const Config& config, int world_rank, int P) {
    int dims[2] = {0, 0};  // dims[0] = process rows (Py); dims[1] = columns (Px).
    MPI_Dims_create(P, 2, dims);
    if (dims[0] > config.N - 1 || dims[1] > config.N - 1) {
        throw std::invalid_argument("block decomposition requires at least one interior row and column per Cartesian process-grid coordinate");
    }
    int periods[2] = {0, 0}; // Explicitly NON-periodic plate.
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0 /* no rank reorder */, &cart);
    if (cart == MPI_COMM_NULL) {
        throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL");
    }
    int cart_rank = -1;
    MPI_Comm_rank(cart, &cart_rank);
    int coords[2] = {0, 0};
    MPI_Cart_coords(cart, cart_rank, 2, coords);
    (void)world_rank; // Rank reordering is disabled; world root remains cart root.

    int south = MPI_PROC_NULL, north = MPI_PROC_NULL;
    int west = MPI_PROC_NULL, east = MPI_PROC_NULL;
    MPI_Cart_shift(cart, 0, 1, &south, &north);
    MPI_Cart_shift(cart, 1, 1, &west, &east);

    Block block;
    block.y = split_axis(config.N, dims[0], coords[0]);
    block.x = split_axis(config.N, dims[1], coords[1]);
    block.pitch = block.x.count + 2;
    const std::size_t rows = static_cast<std::size_t>(block.y.count) + 2U;
    const std::size_t pitch = static_cast<std::size_t>(block.pitch);
    if (rows > std::numeric_limits<std::size_t>::max() / pitch) {
        throw std::invalid_argument("local block allocation overflows size_t");
    }
    const std::size_t local_nodes = rows * pitch;
    std::vector<double> old_values(local_nodes, 0.0);
    std::vector<double> new_values(local_nodes, 0.0);
    initialize_block(config, block, old_values);
    initialize_block(config, block, new_values);

    MPI_Datatype column_type = MPI_DATATYPE_NULL;
    MPI_Type_vector(block.y.count, 1, block.pitch, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    if (config.debug_halos) {
        exchange_halos(old_values, block, cart, south, north, west, east, column_type);
        print_halo_debug(block, cart, cart_rank, P, coords, "initial", old_values);
    }

    MPI_Barrier(cart);
    double comm_seconds = 0.0;
    double compute_seconds = 0.0;
    const double start = MPI_Wtime();
    for (std::uint64_t step = 0; step < config.steps; ++step) {
        const double comm_start = MPI_Wtime();
        exchange_halos(old_values, block, cart, south, north, west, east, column_type);
        comm_seconds += MPI_Wtime() - comm_start;

        const double compute_start = MPI_Wtime();
        for (int j = 1; j <= block.y.count; ++j) {
            for (int i = 1; i <= block.x.count; ++i) {
                const auto k = local_index(i, j, block.pitch);
                // EXACT Part 1/2 stencil order: left + right + bottom + top.
                const double sum = old_values[local_index(i - 1, j, block.pitch)]
                                 + old_values[local_index(i + 1, j, block.pitch)]
                                 + old_values[local_index(i, j - 1, block.pitch)]
                                 + old_values[local_index(i, j + 1, block.pitch)];
                new_values[k] = old_values[k] + config.r * (sum - 4.0 * old_values[k]);
            }
        }
        old_values.swap(new_values);
        compute_seconds += MPI_Wtime() - compute_start;
    }
    const double total_seconds = MPI_Wtime() - start;

    if (config.debug_halos) {
        exchange_halos(old_values, block, cart, south, north, west, east, column_type);
        print_halo_debug(block, cart, cart_rank, P, coords, "final", old_values);
    }

    const double local_times[3] = {total_seconds, compute_seconds, comm_seconds};
    double max_times[3] = {0.0, 0.0, 0.0};
    MPI_Reduce(local_times, max_times, 3, MPI_DOUBLE, MPI_MAX, 0, cart);

    StripResult result;
    if (cart_rank == 0) {
        result.total_seconds = max_times[0];
        result.compute_seconds = max_times[1];
        result.comm_seconds = max_times[2];
    }
    if (!config.output.empty() || config.test_case == 'V') {
        result.field = gather_field(config, block, old_values, cart, cart_rank, P, dims);
    }
    MPI_Type_free(&column_type);
    MPI_Comm_free(&cart);
    return result;
}

} // namespace heat
