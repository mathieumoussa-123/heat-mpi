// MPI ping-pong benchmark for the Parallel 2D Heat Diffusion project.
//
// Run with EXACTLY two MPI processes. Rank 0 sends a message to rank 1,
// which immediately sends it back. Rank 0 times complete round trips.
//
// The FIRST FOUR CSV columns are compatible with scripts/fit_pingpong.py.
// Extra columns summarize variability across independently timed batches.
// Transport selection (shared memory vs TCP) is set with mpirun MCA flags,
// NOT inside this source file.

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kTag = 7;
constexpr std::size_t kMaxBytes = 16U * 1024U * 1024U;

struct Options {
    int batches = 3;
    int warmups = 5;
};

// All ranks receive the same command line from mpirun.
bool parse_options(int argc, char** argv, Options& opt, bool& help) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            help = true;
            return true;
        }
        if (arg != "--batches" && arg != "--warmups") {
            return false;
        }
        if (i + 1 >= argc) {
            return false;
        }
        const std::string number(argv[++i]);
        if (number.empty() || number.find_first_not_of("0123456789") != std::string::npos) {
            return false;
        }
        const long value = std::strtol(number.c_str(), nullptr, 10);
        if (value < 0 || value > 100000) {
            return false;
        }
        if (arg == "--batches") {
            if (value == 0) return false;
            opt.batches = static_cast<int>(value);
        } else {
            opt.warmups = static_cast<int>(value);
        }
    }
    return true;
}

int repetitions_for(std::size_t bytes) {
    if (bytes <= 4096U) return 500;
    if (bytes <= (1U << 20)) return 100;
    return 10;
}

// The matching order eliminates blocking-send deadlock:
// rank 0 SEND -> RECV, rank 1 RECV -> SEND.
void round_trip(int rank, unsigned char* buffer, int count) {
    if (rank == 0) {
        MPI_Send(buffer, count, MPI_BYTE, 1, kTag, MPI_COMM_WORLD);
        MPI_Recv(buffer, count, MPI_BYTE, 1, kTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(buffer, count, MPI_BYTE, 0, kTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(buffer, count, MPI_BYTE, 0, kTag, MPI_COMM_WORLD);
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Error: mpi_pingpong requires exactly two MPI ranks.\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    Options opt;
    bool help = false;
    if (!parse_options(argc, argv, opt, help)) {
        if (rank == 0) {
            std::cerr << "Usage: mpi_pingpong [--batches POSITIVE_INTEGER] "
                         "[--warmups NONNEGATIVE_INTEGER]\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (help) {
        if (rank == 0) {
            std::cout << "Usage: mpi_pingpong [--batches 3] [--warmups 5]\n"
                      << "Exactly 2 ranks. CSV is printed by rank 0.\n";
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    if (rank == 0) {
        // Keep original columns intact for the existing fitting script.
        std::cout << "bytes,repetitions,mean_rtt_s,mean_oneway_s,"
                     "median_oneway_s,min_oneway_s,max_oneway_s,batches\n";
        std::cout << std::setprecision(12);
    }

    for (std::size_t bytes = 8; bytes <= kMaxBytes; bytes *= 2) {
        std::vector<unsigned char> message(bytes, 42);
        const int count = static_cast<int>(bytes);  // 16 MiB fits MPI's int count.
        const int repetitions = repetitions_for(bytes);

        // Warm up both ranks before starting the timing measurements.
        for (int w = 0; w < opt.warmups; ++w) {
            round_trip(rank, message.data(), count);
        }

        std::vector<double> batch_mean_rtt;
        if (rank == 0) batch_mean_rtt.reserve(static_cast<std::size_t>(opt.batches));

        for (int batch = 0; batch < opt.batches; ++batch) {
            MPI_Barrier(MPI_COMM_WORLD);
            // Only rank 0's clock is relevant; ranks don't compare clocks.
            const double start = rank == 0 ? MPI_Wtime() : 0.0;

            for (int trial = 0; trial < repetitions; ++trial) {
                round_trip(rank, message.data(), count);
            }

            if (rank == 0) {
                const double elapsed = MPI_Wtime() - start;
                batch_mean_rtt.push_back(elapsed / repetitions);
            }
        }

        if (rank == 0) {
            const double sum = std::accumulate(batch_mean_rtt.begin(),
                                               batch_mean_rtt.end(), 0.0);
            const double mean_rtt_s = sum / opt.batches;
            std::sort(batch_mean_rtt.begin(), batch_mean_rtt.end());

            const std::size_t n = batch_mean_rtt.size();
            const double median_rtt_s = (n % 2 == 1)
                ? batch_mean_rtt[n / 2]
                : 0.5 * (batch_mean_rtt[n / 2 - 1] + batch_mean_rtt[n / 2]);

            std::cout << bytes << ',' << repetitions << ','
                      << mean_rtt_s << ',' << mean_rtt_s / 2.0 << ','
                      << median_rtt_s / 2.0 << ','
                      << batch_mean_rtt.front() / 2.0 << ','
                      << batch_mean_rtt.back() / 2.0 << ','
                      << opt.batches << '\n';
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
