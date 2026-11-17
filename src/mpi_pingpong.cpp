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
#include <cstdint>
#include <limits>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kTag = 7;
constexpr std::size_t kMinBytes = 8U;
constexpr std::size_t kMaxBytes = 16U * 1024U * 1024U;

struct Options {
    int batches = 3;
    int warmups = 5;
    std::size_t start_bytes = kMinBytes;
    std::size_t end_bytes = kMaxBytes;
};

// Parse a nonnegative integer without silently accepting overflow.
bool parse_uint(const std::string& number, std::uint64_t& value) {
    if (number.empty()) return false;
    value = 0;
    for (const char c : number) {
        if (c < '0' || c > '9') return false;
        const unsigned digit = static_cast<unsigned>(c - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
            return false;
        }
        value = value * 10U + digit;
    }
    return true;
}

bool valid_message_size(std::uint64_t bytes) {
    return bytes >= kMinBytes && bytes <= kMaxBytes &&
           (bytes & (bytes - 1U)) == 0U;
}

// The same CLI options are supplied to both ranks by mpirun.
bool parse_options(int argc, char** argv, Options& opt, bool& help) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            help = true;
            return true;
        }
        if (arg != "--batches" && arg != "--warmups" &&
            arg != "--start-bytes" && arg != "--end-bytes") {
            return false;
        }
        if (i + 1 >= argc) return false;
        std::uint64_t value = 0;
        if (!parse_uint(argv[++i], value)) return false;
        if (arg == "--batches") {
            if (value == 0 || value > 100000) return false;
            opt.batches = static_cast<int>(value);
        } else if (arg == "--warmups") {
            if (value > 100000) return false;
            opt.warmups = static_cast<int>(value);
        } else if (arg == "--start-bytes") {
            if (!valid_message_size(value)) return false;
            opt.start_bytes = static_cast<std::size_t>(value);
        } else {
            if (!valid_message_size(value)) return false;
            opt.end_bytes = static_cast<std::size_t>(value);
        }
    }
    return opt.start_bytes <= opt.end_bytes;
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
                         "[--warmups NONNEGATIVE_INTEGER] "
                         "[--start-bytes POWER_OF_TWO] "
                         "[--end-bytes POWER_OF_TWO]\n"
                      << "Message sizes must be powers of two between 8 and "
                      << kMaxBytes << " bytes, start <= end.\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (help) {
        if (rank == 0) {
            std::cout << "Usage: mpi_pingpong [--batches 3] [--warmups 5] "
                         "[--start-bytes 8] [--end-bytes 16777216]\n"
                      << "Exactly 2 ranks. CSV is printed by rank 0.\n"
                      << "Sizes must be powers of two in [8, 16777216].\n";
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    if (rank == 0) {
        // Keep original columns intact for the existing fitting script.
        std::cout << "bytes,repetitions,mean_rtt_s,mean_oneway_s,"
                     "median_oneway_s,min_oneway_s,max_oneway_s,batches\n";
        std::cout << std::setprecision(12);
        std::cout.flush();  // Complete header is available if the run is interrupted.
    }

    for (std::size_t bytes = opt.start_bytes; bytes <= opt.end_bytes; bytes *= 2) {
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
            std::cout.flush();  // Persist each finished size for resumable runs.
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
