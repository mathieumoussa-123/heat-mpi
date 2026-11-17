#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = -1, size = 0;
    char name[MPI_MAX_PROCESSOR_NAME];
    int name_length = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(name, &name_length);
    std::cout << "rank=" << rank << " size=" << size
              << " host=" << std::string(name, name_length) << '\n';
    MPI_Finalize();
    return 0;
}
