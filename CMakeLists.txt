cmake_minimum_required(VERSION 3.16)
project(HeatMPI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Part 0 programs still require MPI, even though Part 1 'heat' runs serially.
find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(mpi_hello src/mpi_hello.cpp)
add_executable(mpi_pingpong src/mpi_pingpong.cpp)
foreach(target mpi_hello mpi_pingpong)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
endforeach()

# Keep the serial solver independent of MPI so it remains the reference for Parts 2/3.
add_executable(heat src/heat_main.cpp src/heat_serial.cpp src/heat_io.cpp)
foreach(target mpi_hello mpi_pingpong heat)
    target_compile_options(${target} PRIVATE -O2 -ffp-contract=off -Wall -Wextra -Wpedantic)
endforeach()

enable_testing()
add_test(NAME heat_verification_smoke
         COMMAND heat --decomp serial --case V --N 16 --r 0.2 --T 0.05)
add_test(NAME heat_case1_smoke
         COMMAND heat --decomp serial --case 1 --N 8 --r 0.2 --steps 10)
add_test(NAME heat_case2_smoke
         COMMAND heat --decomp serial --case 2 --N 8 --r 0.2 --steps 10)
