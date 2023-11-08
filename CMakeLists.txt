cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_BUILD_TYPE Release)
project(QCU LANGUAGES C CXX CUDA)

include_directories(include)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
add_library(qcu SHARED src/dslash.cu   src/qcu_wilson_dslash.cu)


find_package(MPI REQUIRED)
target_link_libraries(qcu ${MPI_LIBRARIES})
target_compile_definitions(qcu PRIVATE MPI_ENABLED)