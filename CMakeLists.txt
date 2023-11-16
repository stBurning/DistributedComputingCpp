cmake_minimum_required(VERSION 3.26)
project(DistributedComputing)

set(CMAKE_CXX_STANDARD 20)
# find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
add_executable(DistributedComputing main.cpp
        ThreeDiagMatrices.cpp
        ThreeDiagMatrices.h
)

# target_link_libraries(DistributedComputing PRIVATE MPI::MPI_CXX)
