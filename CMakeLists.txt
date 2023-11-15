cmake_minimum_required(VERSION 3.26)
project(DistributedComputing)

set(CMAKE_CXX_STANDARD 20)

add_executable(DistributedComputing main.cpp
        ThreeDiagMatrices.cpp
        ThreeDiagMatrices.h
)
