cmake_minimum_required(VERSION 3.14)
project(MiniNeuron VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

file(GLOB SOURCES MiniNeuron/src/*.cpp)

add_library(MiniNeuron STATIC ${SOURCES})
target_include_directories(MiniNeuron PUBLIC MiniNeuron/includes)

if(OpenMP_CXX_FOUND)
    target_link_libraries(MiniNeuron PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(MiniNeuron PUBLIC MININEURON_OPENMP)
endif()

option(MININEURON_BUILD_DEMOS "Build demo programs" ON)

if(MININEURON_BUILD_DEMOS)
    add_executable(demo_xor demos/xor-demo/src/main.cpp)
    target_link_libraries(demo_xor MiniNeuron)
    target_include_directories(demo_xor PRIVATE MiniNeuron/includes)
endif()
