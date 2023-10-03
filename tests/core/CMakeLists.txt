cmake_minimum_required(VERSION 3.16)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(physics_engine LANGUAGES CXX CUDA)

option(USE_CUDA "Use CUDA" ON)

set(fmt_DIR "${CMAKE_SOURCE_DIR}/../../packages/fmt/")
find_package(fmt REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/../../src/")
set(CORE_DIR "${SOURCE_DIR}/core/")
set(COMMON_DIR "${CORE_DIR}/common/")
set(VECTOR_DIR "${CORE_DIR}/vector/")

file(GLOB cpu_source_files 
  "${CMAKE_SOURCE_DIR}/main.cpp"
  "${SOURCE_DIR}/*.cpp" 
  "${VECTOR_DIR}/*.h"
  "${VECTOR_DIR}/*.cpp"
)

file(GLOB gpu_source_files 
  "${SOURCE_DIR}/*.cu" 
  "${VECTOR_DIR}/*.cu"
  "${VECTOR_DIR}/*.cuh"
)

if(USE_CUDA)
  add_executable(physics_engine ${cpu_source_files} ${gpu_source_files})
  target_compile_definitions(physics_engine PRIVATE USE_CUDA)
  target_link_libraries(physics_engine fmt::fmt)
else()
  add_executable(physics_engine ${cpu_source_files})
  target_link_libraries(physics_engine fmt::fmt)
endif()

target_include_directories(physics_engine PRIVATE "/usr/local/cuda/include/" "${COMMON_DIR}")  

# Specify the precompiled header file

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-but-set-variable" )

target_link_libraries(physics_engine fmt::fmt)
message(STATUS "USE_CUDA: ${USE_CUDA}")

target_precompile_headers(physics_engine PRIVATE "${COMMON_DIR}/pch.h")

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
