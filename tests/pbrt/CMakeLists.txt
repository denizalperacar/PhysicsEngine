cmake_minimum_required(VERSION 3.16)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(physics_engine LANGUAGES CXX CUDA)

set(ROOT_DIR "${CMAKE_SOURCE_DIR}/../../")
include("${ROOT_DIR}cmake.inc")
