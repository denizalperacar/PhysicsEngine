#ifndef C0377D8E_933E_4A8A_8BB3_72F8B519BFC1
#define C0377D8E_933E_4A8A_8BB3_72F8B519BFC1


/*
@file:common.h
@author: Deniz A. ACAR
@brief: 
*/

#define DEBUG 0
#define VERBOSE 1

#include "namespaces.h"
#include "helpers.h"
#include "constants.h"

#include "pcg32.h"

#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <atomic>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <GL/glew.h>
#include <GL/glut.h> // import glut before glew3?
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "fmt/core.h"

#include <cstddef>
#include <type_traits>
#include <initializer_list>

#if defined(__CUDACC__)
#  include <cuda_fp16.h>
#endif

using namespace std::chrono;

PE_BEGIN

#define STRINGIFY(x) #x
#define CUDAERRORSUPRESS true

// utility functions

template <typename T>
PE_HOST_DEVICE inline T degrees_to_radians(T degrees) {
	return degrees * pi / 180.0f;
}

template <typename T>
PE_HOST_DEVICE inline T radians_to_degrees(T rad) {
	return rad * 180.f / pi;
}

constexpr uint32_t NUM_THREADS_1D = NUM_THREADS_128; 

template <typename T>
constexpr PE_HOST_DEVICE uint32_t n_blocks1d(T n_elements, uint32_t n_threads = NUM_THREADS_1D) {
	return (n_elements + n_threads - 1) / n_threads;
}

// common operations

template <typename T> PE_HOST_DEVICE T min(T a, T b) { return std::min(a, b); }
template <typename T> PE_HOST_DEVICE T max(T a, T b) { return std::max(a, b); }
template <typename T> PE_HOST_DEVICE T pow(T a, T b) { return std::pow(a, b); }
template <typename T> PE_HOST_DEVICE T atan2(T a, T b) { return std::atan2(a, b); }
template <typename T> PE_HOST_DEVICE T distance(T a, T b) { return std::abs(a - b); }
template <typename T> PE_HOST_DEVICE T copysign(T a, T b) { return std::copysign(a, b); }
template <typename T> PE_HOST_DEVICE T sign(T a) { return std::copysign((T)1, a); }
template <typename T> PE_HOST_DEVICE T floor(T a) { return std::floor(a); }
template <typename T> PE_HOST_DEVICE T ceil(T a) { return std::ceil(a); }
template <typename T> PE_HOST_DEVICE T abs(T a) { return std::abs(a); }
template <typename T> PE_HOST_DEVICE T sin(T a) { return std::sin(a); }
template <typename T> PE_HOST_DEVICE T asin(T a) { return std::asin(a); }
template <typename T> PE_HOST_DEVICE T cos(T a) { return std::cos(a); }
template <typename T> PE_HOST_DEVICE T acos(T a) { return std::acos(a); }
template <typename T> PE_HOST_DEVICE T tan(T a) { return std::tan(a); }
template <typename T> PE_HOST_DEVICE T atan(T a) { return std::atan(a); }
template <typename T> PE_HOST_DEVICE T tanh(T a) { return std::tanh(a); }
template <typename T> PE_HOST_DEVICE T sqrt(T a) { return std::sqrt(a); }
template <typename T> PE_HOST_DEVICE T exp(T a) { return std::exp(a); }
template <typename T> PE_HOST_DEVICE T log(T a) { return std::log(a); }
template <typename T> PE_HOST_DEVICE T exp2(T a) { return std::exp2(a); }
template <typename T> PE_HOST_DEVICE T log2(T a) { return std::log2(a); }
template <typename T> PE_HOST_DEVICE T log10(T a) { return std::log2(a) / std::log2(10); }
template <typename T> PE_HOST_DEVICE T clamp(T a, T b, T c) { return a < b ? b : (c < a ? c : a); }
template <typename T> PE_HOST_DEVICE T mix(T a, T b, T c) { return a * ((T)1 - c) + b * c; }
template <typename T> PE_HOST_DEVICE T lerp(T x, T a, T b) { return a * ((T)1 - x) + b * x; } // TODO add it to operations

template <typename T> PE_HOST_DEVICE T isfinite(T a) {
#if defined(__CUDA_ARCH__)
  return ::isfinite(a);
#else
  return std::isfinite(a);
#endif
}

inline PE_HOST_DEVICE float fma(float a, float b, float c) { return fmaf(a, b, c); }


// Image properties
constexpr float DEFAULT_IMAGE_ASPECT_RATIO = 16.0f / 9.0f;
constexpr uint32_t DEFAULT_IMAGE_WIDTH = 1440;
constexpr uint32_t DEFAULT_IMAGE_HEIGHT = static_cast<uint32_t>((float)DEFAULT_IMAGE_WIDTH / DEFAULT_IMAGE_ASPECT_RATIO);
constexpr uint32_t DEFAULT_NUMBER_OF_PIXELS = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT;


GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

PE_END


#endif /* C0377D8E_933E_4A8A_8BB3_72F8B519BFC1 */
