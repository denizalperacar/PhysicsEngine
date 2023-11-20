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



// Image properties
constexpr float DEFAULT_IMAGE_ASPECT_RATIO = 4.0f / 3.0f;
constexpr uint32_t DEFAULT_IMAGE_WIDTH = 800;
constexpr uint32_t DEFAULT_IMAGE_HEIGHT = static_cast<uint32_t>((float)DEFAULT_IMAGE_WIDTH / DEFAULT_IMAGE_ASPECT_RATIO);
constexpr uint32_t DEFAULT_NUMBER_OF_PIXELS = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT;


GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

PE_END


#endif /* C0377D8E_933E_4A8A_8BB3_72F8B519BFC1 */
