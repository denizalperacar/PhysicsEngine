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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "fmt/core.h"

using namespace std::chrono;

PE_BEGIN


#define cudaErr(x) fmt::println("CUDA ERR CHECK:/n    {} {}", x, cudaGetErrorString(cudaGetLastError()));
#define STRINGIFY(x) #x

// utility functions

PE_HOST_DEVICE inline float degrees_to_radians(float degrees) {
	return degrees * pi / 180.0f;
}

PE_END


#endif /* C0377D8E_933E_4A8A_8BB3_72F8B519BFC1 */
