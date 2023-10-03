#ifndef E7C30600_505C_48FD_8242_1515363E5F0D
#define E7C30600_505C_48FD_8242_1515363E5F0D


#include "namespaces.h"


PE_BEGIN

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
	#define PE_GLOBAL __global__
	#define PE_HOST_DEVICE __host__ __device__
	#define PE_HOST __host__
	#define PE_DEVICE __device__
	#define PE_SHARED __shared__
	#define PE_CONST __constant__
#else 
	#define PE_GLOBAL 
	#define PE_HOST_DEVICE 
	#define PE_HOST 
	#define PE_DEVICE 
	#define PE_SHARED 
	#define PE_CONST 
#endif

#if defined(__CUDA_ARCH__)
	#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
		#define PE_UNROLL _Pragma("unroll")
		#define PE_NO_UNROLL _Pragma("unroll 1")
	#else
		#define PE_UNROLL #pragma unroll
		#define PE_NO_UNROLL #pragma unroll 1
	#endif
#else
	#define PE_UNROLL 
	#define PE_NO_UNROLL 
#endif


#define PE_VECTOR
#define PE_CAMERA
#define PE_TRANSFERTODEVICE
#define PE_VOXEL

PE_END

#endif /* E7C30600_505C_48FD_8242_1515363E5F0D */
