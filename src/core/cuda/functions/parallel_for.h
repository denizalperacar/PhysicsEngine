#ifndef C35A3755_C94A_4637_9D77_A9D3E857C907
#define C35A3755_C94A_4637_9D77_A9D3E857C907

#include "common.h"


PE_BEGIN


#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))

template <typename K,typename T, typename ...Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args)  {

  if (n_elements == (T)0) return;
  kernel<<<n_blocks1d(n_elements), NUM_THREADS_1D, shmem_size, stream>>> (n_elements, args...);
}

template <typename T>
PE_GLOBAL void parallel_for_kernel(const size_t num_elements, T function) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_elements) {
    return;
  }
  
  function(i);
}

// general form a parallel for loop T can be any class or function 
// overload operator() for class T.
template <typename T>
void parallel_for_kernel(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, T&& function) {
  if (n_elements == 0) return;
  parallel_for_kernel<T><<<n_blocks1d(n_elements), NUM_THREADS_1D, shmem_size, stream>>>(n_elements, function);
}

template <typename T>
void parallel_for_kernel(cudaStream_t stream, size_t n_elements, T&& function) {
  parallel_for_kernel(0, stream, n_elements, std::forward<T>(function));
}

template <typename T>
void parallel_for_kernel(size_t n_elements, T&& function) {
  parallel_for_kernel(nullptr, n_elements, std::forward<T>(function));
}

#endif

PE_END

#endif /* C35A3755_C94A_4637_9D77_A9D3E857C907 */
