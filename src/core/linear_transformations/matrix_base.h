#ifndef E92BA02D_8498_40BC_85B5_DAED223969ED
#define E92BA02D_8498_40BC_85B5_DAED223969ED

#include "../common/common.h"
#include "vector_base.h"


PE_BEGIN

template <typename T, uint32_t R, uint32_t C, typename... Ts>
using enable_if_all_vectors_t = std::enable_if_t<sizeof...(Ts) == R && conjunction<std::is_same<Ts, vector_t<T, C>>...>::value>;

template <typename T, uint32_t M, uint32_t N>
struct matrix_t {

  // let default constructor create empty matrix
  matrix_t() = default;
  
  // fill the matrix diagonal with a single value
  PE_HOST_DEVICE matrix_t(T value) {
    PE_UNROLL
    for (uint32_t i = 0; i < M; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < N; j++) {
        rows[ i ][ j ] = (i == j) ? value : (T) 0;
      }
    }
  }

  PE_HOST_DEVICE matrix_t(const matrix_t<T, M, N>& other) {
    PE_UNROLL
    for (uint32_t i = 0; i < N; i++) {
      rows[ i ] = other.rows[ i ];
    }
  }

  PE_HOST_DEVICE static constexpr matrix_t<T, M, N> identity() {
    return matrix_t<T, M, N>( (T)1 );
  }

  PE_HOST_DEVICE static constexpr matrix_t<T, M, N> zero() {
    return matrix_t<T, M, N>( (T)0 );
  }

  PE_HOST_DEVICE static constexpr matrix_t<T, M, N> eye() {
    return matrix_t<T, M, N>( (T)1 );
  }

  template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(const vector_t<T, N, A>& vec) {
    PE_UNROLL
    for (uint32_t i = 0; i < M; i++) {
      rows[ i ] = vec;
    }
  }

  template<typename... Ts, typename = enable_if_size_and_type_match_t<M*N, T, Ts...>>
  PE_HOST_DEVICE matrix_t(Ts... coeffs) : data{ coeffs... } {}

  PE_HOST_DEVICE matrix_t(const T* d) {
    PE_UNROLL
    for (uint32_t i = 0; i < M; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < N; j++) {
        rows[ i ][ j ] = *(d++);
      }
    }
  }

  template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(
      const vector_t<T, N, A>& vec1, 
      const vector_t<T, N, A>& vec2 ) {
    static_assert(M == 2, "matrix_t<T, M, N>(const vector_t<T, N, A>& vec1, const vector_t<T, N, A>& vec2) requires M == 2");
    rows[ 0 ] = vec1; 
    rows[ 1 ] = vec2;
  }

    template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(
      const vector_t<T, N, A>& vec1, 
      const vector_t<T, N, A>& vec2, 
      const vector_t<T, N, A>& vec3 ) {
    static_assert(M == 3, "matrix_t<T, M, N>(const vector_t<T, N, A>& vec1, const vector_t<T, N, A>& vec2) requires M == 3");
    rows[ 0 ] = vec1; 
    rows[ 1 ] = vec2;
    rows[ 2 ] = vec3;
  }

    template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(
      const vector_t<T, N, A>& vec1, 
      const vector_t<T, N, A>& vec2,
      const vector_t<T, N, A>& vec3, 
      const vector_t<T, N, A>& vec4 ) {
    static_assert(M == 4, "matrix_t<T, M, N>(const vector_t<T, N, A>& vec1, const vector_t<T, N, A>& vec2) requires M == 4");
    rows[ 0 ] = vec1; 
    rows[ 1 ] = vec2;
    rows[ 2 ] = vec3;
    rows[ 3 ] = vec4;
  }

  


  union {
    vector_t<T, M> rows[ N ];
    T data[ M * N ];
  };

};

PE_END

#endif /* E92BA02D_8498_40BC_85B5_DAED223969ED */
