#ifndef E92BA02D_8498_40BC_85B5_DAED223969ED
#define E92BA02D_8498_40BC_85B5_DAED223969ED

#include "../common/common.h"
#include "vector_base.h"


PE_BEGIN

template <typename T, uint32_t R, uint32_t C, typename... Ts>
using enable_if_all_vectors_t = std::enable_if_t<sizeof...(Ts) == R && conjunction<std::is_same<Ts, vector_t<T, C>>...>::value>;

// here M defines a number of col and N defines a number of columns
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
        col[ i ][ j ] = (i == j) ? value : (T) 0;
      }
    }
  }

  PE_HOST_DEVICE matrix_t(const matrix_t<T, M, N>& other) {
    PE_UNROLL
    for (uint32_t i = 0; i < N; i++) {
      col[ i ] = other.col[ i ];
    }
  }

  PE_HOST_DEVICE static constexpr matrix_t<T, M, N> identity() {
    return matrix_t<T, M, N>( (T)1 );
  }

  PE_HOST_DEVICE static constexpr matrix_t<T, M, N> zero() {
    return matrix_t<T, M, N>( (T)0 );
  }

  template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(const vector_t<T, N, A>& vec) {
    PE_UNROLL
    for (uint32_t i = 0; i < M; i++) {
      col[ i ] = vec;
    }
  }

  template<typename... Ts, typename = enable_if_size_and_type_match_t<M*N, T, Ts...>>
  PE_HOST_DEVICE matrix_t(Ts... coeffs) : data_{ coeffs... } {}

  PE_HOST_DEVICE matrix_t(const T* d) {
    PE_UNROLL
    for (uint32_t i = 0; i < M; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < N; j++) {
        col[ i ][ j ] = *(d++);
      }
    }
  }

  template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(
      const vector_t<T, M, A>& vec1, 
      const vector_t<T, M, A>& vec2 ) {
    static_assert(N == 2, "matrix_t<T, M, N>(const vector_t<T, N, A>& vec1, const vector_t<T, N, A>& vec2) requires N == 2");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
  }

    template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(
      const vector_t<T, M, A>& vec1, 
      const vector_t<T, M, A>& vec2, 
      const vector_t<T, M, A>& vec3 ) {
    static_assert(N == 3, "matrix_t<T, M, N>(const vector_t<T, N, A>& vec1, const vector_t<T, N, A>& vec2) requires N == 3");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
    col[ 2 ] = vec3;
  }

    template <size_t A>
  PE_HOST_DEVICE matrix_t<T, M, N>(
      const vector_t<T, M, A>& vec1, 
      const vector_t<T, M, A>& vec2,
      const vector_t<T, M, A>& vec3, 
      const vector_t<T, M, A>& vec4 ) {
    static_assert(N == 4, "matrix_t<T, M, N>(const vector_t<T, N, A>& vec1, const vector_t<T, N, A>& vec2) requires N == 4");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
    col[ 2 ] = vec3;
    col[ 3 ] = vec4;
  }

  // stacks vectors one by one on top of each other
  template<typename... Ts, typename = enable_if_all_vectors_t<T, M, N, Ts...>>
  PE_HOST_DEVICE matrix_t<T, M, N> stack(Ts... vectors) {
    col = { vectors()... };
    return *this;
  }

  template <typename W, uint32_t U, uint32_t V>
  PE_HOST_DEVICE matrix_t(const matrix_t<W, U, V>& other) {
    PE_UNROLL
    for (uint32_t i = 0; i < M; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < N; j++) {
        col[ i ][ j ] = (i < U && j < V) ? other.col[ i ][ j ] : (T) 0;
      }
    }
  }


  // matrix vector multiplication
  PE_HOST_DEVICE vector_t<T, M> operator*(const vector_t<T, N>& vec) const {
    vector_t<T, M> result((T)0.0);
    PE_UNROLL
    for (uint32_t i = 0; i < N; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < M; j++) {
        result[j] += col[i][j] * vec[i];
      }
    }
    return result;
  }
  
  // matrix matrix multiplication
  template <uint32_t P>
  PE_HOST_DEVICE matrix_t<T, M, P> operator*(const matrix_t<T, N, P> & other) {
    matrix_t<T, M, P> result(0);
    PE_UNROLL
    for (uint32_t i = 0; i < P; i++) {
      result.col[ i ] = (*this) * other[ i ];
    }
  }

  // return a column
  PE_HOST_DEVICE vector_t<T, M>& operator[](uint32_t i) {
    return col[ i ];
  }

  PE_HOST_DEVICE const vector_t<T, M>& operator[](uint32_t i) const {
    return col[ i ];
  }

	PE_HOST_DEVICE vector_t<T, M>& at(uint32_t idx) { return col[idx]; }
	PE_HOST_DEVICE vector_t<T, M> at(uint32_t idx) const { return col[idx]; }

  PE_HOST_DEVICE T* data() {
    return data_;
  }

  PE_HOST_DEVICE const T* data() const {
    return data_;
  }

  PE_HOST_DEVICE T& operator()(uint32_t i, uint32_t j) {
    return col[ i ][ j ];
  }

  PE_HOST_DEVICE const T& operator()(uint32_t i, uint32_t j) const {
    return col[ i ][ j ];
  }

  union {
    vector_t<T, M> col[ N ];
    T data_[ M * N ];
  };

};














PE_END

#endif /* E92BA02D_8498_40BC_85B5_DAED223969ED */
