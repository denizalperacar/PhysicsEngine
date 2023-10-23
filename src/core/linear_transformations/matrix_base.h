#ifndef E92BA02D_8498_40BC_85B5_DAED223969ED
#define E92BA02D_8498_40BC_85B5_DAED223969ED

#include "../common/common.h"
#include "vector_base.h"


PE_BEGIN

template <typename T, uint32_t R, uint32_t C, typename... Ts>
using enable_if_all_vectors_t = std::enable_if_t<sizeof...(Ts) == R && conjunction<std::is_same<Ts, vector_t<T, C>>...>::value>;

// here M defines a number of col and N defines a number of columns
template <typename T, uint32_t R, uint32_t C>
struct matrix_t {

  // let default constructor create empty matrix
  matrix_t() = default;
  
  // fill the matrix diagonal with a single value
  PE_HOST_DEVICE matrix_t(T value) {
    PE_UNROLL
    for (uint32_t i = 0; i < R; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < C; j++) {
        col[ i ][ j ] = (i == j) ? value : (T) 0;
      }
    }
  }

  PE_HOST_DEVICE matrix_t(const matrix_t<T, R, C>& other) {
    PE_UNROLL
    for (uint32_t i = 0; i < C; i++) {
      col[ i ] = other.col[ i ];
    }
  }

  PE_HOST_DEVICE static constexpr matrix_t<T, R, C> identity() {
    return matrix_t<T, R, C>( (T)1 );
  }

  PE_HOST_DEVICE static constexpr matrix_t<T, R, C> zero() {
    return matrix_t<T, R, C>( (T)0 );
  }

  template <size_t A>
  PE_HOST_DEVICE matrix_t<T, R, C>(const vector_t<T, C, A>& vec) {
    PE_UNROLL
    for (uint32_t i = 0; i < R; i++) {
      col[ i ] = vec;
    }
  }

  template<typename... Ts, typename = enable_if_size_and_type_match_t<R*C, T, Ts...>>
  PE_HOST_DEVICE matrix_t(Ts... coeffs) : data_{ coeffs... } {}

  PE_HOST_DEVICE matrix_t(const T* d) {
    PE_UNROLL
    for (uint32_t i = 0; i < R; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < C; j++) {
        col[ i ][ j ] = *(d++);
      }
    }
  }

  template <size_t A>
  PE_HOST_DEVICE matrix_t<T, R, C>(
      const vector_t<T, R, A>& vec1, 
      const vector_t<T, R, A>& vec2 ) {
    static_assert(C == 2, "matrix_t<T, R, C>(const vector_t<T, C, A>& vec1, const vector_t<T, C, A>& vec2) requires C == 2");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
  }

    template <size_t A>
  PE_HOST_DEVICE matrix_t<T, R, C>(
      const vector_t<T, R, A>& vec1, 
      const vector_t<T, R, A>& vec2, 
      const vector_t<T, R, A>& vec3 ) {
    static_assert(C == 3, "matrix_t<T, R, C>(const vector_t<T, C, A>& vec1, const vector_t<T, C, A>& vec2) requires C == 3");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
    col[ 2 ] = vec3;
  }

    template <size_t A>
  PE_HOST_DEVICE matrix_t<T, R, C>(
      const vector_t<T, R, A>& vec1, 
      const vector_t<T, R, A>& vec2,
      const vector_t<T, R, A>& vec3, 
      const vector_t<T, R, A>& vec4 ) {
    static_assert(C == 4, "matrix_t<T, R, C>(const vector_t<T, C, A>& vec1, const vector_t<T, C, A>& vec2) requires C == 4");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
    col[ 2 ] = vec3;
    col[ 3 ] = vec4;
  }

  // stacks vectors one by one on top of each other
  template<typename... Ts, typename = enable_if_all_vectors_t<T, R, C, Ts...>>
  PE_HOST_DEVICE matrix_t<T, R, C> stack(Ts... vectors) {
    col = { vectors()... };
    return *this;
  }

  template <typename W, uint32_t U, uint32_t V>
  PE_HOST_DEVICE matrix_t(const matrix_t<W, U, V>& other) {
    PE_UNROLL
    for (uint32_t i = 0; i < R; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < C; j++) {
        col[ i ][ j ] = (i < U && j < V) ? other.col[ i ][ j ] : (T) 0;
      }
    }
  }


  // matrix vector multiplication
  PE_HOST_DEVICE vector_t<T, R> operator*(const vector_t<T, C>& vec) const {
    vector_t<T, R> result((T)0.0);
    PE_UNROLL
    for (uint32_t i = 0; i < C; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < R; j++) {
        result[j] += col[i][j] * vec[i];
      }
    }
    return result;
  }
  
  // matrix matrix multiplication
  template <uint32_t P>
  PE_HOST_DEVICE matrix_t<T, R, P> operator*(const matrix_t<T, C, P> & other) {
    matrix_t<T, R, P> result(0);
    PE_UNROLL
    for (uint32_t i = 0; i < P; i++) {
      result.col[ i ] = (*this) * other[ i ];
    }
  }

  // return a column
  PE_HOST_DEVICE vector_t<T, R>& operator[](uint32_t i) {
    return col[ i ];
  }

  PE_HOST_DEVICE const vector_t<T, R>& operator[](uint32_t i) const {
    return col[ i ];
  }

	PE_HOST_DEVICE vector_t<T, R>& at(uint32_t idx) { return col[idx]; }
	PE_HOST_DEVICE vector_t<T, R> at(uint32_t idx) const { return col[idx]; }

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
    vector_t<T, R> col[ C ];
    T data_[ R * C ];
  };

};

// matrix operation definitions

// outer product
template<typename T, uint32_t R, uint32_t C>
PE_HOST_DEVICE matrix_t<T, R, C> outer(
    const vector_t<T, R> &vec1, 
    const vector_t<T, C> &vec2 ) {
  
  matrix_t<T, R, C> result;
  PE_UNROLL
  for (uint32_t i = 0; i < C; i++) {
    PE_UNROLL
    for (uint32_t j = 0; j < R; j++) {
      result[ i ][ j ] = vec1[ j ] * vec2[ i ];
    }
  }
  return result;
}

#define TMAT matrix_t<T, R, C>
#define TVECR vector_t<T, R>
#define TVECC vector_t<T, C>

// define element wise operations
#define ELEMENTWISE_OP(operation, input_type, output_type, expression, ...) \
template <typename T, uint32_t R, uint32_t C> \
PE_HOST_DEVICE input_type operation(__VA_ARGS__) { \
  output_type result; \
  PE_UNROLL \
  for (uint32_t i = 0; i < C; i++) { \
    PE_UNROLL \
    for (uint32_t j = 0; j < R; j++) { \
      result[i][j] = expression; \
    } \
  } \
  return result; \
}




ELEMENTWISE_OP(operator+, TMAT, TMAT, a[i][j] + b[i][j] ,const TMAT &a, const TMAT &b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, a + b[i][j], T a, const TMAT &b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, b + a[i][j], const TMAT &a, T b)




#undef TMAT
#undef ELEMENTWISE_OP


// helper function to print matrix
template <typename T, uint32_t R, uint32_t C>
void print(const matrix_t<T, R, C> &mat ) {
  PE_UNROLL
  for (uint32_t i = 0; i < R; i++) {
    PE_UNROLL
    for (uint32_t j = 0; j < C; j++) {
      std::cout << mat[ j ][ i ] << " ";
    }
    std::cout << std::endl;
  }
}

PE_END

#endif /* E92BA02D_8498_40BC_85B5_DAED223969ED */
