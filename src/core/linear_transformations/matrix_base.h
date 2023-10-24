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
  PE_HOST_DEVICE matrix_t<T, R, P> operator*(const matrix_t<T, C, P>& other) {
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

  PE_HOST_DEVICE T& operator()(uint32_t r, uint32_t c) {
    return col[ c ][ r ];
  }

  PE_HOST_DEVICE const T& operator()(uint32_t r, uint32_t c) const {
    return col[ c ][ r ];
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
    const vector_t<T, R>& vec1, 
    const vector_t<T, C>& vec2 ) {
  
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

// operator overloads
ELEMENTWISE_OP(operator+, TMAT, TMAT, a[i][j] + b[i][j], const TMAT& a, const TMAT& b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, a + b[i][j], T a, const TMAT& b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, b + a[i][j], const TMAT& a, T b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, b[i] + a[i][j], const TMAT& a, const TVECC& b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, b[i] + b[i][j], const TVECC& a, const TMAT& b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, b[j] + a[i][j], const TMAT& a, const TVECR& b)
ELEMENTWISE_OP(operator+, TMAT, TMAT, b[j] + b[i][j], const TVECR& a, const TMAT& b)

ELEMENTWISE_OP(operator/, TMAT, TMAT, a[i][j] / b[i][j], const TMAT& a, const TMAT& b)
ELEMENTWISE_OP(operator/, TMAT, TMAT, a / b[i][j], T a, const TMAT& b)
ELEMENTWISE_OP(operator/, TMAT, TMAT, b / a[i][j], const TMAT& a, T b)
ELEMENTWISE_OP(operator/, TMAT, TMAT, b[i] / a[i][j], const TMAT& a, const TVECC& b)
ELEMENTWISE_OP(operator/, TMAT, TMAT, b[i] / b[i][j], const TVECC& a, const TMAT& b)
ELEMENTWISE_OP(operator/, TMAT, TMAT, b[j] / a[i][j], const TMAT& a, const TVECR& b)
ELEMENTWISE_OP(operator/, TMAT, TMAT, b[j] / b[i][j], const TVECR& a, const TMAT& b)

ELEMENTWISE_OP(operator-, TMAT, TMAT, a[i][j] - b[i][j], const TMAT& a, const TMAT& b)
ELEMENTWISE_OP(operator-, TMAT, TMAT, a - b[i][j], T a, const TMAT& b)
ELEMENTWISE_OP(operator-, TMAT, TMAT, b - a[i][j], const TMAT& a, T b)
ELEMENTWISE_OP(operator-, TMAT, TMAT, b[i] - a[i][j], const TMAT& a, const TVECC& b)
ELEMENTWISE_OP(operator-, TMAT, TMAT, a[i] - b[i][j], const TVECC& a, const TMAT& b)
ELEMENTWISE_OP(operator-, TMAT, TMAT, b[j] - a[i][j], const TMAT& a, const TVECR& b)
ELEMENTWISE_OP(operator-, TMAT, TMAT, a[j] - b[i][j], const TVECR& a, const TMAT& b)
ELEMENTWISE_OP(operator-, TMAT, TMAT, -a[i][j], const TMAT& a)

// element wise function operations
#define ELEMENTWISE_FOP(op) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i][j]), const TMAT& a, const TMAT& b) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b[i][j]), T a, const TMAT& b) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(b, a[i][j]), const TMAT& a, T b) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(b[i], a[i][j]), const TMAT& a, const TVECC& b) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(b[i], b[i][j]), const TVECC& a, const TMAT& b) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(b[j], a[i][j]), const TMAT& a, const TVECR& b) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(b[j], b[i][j]), const TVECR& a, const TMAT& b)

ELEMENTWISE_FOP(min)
ELEMENTWISE_FOP(max)
ELEMENTWISE_FOP(pow)
ELEMENTWISE_FOP(distance)
ELEMENTWISE_FOP(copysign)
ELEMENTWISE_FOP(mul)

#undef ELEMENTWISE_FOP

#define ELEMENTWISE_SOP(op) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j]), const TMAT& a)

ELEMENTWISE_SOP(sign)
ELEMENTWISE_SOP(floor)
ELEMENTWISE_SOP(ceil)
ELEMENTWISE_SOP(abs)
ELEMENTWISE_SOP(sin)
ELEMENTWISE_SOP(asin)
ELEMENTWISE_SOP(cos)
ELEMENTWISE_SOP(acos)
ELEMENTWISE_SOP(tan)
ELEMENTWISE_SOP(tanh)
ELEMENTWISE_SOP(atan)
ELEMENTWISE_SOP(sqrt)
ELEMENTWISE_SOP(exp)
ELEMENTWISE_SOP(log)
ELEMENTWISE_SOP(log2)
ELEMENTWISE_SOP(log10)

#undef ELEMENTWISE_SOP

#define ELEMENTWISE_TOP(op) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i][j], c[i][j]), const TMAT& a, const TMAT& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b[i][j], c[i][j]), T a, const TMAT& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b, c[i][j]), const TMAT& a, T b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i][j], c), const TMAT& a, const TMAT& b, T c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b, c[i][j]), T a, T b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b[i][j], c), T a, const TMAT& b, T c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b, c), const TMAT& a, T b, T c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i], b[i][j], c[i][j]), const TVECC& a, const TMAT& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i], c[i][j]), const TMAT& a, const TVECC& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i][j], c[i]), const TMAT& a, const TMAT& b, const TVECC& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i], b[i], c[i][j]), const TVECC& a, const TVECC& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i], b[i][j], c[i]), const TVECC& a, const TMAT& b, const TVECC& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i], c[i]), const TMAT& a, const TVECC& b, const TVECC& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b[i], c[i][j]), T a, const TVECC& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i], b, c[i][j]), const TVECC& a, T b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b[i][j], c[i]), T a, const TMAT& b, const TVECC& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i], b[i][j], c), const TVECC& a, const TMAT& b, T c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b, c[i]), const TMAT& a, T b, const TVECC& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i], c), const TMAT& a, const TVECC& b, T c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[j], b[i][j], c[i][j]), const TVECR& a, const TMAT& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[j], c[i][j]), const TMAT& a, const TVECR& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[i][j], c[j]), const TMAT& a, const TMAT& b, const TVECR& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[j], b[j], c[i][j]), const TVECR& a, const TVECR& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[j], b[i][j], c[j]), const TVECR& a, const TMAT& b, const TVECR& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[j], c[j]), const TMAT& a, const TVECR& b, const TVECR& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b[j], c[i][j]), T a, const TVECR& b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[j], b, c[i][j]), const TVECR& a, T b, const TMAT& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a, b[i][j], c[j]), T a, const TMAT& b, const TVECR& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[j], b[i][j], c), const TVECR& a, const TMAT& b, T c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b, c[j]), const TMAT& a, T b, const TVECR& c) \
ELEMENTWISE_OP(op, TMAT, TMAT, op(a[i][j], b[j], c), const TMAT& a, const TVECR& b, T c)


ELEMENTWISE_TOP(clamp)
ELEMENTWISE_TOP(mix)
ELEMENTWISE_TOP(fma)

#undef ELEMENTWISE_TOP

#define INPLACE_OP(operation, type_b, expr) \
template <typename T, uint32_t R, uint32_t C> \
PE_HOST_DEVICE TMAT& operation(TMAT& a, type_b b) { \
	PE_UNROLL \
	for (uint32_t i = 0; i < C; ++i) { \
		PE_UNROLL \
		for (uint32_t j = 0; j < R; ++j) { \
			expr; \
		} \
	} \
	return a; \
}

INPLACE_OP(operator+=, const TMAT&, a[i][j] += b[i][j])
INPLACE_OP(operator+=, const TVECC&, a[i][j] += b[i])
INPLACE_OP(operator+=, const TVECR&, a[i][j] += b[j])
INPLACE_OP(operator+=, T, a[i][j] += b)

INPLACE_OP(operator*=, const TMAT&, a[i][j] *= b[i][j])
INPLACE_OP(operator*=, const TVECC&, a[i][j] *= b[i])
INPLACE_OP(operator*=, const TVECR&, a[i][j] *= b[j])
INPLACE_OP(operator*=, T, a[i][j] *= b)

INPLACE_OP(operator/=, const TMAT&, a[i][j] /= b[i][j])
INPLACE_OP(operator/=, const TVECC&, a[i][j] /= b[i])
INPLACE_OP(operator/=, const TVECR&, a[i][j] /= b[j])
INPLACE_OP(operator/=, T, a[i][j] /= b)

INPLACE_OP(operator-=, const TMAT&, a[i][j] -= b[i][j])
INPLACE_OP(operator-=, const TVECC&, a[i][j] -= b[i])
INPLACE_OP(operator-=, const TVECR&, a[i][j] -= b[j])
INPLACE_OP(operator-=, T, a[i][j] -= b)

#undef INPLACE_OP

#define REDUCTION_OP(operation, type_result, init, expr, ...) \
template <typename T, uint32_t R, uint32_t C> \
PE_HOST_DEVICE type_result operation(__VA_ARGS__) { \
	type_result result = init; \
	PE_UNROLL \
	for (uint32_t i = 0; i < C; ++i) { \
		PE_UNROLL \
		for (uint32_t j = 0; j < R; ++j) { \
			expr; \
		} \
	} \
	return result; \
}

REDUCTION_OP(operator==, bool, true,  result &= a[i][j] == b[i][j], const TMAT& a, const TMAT& b)
REDUCTION_OP(operator!=, bool, false, result |= a[i][j] != b[i][j], const TMAT& a, const TMAT& b)
REDUCTION_OP(isfinite, bool, true, result &= isfinite(a[i][j]), const TMAT& a)

#undef REDUCTION_OP

template <typename T>
PE_HOST_DEVICE T determinant(const matrix_t<T, 2, 2>& mat) {
  return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
}

template <typename T>
PE_HOST_DEVICE T determinant(const matrix_t<T, 3, 3>& mat) {
  return + mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) +
         - mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(2, 0) * mat(1, 2)) +
         + mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));
}

// [TDOO] test this might have made a mistake
template <typename T>
PE_HOST_DEVICE T determinant(const matrix_t<T, 4, 4>& mat) {
  return
    + mat(0, 0) * determinant(matrix_t<T, 3, 3>(mat(1, 1), mat(1, 2), mat(1, 3), mat(2, 1), mat(2, 2), mat(2, 3), mat(3, 1), mat(3, 2), mat(3, 3)))
    - mat(0, 1) * determinant(matrix_t<T, 3, 3>(mat(1, 0), mat(1, 2), mat(1, 3), mat(2, 0), mat(2, 2), mat(2, 3), mat(3, 0), mat(3, 2), mat(3, 3)))
    + mat(0, 2) * determinant(matrix_t<T, 3, 3>(mat(1, 0), mat(1, 1), mat(1, 3), mat(2, 0), mat(2, 1), mat(2, 3), mat(3, 0), mat(3, 1), mat(3, 3)))
    - mat(0, 3) * determinant(matrix_t<T, 3, 3>(mat(1, 0), mat(1, 1), mat(1, 2), mat(2, 0), mat(2, 1), mat(2, 2), mat(3, 0), mat(3, 1), mat(3, 2)));
}

template <typename T, uint32_t N>
PE_HOST_DEVICE matrix_t<T, N, N>& operator*=(matrix_t<T, N, N>& matrix, const matrix_t<T, N, N>& other) {
  matrix = matrix * other;
  return matrix;
} 

template <typename T, uint32_t R, uint32_t C>
PE_HOST_DEVICE T frobenious_norm(const matrix_t<T, R, C>& mat) {
  T result = 0;
  PE_UNROLL
  for (uint32_t i = 0; i < C; i++) {
    result += length2(mat[ i ]);
  }
  return sqrt(result);
}

template <typename T, uint32_t R, uint32_t C>
PE_HOST_DEVICE matrix_t<T, C, R> transpose(const matrix_t<T, R, C>& m) {
  matrix_t<T, C, R> result;
  PE_UNROLL
  for (uint32_t i = 0; i < C; i++) {
    PE_UNROLL
    for (uint32_t j = 0; j < R; j++) {
      result[ j ][ i ] = m[ i ][ j ];
    }
  }
  return result;
}

template <typename T, uint32_t R, uint32_t C>
PE_HOST_DEVICE vector_t<T, C> row(const matrix_t<T, R, C>& matrix, int row_index) {
  vector_t<T, C> result;
  PE_UNROLL
  for (uint32_t i = 0; i < C; i++) {
    result[ i ] = matrix[ i ][ row_index ];
  }
  return result;
}

template <typename T, uint32_t R, uint32_t C, size_t A>
PE_HOST_DEVICE matrix_t<T, R, C> replace_row(const matrix_t<T, R, C>& m, int r, const vector_t<T, C, A>& row) {
  matrix_t<T, R, C> result = m;
  PE_UNROLL
  for (uint32_t i = 0; i < C; i++) {
    result[ i ][ r ] = row[ i ];
  }
  return result;
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 2, 2> adjoint(const matrix_t<T, 2, 2>& m) {
  return {m(1, 1), -m(1, 0), -m(0, 1), m(0, 0)};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> adjoint(const matrix_t<T, 3, 3>& m) {
  const T m00 = determinant(matrix_t<T, 2, 2>{m(1,1), m(1,2), m(2,1), m(2,2)});
  const T m01 = determinant(matrix_t<T, 2, 2>{m(1,0), m(1,2), m(2,0), m(2,2)});
  const T m02 = determinant(matrix_t<T, 2, 2>{m(1,0), m(1,1), m(2,0), m(2,1)});

  const T m10 = determinant(matrix_t<T, 2, 2>{m(0,1), m(0,2), m(2,1), m(2,2)});
  const T m11 = determinant(matrix_t<T, 2, 2>{m(0,0), m(0,2), m(2,0), m(2,2)});
  const T m12 = determinant(matrix_t<T, 2, 2>{m(0,0), m(0,1), m(2,0), m(2,1)});

  const T m20 = determinant(matrix_t<T, 2, 2>{m(0.1), m(0,2), m(1,1), m(1,2)});
  const T m21 = determinant(matrix_t<T, 2, 2>{m(0,0), m(0,2), m(1,0), m(1,2)});
  const T m22 = determinant(matrix_t<T, 2, 2>{m(0,0), m(0,1), m(1,0), m(1,1)});
  
  return {
    m00, -m01, m02,
    -m10, m11, -m12,
    m20, -m21, m22
  };
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 4, 4> adjoint(const matrix_t<T, 4, 4>& m) {
	const T m00 = determinant(matrix_t<T, 3, 3>{m(1,1), m(2,1), m(3,1), m(1,2), m(2,2), m(3,2), m(1,3), m(2,3), m(3,3)});
	const T m01 = determinant(matrix_t<T, 3, 3>{m(0,1), m(2,1), m(3,1), m(0,2), m(2,2), m(3,2), m(0,3), m(2,3), m(3,3)});
	const T m02 = determinant(matrix_t<T, 3, 3>{m(0,1), m(1,1), m(3,1), m(0,2), m(1,2), m(3,2), m(0,3), m(1,3), m(3,3)});
	const T m03 = determinant(matrix_t<T, 3, 3>{m(0,1), m(1,1), m(2,1), m(0,2), m(1,2), m(2,2), m(0,3), m(1,3), m(2,3)});

	const T m10 = determinant(matrix_t<T, 3, 3>{m(1,0), m(2,0), m(3,0), m(1,2), m(2,2), m(3,2), m(1,3), m(2,3), m(3,3)});
	const T m11 = determinant(matrix_t<T, 3, 3>{m(0,0), m(2,0), m(3,0), m(0,2), m(2,2), m(3,2), m(0,3), m(2,3), m(3,3)});
	const T m12 = determinant(matrix_t<T, 3, 3>{m(0,0), m(1,0), m(3,0), m(0,2), m(1,2), m(3,2), m(0,3), m(1,3), m(3,3)});
	const T m13 = determinant(matrix_t<T, 3, 3>{m(0,0), m(1,0), m(2,0), m(0,2), m(1,2), m(2,2), m(0,3), m(1,3), m(2,3)});

	const T m20 = determinant(matrix_t<T, 3, 3>{m(1,0), m(2,0), m(3,0), m(1,1), m(2,1), m(3,1), m(1,3), m(2,3), m(3,3)});
	const T m21 = determinant(matrix_t<T, 3, 3>{m(0,0), m(2,0), m(3,0), m(0,1), m(2,1), m(3,1), m(0,3), m(2,3), m(3,3)});
	const T m22 = determinant(matrix_t<T, 3, 3>{m(0,0), m(1,0), m(3,0), m(0,1), m(1,1), m(3,1), m(0,3), m(1,3), m(3,3)});
	const T m23 = determinant(matrix_t<T, 3, 3>{m(0,0), m(1,0), m(2,0), m(0,1), m(1,1), m(2,1), m(0,3), m(1,3), m(2,3)});

	const T m30 = determinant(matrix_t<T, 3, 3>{m(1,0), m(2,0), m(3,0), m(1,1), m(2,1), m(3,1), m(1,2), m(2,2), m(3,2)});
	const T m31 = determinant(matrix_t<T, 3, 3>{m(0,0), m(2,0), m(3,0), m(0,1), m(2,1), m(3,1), m(0,2), m(2,2), m(3,2)});
	const T m32 = determinant(matrix_t<T, 3, 3>{m(0,0), m(1,0), m(3,0), m(0,1), m(1,1), m(3,1), m(0,2), m(1,2), m(3,2)});
	const T m33 = determinant(matrix_t<T, 3, 3>{m(0,0), m(1,0), m(2,0), m(0,1), m(1,1), m(2,1), m(0,2), m(1,2), m(2,2)});

	return {
		 m00, -m10,  m20, -m30,
		-m01,  m11, -m21,  m31,
		 m02, -m12,  m22, -m32,
		-m03,  m13, -m23,  m33,
	};
}

template <typename T, uint32_t N>
PE_HOST_DEVICE matrix_t<T, N, N> inverse(const matrix_t<T, N, N>& m) {
  return adjoint(m) / determinant(m);
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> skew_symmetric_matrix(const vector_t<T, 3>& v) {
  return {0, v[2], -v[1], -v[2], 0, v[0], v[1], -v[0], 0};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> cross_product_matrix(const vector_t<T, 3>& v) {
  return {0, v[2], -v[1], -v[2], 0, v[0], v[1], -v[0], 0};
}

template <typename T, uint32_t N>
PE_HOST_DEVICE matrix_t<T, N, N> eye() {
  return matrix<T, N, N>::identity();
}


#undef TVECC
#undef TVECR
#undef TMAT
#undef ELEMENTWISE_OP


// helper function to print matrix
template <typename T, uint32_t R, uint32_t C>
void print(const matrix_t<T, R, C>& mat ) {
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
