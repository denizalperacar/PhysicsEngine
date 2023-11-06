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

  using value_type = T;
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

  template <size_t A=sizeof(T)>
  PE_HOST_DEVICE matrix_t<T, R, C>(const vector_t<T, C, A>& vec) {
    PE_UNROLL
    for (uint32_t i = 0; i < R; i++) {
      col[ i ] = vec;
    }
  }

  template<typename... Ts, typename = enable_if_size_and_type_match_t<R*C, T, Ts...>>
  PE_HOST_DEVICE matrix_t(Ts... coeffs) : data_{ coeffs... } {}

  PE_HOST_DEVICE matrix_t(std::initializer_list<T> coeffs) {
    PE_UNROLL
    for (int i = 0; i < R * C; i++) {
      data_[ i ] = (T)*(coeffs.begin() + i);
    }
  }

  PE_HOST_DEVICE matrix_t(const T* d) {
    PE_UNROLL
    for (uint32_t i = 0; i < R; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < C; j++) {
        col[ i ][ j ] = *(d++);
      }
    }
  }

  template <size_t A=sizeof(T)>
  PE_HOST_DEVICE matrix_t<T, R, C>(
      const vector_t<T, R, A>& vec1, 
      const vector_t<T, R, A>& vec2 ) {
    static_assert(C == 2, "matrix_t<T, R, C>(const vector_t<T, C, A>& vec1, const vector_t<T, C, A>& vec2) requires C == 2");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
  }

  template <size_t A=sizeof(T)>
  PE_HOST_DEVICE matrix_t<T, R, C>(
      const vector_t<T, R, A>& vec1, 
      const vector_t<T, R, A>& vec2, 
      const vector_t<T, R, A>& vec3 ) {
    static_assert(C == 3, "matrix_t<T, R, C>(const vector_t<T, C, A>& vec1, const vector_t<T, C, A>& vec2) requires C == 3");
    col[ 0 ] = vec1; 
    col[ 1 ] = vec2;
    col[ 2 ] = vec3;
  }

  template <size_t A=sizeof(T)>
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

// matrix vector multiplication
template<typename T, uint32_t R, uint32_t C>
PE_HOST_DEVICE vector_t<T, R> operator*(const matrix_t<T, R, C>& mat, const vector_t<T, C>& vec) {
  vector_t<T, R> result((T)0.0);
  PE_UNROLL
  for (uint32_t i = 0; i < C; i++) {
    PE_UNROLL
    for (uint32_t j = 0; j < R; j++) {
      result[j] += mat.col[i][j] * vec[i];
    }
  }
  return result;
}

// matrix matrix multiplication
template <typename T, uint32_t R, uint32_t C, uint32_t P>
PE_HOST_DEVICE matrix_t<T, R, P> operator*(const matrix_t<T, R, C>& mat, const matrix_t<T, C, P>& other) {
  matrix_t<T, R, P> result((T)0);
  PE_UNROLL
  for (uint32_t i = 0; i < P; i++) {
    result.col[ i ] = mat * other[ i ];
  }
  return result;
}

// matrix vector multiplication
template<typename T, uint32_t R, uint32_t C>
PE_HOST_DEVICE vector_t<T, R> operator*(const vector_t<T, R>& vec, const matrix_t<T, R, C>& mat) {
  vector_t<T, C> result((T)0.0);
  PE_UNROLL
  for (uint32_t i = 0; i < C; i++) {
    result[i] = dot(mat.col[i] * vec);
  }
  return result;
}

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


//ELEMENTWISE_OP(operator*, TMAT, TMAT, a[i][j] * b[i][j], const TMAT& a, const TMAT& b)
ELEMENTWISE_OP(operator*, TMAT, TMAT, a * b[i][j], T a, const TMAT& b)
ELEMENTWISE_OP(operator*, TMAT, TMAT, b * a[i][j], const TMAT& a, T b)
//ELEMENTWISE_OP(operator*, TMAT, TMAT, b[i] * a[i][j], const TMAT& a, const TVECC& b)
//ELEMENTWISE_OP(operator*, TMAT, TMAT, b[i] * b[i][j], const TVECC& a, const TMAT& b)
//ELEMENTWISE_OP(operator*, TMAT, TMAT, b[j] * a[i][j], const TMAT& a, const TVECR& b)
//ELEMENTWISE_OP(operator*, TMAT, TMAT, b[j] * b[i][j], const TVECR& a, const TMAT& b)

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
ELEMENTWISE_FOP(atan2)

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

template <typename T, uint32_t R, uint32_t C, size_t A=sizeof(T)>
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
  return matrix_t<T, N, N>::identity();
}


template <typename T, size_t A=sizeof(T)>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotmat(T angle, const vector_t<T, 3, A>& axis) {
  T s = sin(angle);
  T c = cos(angle);
  T oc = (T)1 - c;

  return {
		oc * axis.x * axis.x + c,          oc * axis.x * axis.y + axis.z * s, oc * axis.z * axis.x - axis.y * s,
		oc * axis.x * axis.y - axis.z * s, oc * axis.y * axis.y + c,          oc * axis.y * axis.z + axis.x * s,
		oc * axis.z * axis.x + axis.y * s, oc * axis.y * axis.z - axis.x * s, oc * axis.z * axis.z + c,
  };
} 

template <typename T, size_t A=sizeof(T)>
PE_HOST_DEVICE matrix_t<T, 3, 3> rotmat(const vector_t<T, 3, A>& v) {
  T angle = length(v);
  if (angle == 0) {
    return matrix_t<T, 3, 3>::identity();
  }
  
  return rotmat(angle, v);
}

template <typename T, uint32_t N>
PE_HOST_DEVICE matrix_t<T, N, N> mat_sqrt(const matrix_t<T, N, N>& m, T epsilon = (T)1e-10f) {
  matrix_t<T, N, N> A = m - matrix_t<T, N, N>::identity();
  matrix_t<T, N, N> Z = A;
  matrix_t<T, N, N> X = A;

  for (uint32_t i = 0; i < 32; i++) {
    if (frobenious_norm(Z) < epsilon) {
      return X;
    }

    Z = Z * A;
    X += ((T)1 / (T)i) * Z;
  }

  return X;
}

template <typename T, uint32_t N>
PE_HOST_DEVICE matrix_t<T, N, N> mat_exp_pade(const matrix_t<T, N, N>& m) {
	// Pade approximation with scaling; same as Matlab.
	// Pseudocode translated from Hawkins and Grimm [2007]
	matrix_t<T, N, N> mX = matrix_t<T, N, N>::identity();
  matrix_t<T, N, N> mD = matrix_t<T, N, N>::identity();
  matrix_t<T, N, N> mN = matrix_t<T, N, N>::identity();

	T c = (T)1;
	constexpr uint32_t q = 6; // Matlab's default when using this algorithm

	T s = -(T)1;
	for (uint32_t k = 1; k <= q; ++k) {
		c = c * (q - k + 1) / (k * (2 * q - k + 1));
		mX = m * mX;
		auto cmX = c * mX;
		mN = mN + cmX;
		mD = mD + s * cmX;
		s = -s;
	}

	return inverse(mD) * mN;
}

template <typename T, uint32_t N>
PE_HOST_DEVICE matrix_t<T, N, N> mat_log(const matrix_t<T, N, N>& m) {
	matrix_t<T, N, N> result(m);

	uint32_t j = 0;
	for (; j < 32; ++j) {
		if (frobenius_norm(result - matrix_t<T, N, N>::identity()) < (T)1e-5f) {
			break;
		}

		result = mat_sqrt(result);
	}

	result = mat_log_hawkins(result);
	return (T)scalbnf(1.0f, j) * result;
}

template <typename T, uint32_t N>
PE_HOST_DEVICE matrix_t<T, N, N> mat_exp(const matrix_t<T, N, N>& m) {
	uint32_t N_SQUARING = max(0, 1 + (int)floor(log2(frobenius_norm(m))));

	matrix_t<T, N, N> result = (T)scalbnf(1.0f, -N_SQUARING) * m;
	result = mat_exp_pade(result);

	for (uint32_t i = 0; i < N_SQUARING; ++i) {
		result *= result;
	}

	return result;
}


template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> orthogonalize(const matrix_t<T, 3, 3>& m) {
	return matrix_t<T, 3, 3>{
		(T)0.5f * ((T)3 - dot(m[0], m[0])) * m[0],
		(T)0.5f * ((T)3 - dot(m[1], m[1])) * m[1],
		(T)0.5f * ((T)3 - dot(m[2], m[2])) * m[2],
	};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> so3_log(const matrix_t<T, 3, 3>& m) {
	T tr = clamp(m[0][0] + m[1][1] + m[2][2], -(T)1 + std::numeric_limits<T>::epsilon(), (T)1);
	T radians = acosf((tr - (T)1) / (T)2);
	return radians / sqrt(((T)1 + tr) * ((T)3 - tr)) * (m - transpose(m));
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 3, 3> so3_exp(const matrix_t<T, 3, 3>& m) {
	vector_t<T, 3> axis = {-m[2][1], m[2][0], -m[1][0]};
	T radians_sq = length2(axis);
	if (radians_sq == (T)0) {
		return matrix_t<T, 3, 3>::identity();
	}

	T radians = sqrt(radians_sq);
	return matrix_t<T, 3, 3>::identity() + (sin(radians) / radians) * m + (((T)1 - cos(radians)) / radians_sq) * (m * m);
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 4, 3> se3_log(const matrix_t<T, 4, 3>& m) {
	auto omega = so3_log(matrix_t<T, 3, 3>(m));
	vector_t<T, 3> axis = {-omega[2][1], omega[2][0], -omega[1][0]};
	T radians_sq = length2(axis);
	auto inv_trans = matrix_t<T, 3, 3>::identity();
	if (radians_sq > (T)0) {
		T radians = sqrt(radians_sq);
		inv_trans += -(T)0.5 * omega + (((T)1 - (T)0.5 * radians * cos((T)0.5 * radians) / sin((T)0.5 * radians)) / radians_sq) * (omega * omega);
	}

	return {omega[0], omega[1], omega[2], inv_trans * m[3]};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 4, 3> se3_exp(const matrix_t<T, 4, 3>& m) {
	matrix_t<T, 3, 3> omega = m;
	vector_t<T, 3> axis = {-omega[2][1], omega[2][0], -omega[1][0]};
	T radians_sq = length2(axis);
	auto trans = matrix_t<T, 3, 3>::identity();
	if (radians_sq > (T)0) {
		T radians = sqrt(radians_sq);
		trans += (((T)1 - cos(radians)) / radians_sq) * omega + ((radians - sin(radians)) / (radians * radians_sq)) * (omega * omega);
	}

	auto rot = so3_exp(omega);
	return {rot[0], rot[1], rot[2], trans * m[3]};
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 4, 4> se3_log(const matrix_t<T, 4, 4>& m) {
	auto result = matrix_t<T, 4, 4>(se3_log(matrix_t<T, 4, 3>(m)));
	result[3][3] = (T)0;
	return result;
}

template <typename T>
PE_HOST_DEVICE matrix_t<T, 4, 4> se3_exp(const matrix_t<T, 4, 4>& m) {
	return matrix_t<T, 4, 4>(se3_exp(matrix_t<T, 4, 3>(m)));
}


#define DEF_NON_TEMPLATED_MATRIX_TYPES(name, T) \
template <uint32_t R, uint32_t C> \
using name = matrix_t<T, R, C>; \
using name##4x4 = name<4, 4>; \
using name##4x3 = name<4, 3>; \
using name##4x2 = name<4, 2>; \
using name##3x4 = name<3, 4>; \
using name##3x3 = name<3, 3>; \
using name##3x2 = name<3, 2>; \
using name##2x4 = name<2, 4>; \
using name##2x3 = name<2, 3>; \
using name##2x2 = name<2, 2>; \
using name##4 = name##4x4; \
using name##3 = name##3x3; \
using name##2 = name##2x2;

DEF_NON_TEMPLATED_MATRIX_TYPES(mat, float)
DEF_NON_TEMPLATED_MATRIX_TYPES(dmat, double)
#if defined(__CUDACC__)
DEF_NON_TEMPLATED_MATRIX_TYPES(hmat, __half)
#endif

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
