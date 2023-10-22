#ifndef AF1F8277_1C55_4D40_BD8A_1BD3DED792CE
#define AF1F8277_1C55_4D40_BD8A_1BD3DED792CE

#include "common.h"

PE_BEGIN

template <class...> struct conjunction : std::true_type {};
template <class B1> struct conjunction<B1> : B1 {};
template <class B1, class... Bn> struct conjunction<B1, Bn...> : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <uint32_t N, typename T, typename... Ts>
using enable_if_size_and_type_match_t = std::enable_if_t<sizeof...(Ts) == N && conjunction<std::is_same<Ts, T>...>::value>;

#define PE_VECTOR_BASE \
  using value_type = T; \
  \
  vector_t() = default; \
  \
  PE_HOST_DEVICE vector_t( T value ) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = value;\
    }\
  }\
  \
  PE_HOST_DEVICE vector_t( T* values ) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = values[i];\
    }\
  }\
  \
  template <typename U, uint32_t S, uint32_t A>\
  PE_HOST_DEVICE vector_t(const vector_t<U,S,A> &other) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = i < S ? (T) other[i] : (T)0;\
    }\
  }\
  \
  PE_HOST_DEVICE static constexpr vector_t<T, DIM> zeros() {\
    return vector_t<T, DIM>(0);\
  }\
  \
  PE_HOST_DEVICE static constexpr vector_t<T, DIM> ones() {\
    return vector_t<T, DIM>(1);\
  }\
  \
  PE_HOST_DEVICE T& operator[](uint32_t index) {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE const T& operator[](uint32_t index) const {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE T& operator()(uint32_t index) {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE const T& operator()(uint32_t index) const {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE T* data() {\
    return (T*) this;\
  }\
  \
  PE_HOST_DEVICE const T* data() const {\
    return (const T*) this;\
  }\
  \
  PE_HOST_DEVICE T* begin() {\
    return (T*) this;\
  }\
  \
  PE_HOST_DEVICE const T* begin() const {\
    return (const T*) this;\
  } \
  \
  PE_HOST_DEVICE T* end() {\
    return (T*) this + DIM;\
  }\
  \
  PE_HOST_DEVICE const T* end() const {\
    return (const T*) this + DIM;\
  }\
  \
  template <uint32_t start, uint32_t steps> \
  PE_HOST_DEVICE vector_t<T, steps>& slice() { \
    static_assert(start + steps <= DIM, "Slice out of bounds"); \
    return *(vector_t<T, steps>*)(data() + start); \
  } \
  \
  template <uint32_t start, uint32_t steps> \
  PE_HOST_DEVICE const vector_t<T, steps>& slice() const { \
    static_assert(start + steps <= DIM, "Slice out of bounds"); \
    return *(vector_t<T, steps>*)(data() + start); \
  } \
  PE_HOST_DEVICE static constexpr uint32_t size() { return DIM; } \
  PE_HOST_DEVICE static constexpr uint32_t alignment() { return ALIGNMENT; } \


template <typename T, uint32_t DIM, uint32_t ALIGNMENT=sizeof(T)>
struct alignas(ALIGNMENT) vector_t {

public:
  PE_VECTOR_BASE
  T elements[DIM];
  template <typename...Tn, typename = enable_if_size_and_type_match_t<DIM, T, Tn...>>
  PE_HOST_DEVICE vector_t(Tn... values) : elements{values...} {}
  PE_HOST_DEVICE vector_t(std::initializer_list<T> values) {
    assert(values.size() == DIM, "Initializer list size does not match vector size");
    std::copy(values.begin(), values.end(), elements);
  }
};

template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 1, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 1;
  PE_VECTOR_BASE  
  union {T x, r, t; };
};


template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 2, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 2;
  PE_VECTOR_BASE  
  union {T x, r; };
  union {T y, g; };
  PE_HOST_DEVICE vector_t(T a, T b) : x(a), y(b) {}
};

template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 3, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 3;
  PE_VECTOR_BASE  
  union {T x, r; };
  union {T y, g; };
  union {T z, b; };
  PE_HOST_DEVICE vector_t(T a, T b, T c) : x(a), y(b), z(c) {}
  template<size_t A> PE_HOST_DEVICE vector_t(vector_t<T, 2, A> a, T b) : x(a.x), y(a.y), z(b) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, vector_t<T, 2, A> b) : x(a), y(b.x), z(b.y) {}

};


template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 4, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 4;
  PE_VECTOR_BASE  
  union {T x, r; };
  union {T y, g; };
  union {T z, b; };
  union {T w, a; };

  PE_HOST_DEVICE vector_t(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
  template<size_t A> PE_HOST_DEVICE vector_t(vector_t<T, 2, A> a, T b, T c) : x(a.x), y(a.y), z(b), w(c) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, vector_t<T, 2, A> b, T c) : x(a), y(b.x), z(b.y), w(c) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, T b, vector_t<T, 2, A> c) : x(a), y(b), z(c.x), w(c.y) {}
  template<size_t A> PE_HOST_DEVICE vector_t(vector_t<T, 3, A> a, T b) : x(a.x), y(a.y), z(a.z), w(b) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, vector_t<T, 3, A> b) : x(a), y(b.x), z(b.y), w(b.z) {}
  template<size_t A, size_t B> PE_HOST_DEVICE vector_t(vector_t<T, 2, A> a, vector_t<T, 2, B> b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
};

#undef PE_VECTOR_BASE


template <typename T> PE_HOST_DEVICE T min(T a, T b) { return std::min(a, b); }
template <typename T> PE_HOST_DEVICE T max(T a, T b) { return std::max(a, b); }
template <typename T> PE_HOST_DEVICE T pow(T a, T b) { return std::pow(a, b); }
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
template <typename T> PE_HOST_DEVICE T sqrt(T a) { return std::sqrt(a); }
template <typename T> PE_HOST_DEVICE T exp(T a) { return std::exp(a); }
template <typename T> PE_HOST_DEVICE T log(T a) { return std::log(a); }
template <typename T> PE_HOST_DEVICE T exp2(T a) { return std::exp2(a); }
template <typename T> PE_HOST_DEVICE T log2(T a) { return std::log2(a); }
template <typename T> PE_HOST_DEVICE T log10(T a) { return std::log2(a) / std::log2(10); }
template <typename T> PE_HOST_DEVICE T clamp(T a, T b, T c) { return a < b ? b : (c < a ? c : a); }
template <typename T> PE_HOST_DEVICE T mix(T a, T b, T c) { return a * ((T)1 - c) + b * c; }

template <typename T> PE_HOST_DEVICE T isfinite(T a) {
#if defined(__CUDA_ARCH__)
  return ::isfinite(a);
#else
  return std::isfinite(a);
#endif
}

inline PE_HOST_DEVICE float fma(float a, float b, float c) { return fmaf(a, b, c); }


// define the elemtwise operations on vector
#define TVEC vector_t<T, D, A>
#define BVEC vector_t<bool, D, A>

#define ELEMENTWISE_OP(operator, result_type, expression, ...) \
template<typename T, uint32_t D, size_t A> \
PE_HOST_DEVICE result_type operator(__VA_ARGS__) { \
  result_type result; \
  PE_UNROLL \
  for (uint32_t ind = 0; ind < D; ind++) { \
    result[ind] = expression; \
  } \
  return result;\
}

ELEMENTWISE_OP(operator+, TVEC, a[ind] + b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator+, TVEC, a + b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator+, TVEC, a[ind] + b, const TVEC &a, T b)

ELEMENTWISE_OP(operator-, TVEC, a[ind] - b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator-, TVEC, a - b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator-, TVEC, a[ind] - b, const TVEC &a, T b)
ELEMENTWISE_OP(operator-, TVEC, -a[ind], const TVEC &a)

ELEMENTWISE_OP(operator*, TVEC, a[ind] * b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator*, TVEC, a * b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator*, TVEC, a[ind] * b, const TVEC &a, T b)

ELEMENTWISE_OP(operator/, TVEC, a[ind] / b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator/, TVEC, a / b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator/, TVEC, a[ind] / b, const TVEC &a, T b)

ELEMENTWISE_OP(min, TVEC, min(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(min, TVEC, min(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(min, TVEC, min(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(max, TVEC, max(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(max, TVEC, max(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(max, TVEC, max(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(pow, TVEC, pow(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(pow, TVEC, pow(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(pow, TVEC, pow(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(distance, TVEC, distance(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(distance, TVEC, distance(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(distance, TVEC, distance(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(copysign, TVEC, copysign(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(copysign, TVEC, copysign(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(copysign, TVEC, copysign(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(sign, TVEC, sign(a[ind]), const TVEC& a)
ELEMENTWISE_OP(floor, TVEC, floor(a[ind]), const TVEC& a)
ELEMENTWISE_OP(ceil, TVEC, ceil(a[ind]), const TVEC& a)
ELEMENTWISE_OP(abs, TVEC, abs(a[ind]), const TVEC& a)
ELEMENTWISE_OP(sin, TVEC, sin(a[ind]), const TVEC& a)
ELEMENTWISE_OP(asin, TVEC, asin(a[ind]), const TVEC& a)
ELEMENTWISE_OP(cos, TVEC, cos(a[ind]), const TVEC& a)
ELEMENTWISE_OP(acos, TVEC, acos(a[ind]), const TVEC& a)
ELEMENTWISE_OP(tan, TVEC, tan(a[ind]), const TVEC& a)
ELEMENTWISE_OP(atan, TVEC, atan(a[ind]), const TVEC& a)
ELEMENTWISE_OP(sqrt, TVEC, sqrt(a[ind]), const TVEC& a)
ELEMENTWISE_OP(exp, TVEC, exp(a[ind]), const TVEC& a)
ELEMENTWISE_OP(log, TVEC, log(a[ind]), const TVEC& a)
ELEMENTWISE_OP(log2, TVEC, log2(a[ind]), const TVEC& a)
ELEMENTWISE_OP(log10, TVEC, log10(a[ind]), const TVEC& a)
ELEMENTWISE_OP(isfinite, BVEC, isfinite(a[ind]), const TVEC& a)

ELEMENTWISE_OP(clamp, TVEC, clamp(a[ind], b[ind], c[ind]), const TVEC& a, const TVEC& b, const TVEC& c)
ELEMENTWISE_OP(clamp, TVEC, clamp(a[ind], b[ind], c), const TVEC& a, const TVEC& b, T c)
ELEMENTWISE_OP(clamp, TVEC, clamp(a[ind], b, c[ind]), const TVEC& a, T b, const TVEC& c)
ELEMENTWISE_OP(clamp, TVEC, clamp(a[ind], b, c), const TVEC& a, T b, T c)

ELEMENTWISE_OP(mix, TVEC, a[ind] * ((T)1 - c[ind]) + b[ind] * c[ind], const TVEC& a, const TVEC& b, const TVEC& c)
ELEMENTWISE_OP(mix, TVEC, a[ind] * ((T)1 - c) + b[ind] * c, const TVEC& a, const TVEC& b, T c)

ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b[ind], c[ind]), const TVEC& a, const TVEC& b, const TVEC& c)
ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b[ind], c), const TVEC& a, const TVEC& b, T c)
ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b, c[ind]), const TVEC& a, T b, const TVEC& c)
ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b, c), const TVEC& a, T b, T c)
ELEMENTWISE_OP(fma, TVEC, fma(a, b[ind], c[ind]), T a, const TVEC& b, const TVEC& c)
ELEMENTWISE_OP(fma, TVEC, fma(a, b[ind], c), T a, const TVEC& b, T c)
ELEMENTWISE_OP(fma, TVEC, fma(a, b, c[ind]), T a, T b, const TVEC& c)

template <typename T, uint32_t N, size_t A>
PE_DEVICE void atomic_add(T* dst, const vector_t<T, N, A>& a) {
	PE_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		atomicAdd(dst + i, a[i]);
	}
}

#undef ELEMENTWISE_OP

#define INPLACE_EOP(operator, out_type, expression) \
template<typename T, uint32_t D, size_t A> \
PE_HOST_DEVICE TVEC& operator(TVEC &a, out_type b) { \
  PE_UNROLL \
  for (uint32_t ind = 0; ind < D; ind++) { \
    expression; \
  } \
  return a; \
}

INPLACE_EOP(operator*=, const TVEC&, a[ind] *= b[ind])
INPLACE_EOP(operator/=, const TVEC&, a[ind] /= b[ind])
INPLACE_EOP(operator+=, const TVEC&, a[ind] += b[ind])
INPLACE_EOP(operator-=, const TVEC&, a[ind] -= b[ind])

INPLACE_EOP(operator*=, T, a[ind] *= b)
INPLACE_EOP(operator/=, T, a[ind] /= b)

#undef INPLACE_EOP

#define REDUCTION_EOP(operator, result_type, expression, initial_value, ...) \
template<typename T, uint32_t D, size_t A> \
PE_HOST_DEVICE result_type operator(__VA_ARGS__) { \
  result_type result = initial_value; \
  PE_UNROLL \
  for (uint32_t ind = 0; ind < D; ind++) { \
    expression; \
  } \
  return result;\
}

REDUCTION_EOP(dot, T, result += a[ind] * b[ind], (T) 0, const TVEC& a, const TVEC& b)
REDUCTION_EOP(sum, T, result += a[ind] + b[ind], (T) 0, const TVEC& a, const TVEC& b)
REDUCTION_EOP(mean, T, result += a[ind] / D, (T) 0, const TVEC& a)
REDUCTION_EOP(prod, T, result *= a[ind], (T) 1, const TVEC& a)
REDUCTION_EOP(product, T, result *= a[ind], (T) 1, const TVEC& a)
REDUCTION_EOP(min, T, min(result, a[ind]), a[0], const TVEC& a)
REDUCTION_EOP(max, T, max(result, a[ind]), a[0], const TVEC& a)
REDUCTION_EOP(length2, T, result += a[ind] * a[ind], (T) 0, const TVEC& a)
REDUCTION_EOP(squared_sum, T, result += a[ind] * a[ind], (T) 0, const TVEC& a)
REDUCTION_EOP(any, bool, result || a[ind], false, const BVEC& a)
REDUCTION_EOP(all, bool, result && a[ind], true, const BVEC& a)
REDUCTION_EOP(none, bool, result && !a[ind], true, const BVEC& a)
REDUCTION_EOP(has, bool, result || (a[ind] == b), false, const BVEC& a, T b)
REDUCTION_EOP(operator==, bool, result &= a[ind] == b[ind], true, const TVEC& a, const TVEC& b)
REDUCTION_EOP(operator!=, bool, result &= a[ind] != b[ind], true, const TVEC& a, const TVEC& b)
REDUCTION_EOP(close, bool, result &= distance(a[ind], b[ind]) < eps, true, const TVEC& a, const TVEC& b, T eps)

#undef REDUCTION_EOP

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE T length(const TVEC& a) {
  return sqrt(length2(a));
}

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE T distance(const TVEC& a, const TVEC& b) {
  return length(a - b);
}

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE TVEC normalize(const TVEC& a) {
  T len = length(a);
  
  if (len <= (T)0) {
    TVEC result{T(0)};
    result[0] = (T)1;
    return result;
  }

  return a / length(a);
}

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE TVEC cross(const TVEC& a, const TVEC& b) {
  static_assert(D == 3, "Cross product is only defined for 3D vectors");
  return TVEC(a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE TVEC reflect(const TVEC& a, const TVEC& n) {
  return a - (T)2 * dot(a, n) * n;
}

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE TVEC project(const TVEC& a, const TVEC& n) {
  return a - dot(a, n) * n;
}

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE TVEC refract(const TVEC& a, const TVEC& n, T eta) {
  T d = dot(a, n);
  T k = (T)1 - eta * eta * ((T)1 - d * d);
  if (k < (T)0) {
    return TVEC(T(0));
  }
  return eta * a - (eta * d + sqrt(k)) * n;
}

template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE TVEC faceforward(const TVEC& n, const TVEC& i, const TVEC& nref) {
	return n * -copysign((T)1, dot(i, nref));
}


template <typename T, uint32_t D, size_t A>
PE_HOST_DEVICE TVEC unit_vector(int direction) {
  TVEC result{T(0)};
  result[direction] = (T)1;
  return result;
}


/*
  Define the unit vectors in 3D space
*/
#define UNIT_VECTOR_3D(var_type, dir_name, direction) \
template <typename T, size_t A> \
PE_HOST_DEVICE vector_t<T, 3, A> var_type##dir_name() { \
  int dir = direction; \
  vector_t<T, 3, A> result{T(0)}; \
  result[dir - 1] = (T)1; \
  return result; \
}

#define VECTOR_SPACE_UNIT_VECTOR_3D(varname) \
UNIT_VECTOR_3D(varname, 1, 1) \
UNIT_VECTOR_3D(varname, x, 1) \
UNIT_VECTOR_3D(varname, 2, 2) \
UNIT_VECTOR_3D(varname, y, 2) \
UNIT_VECTOR_3D(varname, 3, 3) \
UNIT_VECTOR_3D(varname, z, 3) 

VECTOR_SPACE_UNIT_VECTOR_3D(u) // principal coordinates
VECTOR_SPACE_UNIT_VECTOR_3D(p) // positional coordinates
VECTOR_SPACE_UNIT_VECTOR_3D(v) // velocity
VECTOR_SPACE_UNIT_VECTOR_3D(a) // acceleration
VECTOR_SPACE_UNIT_VECTOR_3D(F) // force
VECTOR_SPACE_UNIT_VECTOR_3D(M) // moment

#undef VECTOR_SPACE_UNIT_VECTOR_3D
#undef UNIT_VECTOR_3D
#undef TVEC
#undef BVEC

#define NON_TEMPLATED_VECTOR_TYPES(name, T) \
template <uint32_t D> using name = vector_t<T, D>; \
template <uint32_t D> using aligned##name = vector_t<T, D, sizeof(T) * D>; \
template <uint32_t D> using a##name = vector_t<T, D, sizeof(T) * D>; \
using name##1 = vector_t<T, 1>; \
using name##2 = vector_t<T, 2>; \
using name##3 = vector_t<T, 3>; \
using name##4 = vector_t<T, 4>;

NON_TEMPLATED_VECTOR_TYPES(bvec, bool)
NON_TEMPLATED_VECTOR_TYPES(vec, float)
NON_TEMPLATED_VECTOR_TYPES(dvec, double)
NON_TEMPLATED_VECTOR_TYPES(ivec, int32_t)
NON_TEMPLATED_VECTOR_TYPES(uvec, uint32_t)
NON_TEMPLATED_VECTOR_TYPES(u16vec, uint16_t)
NON_TEMPLATED_VECTOR_TYPES(u64vec, uint64_t)
NON_TEMPLATED_VECTOR_TYPES(i16vec, int16_t)
NON_TEMPLATED_VECTOR_TYPES(i64vec, int64_t)

#if defined(__CUDACC__)
  NON_TEMPLATED_VECTOR_TYPES(hvec, __half)
#endif

#undef NON_TEMPLATED_VECTOR_TYPES


#if defined(__CUDACC__)
inline PE_HOST_DEVICE float4 to_float4(const vec4& x) { return {x.x, x.y, x.z, x.w}; }
inline PE_HOST_DEVICE float3 to_float3(const vec3& x) { return {x.x, x.y, x.z}; }
inline PE_HOST_DEVICE float2 to_float2(const vec2& x) { return {x.x, x.y}; }
inline PE_HOST_DEVICE vec4 to_vec4(const float4& x) { return {x.x, x.y, x.z, x.w}; }
inline PE_HOST_DEVICE vec3 to_vec3(const float3& x) { return {x.x, x.y, x.z}; }
inline PE_HOST_DEVICE vec2 to_vec2(const float2& x) { return {x.x, x.y}; }
#endif

PE_END

#endif /* AF1F8277_1C55_4D40_BD8A_1BD3DED792CE */
